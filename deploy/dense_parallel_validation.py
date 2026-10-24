"""Real-checkpoint Gemma/Qwen attention and dense FFN tensor parallelism."""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.distributed as dist

from deploy.jit_cache import ensure_kernel_cache_dirs, install_flashinfer_hash_cache
from resin import (
    BatchPhase,
    Engine,
    Parallelism,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups


class _Observed:
    runner_config: RunnerConfig

    def __init__(self, runner):
        self.runner_config = runner.runner_config
        self.runner, self.calls, self.released = runner, [], []

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def forward(self, plan):
        self.calls.append(
            dict(
                phase=str(plan.phase),
                sequences=plan.num_sequences,
                logprobs=plan.needs_logprobs,
            )
        )
        return self.runner.forward(plan)

    def release(self, sequence_ids):
        self.released.extend(map(int, sequence_ids))
        self.runner.release(sequence_ids)


def _runner(model, model_path, assistant_path, rank, groups, spec, graphs):
    base: Any
    if model == "gemma4":
        from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

        base = Gemma4CudaRunner
        options = dict(
            cache_parent_capacity=128,
            assistant_model=assistant_path if spec else None,
        )
        width = 8
    elif model == "qwen3_5":
        from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

        base = Qwen3_5CudaRunner
        options = dict(max_seq_len=512, max_batch_size=2, cache_slab_capacity=128)
        width = 512
    else:
        raise ValueError(f"unknown dense validation model: {model}")

    class BoundedRunner(cast(type[Any], base)):
        def _configure_prefill_graphs(self, operations, **kwargs):
            executor = self._require_cuda_graph_executor()
            register = executor.add_mode

            def bounded(name, mode):
                register(
                    name,
                    replace(
                        mode, shapes=tuple(s for s in mode.shapes if s[0] in (32, 64))
                    ),
                )

            # Bound startup cost without changing the production graph catalog.
            with patch.object(executor, "add_mode", bounded):
                callbacks = super()._configure_prefill_graphs(operations, **kwargs)
            if graphs:
                self._prefill_tensor_graphs["prefill"].buckets = (32, 64)
            return callbacks

    return BoundedRunner(
        model_path,
        device=f"cuda:{rank}",
        parallel_groups=groups,
        speculative_tokens=spec,
        cuda_graph_prefill="breakable" if graphs else "disabled",
        cuda_graph_decode="full" if graphs else "disabled",
        cuda_graph_decode_shapes=((1, width), (2, width)) if graphs else (),
        **options,
    )


def _score(value):
    if value is None:
        return None
    return dict(
        logprob=float(value.logprob),
        top=[list(item) for item in value.top_logprobs],
        targets=[list(item) for item in value.target_logprobs],
    )


def _drain(engine, events, *, after_step=None):
    steps = 0
    while engine.has_work():
        for event in engine.step(allow_overlap=False):
            events.setdefault(int(event.request_id), []).append(
                dict(
                    token=int(event.token_id),
                    index=int(event.index),
                    score=_score(event.logprobs),
                )
            )
        steps += 1
        if steps > 96:
            raise RuntimeError("dense validation exceeded 96 steps")
        if after_step is not None:
            after_step(steps)
    return steps


def _workload(engine, prompts):
    events, requests, requested_scores = {}, {}, {}
    sampling = SamplingParams(temperature=0.0, logprobs=True, top_logprobs=3)
    unscored = SamplingParams(temperature=0.0)

    def submit(name, prompt, tokens=8, *, scored=True, **kwargs):
        request = engine.submit(
            prompt,
            max_new_tokens=tokens,
            sampling_params=sampling if scored else unscored,
            **kwargs,
        )
        requests[name] = request
        requested_scores[name] = scored
        return request

    def cancel(request):
        if not engine.cancel(request) or not request.cancelled or not request.done:
            raise AssertionError("cancellation did not finish and release the request")

    submit("short", prompts["short"])
    submit("long", prompts["long"])
    queued = engine.submit(prompts["long"], max_new_tokens=8, sampling_params=sampling)
    cancel(queued)
    _drain(engine, events)

    # Every driver exercises a real two-row decode, then reuses released pages.
    submit("batch_short", prompts["short"])
    submit("batch_long", prompts["long"])
    _drain(engine, events)
    # These runners intentionally use eager decode for generation logprobs.
    # Separate unscored requests must exercise captured decode at batch two.
    submit("graph_short", prompts["short"], scored=False)
    submit("graph_long", prompts["long"], scored=False)
    _drain(engine, events)
    active = engine.submit(prompts["long"], max_new_tokens=8, sampling_params=sampling)
    _drain(
        engine, events, after_step=lambda step: cancel(active) if step == 1 else None
    )
    scored = engine.submit(
        prompts["long"],
        max_new_tokens=0,
        sampling_params=SamplingParams(
            temperature=0.0, prompt_logprobs=True, prompt_top_logprobs=3
        ),
        prompt_target_logprobs=[[], *[[token, 0] for token in prompts["long"][1:]]],
    )
    _drain(engine, events)
    local = getattr(engine, "local", engine)
    if local.cache_free_parent_count != local.cache_parent_capacity:
        raise AssertionError("completed and cancelled requests leaked cache parents")
    outputs = {}
    for name, request in requests.items():
        output = list(request.take_result())
        trace = events[int(request.request_id)]
        if len(output) != 8 or output != [row["token"] for row in trace]:
            raise AssertionError("streaming events differ from completed token IDs")
        if [row["index"] for row in trace] != list(range(8)):
            raise AssertionError("streaming token indices are not contiguous")
        if any((row["score"] is not None) != requested_scores[name] for row in trace):
            raise AssertionError("generation logprob availability differs from request")
        outputs[name] = dict(tokens=output, events=trace)
    prompt_scores = scored.prompt_logprobs
    if prompt_scores is None or len(prompt_scores) != len(prompts["long"]):
        raise AssertionError("prompt scoring omitted positions")
    if prompt_scores[0] is not None or any(
        score is None for score in prompt_scores[1:]
    ):
        raise AssertionError("prompt scoring lost chunk-boundary predecessor state")
    if scored.take_result():
        raise AssertionError("prompt-only scoring generated output tokens")
    return dict(
        outputs=outputs,
        prompt_scores=[_score(value) for value in prompt_scores],
        cancelled_queued=queued.cancelled,
        cancelled_active=active.cancelled,
        free_parents=local.cache_free_parent_count,
        capacity=local.cache_parent_capacity,
    )


def _rank(rank, port, directory, model, model_path, assistant_path, prompts, case):
    torch.cuda.set_device(rank)
    ensure_kernel_cache_dirs()
    install_flashinfer_hash_cache()
    groups = runner = engine = None
    try:
        if case["attn_tp"] is not None:
            os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
            dist.init_process_group(
                "nccl",
                rank=rank,
                world_size=2,
                timeout=timedelta(minutes=10),
                device_id=torch.device(f"cuda:{rank}"),
            )
            groups = ParallelGroups.create(
                Parallelism(world_size=2, attn_tp=case["attn_tp"])
            )
        started = time.monotonic()
        runner = _runner(
            model,
            model_path,
            assistant_path,
            rank,
            groups,
            case["spec"],
            case["graphs"],
        )
        print(
            f"[dense-validation] {model} {case['name']} rank {rank} loaded", flush=True
        )
        if case["graphs"]:
            runner.capture_cuda_graphs()
            print(
                f"[dense-validation] {model} {case['name']} rank {rank} captured",
                flush=True,
            )
        observed = _Observed(runner)
        options: dict[str, Any] = dict(
            max_running_requests=2,
            context_length=512,
            eos_token_id=None,
            prefix_cache=False,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=64, prefill_chunk_size=32
            ),
        )
        worker = None if groups is None else DistributedWorker(observed, groups)
        if groups is None or groups.is_driver:
            engine = (
                Engine(observed, **options)
                if worker is None
                else DistributedEngine(worker, **options)
            )
            output = _workload(engine, prompts)
            engine.close()
            engine = None
        else:
            assert worker is not None
            worker.serve()
            output = None
        torch.cuda.synchronize()
        stats = runner.cuda_graph_stats
        if groups is not None:
            dist.barrier()
        Path(directory, f"{case['name']}-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    elapsed_s=time.monotonic() - started,
                    output=output,
                    calls=observed.calls,
                    released_ids=observed.released,
                    graphs=stats,
                    speculation=runner.speculative_stats,
                )
            )
        )
    finally:
        if engine is not None:
            engine.close()
        if runner is not None:
            runner.close_cuda_graphs()
        if groups is not None:
            groups.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _error_stats(values):
    return dict(
        count=len(values),
        mean=math.fsum(values) / len(values) if values else None,
        max=max(values, default=None),
    )


def _numerical_comparison(actual, expected):
    """Describe arithmetic differences; this is not a model-quality evaluation."""
    generation = {}
    for name, result in actual["outputs"].items():
        reference = expected["outputs"][name]
        pairs = list(zip(result["tokens"], reference["tokens"], strict=True))
        changed = [position for position, (a, b) in enumerate(pairs) if a != b]
        prefix = changed[0] if changed else len(pairs)
        errors = [
            abs(row["score"]["logprob"] - original["score"]["logprob"])
            for row, original in zip(
                result["events"][:prefix], reference["events"][:prefix], strict=True
            )
            if row["score"] is not None
        ]
        generation[name] = dict(
            token_ids=result["tokens"],
            reference_token_ids=reference["tokens"],
            changed_positions=changed,
            matched_prefix_tokens=prefix,
            matched_prefix_observed_logprob_error=_error_stats(errors),
        )
    observed, targets, top1_changed = [], [], []
    for position, (score, original) in enumerate(
        zip(actual["prompt_scores"], expected["prompt_scores"], strict=True)
    ):
        if score is None:
            continue
        observed.append(abs(score["logprob"] - original["logprob"]))
        targets.extend(
            abs(a[1] - b[1])
            for a, b in zip(score["targets"], original["targets"], strict=True)
        )
        if score["top"][0][0] != original["top"][0][0]:
            top1_changed.append(position)
    return dict(
        baseline_equivalent=actual == expected,
        generation=generation,
        prompt_observed_logprob_error=_error_stats(observed),
        prompt_target_logprob_error=_error_stats(targets),
        prompt_top1_changed_positions=top1_changed,
    )


def _validate_record(
    record, *, spec: int, graphs: bool = False, prompt_tokens: list[int] | None = None
):
    """Check coverage and requested score fields, also for retained GPU JSON."""
    if not any(
        call["phase"] == str(BatchPhase.DECODE)
        and call["sequences"] == 2
        and not call["logprobs"]
        for call in record["calls"]
    ):
        raise AssertionError("workload did not execute an unscored two-row decode")
    if graphs:
        stats = record["graphs"]
        if any(stats[mode]["replay_count"] == 0 for mode in ("prefill", "decode")):
            raise AssertionError("workload did not replay prefill and decode graphs")
        if not any(
            shape["shape"][0] == 2 and shape["replay_count"] > 0
            for shape in stats["decode"]["shape_stats"]
        ):
            raise AssertionError("workload did not replay a two-row decode graph")
        unexpected = {
            reason: count
            for reason, count in stats["decode"]["fallback_counts"].items()
            if count and reason != "logprobs"
        }
        if unexpected:
            raise AssertionError(f"unexpected decode eager fallback: {unexpected}")
    counters = record["speculation"]
    if counters["draft_tokens"] != spec:
        raise AssertionError("runner used the wrong speculative length")
    if spec and (counters["blocks"] <= 0 or counters["drafts"] <= 0):
        raise AssertionError("speculative workload did not verify any drafts")
    if not spec and counters["drafts"]:
        raise AssertionError("non-speculative workload verified drafts")
    output = record["output"]
    if output is None:
        return
    if not output["cancelled_queued"] or not output["cancelled_active"]:
        raise AssertionError("workload did not cancel queued and active requests")
    if output["free_parents"] != output["capacity"]:
        raise AssertionError("workload leaked cache parents")
    unscored = {"graph_short", "graph_long"}
    if set(output["outputs"]) != {
        "short",
        "long",
        "batch_short",
        "batch_long",
        *unscored,
    }:
        raise AssertionError("workload omitted a generation request")

    def check_top(score):
        values = [
            score["logprob"],
            *[v for _, v in score["top"]],
            *[v for _, v in score["targets"]],
        ]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("nonfinite generation or prompt score")
        top = score["top"]
        if len(top) != 3 or len({row[0] for row in top}) != 3:
            raise AssertionError("requested three distinct top-logprob entries")

    for name, result in output["outputs"].items():
        events = result["events"]
        if (
            len(result["tokens"]) != 8
            or result["tokens"] != [event["token"] for event in events]
            or [event["index"] for event in events] != list(range(8))
        ):
            raise AssertionError("generation tokens and streaming events disagree")
        for event in result["events"]:
            score = event["score"]
            if name in unscored:
                if score is not None:
                    raise AssertionError(
                        "unscored request unexpectedly returned scores"
                    )
                continue
            check_top(score)
            if score["targets"]:
                raise AssertionError("generation unexpectedly returned prompt targets")
    scores = output["prompt_scores"]
    if not scores or scores[0] is not None or any(s is None for s in scores[1:]):
        raise AssertionError("prompt scores omitted a predecessor or chunk boundary")
    if prompt_tokens is not None and len(scores) != len(prompt_tokens):
        raise AssertionError("prompt score count differs from prompt token count")
    for position, score in enumerate(scores[1:], start=1):
        check_top(score)
        targets = score["targets"]
        if len(targets) != 2 or targets[1][0] != 0:
            raise AssertionError("prompt target IDs omitted a requested position")
        if prompt_tokens is not None and targets[0][0] != prompt_tokens[position]:
            raise AssertionError("prompt target IDs differ from requested token IDs")
        if not math.isclose(targets[0][1], score["logprob"], abs_tol=1e-6):
            raise AssertionError("observed-token score differs from its target score")


def _cases(speculative: bool) -> list[dict[str, Any]]:
    cases = [
        dict(name="baseline", attn_tp=None, spec=0, graphs=False),
        dict(name="single-graphs-spec0", attn_tp=None, spec=0, graphs=True),
    ]
    for tp in (2, 1):
        cases.append(
            dict(
                name=f"attn-tp{tp}-ffn-tp2-eager-spec0",
                attn_tp=tp,
                spec=0,
                graphs=False,
            )
        )
        cases.extend(
            dict(
                name=f"attn-tp{tp}-ffn-tp2-spec{spec}",
                attn_tp=tp,
                spec=spec,
                graphs=True,
            )
            for spec in ((0, 2) if speculative else (0,))
        )
    return cases


def evaluate_records(records, prompts):
    """Validate execution and graph equivalence, then report TP/spec arithmetic."""
    for case in _cases(any(name.endswith("spec2") for name in records)):
        values = records[case["name"]]
        if [value["rank"] for value in values] != [0, 1]:
            raise AssertionError("case must contain both GPU ranks in order")
        for value in values:
            _validate_record(
                value,
                spec=case["spec"],
                graphs=case["graphs"],
                prompt_tokens=prompts["long"],
            )
    baseline = records["baseline"]
    if baseline[0]["output"] != baseline[1]["output"]:
        raise AssertionError("independent eager baseline ranks disagree")
    pairs = [("single-graphs-spec0", "baseline")]
    pairs.extend(
        (f"attn-tp{tp}-ffn-tp2-spec0", f"attn-tp{tp}-ffn-tp2-eager-spec0")
        for tp in (1, 2)
    )
    for graph, eager in pairs:
        for actual, expected in zip(records[graph], records[eager], strict=True):
            if actual["output"] != expected["output"]:
                raise AssertionError(
                    f"same-topology graph/eager outputs differ: {graph}"
                )
    comparisons, speculative = {}, {}
    for name, values in records.items():
        comparisons[name] = _numerical_comparison(
            values[0]["output"], baseline[0]["output"]
        )
        if name.endswith("spec2"):
            speculative[name] = _numerical_comparison(
                values[0]["output"],
                records[name.removesuffix("spec2") + "spec0"][0]["output"],
            )
    return dict(
        execution_passed=True,
        same_topology_graph_exact=True,
        baseline_equivalent=all(
            value["baseline_equivalent"] for value in comparisons.values()
        ),
        numerical_stats=comparisons,
        speculative_vs_non_speculative=speculative,
    )


def run_validation(model: str, model_path: str, assistant_path: str | None = None):
    """Fresh processes isolate every Engine/cache binding and speculation mode."""
    from transformers import AutoTokenizer

    if torch.cuda.device_count() < 2:
        raise ValueError("dense parallel validation requires two GPUs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer is None:
        raise RuntimeError("checkpoint tokenizer could not be loaded")
    tokens = tokenizer.encode(
        "The capital of France is Paris. The capital of Japan is Tokyo. " * 12
    )
    prompts = {"short": tokens[:17], "long": tokens[:49]}
    cases = _cases(model == "qwen3_5" or assistant_path is not None)
    records = {}
    with tempfile.TemporaryDirectory(prefix="resin-dense-parallel-") as directory:
        for case in cases:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            print(f"[dense-validation] {model} {case['name']}", flush=True)
            processes = torch.multiprocessing.spawn(
                _rank,
                args=(
                    port,
                    directory,
                    model,
                    str(model_path),
                    assistant_path,
                    prompts,
                    case,
                ),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 20 * 60
            try:
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{case['name']} exceeded 20 minutes")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
            values = [
                json.loads(Path(directory, f"{case['name']}-{rank}.json").read_text())
                for rank in range(2)
            ]
            records[case["name"]] = values
            Path(f"/tmp/{model}-dense-tp-cases.json").write_text(
                json.dumps(records, indent=2)
            )
            for value in values:
                _validate_record(
                    value,
                    spec=cast(int, case["spec"]),
                    graphs=cast(bool, case["graphs"]),
                    prompt_tokens=prompts["long"],
                )
    return dict(
        model=model,
        prompts=prompts,
        cases=records,
        **evaluate_records(records, prompts),
    )
