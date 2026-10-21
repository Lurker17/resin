"""Real-checkpoint GPT-OSS EP validation against independent GPU baselines."""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from deploy.graph_validation import (
    prefill_plan_metadata,
    validate_prefill_attention,
    validate_prefill_attention_regions,
)
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

CACHE_PARENTS = 128
CONTEXT_LENGTH = 512
PREFILL_CHUNK_SIZE = 32
GENERATION_TOKENS = 8


def _initialize(rank, port, world_size):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
        device_id=torch.device(f"cuda:{rank}"),
    )
    ensure_kernel_cache_dirs()
    install_flashinfer_hash_cache()


def _new_runner(rank, model_path, *, groups=None, graphs=False):
    from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
    from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

    class ValidationGptRunner(GptOssCudaRunner):
        prefill_chunk_size = PREFILL_CHUNK_SIZE

    options = {} if groups is None else {"parallel_groups": groups}
    if graphs:
        options["cuda_graph_decode_shapes"] = ((1, 4), (2, 4))
    return ValidationGptRunner(
        model_path,
        device=f"cuda:{rank}",
        cache_parent_capacity=CACHE_PARENTS,
        kernel_moe=FlashInferSM90MXFP4Kernel(),
        fast_allreduce=False,
        cuda_graph_prefill="breakable" if graphs else "disabled",
        cuda_graph_decode="full" if graphs else "disabled",
        **options,
    )


def _sampling():
    return SamplingParams(temperature=0.0, logprobs=True, top_logprobs=3)


def _engine_options():
    return {
        "max_running_requests": 2,
        "context_length": CONTEXT_LENGTH,
        "eos_token_id": None,
        "scheduler_config": SchedulerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
        ),
    }


def _drain(engine, *, inject=None):
    events = []
    steps = 0
    while engine.has_work():
        for event in engine.step():
            score = event.logprobs
            events.append(
                {
                    "request_id": int(event.request_id),
                    "token_id": int(event.token_id),
                    "index": int(event.index),
                    "logprob": None if score is None else float(score.logprob),
                    "top_logprobs": [] if score is None else score.top_logprobs,
                }
            )
        steps += 1
        if steps > 128:
            raise RuntimeError("GPT-OSS validation exceeded 128 coordinated steps")
        if inject is not None:
            inject(steps)
    return {"steps": steps, "events": events}


class _Capture:
    def __init__(self, runner, *, expert=False):
        self.runner = runner
        self.original_sampler = None
        self.history = []
        self.logits = None
        self.expert = None
        self.expected_tokens = None
        self.force_tokens = True

        def head_hook(_module, _inputs, output):
            self.logits = output.detach().float().cpu()

        self.hooks = [runner._model.lm_head.register_forward_hook(head_hook)]
        if expert:

            def expert_hook(_module, inputs, output):
                if self.expert is None:
                    self.expert = {
                        name: tensor.detach().cpu()
                        for name, tensor in zip(
                            ("hidden", "indices", "weights", "logits"),
                            inputs,
                            strict=True,
                        )
                    }
                    self.expert["output"] = output.detach().cpu()

            self.hooks.append(
                runner._model.model.layers[0].mlp.experts.register_forward_hook(
                    expert_hook
                )
            )

    def record(self, logits, plan, tokens):
        self.logits = logits.detach().float().cpu()
        if (
            self.original_sampler is not None
            and plan is not None
            and plan.num_sequences == 1
            and plan.sampling_mask[0]
        ):
            index = len(self.history)
            self.history.append(self.logits.clone())
            if self.expected_tokens is not None:
                if index >= len(self.expected_tokens):
                    raise RuntimeError("decode exceeded baseline history")
                if not self.force_tokens:
                    ids = getattr(tokens, "token_ids", tokens)
                    actual = int(ids[0])
                    expected = self.expected_tokens[index]
                    if actual != expected:
                        raise AssertionError(
                            f"greedy graph token {index} changed baseline context: "
                            f"{actual} != {expected}"
                        )

    def begin_history(self, forced_tokens=None, *, force=True):
        if self.original_sampler is not None:
            raise RuntimeError("sampler history capture is already active")
        self.history = []
        self.expected_tokens = forced_tokens
        self.force_tokens = force
        original_sampler = self.runner._sampler
        self.original_sampler = original_sampler
        capture = self

        class Sampler:
            def __getattr__(self, name):
                return getattr(original_sampler, name)

            def __call__(self, logits, plan=None, **kwargs):
                tokens = original_sampler(logits, plan, **kwargs)
                capture.record(logits, plan, tokens)
                if (
                    force
                    and forced_tokens is not None
                    and plan is not None
                    and plan.num_sequences == 1
                    and plan.sampling_mask[0]
                ):
                    tokens = tokens.clone()
                    tokens[0] = forced_tokens[len(capture.history) - 1]
                return tokens

        self.runner._sampler = Sampler()

    def end_history(self):
        if self.original_sampler is not None:
            self.runner._sampler = self.original_sampler
            self.original_sampler = None
        return self.history

    def close(self):
        self.end_history()
        for hook in self.hooks:
            hook.remove()


class _ObservedRunner:
    runner_config: RunnerConfig

    def __init__(self, runner, capture):
        self.runner = runner
        self.capture = capture
        self.calls, self.pool_ids, self.released = [], [], []

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def bind_resources(self, binding):
        self.pool_ids.append(int(binding.cache_pool_id))
        return self.runner.bind_resources(binding)

    def release(self, sequence_ids):
        self.released.extend(map(int, sequence_ids))
        release = getattr(self.runner, "release", None)
        if release is not None:
            return release(sequence_ids)

    def validate_distributed(self, plan, context):
        return self.runner.validate_distributed(plan, context)

    def forward_distributed(self, plan, context):
        before = {
            mode: stats["replay_count"]
            for mode, stats in self.runner.cuda_graph_stats.items()
        }
        call = {
            "step": context.step_id,
            "phase": "prefill" if context.phase == BatchPhase.PREFILL else "decode",
            "placeholder": context.placeholder,
            "rows": context.num_sequences,
            "tokens": context.num_tokens,
            "replica_sizes": context.replica_sizes,
            "sequence_ids": [] if plan is None else list(map(int, plan.sequence_ids)),
            **(
                prefill_plan_metadata(plan)
                if context.phase == BatchPhase.PREFILL
                else {}
            ),
        }
        self.calls.append(call)
        self.runner._last_graph_logits = None
        result = self.runner.forward_distributed(plan, context)
        call["graph_replays"] = {
            mode: stats["replay_count"] - before.get(mode, 0)
            for mode, stats in self.runner.cuda_graph_stats.items()
        }
        logits = self.runner._last_graph_logits
        if plan is not None and logits is not None:
            self.capture.record(logits[: plan.num_sequences], plan, result)
        return result


def _metrics(actual, expected, *, maximum, relative):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    difference = actual - expected
    result = {
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "rms": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
        "actual_rms": float(actual.square().mean().sqrt()) if actual.numel() else 0.0,
        "expected_rms": float(expected.square().mean().sqrt())
        if expected.numel()
        else 0.0,
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-12)),
        "max_abs_tolerance": maximum,
        "relative_l2_tolerance": relative,
    }
    result["passed"] = (
        bool(torch.isfinite(actual).all())
        and result["max_abs"] <= maximum
        and result["relative_l2"] <= relative
    )
    return result


def _baseline_rank(rank, port, directory, model_path, prompts, world_size):
    _initialize(rank, port, world_size)
    start = time.perf_counter()
    runner = _new_runner(rank, model_path)
    capture = _Capture(runner, expert=True)
    engine = Engine(runner, **_engine_options())
    try:
        teachers, teacher_reports = {}, {}
        for name in ("short", "long", "probe"):
            request = engine.submit(
                prompts[name], max_new_tokens=1, sampling_params=_sampling()
            )
            progress = _drain(engine)
            assert capture.logits is not None
            teachers[name] = capture.logits.clone()
            teacher_reports[name] = {"token_ids": request.take_result(), **progress}
        capture.begin_history()
        request = engine.submit(
            prompts["probe"],
            max_new_tokens=GENERATION_TOKENS,
            sampling_params=_sampling(),
        )
        generation = {**_drain(engine), "token_ids": request.take_result()}
        generation_logits = capture.end_history()
        if len(generation_logits) != GENERATION_TOKENS:
            raise AssertionError("baseline did not capture every generated-token logit")
        repeated_teachers = {}
        for name in ("short", "long", "probe"):
            request = engine.submit(
                prompts[name], max_new_tokens=1, sampling_params=_sampling()
            )
            _drain(engine)
            repeated_teachers[name] = {
                "logits": _metrics(
                    capture.logits, teachers[name], maximum=0.5, relative=0.02
                ),
                "token_ids": request.take_result(),
            }
            repeated_teachers[name]["matches_first_run"] = (
                repeated_teachers[name]["token_ids"]
                == teacher_reports[name]["token_ids"]
            )
        capture.begin_history(forced_tokens=generation["token_ids"])
        repeated_request = engine.submit(
            prompts["probe"],
            max_new_tokens=GENERATION_TOKENS,
            sampling_params=_sampling(),
        )
        _drain(engine)
        repeated_ids = repeated_request.take_result()
        repeated_logits = capture.end_history()
        repeatability = {
            "teacher_forced": repeated_teachers,
            "teacher_forced_decode": {
                "token_ids": repeated_ids,
                "matches_forced_history": repeated_ids == generation["token_ids"],
                "comparisons": [
                    {
                        "token_index": index,
                        "phase": "prefill" if index == 0 else "decode",
                        **_metrics(actual, expected, maximum=0.5, relative=0.02),
                    }
                    for index, (actual, expected) in enumerate(
                        zip(repeated_logits, generation_logits, strict=True)
                    )
                ],
            },
        }
        if capture.expert is None:
            raise AssertionError(
                "baseline did not capture real checkpoint expert execution"
            )
        torch.save(
            {
                "teachers": teachers,
                "expert": capture.expert,
                "generation_logits": generation_logits,
            },
            Path(directory, f"baseline-{rank}.pt"),
        )
        report = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.perf_counter() - start,
            "teacher_forced": teacher_reports,
            "generation": generation,
            "repeatability": repeatability,
            "cache_free_parents": engine.cache_free_parent_count,
            "cache_parent_capacity": engine.cache_parent_capacity,
        }
        Path(directory, f"baseline-{rank}.json").write_text(json.dumps(report))
        dist.barrier()
    finally:
        engine.close()
        capture.close()
        dist.destroy_process_group()


def _expert_comparisons(runner, groups, reference):
    expert = runner._model.model.layers[0].mlp.experts
    results = []
    count = reference["hidden"].shape[0]
    dp = groups.topology.coordinates(groups.rank)[0]
    dp_size = groups.topology.attn_dp
    selections = [
        torch.arange(count) if dp_size == 1 else torch.arange(dp, count, dp_size),
        torch.arange(0) if dp_size > 1 and dp == 0 else torch.arange(min(count, 5)),
    ]
    # Replicated attention must use the same rows on every rank.
    if groups.topology.attn_dp == 1:
        selections[1] = torch.arange(min(count, 5))
    for selection in selections:
        inputs = [
            reference[name][selection].to(runner.device).contiguous()
            for name in ("hidden", "indices", "weights", "logits")
        ]
        partials = []
        original_local = expert._dispatch.local

        def capture_partial(hidden, indices, weights):
            partial = original_local(hidden, indices, weights)
            values = partial.detach().float()
            partials.append(
                {
                    "rows": len(partial),
                    "dtype": str(partial.dtype),
                    "max_abs": float(values.abs().max()) if values.numel() else 0.0,
                    "rms": float(values.square().mean().sqrt())
                    if values.numel()
                    else 0.0,
                    "mean_abs": float(values.abs().mean()) if values.numel() else 0.0,
                }
            )
            return partial

        expert._dispatch.local = capture_partial
        try:
            with torch.inference_mode():
                output = expert(*inputs)
        finally:
            expert._dispatch.local = original_local
        actual = output.detach().float().cpu()
        expected = reference["output"][selection].float()
        difference = (actual - expected).abs()
        positions = difference.flatten().topk(min(10, difference.numel())).indices
        errors = []
        for position in positions.tolist():
            row, column = divmod(position, actual.shape[1])
            source_row = int(selection[row])
            errors.append(
                {
                    "row": row,
                    "hidden_column": column,
                    "source_row": source_row,
                    "actual": float(actual[row, column]),
                    "expected": float(expected[row, column]),
                    "abs_error": float(difference[row, column]),
                    "global_experts": reference["indices"][source_row].tolist(),
                    "routing_weights": reference["weights"][source_row].tolist(),
                }
            )
        results.append(
            {
                "rows": len(selection),
                "local_partials": partials,
                "largest_errors": errors,
                **_metrics(
                    output, reference["output"][selection], maximum=0.25, relative=0.025
                ),
            }
        )
    return results


def _driver_workload(worker, capture, reference, prompts, baseline, *, graphs=False):
    groups = worker.groups
    dp = groups.topology.coordinates(groups.rank)[0]
    engine = DistributedEngine(worker, **_engine_options())
    try:
        teachers = {}
        names = ("short", "long", "probe")
        for index in range(len(names)):
            name = names[(index + dp) % len(names)]
            request = engine.submit(
                prompts[name], max_new_tokens=1, sampling_params=_sampling()
            )
            progress = _drain(engine)
            teachers[name] = {
                "token_ids": request.take_result(),
                **progress,
                "logits": _metrics(
                    capture.logits,
                    reference["teachers"][name],
                    maximum=0.5,
                    relative=0.02,
                ),
            }
            teachers[name]["matches_baseline"] = (
                teachers[name]["token_ids"]
                == baseline["teacher_forced"][name]["token_ids"]
            )

        capture.begin_history(
            forced_tokens=baseline["generation"]["token_ids"], force=not graphs
        )
        forced_request = engine.submit(
            prompts["probe"],
            max_new_tokens=GENERATION_TOKENS,
            sampling_params=_sampling(),
        )
        forced_progress = _drain(engine)
        forced_ids = forced_request.take_result()
        decode_logits = capture.end_history()
        # Graph decode uses the ordinary captured sampler and checks each
        # greedy token before comparing logits for the same baseline context.
        decode_comparisons = [
            {
                "token_index": index,
                "phase": "prefill" if index == 0 else "decode",
                **_metrics(actual, expected, maximum=0.5, relative=0.02),
            }
            for index, (actual, expected) in enumerate(
                zip(decode_logits, reference["generation_logits"], strict=True)
            )
        ]

        requests, cancellations = {}, []

        def submit(name, prompt, limit):
            request = engine.submit(
                prompts[prompt], max_new_tokens=limit, sampling_params=_sampling()
            )
            requests[name] = (request, limit)
            return request

        def cancel(name, request):
            accepted = engine.cancel(request)
            cancellations.append(
                {
                    "name": name,
                    "accepted": accepted,
                    "done": request.done,
                    "cancelled": request.cancelled,
                    "output": request.take_result() if request.done else None,
                }
            )

        if dp == 0:
            submit("short", "short", 3)
            queued = engine.submit(
                prompts["late"], max_new_tokens=4, sampling_params=_sampling()
            )
            cancel("queued", queued)
        if dp == 1 or groups.topology.attn_dp == 1:
            submit("long", "long", 4)
        active = submit("active_cancel", "long", 6) if dp == 1 else None

        def inject(step):
            if step == 1 and active is not None:
                requests.pop("active_cancel")
                cancel("active", active)
                submit("late", "late", 2)

        progress = _drain(engine, inject=inject)
        outputs = {
            name: {"token_ids": request.take_result(), "expected_tokens": limit}
            for name, (request, limit) in requests.items()
        }
        free_after_workload = engine.local.cache_free_parent_count
        probes = []
        for _ in range(2):
            request = engine.submit(
                prompts["probe"],
                max_new_tokens=GENERATION_TOKENS,
                sampling_params=_sampling(),
            )
            probe = {**_drain(engine), "token_ids": request.take_result()}
            probe["matches_baseline"] = (
                probe["token_ids"] == baseline["generation"]["token_ids"]
            )
            probes.append(probe)
        return {
            "dp": dp,
            "teacher_forced": teachers,
            "teacher_forced_decode": {
                "history_mode": "checked_greedy" if graphs else "teacher_forced",
                "token_ids": forced_ids,
                "matches_forced_history": forced_ids
                == baseline["generation"]["token_ids"],
                "comparisons": decode_comparisons,
                **forced_progress,
            },
            "outputs": outputs,
            "workload": progress,
            "cancellations": cancellations,
            "probes": probes,
            "free_after_workload": free_after_workload,
            "free_after_replay": engine.local.cache_free_parent_count,
            "cache_parent_capacity": engine.local.cache_parent_capacity,
        }
    finally:
        engine.close()


def _ep_rank(
    rank, port, directory, model_path, prompts, world_size, case, graphs=False
):
    _initialize(rank, port, world_size)
    groups = ParallelGroups.create(Parallelism(**case["parallelism"]))
    start = time.perf_counter()
    runner = capture = None
    try:
        runner = _new_runner(rank, model_path, groups=groups, graphs=graphs)
        reference = torch.load(Path(directory, "baseline-0.pt"), weights_only=True)
        baseline = json.loads(Path(directory, "baseline-0.json").read_text())
        expert = _expert_comparisons(runner, groups, reference["expert"])
        # Hooks that transfer tensors to the host are installed only after
        # capture. Every rank captures the same catalog before serving steps.
        captured = runner.capture_cuda_graphs() if graphs else runner.cuda_graph_stats
        projection_probes = validate_prefill_attention_regions(runner) if graphs else []
        capture = _Capture(runner)
        observed = _ObservedRunner(runner, capture)
        worker = DistributedWorker(observed, groups)
        if groups.is_driver:
            result = _driver_workload(
                worker, capture, reference, prompts, baseline, graphs=graphs
            )
        else:
            worker.serve()
            result = None
        torch.cuda.synchronize()
        dist.barrier()
        Path(directory, f"{case['name']}-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "gpu": torch.cuda.get_device_name(),
                    "driver": groups.is_driver,
                    "elapsed_seconds": time.perf_counter() - start,
                    "expert_comparisons": expert,
                    "pool_ids": observed.pool_ids,
                    "released_ids": observed.released,
                    "calls": observed.calls,
                    "graphs_after_capture": captured,
                    "prefill_attention_same_path": projection_probes,
                    "graphs_after_workload": runner.cuda_graph_stats,
                    "driver_result": result,
                }
            )
        )
    finally:
        if capture is not None:
            capture.close()
        if runner is not None:
            runner.close_cuda_graphs()
        groups.close()
        dist.destroy_process_group()


def _spawn(target, directory, model_path, prompts, world_size, *extra):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = torch.multiprocessing.spawn(
        target,
        args=(port, directory, str(model_path), prompts, world_size, *extra),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 30 * 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Hopper checkpoint validation stage exceeded 30 minutes"
                )
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)


def _check_case(records, case, vocab_size, *, graphs=False, collect_numerical=False):
    parallelism = Parallelism(**case["parallelism"])
    attn_dp = parallelism.attn_dp
    numerical_errors = []
    prefill_attention = (
        validate_prefill_attention(
            records,
            stats_key="graphs_after_workload",
            required_operations=("qkv_rope",),
        )
        if graphs
        else None
    )
    phases = [(call["step"], call["phase"]) for call in records[0]["calls"]]
    if not phases or any(
        [(call["step"], call["phase"]) for call in record["calls"]] != phases
        for record in records
    ):
        raise AssertionError("EP ranks executed different coordinated phases")
    if {phase for _, phase in phases} != {"prefill", "decode"}:
        raise AssertionError("Hopper validation did not exercise both phases")
    drivers = [record["driver_result"] for record in records if record["driver"]]
    for record in records:
        if graphs:
            for mode in ("prefill", "decode"):
                capture = record["graphs_after_capture"][mode]
                final = record["graphs_after_workload"][mode]
                if (
                    not capture["capture_count"]
                    or capture["captured_shapes"] != capture["configured_shapes"]
                    or final["replay_count"] <= capture["replay_count"]
                ):
                    raise AssertionError(
                        f"rank {record['rank']} did not capture and replay {mode} graphs"
                    )
                if final["eager_fallback_count"]:
                    raise AssertionError(
                        f"rank {record['rank']} {mode} silently fell back to eager: "
                        f"{final['fallback_counts']}"
                    )
        for comparison in record["expert_comparisons"]:
            if not comparison["passed"]:
                numerical_errors.append(
                    {
                        "kind": "fused_expert",
                        "rank": record["rank"],
                        "comparison": comparison,
                    }
                )
    if len(drivers) != attn_dp or any(
        len(record["pool_ids"]) != 1 for record in records
    ):
        raise AssertionError("request Engine/cache namespace count is incorrect")
    if attn_dp > 1 and not any(
        call["placeholder"] for record in records for call in record["calls"]
    ):
        raise AssertionError("attention-DP empty participation was not exercised")
    if graphs and attn_dp > 1:
        idle_modes = {
            mode
            for record in records
            for call in record["calls"]
            if call["placeholder"]
            for mode, replays in call["graph_replays"].items()
            if replays
        }
        if not {"prefill", "decode"}.issubset(idle_modes):
            raise AssertionError(
                "idle attention-DP ranks did not replay both graph modes"
            )
    for driver in drivers:
        for teacher in driver["teacher_forced"].values():
            if not teacher["logits"]["passed"]:
                numerical_errors.append(
                    {
                        "kind": "teacher_forced_prefill",
                        "dp": driver["dp"],
                        "comparison": teacher["logits"],
                    }
                )
        forced_decode = driver["teacher_forced_decode"]
        if (
            not forced_decode["matches_forced_history"]
            or len(forced_decode["comparisons"]) != GENERATION_TOKENS
        ):
            raise AssertionError(
                "teacher-forced decode did not preserve baseline contexts"
            )
        for comparison in forced_decode["comparisons"]:
            if not comparison["passed"]:
                numerical_errors.append(
                    {
                        "kind": "teacher_forced_decode",
                        "dp": driver["dp"],
                        "comparison": comparison,
                    }
                )
        if (
            driver["free_after_workload"] != driver["cache_parent_capacity"]
            or driver["free_after_replay"] != driver["cache_parent_capacity"]
        ):
            raise AssertionError("completed or cancelled requests leaked KV parents")
        if driver["probes"][0]["token_ids"] != driver["probes"][1]["token_ids"]:
            raise AssertionError("isolated greedy replay changed after cache reuse")
        for output in driver["outputs"].values():
            if len(output["token_ids"]) != output["expected_tokens"] or any(
                not 0 <= token < vocab_size for token in output["token_ids"]
            ):
                raise AssertionError(
                    "full-model generation returned invalid token IDs/counts"
                )
        for cancellation in driver["cancellations"]:
            if not all(
                cancellation[field] for field in ("accepted", "done", "cancelled")
            ):
                raise AssertionError("request cancellation did not complete")
        for event in [
            *driver["workload"]["events"],
            *(event for probe in driver["probes"] for event in probe["events"]),
        ]:
            if event["logprob"] is None or not math.isfinite(event["logprob"]):
                raise AssertionError("generation omitted finite token logprobs")
    if numerical_errors and not collect_numerical:
        raise AssertionError(
            f"checkpoint numerical comparisons failed: {numerical_errors}"
        )
    return {
        "name": case["name"],
        "parallelism": {**case["parallelism"], "expert_tp": parallelism.expert_tp},
        "coordinated_steps": len(phases),
        "numerical_errors": numerical_errors,
        "prefill_attention": prefill_attention,
        "ranks": records,
    }


def _case_summary(records, case):
    return {
        **case,
        "ranks": [
            {
                "rank": record["rank"],
                "expert_comparisons": record["expert_comparisons"],
                "graphs": {
                    mode: {
                        "captures": stats["capture_count"],
                        "replays": stats["replay_count"],
                        "eager_fallbacks": stats["eager_fallback_count"],
                    }
                    for mode, stats in record["graphs_after_workload"].items()
                },
                **(
                    {}
                    if record["driver_result"] is None
                    else {
                        "teacher_forced": {
                            name: {
                                "logits": teacher["logits"],
                                "matches_baseline": teacher["matches_baseline"],
                                "token_ids": teacher["token_ids"],
                            }
                            for name, teacher in record["driver_result"][
                                "teacher_forced"
                            ].items()
                        },
                        "teacher_forced_decode": record["driver_result"][
                            "teacher_forced_decode"
                        ]["comparisons"],
                        "greedy_probes": [
                            {
                                "token_ids": probe["token_ids"],
                                "matches_baseline": probe["matches_baseline"],
                            }
                            for probe in record["driver_result"]["probes"]
                        ],
                    }
                ),
            }
            for record in records
        ],
    }


def _validation_cases(world_size=2, *, composable=False):
    if world_size not in ((2, 4, 8) if composable else (2,)):
        raise ValueError(
            "validation uses two GPUs, or two/four/eight with composable=True"
        )

    def case(name, **axes):
        topology = Parallelism(world_size=world_size, **axes)
        return {
            "name": name,
            "parallelism": {
                axis: getattr(topology, axis)
                for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
            },
        }

    if not composable:
        return [
            case(f"ep{world_size}-attn-dp1", ep=world_size),
            case(
                f"ep{world_size}-attn-dp{world_size}", ep=world_size, attn_dp=world_size
            ),
        ]
    if world_size == 2:
        return [
            case("expert-tp2-attn-tp2", attn_tp=2),
            case("expert-tp2-cp2", cp=2),
            case("expert-tp2-attn-dp2", attn_dp=2),
            case("ep2-attn-tp2", ep=2, attn_tp=2),
            case("ep2-cp2", ep=2, cp=2),
            case("ep2-attn-dp2", ep=2, attn_dp=2),
        ]
    cases = [
        case("ep2-attn-tp2", ep=2, attn_tp=2),
        case("ep2-attn-dp2-tp2", ep=2, attn_dp=2, attn_tp=2),
        case("ep2-cp2-attn-tp2", ep=2, cp=2, attn_tp=2),
        case("ep2-cp2-attn-dp2", ep=2, cp=2, attn_dp=2),
        case(f"ep{world_size}-attn-dp{world_size}", ep=world_size, attn_dp=world_size),
    ]
    if world_size == 8:
        cases.append(case("all-axes", ep=2, attn_dp=2, cp=2, attn_tp=2))
    return cases


def _selected_validation_cases(world_size=2, *, composable=False, case_names=None):
    cases = _validation_cases(world_size, composable=composable)
    if case_names is None:
        return cases
    catalog = {case["name"]: case for case in cases}
    selected = tuple(case_names)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in catalog for name in selected)
    ):
        raise ValueError(f"case_names must be unique choices from {tuple(catalog)}")
    return [catalog[name] for name in selected]


def run_validation(
    model_path,
    prompts,
    *,
    vocab_size,
    diagnostic_path=None,
    graphs=False,
    composable=False,
    world_size=2,
    case_names=None,
):
    configurations = _selected_validation_cases(
        world_size, composable=composable, case_names=case_names
    )
    graphs = graphs or composable
    if torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"Hopper validation requires exactly {world_size} visible GPUs"
        )
    if any(
        not ids or len(ids) + GENERATION_TOKENS > CONTEXT_LENGTH
        for ids in prompts.values()
    ):
        raise ValueError("validation prompts do not fit the bounded context")
    with tempfile.TemporaryDirectory(prefix="resin-gpt-oss-ep-") as directory:
        print(
            "[parallel-validation] loading independent full20B single-GPU baselines",
            flush=True,
        )
        _spawn(_baseline_rank, directory, model_path, prompts, world_size)
        baseline = [
            json.loads(Path(directory, f"baseline-{rank}.json").read_text())
            for rank in range(world_size)
        ]
        for record in baseline:
            print(
                "[parallel-validation] baseline repeatability "
                + json.dumps(
                    {"rank": record["rank"], **record["repeatability"]},
                    sort_keys=True,
                ),
                flush=True,
            )
        diagnostic = {
            "world_size": world_size,
            "composable": composable,
            "selected_cases": [case["name"] for case in configurations],
            "graphs": "prefill_breakable_decode_full" if graphs else "disabled",
            "baseline": baseline,
            "cases": [],
            "status": "running",
        }
        if diagnostic_path is not None:
            artifact_directory = Path(diagnostic_path).with_suffix("")
            artifact_directory.mkdir(parents=True, exist_ok=True)
            diagnostic["expert_reference_artifacts"] = []
            for rank in range(world_size):
                reference = torch.load(
                    Path(directory, f"baseline-{rank}.pt"), weights_only=True
                )
                artifact = artifact_directory / f"baseline-{rank}-expert.pt"
                torch.save(reference["expert"], artifact)
                diagnostic["expert_reference_artifacts"].append(str(artifact))

        def save_diagnostic():
            if diagnostic_path is not None:
                Path(diagnostic_path).write_text(
                    json.dumps(diagnostic, indent=2, sort_keys=True)
                )

        save_diagnostic()
        cases = []
        for case in configurations:
            print(
                f"[parallel-validation] loading checkpoint {case['name']}: {case['parallelism']}",
                flush=True,
            )
            _spawn(_ep_rank, directory, model_path, prompts, world_size, case, graphs)
            records = [
                json.loads(Path(directory, f"{case['name']}-{rank}.json").read_text())
                for rank in range(world_size)
            ]
            diagnostic["cases"].append({**case, "ranks": records})
            print(
                "[parallel-validation] numerical diagnostics "
                + json.dumps(_case_summary(records, case), sort_keys=True),
                flush=True,
            )
            save_diagnostic()
            try:
                cases.append(
                    _check_case(
                        records,
                        case,
                        vocab_size,
                        graphs=graphs,
                        collect_numerical=True,
                    )
                )
            except Exception as error:
                diagnostic["status"] = "failed"
                diagnostic["validation_error"] = str(error)
                save_diagnostic()
                raise
        numerical_errors = [
            {"case": case["name"], **error}
            for case in cases
            for error in case["numerical_errors"]
        ]
        if numerical_errors:
            diagnostic["status"] = "failed"
            diagnostic["numerical_errors"] = numerical_errors
            diagnostic["validation_error"] = (
                f"{len(numerical_errors)} checkpoint numerical comparisons failed"
            )
            save_diagnostic()
            raise AssertionError(diagnostic["validation_error"])
        return {
            "world_size": world_size,
            "composable": composable,
            "selected_cases": [case["name"] for case in configurations],
            "graphs": "prefill_breakable_decode_full" if graphs else "disabled",
            "baseline": baseline,
            "cases": cases,
            "comparison_policy": "same-context checkpoint logits and real fused expert outputs use fixed BF16 tolerances; greedy token matches are reported",
        }
