"""Bounded full-checkpoint validation over composed CUDA rank meshes."""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import tempfile
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from resin import BatchPhase, Engine, Parallelism, SamplingParams, SchedulerConfig
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups

WORLD_SIZE = 4
CONTEXT_LENGTH = 512
PREFILL_CHUNK_SIZE = 32
CACHE_PARENTS = 64
MAX_STEPS = 96
PROCESS_TIMEOUT_SECONDS = 40 * 60
SPECULATIVE_PROBE_TOKENS = 16


def _output_lengths(output):
    if output is None:
        return []
    if hasattr(output, "token_indptr"):
        offsets = output.token_indptr
        return [int(end - start) for start, end in zip(offsets[:-1], offsets[1:])]
    return [len(row) for row in output]


class _ObservedRunner:
    def __init__(self, runner):
        self.runner = runner
        self.calls: list[dict[str, object]] = []
        self.pool_ids: list[int] = []
        self.released: list[int] = []
        self.decode_trace = None

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
        validate = getattr(self.runner, "validate_distributed", None)
        if validate is not None:
            validate(plan, context)

    def forward(self, plan):
        if self.decode_trace is not None:
            return self.decode_trace.step(plan, lambda: self.runner.forward(plan))
        return self.runner.forward(plan)

    def forward_distributed(self, plan, context):
        self.calls.append(
            {
                "step": context.step_id,
                "phase": "prefill" if context.phase == BatchPhase.PREFILL else "decode",
                "placeholder": context.placeholder,
                "rows": context.num_sequences,
                "tokens": context.num_tokens,
                "replica_sizes": context.replica_sizes,
                "sequence_ids": []
                if plan is None
                else list(map(int, plan.sequence_ids)),
                "contexts": []
                if plan is None
                else list(map(int, plan.context_lengths)),
            }
        )
        if self.decode_trace is not None:
            output = self.decode_trace.step(
                plan, lambda: self.runner.forward_distributed(plan, context)
            )
        else:
            output = self.runner.forward_distributed(plan, context)
        if self.runner.speculative_tokens and context.phase == BatchPhase.DECODE:
            self.calls[-1]["output_lengths"] = _output_lengths(output)
        return output


class _DecodeTrace:
    """Observe genuine model/sampler output before fixing the next engine token.

    Graphs keep their captured sampler. Only the token returned to the Engine
    is replaced after replay and logit capture, so every decode uses the same
    baseline context even when the genuine greedy outputs differ.
    """

    def __init__(self, observed, forced_tokens=None):
        self.observed, self.runner = observed, observed.runner
        self.forced_tokens = forced_tokens
        self.latest = None
        self.graph_shape = None
        self.logits, self.token_ids, self.advanced_token_ids = [], [], []
        self.phases, self.input_token_ids, self.positions, self.graph_shapes = (
            [],
            [],
            [],
            [],
        )

    def __enter__(self):
        if self.observed.decode_trace is not None:
            raise RuntimeError("decode trace is already active")
        self.head_hook = self.runner._model.head.register_forward_hook(
            lambda module, inputs, output: setattr(self, "latest", output.detach())
        )
        self.original_graph = self.runner._run_cuda_graph

        def graph(mode, shape, inputs):
            output = self.original_graph(mode, shape, inputs)
            if mode == "decode":
                self.latest = self.runner._graph_logits[shape]
                self.graph_shape = shape
            return output

        self.runner._run_cuda_graph = graph
        self.observed.decode_trace = self
        return self

    def __exit__(self, *error):
        self.observed.decode_trace = None
        self.runner._run_cuda_graph = self.original_graph
        self.head_hook.remove()

    def step(self, plan, operation):
        self.latest, self.graph_shape = None, None
        output = operation()
        if plan is None or plan.num_sequences != 1 or not plan.sampling_mask[0]:
            return output
        if self.latest is None:
            raise AssertionError("generation omitted eager or graph decode logits")
        index = len(self.logits)
        values = self.latest.detach().float().cpu()[:1].clone()
        if values.ndim != 2:
            raise AssertionError("decode trace requires one vocabulary row")
        ids = getattr(output, "token_ids", output)
        genuine = int(ids[0])
        advanced = genuine
        if self.forced_tokens is not None:
            if index >= len(self.forced_tokens):
                raise AssertionError("decode exceeded the baseline token history")
            if plan.needs_logprobs:
                raise AssertionError("forced decode tracing uses token-only requests")
            advanced = int(self.forced_tokens[index])
        self.logits.append(values.squeeze(0))
        self.token_ids.append(genuine)
        self.advanced_token_ids.append(advanced)
        self.phases.append("prefill" if plan.phase == BatchPhase.PREFILL else "decode")
        self.input_token_ids.append(list(map(int, plan.token_ids)))
        self.positions.append(list(map(int, plan.positions)))
        self.graph_shapes.append(self.graph_shape)
        return output if self.forced_tokens is None else [advanced]

    def result(self):
        if not self.logits:
            raise AssertionError("decode trace did not record any sampled rows")
        if self.forced_tokens is not None and len(self.logits) != len(
            self.forced_tokens
        ):
            raise AssertionError("decode did not cover the baseline token history")
        return {
            "logits": torch.stack(self.logits),
            "sampler_token_ids": self.token_ids,
            "advanced_token_ids": self.advanced_token_ids,
            "phases": self.phases,
            "input_token_ids": self.input_token_ids,
            "positions": self.positions,
            "graph_shapes": self.graph_shapes,
        }


def _initialize(rank: int, port: int, world_size: int = WORLD_SIZE) -> None:
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


def _new_runner(
    rank,
    model_path,
    *,
    groups=None,
    world_size=WORLD_SIZE,
    graphs=False,
    decode_capacity=CONTEXT_LENGTH,
    speculative_tokens=0,
):
    from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

    parallel = (
        {"parallel_groups": groups}
        if groups is not None
        else {
            "tensor_parallel": TensorParallelContext(
                TensorParallelLayout(rank=rank, size=world_size), dist.group.WORLD
            )
        }
    )
    return DeepSeekV4CudaRunner(
        model_path,
        max_seq_len=CONTEXT_LENGTH,
        max_batch_size=2,
        device=f"cuda:{rank}",
        cuda_graph_prefill="breakable" if graphs else "disabled",
        cuda_graph_decode="full" if graphs else "disabled",
        cuda_graph_decode_shapes=((1, decode_capacity), (2, CONTEXT_LENGTH))
        if graphs
        else None,
        speculative_tokens=speculative_tokens,
        fast_allreduce=False,
        cache_parent_capacity=CACHE_PARENTS,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
        **parallel,
    )


def _sampling():
    return SamplingParams(temperature=0.0, logprobs=True, top_logprobs=3)


def _engine_options():
    return {
        "max_running_requests": 2,
        "context_length": CONTEXT_LENGTH,
        "eos_token_id": None,
        "scheduler_config": SchedulerConfig(
            max_batch_tokens=2 * PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
        ),
    }


def _drain(engine, events, *, inject=None):
    steps = 0
    while engine.has_work():
        for event in engine.step():
            scores = event.logprobs
            events.append(
                {
                    "request_id": int(event.request_id),
                    "token_id": int(event.token_id),
                    "index": int(event.index),
                    "step": steps,
                    "logprob": None if scores is None else float(scores.logprob),
                    "top_logprobs": [] if scores is None else scores.top_logprobs,
                }
            )
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError("parallel validation exceeded its step bound")
        if inject is not None:
            inject(steps)
    return steps


def _teacher_logits(engine, runner, prompts):
    """Record full-vocabulary logits on identical fixed token contexts."""

    latest = []
    handle = runner._model.head.register_forward_hook(
        lambda module, inputs, output: latest.append(output.detach())
    )
    values = []
    try:
        for suffix in ((), (2,), (2, 3), (2, 3, 4)):
            latest.clear()
            request = engine.submit(
                [*prompts["probe"], *suffix],
                max_new_tokens=1,
                sampling_params=_sampling(),
            )
            _drain(engine, [])
            request.take_result()
            if not latest:
                raise AssertionError("teacher-forced prefill omitted model logits")
            values.append(latest[-1].float().cpu().squeeze(0).clone())
    finally:
        handle.remove()
    return torch.stack(values)


def _driver_workload(
    worker,
    prompts,
    *,
    numerical_directory=None,
    decode_reference=None,
    speculative_tokens=0,
):
    groups = worker.groups
    dp = groups.topology.coordinates(groups.rank)[0]
    engine = DistributedEngine(worker, **_engine_options())
    requests, events, cancellations = {}, [], []
    sampling = _sampling()
    probe_tokens = SPECULATIVE_PROBE_TOKENS if speculative_tokens else 4
    try:

        def submit(name, prompt, limit):
            request = engine.submit(
                prompt, max_new_tokens=limit, sampling_params=sampling
            )
            requests[name] = (request, limit)
            return request

        def cancel(name, request):
            accepted = engine.cancel(request)
            output = request.take_result() if request.done else None
            cancellations.append(
                {
                    "name": name,
                    "accepted": accepted,
                    "done": request.done,
                    "cancelled": request.cancelled,
                    "output": output,
                }
            )

        if dp == 0:
            submit("short", prompts["short"], 3)
            queued = engine.submit(
                prompts["late"], max_new_tokens=4, sampling_params=sampling
            )
            cancel("queued", queued)
        if dp == 1 or groups.topology.attn_dp == 1:
            submit("long", prompts["long"], 4)
        active_cancel = submit("active_cancel", prompts["long"], 6) if dp == 1 else None

        def inject(step):
            if step == 1 and active_cancel is not None:
                request, _ = requests.pop("active_cancel")
                cancel("active", request)
            if step == 1 and dp == 2:
                submit("late", prompts["late"], 2)

        workload_steps = _drain(engine, events, inject=inject)
        outputs = {
            name: {"token_ids": request.take_result(), "expected_tokens": limit}
            for name, (request, limit) in requests.items()
        }
        free_after_workload = engine.local.cache_free_parent_count
        probes = []
        for _ in range(2):
            probe = engine.submit(
                prompts["probe"],
                max_new_tokens=probe_tokens,
                sampling_params=sampling,
            )
            probe_events = []
            _drain(engine, probe_events)
            probes.append({"token_ids": probe.take_result(), "events": probe_events})
        if numerical_directory is not None and not speculative_tokens:
            torch.save(
                _teacher_logits(engine, worker.runner.runner, prompts),
                Path(numerical_directory, f"{groups.rank}.pt"),
            )
            if decode_reference is not None:
                with _DecodeTrace(
                    worker.runner, decode_reference["advanced_token_ids"]
                ) as trace:
                    request = engine.submit(
                        prompts["probe"],
                        max_new_tokens=len(decode_reference["advanced_token_ids"]),
                        sampling_params=SamplingParams(temperature=0.0),
                    )
                    _drain(engine, [])
                    advanced = request.take_result()
                traced = trace.result()
                if advanced != traced["advanced_token_ids"]:
                    raise AssertionError(
                        "Engine did not advance the fixed decode history"
                    )
                torch.save(
                    traced, Path(numerical_directory, f"decode-{groups.rank}.pt")
                )
        return {
            "dp": dp,
            "workload_steps": workload_steps,
            "outputs": outputs,
            "events": events,
            "cancellations": cancellations,
            "free_after_workload": free_after_workload,
            "free_after_replay": engine.local.cache_free_parent_count,
            "parent_capacity": engine.local.cache_parent_capacity,
            "probes": probes,
        }
    finally:
        engine.close()


def _run_rank(
    rank,
    port,
    report_directory,
    model_path,
    prompts,
    attn_dp,
    topology=None,
    graphs=False,
    decode_reference=None,
    decode_capacity=CONTEXT_LENGTH,
    speculative_tokens=0,
):
    parallelism = Parallelism(
        **(topology or {"world_size": WORLD_SIZE, "ep": WORLD_SIZE, "attn_dp": attn_dp})
    )
    _initialize(rank, port, parallelism.world_size)
    groups = ParallelGroups.create(parallelism)
    runner = None
    try:
        start = time.perf_counter()
        print(f"[deepseek-v4 rank={rank}] loading {parallelism}", flush=True)
        runner = _ObservedRunner(
            _new_runner(
                rank,
                model_path,
                groups=groups,
                graphs=graphs,
                decode_capacity=decode_capacity,
                speculative_tokens=speculative_tokens,
            )
        )
        if graphs:
            if not all(layer.ffn.routed.fused for layer in runner._model.layers):
                raise AssertionError(
                    "composed graph validation requires fused FP4 experts"
                )
            print(
                f"[deepseek-v4 rank={rank}] capturing prefill/decode graphs", flush=True
            )
            runner.capture_cuda_graphs()
            print(f"[deepseek-v4 rank={rank}] graph startup complete", flush=True)
        torch.cuda.synchronize()
        dist.barrier()
        loaded_seconds = time.perf_counter() - start
        worker = DistributedWorker(runner, groups)
        if groups.is_driver:
            result = _driver_workload(
                worker,
                prompts,
                numerical_directory=report_directory if topology else None,
                decode_reference=decode_reference,
                speculative_tokens=speculative_tokens,
            )
        else:
            worker.serve()
            result = None
        torch.cuda.synchronize()
        dist.barrier()
        report = {
            "rank": rank,
            "replica_ranks": groups.replica.ranks,
            "driver": groups.is_driver,
            "gpu": torch.cuda.get_device_name(),
            "loaded_seconds": loaded_seconds,
            "elapsed_seconds": time.perf_counter() - start,
            "weight_load": runner.weight_load,
            "cuda_graph_stats": runner.cuda_graph_stats,
            "speculative_stats": runner.speculative_stats,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_memory_bytes": torch.cuda.mem_get_info()[0],
            "pool_ids": runner.pool_ids,
            "released_ids": runner.released,
            "calls": runner.calls,
            "driver_result": result,
        }
        Path(report_directory, f"{rank}.json").write_text(json.dumps(report))
    except BaseException as error:
        if runner is not None:
            failure = {
                "rank": rank,
                "error": f"{type(error).__name__}: {error}",
                "calls": runner.calls,
                "speculative_stats": runner.speculative_stats,
                "cuda_graph_stats": runner.cuda_graph_stats,
            }
            try:
                torch.save(
                    {
                        shape: values.detach().float().cpu()
                        for shape, values in runner._graph_logits.items()
                    },
                    Path(report_directory, f"error-{rank}-graph-logits.pt"),
                )
            except BaseException as dump_error:
                failure["logit_dump_error"] = str(dump_error)
            Path(report_directory, f"error-{rank}.json").write_text(json.dumps(failure))
        raise
    finally:
        if runner is not None:
            runner.close_cuda_graphs()
        groups.close()
        dist.destroy_process_group()


def _run_tp_baseline_rank(
    rank,
    port,
    report_directory,
    model_path,
    prompts,
    world_size=WORLD_SIZE,
    numerical=False,
    speculative_tokens=0,
    probe_tokens=4,
):
    _initialize(rank, port, world_size)
    runner = _ObservedRunner(
        _new_runner(
            rank,
            model_path,
            world_size=world_size,
            speculative_tokens=speculative_tokens,
        )
    )
    engine = Engine(runner, **_engine_options())
    events, replay_events, result = [], [], None
    try:
        with _DecodeTrace(runner) if numerical else nullcontext() as trace:
            request = engine.submit(
                prompts["probe"],
                max_new_tokens=probe_tokens,
                sampling_params=_sampling(),
            )
            _drain(engine, events)
        result = {
            "token_ids": request.take_result(),
            "events": events,
            "speculative_stats": runner.speculative_stats,
        }
        if speculative_tokens:
            free_after_reference = engine.cache_free_parent_count
            replay = engine.submit(
                prompts["probe"],
                max_new_tokens=probe_tokens,
                sampling_params=_sampling(),
            )
            _drain(engine, replay_events)
            replay_ids = replay.take_result()
            result.update(
                replay_token_ids=replay_ids,
                replay_events=replay_events,
                free_after_reference=free_after_reference,
                free_after_replay=engine.cache_free_parent_count,
                parent_capacity=engine.cache_parent_capacity,
                speculative_stats=runner.speculative_stats,
            )
            if replay_ids != result["token_ids"]:
                raise AssertionError("legacy DSpark replay changed after cache reuse")
            if len(events) != probe_tokens or len(replay_events) != probe_tokens:
                raise AssertionError("legacy DSpark reference omitted streamed tokens")
            if any(
                event["logprob"] is None
                or not math.isfinite(event["logprob"])
                or len(event["top_logprobs"]) != 3
                or any(not math.isfinite(value) for _, value in event["top_logprobs"])
                for event in [*events, *replay_events]
            ):
                raise AssertionError(
                    "legacy DSpark reference omitted finite token scores"
                )
            if any(
                result[field] != result["parent_capacity"]
                for field in ("free_after_reference", "free_after_replay")
            ):
                raise AssertionError("legacy DSpark reference leaked cache parents")
        if numerical:
            torch.save(
                _teacher_logits(engine, runner, prompts),
                Path(report_directory, f"{rank}.pt"),
            )
            torch.save(trace.result(), Path(report_directory, f"decode-{rank}.pt"))
        torch.cuda.synchronize()
        dist.barrier()
        Path(report_directory, f"{rank}.json").write_text(json.dumps(result))
    except BaseException as error:
        Path(report_directory, f"error-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "error": f"{type(error).__name__}: {error}",
                    "events": events,
                    "replay_events": replay_events,
                    "report": result,
                    "speculative_stats": runner.speculative_stats,
                }
            )
        )
        raise
    finally:
        engine.close()
        runner.close_cuda_graphs()
        dist.destroy_process_group()


def _spawn(
    target, model_path, prompts, *extra, world_size=WORLD_SIZE, failure_directory=None
):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="resin-dsv4-parallel-") as directory:
        context = torch.multiprocessing.spawn(
            target,
            args=(port, directory, str(model_path), prompts, *extra),
            nprocs=world_size,
            join=False,
        )
        deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "DeepSeek-V4 parallel validation exceeded 40 minutes"
                    )
        except BaseException:
            if failure_directory is not None:
                shutil.copytree(directory, failure_directory, dirs_exist_ok=True)
            raise
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
        records = []
        for rank in range(world_size):
            record = json.loads(Path(directory, f"{rank}.json").read_text())
            logits_path = Path(directory, f"{rank}.pt")
            if logits_path.exists():
                record["_teacher_logits"] = torch.load(logits_path, weights_only=True)
            decode_path = Path(directory, f"decode-{rank}.pt")
            if decode_path.exists():
                record["_decode_trace"] = torch.load(decode_path, weights_only=True)
            records.append(record)
        return records


def _validate_records(records, attn_dp, vocab_size, *, topology=None, graphs=False):
    drivers = [record["driver_result"] for record in records if record["driver"]]
    if len(drivers) != attn_dp:
        raise AssertionError("request Engine count does not match attention DP")
    steps = [(call["step"], call["phase"]) for call in records[0]["calls"]]
    if not steps or any(
        [(call["step"], call["phase"]) for call in record["calls"]] != steps
        for record in records
    ):
        raise AssertionError("ranks did not execute the same coordinated phases")
    if {phase for _, phase in steps} != {"prefill", "decode"}:
        raise AssertionError("full-model validation did not execute both phases")
    for record in records:
        if len(record["pool_ids"]) != 1:
            raise AssertionError("rank did not bind exactly one replica cache pool")
    for driver in drivers:
        for output in driver["outputs"].values():
            ids = output["token_ids"]
            if len(ids) != output["expected_tokens"] or any(
                not 0 <= token < vocab_size for token in ids
            ):
                raise AssertionError(
                    "full-model output has an invalid token count or ID"
                )
        if (
            driver["free_after_workload"] != driver["parent_capacity"]
            or driver["free_after_replay"] != driver["parent_capacity"]
        ):
            raise AssertionError("completed/cancelled requests leaked cache parents")
        if driver["probes"][0]["token_ids"] != driver["probes"][1]["token_ids"]:
            raise AssertionError("greedy isolated replay changed after cache reuse")
        scored_events = [
            *driver["events"],
            *(event for probe in driver["probes"] for event in probe["events"]),
        ]
        for event in scored_events:
            if event["logprob"] is None or not math.isfinite(event["logprob"]):
                raise AssertionError("full-model generation omitted finite logprobs")
        for cancellation in driver["cancellations"]:
            if not all(
                cancellation[field] for field in ("accepted", "done", "cancelled")
            ):
                raise AssertionError("request cancellation did not complete")
    if attn_dp > 1:
        if not any(
            call["placeholder"] for record in records for call in record["calls"]
        ):
            raise AssertionError("idle attention-DP rank did not participate empty")
        if not any(
            len({tuple(size) for size in call["replica_sizes"]}) > 1
            for call in records[0]["calls"]
        ):
            raise AssertionError("workload did not exercise unequal replica capacities")
        if not any(
            item["name"] == "active"
            for driver in drivers
            for item in driver["cancellations"]
        ):
            raise AssertionError("active cancellation was not exercised")
    if graphs:
        for record in records:
            for mode in ("prefill", "decode"):
                stats = record["cuda_graph_stats"][mode]
                if not stats["capture_count"] or not stats["replay_count"]:
                    raise AssertionError(
                        f"{mode} graphs were not captured and replayed"
                    )
                if stats["eager_fallback_count"] or stats["poisoned"]:
                    raise AssertionError(
                        f"{mode} graph validation fell back or was poisoned"
                    )
    return {
        "parallelism": topology
        or {
            "world_size": WORLD_SIZE,
            "ep": WORLD_SIZE,
            "attn_dp": attn_dp,
            "attn_tp": 1,
            "cp": 1,
        },
        "coordinated_steps": len(steps),
        "checks": {
            "phases_aligned": True,
            "finite_scores": True,
            "cache_reclaimed": True,
            "isolated_replay_identical": True,
            "independent_request_engines": attn_dp,
            "graphs_captured_and_replayed": graphs,
        },
        "ranks": records,
    }


def run_parallel_validation(
    model_path, prompts, *, vocab_size, include_tp_baseline=False
):
    """Load the complete pinned model separately for EP-only and EP+DP runs."""
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            "parallel validation requires exactly four visible CUDA devices"
        )
    if any(
        not prompt or len(prompt) + 6 > CONTEXT_LENGTH for prompt in prompts.values()
    ):
        raise ValueError("validation prompts must fit the bounded context")
    reports = [
        _validate_records(
            _spawn(_run_rank, model_path, prompts, attn_dp), attn_dp, vocab_size
        )
        for attn_dp in (1, WORLD_SIZE)
    ]
    result = {"graphs": "disabled", "speculative_tokens": 0, "cases": reports}
    if include_tp_baseline:
        baseline = _spawn(_run_tp_baseline_rank, model_path, prompts)[0]
        comparisons = []
        for report in reports:
            for record in report["ranks"]:
                if record["driver"]:
                    ids = record["driver_result"]["probes"][0]["token_ids"]
                    comparisons.append(
                        {
                            "attn_dp": report["parallelism"]["attn_dp"],
                            "rank": record["rank"],
                            "token_ids": ids,
                            "matches_tp4": ids == baseline["token_ids"],
                        }
                    )
        result["tp4_baseline"] = baseline
        result["baseline_comparisons"] = comparisons
        result["baseline_comparison_policy"] = (
            "observational; different quantized execution layouts may change greedy tokens"
        )
    return result


def _composable_cases(world_size):
    axes = [
        ("attention-tp-expert-tp", {"ep": 1, "attn_tp": 2}),
        ("context-expert-tp", {"ep": 1, "cp": 2}),
        ("attention-dp-expert-tp", {"ep": 1, "attn_dp": 2}),
        ("ep-attention-tp", {"ep": 2, "attn_tp": 2}),
        ("ep-context", {"ep": 2, "cp": 2}),
        ("ep-attention-dp", {"ep": 2, "attn_dp": 2}),
    ]
    if world_size == 4:
        axes.extend(
            [
                ("ep-context-attention-tp", {"ep": 2, "cp": 2, "attn_tp": 2}),
                ("ep-attention-dp-tp", {"ep": 2, "attn_dp": 2, "attn_tp": 2}),
                ("ep-attention-dp-context", {"ep": 2, "attn_dp": 2, "cp": 2}),
            ]
        )
    cases = []
    for name, values in axes:
        topology = {
            "world_size": world_size,
            "attn_dp": 1,
            "attn_tp": 1,
            "cp": 1,
            "ep": 1,
            **values,
        }
        Parallelism(**topology)  # Validate the matrix before allocating GPUs.
        cases.append((name, topology))
    return cases


def _logit_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError("teacher-forced contexts produced different logit shapes")
    difference = actual.float() - expected.float()
    maximum = float(difference.abs().max())
    relative = float(difference.norm() / expected.float().norm().clamp_min(1e-12))
    offsets = difference.mean(dim=-1)
    expected_logprobs = expected.float().log_softmax(dim=-1)
    actual_logprobs = actual.float().log_softmax(dim=-1)
    result = {
        "contexts": actual.shape[0],
        "max_abs": maximum,
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": relative,
        "logit_offsets": offsets.tolist(),
        "centered_rms": float((difference - offsets[..., None]).square().mean().sqrt()),
        "kl_expected_actual": (
            expected_logprobs.exp() * (expected_logprobs - actual_logprobs)
        )
        .sum(dim=-1)
        .tolist(),
        "expected_top_token_ids": expected.argmax(dim=-1).tolist(),
        "actual_top_token_ids": actual.argmax(dim=-1).tolist(),
        "max_abs_tolerance": 0.5,
        "relative_l2_tolerance": 0.02,
        "passed": bool(torch.isfinite(actual).all())
        and maximum <= 0.5
        and relative <= 0.02,
    }
    return result


def _decode_comparison(actual, expected):
    """Compare distributions only after checking the fixed token contexts."""

    contexts_match = (
        actual["advanced_token_ids"] == expected["advanced_token_ids"]
        and actual["phases"] == expected["phases"]
        and all(
            actual[field][index] == expected[field][index]
            for index, phase in enumerate(expected["phases"])
            if phase == "decode"
            for field in ("input_token_ids", "positions")
        )
    )
    comparisons = [
        {
            "token_index": index,
            "phase": expected["phases"][index],
            **_logit_metrics(value[None], reference[None]),
        }
        for index, (value, reference) in enumerate(
            zip(actual["logits"], expected["logits"], strict=True)
        )
    ]
    return {
        **{key: value for key, value in actual.items() if key != "logits"},
        "contexts_match": contexts_match,
        "sampler_matches_baseline": actual["sampler_token_ids"]
        == expected["sampler_token_ids"],
        "comparisons": comparisons,
        "passed": contexts_match and all(value["passed"] for value in comparisons),
    }


def _validate_speculative_records(records, speculative_tokens):
    """Check real variable-length acceptance, including idle replica recovery."""

    accepted = 0
    blocks = 0
    for record in records:
        lengths = []
        for call in record["calls"]:
            if call["phase"] != "decode":
                continue
            output_lengths = call["output_lengths"]
            if len(output_lengths) != call["rows"] or any(
                not 1 <= size <= speculative_tokens + 1 for size in output_lengths
            ):
                raise AssertionError(
                    "speculative output rows do not match the block budget"
                )
            lengths.extend(output_lengths)
        stats = record["speculative_stats"]
        if (
            stats["blocks"] != len(lengths)
            or stats["drafts"] != len(lengths) * speculative_tokens
            or stats["emitted"] != sum(lengths)
            or stats["accepted"] != sum(size - 1 for size in lengths)
        ):
            raise AssertionError(
                "speculative counters do not match verified output blocks"
            )
        if not record["driver"]:
            continue
        blocks += stats["blocks"]
        accepted += stats["accepted"]
        driver = record["driver_result"]
        for events in [
            driver["events"],
            *(probe["events"] for probe in driver["probes"]),
        ]:
            by_request = {}
            for event in events:
                by_request.setdefault(event["request_id"], []).append(event["index"])
                if (
                    event["logprob"] is None
                    or not math.isfinite(event["logprob"])
                    or len(event["top_logprobs"]) != 3
                    or any(
                        not math.isfinite(value) for _, value in event["top_logprobs"]
                    )
                ):
                    raise AssertionError("speculative logprob columns are incomplete")
            if any(
                indices != list(range(len(indices))) for indices in by_request.values()
            ):
                raise AssertionError(
                    "accepted blocks skipped or repeated a streamed token index"
                )
        if any(
            len(probe["token_ids"]) != SPECULATIVE_PROBE_TOKENS
            for probe in driver["probes"]
        ):
            raise AssertionError(
                "speculative output did not honor the request token budget"
            )
        if any(
            [event["token_id"] for event in probe["events"]] != probe["token_ids"]
            for probe in driver["probes"]
        ):
            raise AssertionError("speculative stream omitted accepted output tokens")
    if not blocks or not accepted:
        raise AssertionError(
            "speculative validation must verify blocks and accept draft tokens"
        )
    if len([record for record in records if record["driver"]]) > 1:
        if not any(
            any(call["placeholder"] for call in record["calls"][:index])
            and not call["placeholder"]
            for record in records
            for index, call in enumerate(record["calls"])
        ):
            raise AssertionError("idle attention-DP replicas did not resume generation")
    return {
        "verified_blocks": blocks,
        "accepted_draft_tokens": accepted,
        "variable_output_rows_valid": True,
        "stream_indices_contiguous": True,
        "accepted_tokens_have_finite_scores": True,
        "idle_replica_resumed": len([record for record in records if record["driver"]])
        > 1,
    }


def _token_comparison(actual, expected):
    differences = [
        index
        for index, (token, reference) in enumerate(zip(actual, expected))
        if token != reference
    ]
    if len(actual) != len(expected):
        differences.append(min(len(actual), len(expected)))
    return {
        "matches": not differences,
        "first_difference": differences[0] if differences else None,
        "token_ids": actual,
        "reference_token_ids": expected,
    }


def _run_speculative_validation(
    model_path,
    prompts,
    *,
    vocab_size,
    diagnostic_path,
    selected,
    catalog,
    baseline_path,
    graphs,
    decode_capacity,
    speculative_tokens,
    world_size,
):
    """Validate DSpark as variable output blocks, never as one-token decode steps."""

    directory = None if diagnostic_path is None else Path(diagnostic_path)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "model_path": str(Path(model_path).resolve()),
        "world_size": world_size,
        "prompts": prompts,
        "speculative_tokens": speculative_tokens,
        "speculative_contract": "masked-compressor-ring-v1",
        "probe_tokens": SPECULATIVE_PROBE_TOKENS,
        "attention_contract": "fixed-swa128-canonical-selected-capacity",
    }
    if baseline_path is None:
        references = []
        for count in (0, speculative_tokens):
            print(
                f"[deepseek-v4] loading eager TP{world_size} reference, DSpark K={count}",
                flush=True,
            )
            references.append(
                _spawn(
                    _run_tp_baseline_rank,
                    model_path,
                    prompts,
                    world_size,
                    False,
                    count,
                    SPECULATIVE_PROBE_TOKENS,
                    world_size=world_size,
                    failure_directory=None
                    if directory is None
                    else directory / f"baseline-k{count}-workers",
                )[0]
            )
        target, baseline = references
    else:
        saved = torch.load(baseline_path, weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(
                "reused speculative baseline must match checkpoint, GPU count, prompts, K and attention contract"
            )
        target, baseline = saved["target_report"], saved["report"]
    if directory is not None:
        torch.save(
            {"identity": identity, "target_report": target, "report": baseline},
            directory / "baseline.pt",
        )
    result = {
        "speculative_tokens": speculative_tokens,
        "world_size": world_size,
        "selected_cases": selected,
        "diagnostic_path": None if directory is None else str(directory),
        "graphs": {
            "prefill": "breakable" if graphs else "disabled",
            "decode": "full" if graphs else "disabled",
        },
        "decode_shapes": ((1, decode_capacity), (2, CONTEXT_LENGTH)) if graphs else (),
        "eager_tp_baseline": target,
        "eager_dspark_baseline": baseline,
        "baseline_comparison": _token_comparison(
            baseline["token_ids"], target["token_ids"]
        ),
        "comparison_policy": (
            "Complete isolated greedy sequences are compared with eager target and legacy DSpark references. "
            "Token differences are numerical diagnostics across quantized layouts; functional checks require "
            "real accepted drafts, streamed block indices, graph replay, idle/resume and cache recovery."
        ),
        "cases": [],
        "status": "running",
    }

    def persist():
        if directory is not None:
            (directory / "report.json").write_text(json.dumps(result, indent=2))

    persist()
    for name in selected:
        topology = catalog[name]
        print(
            f"[deepseek-v4] validating DSpark K={speculative_tokens}, {name}: {topology}",
            flush=True,
        )
        try:
            records = _spawn(
                _run_rank,
                model_path,
                prompts,
                topology["attn_dp"],
                topology,
                graphs,
                None,
                decode_capacity,
                speculative_tokens,
                world_size=world_size,
                failure_directory=None
                if directory is None
                else directory / f"{name}-workers",
            )
            report = {"name": name, "ranks": records}
            result["cases"].append(report)
            persist()
            report.update(
                _validate_records(
                    records,
                    topology["attn_dp"],
                    vocab_size,
                    topology=topology,
                    graphs=graphs,
                )
            )
            report["speculative_checks"] = _validate_speculative_records(
                records, speculative_tokens
            )
            report["greedy_comparisons"] = [
                {
                    "rank": record["rank"],
                    "target": _token_comparison(
                        record["driver_result"]["probes"][0]["token_ids"],
                        target["token_ids"],
                    ),
                    "legacy_dspark": _token_comparison(
                        record["driver_result"]["probes"][0]["token_ids"],
                        baseline["token_ids"],
                    ),
                }
                for record in records
                if record["driver"]
            ]
            report["status"] = "passed"
            if directory is not None:
                (directory / f"{name}.json").write_text(json.dumps(report, indent=2))
            persist()
        except BaseException as error:
            result.update(
                status="failed",
                failed_case=name,
                error=f"{type(error).__name__}: {error}",
            )
            persist()
            raise
    result["status"] = "passed"
    persist()
    return result


def run_composable_validation(
    model_path,
    prompts,
    *,
    vocab_size,
    diagnostic_path=None,
    case_names=None,
    baseline_path=None,
    graphs=True,
    decode_capacity=CONTEXT_LENGTH,
    speculative_tokens=0,
):
    """Compare composed graph layouts with an independent eager TP reference."""

    world_size = torch.cuda.device_count()
    if world_size not in (2, 4):
        raise RuntimeError("composable validation requires two or four CUDA devices")
    if type(decode_capacity) is not int or not 2 <= decode_capacity <= CONTEXT_LENGTH:
        raise ValueError(f"decode_capacity must be between 2 and {CONTEXT_LENGTH}")
    if any(
        not prompt or len(prompt) + 6 > CONTEXT_LENGTH for prompt in prompts.values()
    ):
        raise ValueError("validation prompts must fit the bounded context")
    catalog = dict(_composable_cases(world_size))
    selected = tuple(catalog) if case_names is None else tuple(case_names)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - catalog.keys()
    ):
        raise ValueError(f"case_names must be unique choices from {tuple(catalog)}")
    if type(speculative_tokens) is not int or speculative_tokens < 0:
        raise ValueError("speculative_tokens must be a non-negative integer")
    if speculative_tokens:
        return _run_speculative_validation(
            model_path,
            prompts,
            vocab_size=vocab_size,
            diagnostic_path=diagnostic_path,
            selected=selected,
            catalog=catalog,
            baseline_path=baseline_path,
            graphs=graphs,
            decode_capacity=decode_capacity,
            speculative_tokens=speculative_tokens,
            world_size=world_size,
        )
    directory = None if diagnostic_path is None else Path(diagnostic_path)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "model_path": str(Path(model_path).resolve()),
        "world_size": world_size,
        "prompts": prompts,
        "attention_contract": "fixed-swa128-canonical-selected-capacity",
    }
    if baseline_path is None:
        print(
            f"[deepseek-v4] loading independent eager TP{world_size} reference",
            flush=True,
        )
        baseline = _spawn(
            _run_tp_baseline_rank,
            model_path,
            prompts,
            world_size,
            True,
            world_size=world_size,
        )[0]
        expected = baseline.pop("_teacher_logits")
        decode_reference = baseline.pop("_decode_trace", None)
    else:
        saved = torch.load(baseline_path, weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(
                "reused baseline must match checkpoint, GPU count, prompts and attention contract"
            )
        baseline, expected = saved["report"], saved["teacher_logits"]
        decode_reference = saved.get("decode_trace")
        print(f"[deepseek-v4] reusing eager baseline from {baseline_path}", flush=True)
    if directory is not None:
        torch.save(
            {
                "numerical_version": 2 if decode_reference is not None else 1,
                "identity": identity,
                "report": baseline,
                "teacher_logits": expected,
                "decode_trace": decode_reference,
            },
            directory / "baseline.pt",
        )
        (directory / "baseline.json").write_text(json.dumps(baseline, indent=2))
    reports = []
    result = {
        "graphs": {
            "prefill": "breakable" if graphs else "disabled",
            "decode": "full" if graphs else "disabled",
        },
        "decode_shapes": ((1, decode_capacity), (2, CONTEXT_LENGTH)) if graphs else (),
        "speculative_tokens": 0,
        "world_size": world_size,
        "selected_cases": selected,
        "diagnostic_path": None if directory is None else str(directory),
        "eager_tp_baseline": baseline,
        "decode_reference_available": decode_reference is not None,
        "comparison_policy": (
            "Fixed prefill and decode contexts use unchanged BF16 logit tolerances. "
            "Decode records genuine sampler outputs before fixing only the Engine's "
            "next token; graph samplers remain captured. Legacy baselines omit decode "
            "validation. CP replicates KV/compressor state."
        ),
        "cases": reports,
        "numerical_errors": [],
        "status": "running",
    }

    def persist():
        if directory is not None:
            (directory / "report.json").write_text(json.dumps(result, indent=2))

    persist()
    for name in selected:
        topology = catalog[name]
        print(f"[deepseek-v4] validating {name}: {topology}", flush=True)
        try:
            records = _spawn(
                _run_rank,
                model_path,
                prompts,
                topology["attn_dp"],
                topology,
                graphs,
                decode_reference,
                decode_capacity,
                world_size=world_size,
            )
            actual_logits = {
                record["rank"]: record.pop("_teacher_logits")
                for record in records
                if "_teacher_logits" in record
            }
            if directory is not None:
                torch.save(actual_logits, directory / f"{name}-logits.pt")
            decode_traces = {
                record["rank"]: record.pop("_decode_trace")
                for record in records
                if "_decode_trace" in record
            }
            if directory is not None:
                torch.save(decode_traces, directory / f"{name}-decode.pt")
            comparisons = []
            decode_comparisons = []
            for record in records:
                if record["driver"]:
                    actual = actual_logits.get(record["rank"])
                    if actual is None:
                        raise AssertionError(
                            "replica driver omitted fixed-context logits"
                        )
                    comparisons.append(
                        {"rank": record["rank"], **_logit_metrics(actual, expected)}
                    )
                    if decode_reference is not None:
                        traced = decode_traces.get(record["rank"])
                        if traced is None:
                            raise AssertionError("replica driver omitted decode trace")
                        decode_comparisons.append(
                            {
                                "rank": record["rank"],
                                **_decode_comparison(traced, decode_reference),
                            }
                        )
            report = {
                "name": name,
                "parallelism": topology,
                "ranks": records,
                "teacher_forced_comparisons": comparisons,
                "teacher_forced_decode": {
                    "status": (
                        "passed"
                        if all(value["passed"] for value in decode_comparisons)
                        else "failed"
                    )
                    if decode_reference is not None
                    else "skipped_legacy_baseline",
                    "ranks": decode_comparisons,
                },
            }
            reports.append(report)
            if directory is not None:
                (directory / f"{name}.json").write_text(json.dumps(report, indent=2))
            persist()
            report.update(
                _validate_records(
                    records,
                    topology["attn_dp"],
                    vocab_size,
                    topology=topology,
                    graphs=graphs,
                )
            )
            report["probe_matches_eager_tp"] = [
                record["driver_result"]["probes"][0]["token_ids"]
                == baseline["token_ids"]
                for record in records
                if record["driver"]
            ]
            if any(not value["contexts_match"] for value in decode_comparisons):
                raise AssertionError("decode did not preserve baseline token contexts")
            report["numerical_errors"] = [
                {
                    "kind": "same-context model logits",
                    "rank": comparison["rank"],
                    "comparison": comparison,
                }
                for comparison in comparisons
                if not comparison["passed"]
            ] + [
                {
                    "kind": "fixed-context decode logits",
                    "rank": decoded["rank"],
                    "comparison": comparison,
                }
                for decoded in decode_comparisons
                for comparison in decoded["comparisons"]
                if not comparison["passed"]
            ]
            report["status"] = "failed" if report["numerical_errors"] else "passed"
            result["numerical_errors"].extend(
                {"case": name, **error} for error in report["numerical_errors"]
            )
            if directory is not None:
                (directory / f"{name}.json").write_text(json.dumps(report, indent=2))
            persist()
        except BaseException as error:
            result.update(
                status="failed",
                failed_case=name,
                error=f"{type(error).__name__}: {error}",
            )
            persist()
            raise
    if result["numerical_errors"]:
        failed = [report["name"] for report in reports if report["numerical_errors"]]
        failures = tuple(
            dict.fromkeys(
                f"{error['case']}: {error['kind']} exceed tolerance"
                for error in result["numerical_errors"]
            )
        )
        result.update(
            status="failed",
            failed_case=failed[0],
            failed_cases=failed,
            error="; ".join(failures),
        )
        persist()
        raise AssertionError(result["error"])
    result["status"] = "passed"
    persist()
    return result
