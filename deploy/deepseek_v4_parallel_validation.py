"""Bounded full-checkpoint validation over composed CUDA rank meshes."""

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


class _ObservedRunner:
    def __init__(self, runner):
        self.runner = runner
        self.calls: list[dict[str, object]] = []
        self.pool_ids: list[int] = []
        self.released: list[int] = []

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
        return self.runner.forward_distributed(plan, context)


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


def _new_runner(rank, model_path, *, groups=None, world_size=WORLD_SIZE, graphs=False):
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
        cuda_graph_decode_shapes=((1, CONTEXT_LENGTH), (2, CONTEXT_LENGTH))
        if graphs
        else None,
        speculative_tokens=0,
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


def _driver_workload(worker, prompts, *, numerical_directory=None):
    groups = worker.groups
    dp = groups.topology.coordinates(groups.rank)[0]
    engine = DistributedEngine(worker, **_engine_options())
    requests, events, cancellations = {}, [], []
    try:

        def submit(name, prompt, limit):
            request = engine.submit(
                prompt, max_new_tokens=limit, sampling_params=_sampling()
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
                prompts["late"], max_new_tokens=4, sampling_params=_sampling()
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
                prompts["probe"], max_new_tokens=4, sampling_params=_sampling()
            )
            probe_events = []
            _drain(engine, probe_events)
            probes.append({"token_ids": probe.take_result(), "events": probe_events})
        if numerical_directory is not None:
            torch.save(
                _teacher_logits(engine, worker.runner.runner, prompts),
                Path(numerical_directory, f"{groups.rank}.pt"),
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
            _new_runner(rank, model_path, groups=groups, graphs=graphs)
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
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_memory_bytes": torch.cuda.mem_get_info()[0],
            "pool_ids": runner.pool_ids,
            "released_ids": runner.released,
            "calls": runner.calls,
            "driver_result": result,
        }
        Path(report_directory, f"{rank}.json").write_text(json.dumps(report))
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
):
    _initialize(rank, port, world_size)
    runner = _new_runner(rank, model_path, world_size=world_size)
    engine = Engine(runner, **_engine_options())
    try:
        request = engine.submit(
            prompts["probe"], max_new_tokens=4, sampling_params=_sampling()
        )
        events = []
        _drain(engine, events)
        result = {"token_ids": request.take_result(), "events": events}
        if numerical:
            torch.save(
                _teacher_logits(engine, runner, prompts),
                Path(report_directory, f"{rank}.pt"),
            )
        torch.cuda.synchronize()
        dist.barrier()
        Path(report_directory, f"{rank}.json").write_text(json.dumps(result))
    finally:
        engine.close()
        runner.close_cuda_graphs()
        dist.destroy_process_group()


def _spawn(target, model_path, prompts, *extra, world_size=WORLD_SIZE):
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
        for event in [
            *driver["events"],
            *(event for probe in driver["probes"] for event in probe["events"]),
        ]:
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
    result = {
        "contexts": actual.shape[0],
        "max_abs": maximum,
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": relative,
        "max_abs_tolerance": 0.5,
        "relative_l2_tolerance": 0.02,
    }
    if not torch.isfinite(actual).all() or maximum > 0.5 or relative > 0.02:
        raise AssertionError(
            f"same-context BF16 model logits exceed tolerance: {result}"
        )
    return result


def run_composable_validation(model_path, prompts, *, vocab_size):
    """Compare composed graph layouts with an independent eager TP reference."""

    world_size = torch.cuda.device_count()
    if world_size not in (2, 4):
        raise RuntimeError("composable validation requires two or four CUDA devices")
    if any(
        not prompt or len(prompt) + 6 > CONTEXT_LENGTH for prompt in prompts.values()
    ):
        raise ValueError("validation prompts must fit the bounded context")
    print(
        f"[deepseek-v4] loading independent eager TP{world_size} reference", flush=True
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
    reports = []
    for name, topology in _composable_cases(world_size):
        print(f"[deepseek-v4] validating {name}: {topology}", flush=True)
        records = _spawn(
            _run_rank,
            model_path,
            prompts,
            topology["attn_dp"],
            topology,
            True,
            world_size=world_size,
        )
        comparisons = []
        for record in records:
            actual = record.pop("_teacher_logits", None)
            if record["driver"]:
                if actual is None:
                    raise AssertionError("replica driver omitted fixed-context logits")
                comparisons.append(
                    {"rank": record["rank"], **_logit_metrics(actual, expected)}
                )
        report = _validate_records(
            records,
            topology["attn_dp"],
            vocab_size,
            topology=topology,
            graphs=True,
        )
        report["name"] = name
        report["teacher_forced_comparisons"] = comparisons
        report["probe_matches_eager_tp"] = [
            record["driver_result"]["probes"][0]["token_ids"] == baseline["token_ids"]
            for record in records
            if record["driver"]
        ]
        reports.append(report)
    return {
        "graphs": {"prefill": "breakable", "decode": "full"},
        "speculative_tokens": 0,
        "world_size": world_size,
        "eager_tp_baseline": baseline,
        "comparison_policy": (
            "Four identical teacher-forced contexts use fixed BF16 logit tolerances; "
            "greedy sequence matches are reported separately. CP replicates KV/compressor state."
        ),
        "cases": reports,
    }
