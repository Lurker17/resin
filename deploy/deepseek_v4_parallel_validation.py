"""Bounded full-checkpoint EP/attention-DP validation on four CUDA ranks."""

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


def _initialize(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=10),
        device_id=torch.device(f"cuda:{rank}"),
    )


def _new_runner(rank, model_path, *, groups=None):
    from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

    parallel = (
        {"parallel_groups": groups}
        if groups is not None
        else {
            "tensor_parallel": TensorParallelContext(
                TensorParallelLayout(rank=rank, size=WORLD_SIZE), dist.group.WORLD
            )
        }
    )
    return DeepSeekV4CudaRunner(
        model_path,
        max_seq_len=CONTEXT_LENGTH,
        max_batch_size=2,
        device=f"cuda:{rank}",
        cuda_graph_prefill="disabled",
        cuda_graph_decode="disabled",
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


def _driver_workload(worker, prompts):
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


def _run_rank(rank, port, report_directory, model_path, prompts, attn_dp):
    _initialize(rank, port)
    groups = ParallelGroups.create(
        Parallelism(world_size=WORLD_SIZE, ep=WORLD_SIZE, attn_dp=attn_dp)
    )
    try:
        start = time.perf_counter()
        runner = _ObservedRunner(_new_runner(rank, model_path, groups=groups))
        torch.cuda.synchronize()
        dist.barrier()
        loaded_seconds = time.perf_counter() - start
        worker = DistributedWorker(runner, groups)
        if groups.is_driver:
            result = _driver_workload(worker, prompts)
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
            "pool_ids": runner.pool_ids,
            "released_ids": runner.released,
            "calls": runner.calls,
            "driver_result": result,
        }
        Path(report_directory, f"{rank}.json").write_text(json.dumps(report))
    finally:
        groups.close()
        dist.destroy_process_group()


def _run_tp_baseline_rank(rank, port, report_directory, model_path, prompts):
    _initialize(rank, port)
    runner = _new_runner(rank, model_path)
    engine = Engine(runner, **_engine_options())
    try:
        request = engine.submit(
            prompts["probe"], max_new_tokens=4, sampling_params=_sampling()
        )
        events = []
        _drain(engine, events)
        result = {"token_ids": request.take_result(), "events": events}
        torch.cuda.synchronize()
        dist.barrier()
        Path(report_directory, f"{rank}.json").write_text(json.dumps(result))
    finally:
        engine.close()
        dist.destroy_process_group()


def _spawn(target, model_path, prompts, *extra):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="resin-dsv4-parallel-") as directory:
        context = torch.multiprocessing.spawn(
            target,
            args=(port, directory, str(model_path), prompts, *extra),
            nprocs=WORLD_SIZE,
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
        return [
            json.loads(Path(directory, f"{rank}.json").read_text())
            for rank in range(WORLD_SIZE)
        ]


def _validate_records(records, attn_dp, vocab_size):
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
        if not any(call["placeholder"] for call in records[3]["calls"]):
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
    return {
        "parallelism": {
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
