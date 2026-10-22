"""Load one CUDA model per rank and run a resident, synchronous Engine sweep."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from resin import Engine, Parallelism, SchedulerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups


def _runner(model, checkpoint, *, device, groups, cases, options, context_capacity):
    common: dict[str, Any] = dict(
        device=device,
        parallel_groups=groups,
        cuda_graph_prefill=options["prefill_graph"],
        cuda_graph_decode=options["decode_graph"],
        gpu_memory_utilization=options["gpu_memory_utilization"],
        fast_allreduce=False,
    )
    batches = sorted({case.batch_size for case in cases})
    if model.startswith("gpt-oss"):
        from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        if options["draft_tokens"]:
            raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")

        class BenchGptRunner(GptOssCudaRunner):
            prefill_chunk_size = options["prefill_chunk_size"]

        return BenchGptRunner(
            checkpoint,
            kernel_moe=FlashInferSM90MXFP4Kernel(),
            cuda_graph_decode_shapes=(
                tuple((batch, (context_capacity + 127) // 128) for batch in batches)
                if options["decode_graph"] != "disabled"
                else ()
            ),
            **common,
        )
    if model == "deepseek-v4":
        from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

        runner_type = DeepSeekV4CudaRunner
    else:
        raise ValueError(f"unknown benchmark model: {model}")
    return runner_type(
        checkpoint,
        max_seq_len=context_capacity,
        max_batch_size=max(batches),
        speculative_tokens=options["draft_tokens"],
        prefill_chunk_size=options["prefill_chunk_size"],
        cuda_graph_decode_shapes=(
            tuple((batch, context_capacity) for batch in batches)
            if options["decode_graph"] != "disabled"
            else ()
        ),
        **common,
    )


def _rank(rank, port, report_path, model, checkpoint, case_values, topology, options):
    from resin.bench import (
        BenchCase,
        BenchmarkRunner,
        distributed_reducer,
        run_sweep,
    )

    if model.startswith("gpt-oss"):
        from deploy.jit_cache import install_flashinfer_hash_cache

        install_flashinfer_hash_cache()
    torch.cuda.set_device(rank)
    groups = None
    if topology.world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=topology.world_size,
            device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=30),
        )
        groups = ParallelGroups.create(topology)
    cases = [BenchCase(**case) for case in case_values]
    # Reserve real generated positions, including draft write lookahead. Context
    # lengths on the charts remain the prompt lengths recorded by the harness.
    headroom = (options["warmup_steps"] + options["measure_steps"] + 2) * (
        options["draft_tokens"] + 1
    )
    capacity = max(case.context_length for case in cases) + headroom
    started = time.perf_counter()
    raw = _runner(
        model,
        checkpoint,
        device=f"cuda:{rank}",
        groups=groups,
        cases=cases,
        options=options,
        context_capacity=capacity,
    )
    observed = BenchmarkRunner(
        raw,
        accepted_drafts=options["accepted_drafts"],
        balanced_experts=options["balanced_experts"],
    )
    torch.cuda.synchronize()
    loaded_seconds = time.perf_counter() - started
    started = time.perf_counter()
    raw.capture_cuda_graphs()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    metadata = {
        **options["metadata"],
        "model": model,
        "checkpoint": str(checkpoint),
        "hardware": torch.cuda.get_device_name(rank),
        "parallelism": {
            axis: getattr(topology, axis)
            for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        },
        "graphs": {
            "prefill": options["prefill_graph"],
            "decode": options["decode_graph"],
        },
        "quant": "checkpoint",
        "load_seconds": loaded_seconds,
        "capture_seconds": capture_seconds,
        "prefill_chunk_size": options["prefill_chunk_size"],
    }
    engine_options: dict[str, Any] = dict(
        max_running_requests=max(case.batch_size for case in cases),
        context_length=capacity,
        eos_token_id=None,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=options["prefill_chunk_size"],
            prefill_chunk_size=options["prefill_chunk_size"],
        ),
    )
    worker = None if groups is None else DistributedWorker(observed, groups)
    if groups is not None and worker is not None and not groups.is_driver:
        worker.serve()
        os._exit(0)
    engine = (
        Engine(observed, **engine_options)
        if worker is None
        else DistributedEngine(worker, **engine_options)
    )
    try:
        # A seeded synthetic prompt avoids tokenization overhead and special
        # token stops. Every replica has distinct, reproducible input tokens.
        generator = torch.Generator().manual_seed(41 + rank)
        prompt = torch.randint(
            1000, min(30_000, raw.vocab_size), (capacity,), generator=generator
        ).tolist()
        records = run_sweep(
            engine,
            cases,
            runner=observed,
            prompt_tokens=prompt,
            warmup_steps=options["warmup_steps"],
            measure_steps=options["measure_steps"],
            gpu_count=topology.world_size,
            attn_dp=topology.attn_dp,
            synchronize=torch.cuda.synchronize,
            reduce_step=None if groups is None else distributed_reducer(groups),
            metadata=metadata,
        )
        if rank == 0:
            Path(report_path).write_text(json.dumps(records))
    finally:
        engine.close()
    # Spawned process exit releases graph-owned NCCL communicators together.
    if groups is not None:
        os._exit(0)


def run_benchmark(
    model: str,
    checkpoint: str | Path,
    cases: list[dict],
    *,
    parallelism: Parallelism | None = None,
    draft_tokens: int = 4,
    accepted_drafts: int | None = 3,
    balanced_experts: bool = True,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_chunk_size: int = 4096,
    prefill_graph: str = "breakable",
    decode_graph: str = "full",
    gpu_memory_utilization: float = 0.9,
    metadata: dict | None = None,
) -> list[dict]:
    """One load and graph catalog for all compatible points in this sweep."""
    from resin.bench import BenchCase

    topology = parallelism or Parallelism()
    if not cases:
        raise ValueError("the benchmark needs at least one case")
    for values in cases:
        BenchCase(**values)
    if (
        draft_tokens < 0
        or accepted_drafts is not None
        and not 0 <= accepted_drafts <= draft_tokens
    ):
        raise ValueError("accepted drafts must be between zero and draft tokens")
    if model.startswith("gpt-oss") and draft_tokens:
        raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")
    if torch.cuda.device_count() < topology.world_size:
        raise ValueError(f"the sweep requires {topology.world_size} visible GPUs")
    options = dict(
        draft_tokens=draft_tokens,
        accepted_drafts=accepted_drafts,
        balanced_experts=balanced_experts,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        prefill_chunk_size=prefill_chunk_size,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata=metadata or {},
    )
    with tempfile.TemporaryDirectory(prefix="resin-bench-") as directory:
        output = Path(directory) / "results.json"
        if topology.world_size == 1:
            _rank(0, 0, output, model, checkpoint, cases, topology, options)
        else:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            torch.multiprocessing.spawn(
                _rank,
                args=(port, output, model, checkpoint, cases, topology, options),
                nprocs=topology.world_size,
                join=True,
            )
        return json.loads(output.read_text())


def write_results(path: str | Path, records: list[dict]) -> None:
    """Append raw points; plotting deliberately resolves repeated points later."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["run_benchmark", "write_results"]
