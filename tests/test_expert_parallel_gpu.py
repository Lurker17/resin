"""Real NCCL expert dispatch checked against independent arithmetic."""

from __future__ import annotations

import faulthandler
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin import Parallelism
from resin.backend.topology import ParallelGroups

WORLD_SIZE = int(os.environ.get("RESIN_EP_TEST_WORLD_SIZE", "4"))

pytestmark = pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices and NCCL",
)


def _dispatch_rank(rank, rendezvous, directory, replicated):
    from resin.backend.expert_parallel import ExpertParallelDispatch

    faulthandler.dump_traceback_later(45, repeat=True)

    def progress(message):
        print(
            f"[NCCL reference rank={rank}, replicated={replicated}] {message}",
            flush=True,
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=2),
        device_id=device,
    )
    progress("world ready")
    groups = ParallelGroups.create(
        Parallelism(
            world_size=WORLD_SIZE,
            ep=WORLD_SIZE,
            attn_dp=WORLD_SIZE // 2 if replicated else WORLD_SIZE,
        )
    )
    progress("groups ready")
    try:
        local_experts = 8 // WORLD_SIZE
        start = local_experts * groups.ep.local_rank
        stop = start + local_experts
        calls = []

        def local(hidden, indices, weights):
            calls.append(int(hidden.shape[0]))
            partial = torch.zeros_like(hidden)
            for expert in range(start, stop):
                rows, slots = torch.where(indices == expert)
                partial.index_add_(
                    0,
                    rows,
                    hidden[rows] * (expert + 1) * weights[rows, slots, None],
                )
            return partial

        dispatch = ExpertParallelDispatch(
            local,
            num_experts=8,
            group=groups.ep,
            replication=groups.attn_replication if replicated else None,
        )
        owner = groups.topology.coordinates(rank)[0]
        sizes = (
            ((3,) if WORLD_SIZE == 2 else (0, 3))
            if replicated
            else (0, 5, 1, 3)[:WORLD_SIZE]
        )
        errors = []
        for phase, count in enumerate((sizes[owner], 0, sizes[owner])):
            progress(f"phase {phase}: {count} local rows")
            hidden = (
                torch.arange(count * 16, device=device, dtype=torch.float32).reshape(
                    count, 16
                )
                / 32
                + owner
            )
            rows = torch.arange(count, device=device, dtype=torch.int64)
            # Include repeated experts and multiple experts on one destination.
            indices = torch.stack((rows % 8, rows % 8, (rows * 3 + 7) % 8), dim=1)
            weights = (
                torch.tensor((0.25, 0.125, 0.625), device=device)
                .expand(count, 3)
                .contiguous()
            )
            expected = (
                hidden[:, None, :] * (indices + 1)[:, :, None] * weights[:, :, None]
            ).sum(dim=1)
            actual = dispatch(hidden, indices, weights)
            progress(f"phase {phase}: dispatch returned")
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            errors.append(0.0 if not count else float((actual - expected).abs().max()))
        torch.cuda.synchronize()
        progress("device synchronized")
        dist.barrier()
        progress("barrier complete")
        Path(directory, f"{rank}.json").write_text(
            json.dumps({"rank": rank, "max_abs_errors": errors, "received_rows": calls})
        )
    finally:
        groups.close()
        progress("groups closed")
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("replicated", [False, True])
def test_nccl_expert_dispatch_matches_reference_with_unequal_and_empty_rows(
    tmp_path, replicated
):
    context = mp.spawn(
        _dispatch_rank,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), replicated),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("NCCL expert dispatch exceeded its 150 second bound")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    records = [
        json.loads((tmp_path / f"{rank}.json").read_text())
        for rank in range(WORLD_SIZE)
    ]
    assert all(len(record["max_abs_errors"]) == 3 for record in records)
    assert any(any(record["received_rows"]) for record in records)


def _graph_dispatch_rank(rank, rendezvous, directory, replicated):
    from resin.backend.expert_parallel import ExpertParallelDispatch

    faulthandler.dump_traceback_later(30, repeat=True)

    def progress(message):
        print(
            f"[NCCL graph rank={rank}, replicated={replicated}] {message}",
            flush=True,
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=2),
        device_id=device,
    )
    progress("world ready")
    groups = ParallelGroups.create(
        Parallelism(
            world_size=WORLD_SIZE,
            ep=WORLD_SIZE,
            attn_dp=WORLD_SIZE // 2 if replicated else WORLD_SIZE,
        )
    )
    progress("groups ready")
    graph = None
    try:
        capacity, width, top_k = 5, 16, 3
        local_experts = 8 // WORLD_SIZE
        start = local_experts * groups.ep.local_rank
        stop = start + local_experts
        coefficients = torch.arange(1, 9, device=device, dtype=torch.float32)
        # Distinct owners must retain these FP32 partials until combination.
        coefficients[0], coefficients[4] = 1000.75, -998.75

        def local(hidden, indices, weights):
            owned = (indices >= start) & (indices < stop)
            factors = coefficients[indices.long()] * weights * owned
            return hidden.float() * factors.sum(1, keepdim=True)

        dispatch = ExpertParallelDispatch(
            local,
            num_experts=8,
            group=groups.ep,
            replication=groups.attn_replication if replicated else None,
        )
        hidden = torch.empty(capacity, width, device=device, dtype=torch.bfloat16)
        indices = torch.empty(capacity, top_k, device=device, dtype=torch.int32)
        weights = torch.empty(capacity, top_k, device=device, dtype=torch.float32)
        valid_rows = torch.empty((), device=device, dtype=torch.int32)
        owner = groups.topology.coordinates(rank)[0]

        def stage(phase):
            hidden.fill_(1)
            rows = torch.arange(capacity, device=device, dtype=torch.int32)
            if phase == 2:
                indices[:, 0], indices[:, 1], indices[:, 2] = 0, 4, 0
                weights[:, 0], weights[:, 1], weights[:, 2] = 1, 1, 0
                valid_rows.fill_(capacity)
                return
            hidden.copy_(
                torch.arange(capacity * width, device=device).reshape(capacity, width)
                / 32
                + owner
                + phase
            )
            indices.copy_(
                torch.stack((rows % 8, rows % 8, (rows * 3 + phase) % 8), dim=1)
            )
            weights.copy_(
                torch.tensor((0.25, 0.125, 0.625), device=device).expand(
                    capacity, top_k
                )
            )
            # Change active rows without changing graph shapes. This includes
            # completely idle replicas, followed by active work on replay.
            count = 0 if phase == 1 else (owner + 2) % (capacity + 1)
            valid_rows.fill_(count)

        stage(0)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.cuda.stream(stream),
            dispatch.fixed_capacity(capacity, valid_rows=valid_rows),
        ):
            for warmup in range(3):
                progress(f"warmup {warmup}: launch")
                dispatch(hidden, indices, weights)
                progress(f"warmup {warmup}: enqueued")
        progress("warmup: synchronizing")
        stream.synchronize()
        progress("warmup: synchronized; entering barrier")
        dist.barrier()
        progress("warmup: barrier complete; entering capture")
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=stream),
            dispatch.fixed_capacity(capacity, valid_rows=valid_rows),
        ):
            progress("capture: dispatch")
            actual = dispatch(hidden, indices, weights)
            progress("capture: dispatch enqueued")
        progress("capture: complete")
        errors = []
        for phase in (0, 1, 2, 3):
            progress(f"replay {phase}: staging")
            stage(phase)
            progress(f"replay {phase}: launch")
            graph.replay()
            progress(f"replay {phase}: enqueued; comparing")
            expected = (
                hidden.float()
                * (coefficients[indices.long()] * weights).sum(1, keepdim=True)
            ).to(hidden.dtype)
            expected.masked_fill_(
                (torch.arange(capacity, device=device) >= valid_rows)[:, None], 0
            )
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            progress(f"replay {phase}: assertion passed")
            errors.append(float((actual.float() - expected.float()).abs().max()))
        progress("replays: synchronizing")
        torch.cuda.synchronize()
        progress("replays: synchronized; entering barrier")
        dist.barrier()
        progress("replays: barrier complete")
        Path(directory, f"{rank}.json").write_text(
            json.dumps({"rank": rank, "replays": 4, "max_abs_errors": errors})
        )
    finally:
        if graph is not None:
            progress("releasing captured NCCL graph")
            torch.cuda.synchronize()
            graph.reset()
            progress("captured NCCL graph released")
        progress("closing groups")
        groups.close()
        progress("groups closed; closing world")
        dist.destroy_process_group()
        progress("world closed")
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("replicated", [False, True])
def test_nccl_expert_dispatch_cuda_graph_replays_routes_and_idle_rows(
    tmp_path, replicated
):
    context = mp.spawn(
        _graph_dispatch_rank,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), replicated),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("NCCL graph dispatch exceeded its 150 second bound")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    records = [
        json.loads((tmp_path / f"{rank}.json").read_text())
        for rank in range(WORLD_SIZE)
    ]
    assert all(record["replays"] == 4 for record in records)
    assert all(record["max_abs_errors"] == [0.0] * 4 for record in records)
