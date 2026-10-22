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
            (
                (3,)
                if WORLD_SIZE == 2
                else tuple((0, 3)[i % 2] for i in range(WORLD_SIZE // 2))
            )
            if replicated
            else tuple((0, 5, 1, 3)[i % 4] for i in range(WORLD_SIZE))
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

        # Repeated eager combines must preserve owner-rank summation order,
        # including a completely idle source and non-source attention copies.
        values = (
            (1000.75, -998.75)
            if WORLD_SIZE == 2
            else (2**24, 1, -(2**24), 1) + (0,) * (WORLD_SIZE - 4)
        )

        def ordered_local(hidden, indices, weights):
            owned = (indices >= start) & (indices < stop)
            return (
                hidden.float() * (owned * weights).sum(1, keepdim=True) * values[rank]
            )

        ordered = ExpertParallelDispatch(
            ordered_local,
            num_experts=8,
            group=groups.ep,
            replication=groups.attn_replication if replicated else None,
        )
        ordered_replays = 0
        for count in (sizes[owner], 0, sizes[owner]):
            hidden = torch.ones(count, 16, device=device, dtype=torch.bfloat16)
            indices = (
                torch.arange(WORLD_SIZE, device=device, dtype=torch.int32)
                .mul(local_experts)
                .expand(count, WORLD_SIZE)
            )
            weights = torch.ones(count, WORLD_SIZE, device=device)
            expected = torch.zeros_like(hidden, dtype=torch.float32)
            for value in values:
                expected += hidden.float() * value
            for _ in range(8):
                actual = ordered(hidden, indices, weights)
                torch.testing.assert_close(
                    actual, expected.to(hidden.dtype), rtol=0, atol=0
                )
                ordered_replays += 1
        torch.cuda.synchronize()
        progress("device synchronized")
        dist.barrier()
        progress("barrier complete")
        Path(directory, f"{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "max_abs_errors": errors,
                    "received_rows": calls,
                    "rank_ordered_replays": ordered_replays,
                }
            )
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
                # ty misbinds pytest's decorated helper and drops its reason parameter.
                pytest.fail("NCCL expert dispatch exceeded its 150 second bound")  # ty: ignore[invalid-argument-type]
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
    assert all(record["rank_ordered_replays"] == 24 for record in records)
    assert any(any(record["received_rows"]) for record in records)


def _graph_dispatch_rank(
    rank,
    rendezvous,
    directory,
    replicated,
    layout=None,
    route_limit=None,
    weight_dtype=torch.float32,
):
    from resin.backend.expert_parallel import ExpertParallelDispatch

    if route_limit is not None:
        import resin.backend.cuda.expert_parallel as transport

        transport._MAX_DESTINATIONS_PER_PASS = route_limit

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
            **(
                layout
                or {
                    "ep": WORLD_SIZE,
                    "attn_dp": WORLD_SIZE // 2 if replicated else WORLD_SIZE,
                }
            ),
        )
    )
    progress("groups ready")
    graph = None
    try:
        capacity, width, top_k = 5, 16, 3
        local_experts = 8 // groups.ep.size
        start = local_experts * groups.ep.local_rank
        stop = start + local_experts
        coefficients = torch.arange(1, 9, device=device, dtype=torch.float32)
        # Distinct owners must retain these FP32 partials until combination.
        coefficients[0], coefficients[4] = 1000.75, -998.75
        inspected = []

        def local(hidden, indices, weights):
            assert weights.dtype == weight_dtype
            inspected.append((hidden, indices))
            owned = (indices >= start) & (indices < stop)
            factors = coefficients[indices.long().clamp_max(7)] * weights * owned
            return hidden.float() * factors.sum(1, keepdim=True) / groups.expert_tp.size

        dispatch = ExpertParallelDispatch(
            local,
            num_experts=8,
            groups=groups if layout else None,
            group=groups.ep if not layout else None,
            replication=(groups.attn_tp, groups.attn_replication)
            if layout
            else groups.attn_replication
            if replicated
            else None,
        )
        hidden = torch.empty(capacity, width, device=device, dtype=torch.bfloat16)
        indices = torch.empty(capacity, top_k, device=device, dtype=torch.int32)
        weights = torch.empty(capacity, top_k, device=device, dtype=weight_dtype)
        valid_rows = torch.empty((), device=device, dtype=torch.int32)
        dp_rank, cp_rank, _, _ = groups.topology.coordinates(rank)
        owner = dp_rank * groups.cp.size + cp_rank

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
        foreign_device = (
            (rank + 1) % WORLD_SIZE if layout is None and not replicated else rank
        )

        def run_dispatch():
            # Device-explicit runners must work even when their caller has a
            # different current device. This covers first allocation and capture.
            with torch.cuda.device(foreign_device):
                output = dispatch(hidden, indices, weights)
                assert torch.cuda.current_device() == foreign_device
                return output

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.cuda.stream(stream),
            dispatch.fixed_capacity(capacity, valid_rows=valid_rows),
        ):
            for warmup in range(3):
                progress(f"warmup {warmup}: launch")
                run_dispatch()
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
            actual = run_dispatch()
            progress("capture: dispatch enqueued")
        progress("capture: complete")
        errors, network_rows = [], []
        from flashinfer.comm.trtllm_moe_alltoall import get_moe_alltoall_module

        transport = next(iter(dispatch._transports.values()))
        assert transport.workspace.device == hidden.device
        assert all(key[-1] == hidden.device for key in dispatch._transports)
        names, offsets = get_moe_alltoall_module().moe_a2a_get_metainfo_index_pairs()
        metadata = dict(zip(names, offsets, strict=True))
        offset = int(
            transport.metainfo[int(metadata["MOE_A2A_SEND_COUNTERS_OFFSET_INDEX"])]
        )
        send_counts = transport.workspace[rank, offset : offset + 4 * WORLD_SIZE].view(
            torch.int32
        )
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
            received_hidden, received_indices = inspected[-1]
            padding = (received_hidden == 0).all(dim=1)
            assert torch.all(received_indices[padding] == 8), (
                "padding must select no expert GEMM"
            )
            # Count actual GPU-side copies, independently of the static receive
            # allocation. Padding and redundant attention copies must stay local;
            # a routed row is sent once to each selected physical expert owner.
            counts = [0] * WORLD_SIZE
            source = all(copy.local_rank == 0 for copy in dispatch.replication)
            for row, expert_ids in enumerate(indices.cpu().tolist()):
                destinations = {rank}
                if source and row < int(valid_rows):
                    destinations = {
                        (expert // local_experts) * groups.expert_tp.size + lane
                        for expert in expert_ids
                        for lane in range(groups.expert_tp.size)
                    }
                for destination in destinations:
                    counts[destination] += 1
            if route_limit is None:
                assert send_counts.cpu().tolist() == counts
                network_rows.append(sum(counts) - counts[rank])
            progress(f"replay {phase}: assertion passed")
            errors.append(float((actual.float() - expected.float()).abs().max()))
        progress("replays: synchronizing")
        torch.cuda.synchronize()
        progress("replays: synchronized; entering barrier")
        dist.barrier()
        progress("replays: barrier complete")
        Path(directory, f"{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "replays": 4,
                    "max_abs_errors": errors,
                    "network_rows": network_rows,
                    "network_payload_bytes": [
                        count * (width * 6 + top_k * 8) for count in network_rows
                    ],
                    "dense_payload_bytes": capacity
                    * (WORLD_SIZE - 1)
                    * (width * 6 + top_k * 8),
                }
            )
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
    tmp_path, replicated, layout=None, route_limit=None, weight_dtype=torch.float32
):
    context = mp.spawn(
        _graph_dispatch_rank,
        args=(
            f"file://{tmp_path / 'rendezvous'}",
            str(tmp_path),
            replicated,
            layout,
            route_limit,
            weight_dtype,
        ),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("NCCL graph dispatch exceeded its 150 second bound")  # ty: ignore[invalid-argument-type]
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
    assert all(
        not record["network_rows"] or record["network_rows"][1] == 0
        for record in records
    )
    print(
        "[sparse-expert-payload] "
        + json.dumps({"layout": layout, "replicated": replicated, "ranks": records})
    )


@pytest.mark.skipif(
    WORLD_SIZE not in (2, 4, 8), reason="composition checks use two/four/eight GPUs"
)
@pytest.mark.parametrize(
    "layout",
    (
        [
            {"ep": 2, "attn_dp": 2, "attn_tp": 2},
            {"ep": 1, "attn_dp": 2, "attn_tp": 2},
            {"ep": 2, "cp": 2, "attn_tp": 2},
            {"ep": 2, "attn_tp": 2},
        ]
        if WORLD_SIZE == 4
        else [{"ep": 2, "attn_dp": 2, "cp": 2, "attn_tp": 2}]
        if WORLD_SIZE == 8
        else [{"ep": 1, "attn_dp": 2}, {"ep": 2, "attn_tp": 2}, {"ep": 2, "cp": 2}]
    ),
)
def test_sparse_expert_graph_composes_attention_mesh_and_expert_tp(tmp_path, layout):
    test_nccl_expert_dispatch_cuda_graph_replays_routes_and_idle_rows(
        tmp_path, False, layout
    )


@pytest.mark.skipif(
    WORLD_SIZE not in (2, 4, 8),
    reason="forced multi-pass check uses two/four/eight GPUs",
)
def test_sparse_expert_graph_rank_partitioned_multipass(tmp_path):
    layout = (
        {"ep": 1, "attn_dp": 2}
        if WORLD_SIZE == 2
        else {"ep": 2, "attn_dp": 2, "attn_tp": 2}
        if WORLD_SIZE == 4
        else {"ep": 2, "attn_dp": 2, "cp": 2, "attn_tp": 2}
    )
    test_nccl_expert_dispatch_cuda_graph_replays_routes_and_idle_rows(
        tmp_path, False, layout, route_limit=1 if WORLD_SIZE == 2 else 2
    )


def test_sparse_expert_graph_preserves_bf16_router_weights(tmp_path):
    test_nccl_expert_dispatch_cuda_graph_replays_routes_and_idle_rows(
        tmp_path, False, weight_dtype=torch.bfloat16
    )
