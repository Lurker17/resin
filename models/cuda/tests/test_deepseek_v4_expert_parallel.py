"""Real CPU collectives exercise the CUDA model's eager EP routing contract."""

from __future__ import annotations

import multiprocessing
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from resin.backend.topology import RankGroup
from resin_models_cuda.deepseek_v4.experts import ExpertParallelDispatch


def _inputs(source: int, rows: int, *, one_owner: bool = False):
    hidden = torch.arange(rows * 4, dtype=torch.float32).view(rows, 4) / 8 + source
    indices = (torch.arange(rows * 3, dtype=torch.int64).view(rows, 3) + source * 2) % 8
    if one_owner:
        # Two experts owned by the same rank must cause only one token transfer.
        indices = indices % 2
    weights = torch.tensor([0.125, 0.375, 0.5]).expand(rows, 3).contiguous()
    return hidden, indices, weights


def _reference(hidden, indices, weights):
    coefficients = indices.float() + 1
    return (
        (hidden[:, None, :] * coefficients[..., None] + coefficients[..., None] / 8)
        * weights[..., None]
    ).sum(1)


def _ep_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        group = RankGroup(tuple(range(4)), dist.group.WORLD, rank)
        received_rows = []

        def local(hidden, indices, weights):
            received_rows.append(hidden.shape[0])
            result = torch.zeros_like(hidden)
            for expert in range(rank * 2, rank * 2 + 2):
                coefficient = (((indices == expert) * weights).sum(1) * (expert + 1))[
                    :, None
                ]
                result += (hidden + 0.125) * coefficient
            return result

        def check(row_counts, *, replication=None, one_owner=False):
            source = rank if replication is None else rank // 2
            rows = row_counts[source]
            hidden, indices, weights = _inputs(source, rows, one_owner=one_owner)
            dispatch = ExpertParallelDispatch(
                local, num_experts=8, group=group, replication=replication
            )
            output = dispatch(hidden, indices, weights)
            torch.testing.assert_close(output, _reference(hidden, indices, weights))
            counts = [None] * 4
            dist.all_gather_object(counts, received_rows[-1])
            expected_transfers = 0
            for source, count in enumerate(row_counts):
                _, expert_ids, _ = _inputs(source, count, one_owner=one_owner)
                expected_transfers += sum(
                    len(set(row)) for row in (expert_ids // 2).tolist()
                )
            assert sum(counts) == expected_transfers

        # Different requests on every source, one completely idle source, and
        # several selected experts sharing an owner all exercise real splits.
        check([3, 2, 0, 1])
        check([1, 0, 4, 0], one_owner=True)
        check([0, 0, 0, 0])
        replication = None
        handles = []
        for ranks in [(0, 1), (2, 3)]:
            handle = dist.new_group(list(ranks), backend="gloo")
            handles.append(handle)
            if rank in ranks:
                replication = RankGroup(ranks, handle, ranks.index(rank))
        # Replicated attention sends once and broadcasts complete result rows.
        # Non-source ranks remain expert owners and receive remote tokens.
        check([3, 1], replication=replication)
        check([0, 2], replication=replication)
        for handle in handles:
            if handle != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(handle)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_expert_dispatch_routes_unequal_batches_empty_sources_and_replication(tmp_path):
    context = multiprocessing.get_context("spawn")
    rendezvous = str(tmp_path / "expert-parallel")
    processes = [
        context.Process(target=_ep_worker, args=(rank, rendezvous)) for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
