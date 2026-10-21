"""Real CPU collectives exercise variable and fixed-capacity expert dispatch."""

from __future__ import annotations

import multiprocessing
import time
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from resin.backend.expert_parallel import ExpertParallelDispatch, fixed_expert_capacity
from resin.backend.topology import RankGroup


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

        def check(
            row_counts,
            *,
            replication=None,
            one_owner=False,
            capacity=None,
            masked=False,
        ):
            source = rank if replication is None else rank // 2
            rows = row_counts[source]
            hidden, indices, weights = _inputs(source, rows, one_owner=one_owner)
            dispatch = ExpertParallelDispatch(
                local, num_experts=8, group=group, replication=replication
            )
            context = (
                nullcontext()
                if capacity is None
                else dispatch.fixed_capacity(
                    capacity,
                    valid_rows=torch.tensor(max(0, rows - 1)) if masked else None,
                )
            )
            with context:
                output = dispatch(hidden, indices, weights)
            if masked:
                weights[max(0, rows - 1) :].zero_()
            torch.testing.assert_close(output, _reference(hidden, indices, weights))
            counts: list[int] = [0] * 4
            dist.all_gather_object(counts, received_rows[-1])
            if capacity is not None:
                assert counts == [4 * capacity] * 4
                return
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
        check([3, 2, 0, 1], capacity=4)
        check([1, 0, 4, 0], one_owner=True, capacity=4)
        check([0, 0, 0, 0], capacity=4)
        check([3, 2, 0, 1], capacity=4, masked=True)
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
        check([3, 1], replication=replication, capacity=4)
        check([0, 2], replication=replication, capacity=4)
        check([3, 1], replication=replication, capacity=4, masked=True)

        def precise_local(hidden, indices, weights):
            result = torch.zeros_like(hidden, dtype=torch.float32)
            # BF16 owner sums would lose this 2.0 residual. Idle owners must
            # also return FP32 so every reverse exchange uses the same dtype.
            for expert, value in [(0, 1000.75), (2, -998.75)]:
                if expert // 2 == rank:
                    coefficient = ((indices == expert) * weights).sum(1)
                    result += hidden.float() * coefficient[:, None] * value
            return result

        count = int(rank == 0)
        hidden = torch.ones(count, 4, dtype=torch.bfloat16)
        indices = torch.tensor([[0, 2]], dtype=torch.int64)[:count]
        weights = torch.ones(count, 2, dtype=torch.float32)
        precise = ExpertParallelDispatch(precise_local, num_experts=8, group=group)
        for context in (nullcontext(), precise.fixed_capacity(2)):
            with context:
                result = precise(hidden, indices, weights)
            assert result.dtype == hidden.dtype
            torch.testing.assert_close(result, torch.full_like(hidden, 2))

        def ordered_local(hidden, indices, weights):
            value = (2**24, 1, -(2**24), 1)[rank]
            owned = (indices // 2 == rank).float() * weights
            return hidden.float() * owned.sum(1, keepdim=True) * value

        # FP32 arithmetic in rank order leaves one; summing the two small
        # partials together before cancellation would leave two instead.
        for replicas in (None, replication):
            ordered = ExpertParallelDispatch(
                ordered_local, num_experts=8, group=group, replication=replicas
            )
            source = rank if replicas is None else rank // 2
            count = (2, 0, 3, 1)[source]
            for live in (count, 0, count):
                hidden = torch.ones(live, 4, dtype=torch.bfloat16)
                indices = torch.tensor([0, 2, 4, 6]).expand(live, 4)
                weights = torch.ones(live, 4)
                actual = ordered(hidden, indices, weights)
                torch.testing.assert_close(
                    actual, torch.ones_like(hidden), rtol=0, atol=0
                )
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


def test_fixed_capacity_context_validates_and_restores():
    dispatch = ExpertParallelDispatch(
        lambda hidden, indices, weights: hidden,
        num_experts=2,
        group=RankGroup((0, 1), None, 0),
    )
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            # Invalid capacities exercise runtime validation.
            with dispatch.fixed_capacity(value):  # ty: ignore[invalid-argument-type]
                pass
    with dispatch.fixed_capacity(3):
        assert dispatch._capacity == 3
        with dispatch.fixed_capacity(2):
            assert dispatch._capacity == 2
        assert dispatch._capacity == 3
        with pytest.raises(ValueError, match="4 rows but capacity is 3"):
            dispatch(torch.empty(4, 2), torch.empty(4, 1), torch.empty(4, 1))
    assert dispatch._capacity is None
    with pytest.raises(ValueError, match="one integer tensor value"):
        with dispatch.fixed_capacity(3, valid_rows=torch.tensor(1.5)):
            pass


def test_model_context_selects_and_restores_all_expert_transports():
    class Layer(torch.nn.Module):
        dispatch: ExpertParallelDispatch

    layers = [Layer(), Layer()]
    model = torch.nn.Module()
    setattr(model, "layers", torch.nn.ModuleList(layers))
    for layer in layers:
        setattr(
            layer,
            "dispatch",
            ExpertParallelDispatch(
                lambda hidden, indices, weights: hidden,
                num_experts=2,
                group=RankGroup((0, 1), None, 0),
            ),
        )
    valid_rows = torch.tensor(2, dtype=torch.int32)
    with fixed_expert_capacity(model, 4, valid_rows=valid_rows):
        for layer in layers:
            assert layer.dispatch._capacity == 4
            assert layer.dispatch._valid_rows is valid_rows
    for layer in layers:
        assert layer.dispatch._capacity is None
        assert layer.dispatch._valid_rows is None
