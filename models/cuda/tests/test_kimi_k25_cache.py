from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from resin import BatchPhase, CacheRetention
from resin_models_cuda.kimi_k25.cache import (
    CacheArena,
    CacheGeometry,
    CachePlan,
    GroupPlan,
    build_cache_plan,
    plan_cache_capacity,
)


def test_target_fp8_and_draft_bf16_share_contiguous_parent_bytes() -> None:
    geometry = CacheGeometry.build(61)
    layout = geometry.layout(10)

    assert geometry.parent_units == 64
    assert geometry.draft_blocks_per_parent == 32
    assert geometry.bytes_per_parent == 64 * 64 * 576
    assert geometry.aux_blocks_per_parent == 2
    assert tuple(group.blocks_per_slab for group in layout.groups) == (1, 32, 2)
    assert all(group.retention is CacheRetention.HISTORY for group in layout.groups)

    arena = CacheArena(torch, "cpu", layout, geometry)
    assert arena.target.shape == (14 * 64, 64, 576)
    assert arena.draft.shape == (14 * 32, 64, 576)
    assert arena.aux.shape == (14, 2, 64, 7168)
    assert (
        arena.target.untyped_storage().data_ptr()
        == arena.draft.untyped_storage().data_ptr()
    )


def test_post_load_budget_accounts_for_null_and_graph_parents() -> None:
    geometry = CacheGeometry.build(61)
    plan = plan_cache_capacity(
        288 << 30,
        100 << 30,
        0.95,
        geometry,
        reserved_parent_count=3,
    )
    budget = (100 << 30) - math.ceil((288 << 30) * 0.05)

    assert plan.parent_capacity + 3 == budget // geometry.bytes_per_parent
    assert (
        plan.allocation_bytes == (plan.parent_capacity + 3) * geometry.bytes_per_parent
    )


def test_graph_page_tables_are_full_capacity_and_flashinfer_aligned() -> None:
    geometry = CacheGeometry.build(61)
    arena = CacheArena(torch, "cpu", geometry.layout(10), geometry)

    short = arena.graph_cache(1, 1)
    full = arena.graph_cache(2, 1025)

    assert short.groups[0].parent_ids.shape == (1, 2)
    assert full.groups[0].parent_ids.shape == (2, 18)


def test_plan_preserves_structured_target_and_draft_addresses() -> None:
    layout = CacheGeometry.build(61).layout(20)
    groups = tuple(
        SimpleNamespace(
            group_index=index,
            blocks_per_slab=descriptor.blocks_per_slab,
            block_indptr=np.array([0, 1], dtype=np.uint32),
            block_slab_ids=np.array([index + 3], dtype=np.uint32),
            block_slotnums=np.array(
                [min(index * 7, descriptor.blocks_per_slab - 1)],
                dtype=np.uint32,
            ),
            block_table_start_indices=np.array([0], dtype=np.uint32),
            append_slab_ids=np.array([index + 3], dtype=np.uint32),
            append_slotnums=np.array(
                [min(index * 7, descriptor.blocks_per_slab - 1)],
                dtype=np.uint32,
            ),
            append_block_offsets=np.array([0], dtype=np.uint32),
        )
        for index, descriptor in enumerate(layout.groups)
    )
    plan = SimpleNamespace(
        num_sequences=1,
        token_indptr=np.array([0, 1], dtype=np.uint32),
        positions=np.array([0], dtype=np.uint32),
        context_lengths=np.array([0], dtype=np.uint32),
        cache_groups=groups,
        phase=BatchPhase.PREFILL,
    )

    cache = build_cache_plan(plan, layout, decode_lookahead=3)

    assert cache.groups[0].parent_ids[0].tolist() == [3]
    assert cache.groups[0].child_slots[0].tolist() == [0]
    assert cache.groups[1].parent_ids[0].tolist() == [4]
    assert cache.groups[1].child_slots[0].tolist() == [7]
    assert cache.groups[2].parent_ids[0].tolist() == [5]
    assert cache.groups[2].child_slots[0].tolist() == [1]


def test_new_cache_blocks_are_cleared_without_touching_neighbor_slots() -> None:
    geometry = CacheGeometry.build(61)
    layout = geometry.layout(20)
    groups = tuple(
        SimpleNamespace(
            group_index=index,
            blocks_per_slab=descriptor.blocks_per_slab,
            block_indptr=np.array([0, 1], dtype=np.uint32),
            block_slab_ids=np.array([index + 3], dtype=np.uint32),
            block_slotnums=np.array(
                [min(index * 7, descriptor.blocks_per_slab - 1)],
                dtype=np.uint32,
            ),
            block_table_start_indices=np.array([0], dtype=np.uint32),
            append_slab_ids=np.array([index + 3], dtype=np.uint32),
            append_slotnums=np.array(
                [min(index * 7, descriptor.blocks_per_slab - 1)],
                dtype=np.uint32,
            ),
            append_block_offsets=np.array([0], dtype=np.uint32),
        )
        for index, descriptor in enumerate(layout.groups)
    )
    plan = SimpleNamespace(
        num_sequences=1,
        token_indptr=np.array([0, 1], dtype=np.uint32),
        positions=np.array([0], dtype=np.uint32),
        context_lengths=np.array([0], dtype=np.uint32),
        cache_groups=groups,
        phase=BatchPhase.PREFILL,
    )
    cache = build_cache_plan(plan, layout, decode_lookahead=3)
    arena = CacheArena(torch, "cpu", layout, geometry)
    arena.storage.fill_(0x55)

    arena.prepare_step(cache, plan.context_lengths)

    assert torch.count_nonzero(arena.storage[3]) == 0
    assert torch.count_nonzero(arena.draft[4 * 32 + 7]) == 0
    assert torch.count_nonzero(arena.draft[4 * 32 + 6]) > 0
    assert torch.count_nonzero(arena.aux[5, 1]) == 0
    assert torch.count_nonzero(arena.aux[5, 0]) > 0


def test_speculative_lookahead_page_is_cleared_before_first_append() -> None:
    geometry = CacheGeometry.build(61)
    arena = CacheArena(torch, "cpu", geometry.layout(20), geometry)

    def group(parents: list[int], slots: list[int]) -> GroupPlan:
        return GroupPlan(
            (np.asarray(parents, dtype=np.uint32),),
            (np.asarray(slots, dtype=np.uint32),),
            np.array([0], dtype=np.uint32),
            np.array([parents[0]], dtype=np.uint32),
            np.array([slots[0]], dtype=np.uint32),
            np.array([63], dtype=np.uint32),
        )

    plan = CachePlan(
        (
            group([3, 4], [0, 0]),
            group([5, 5], [7, 8]),
            group([6, 6], [0, 1]),
        )
    )
    arena.storage.fill_(0x55)

    arena.prepare_step(plan, np.array([63], dtype=np.uint32))

    assert torch.count_nonzero(arena.storage[3]) > 0
    assert torch.count_nonzero(arena.storage[4]) == 0
    assert torch.count_nonzero(arena.draft[5 * 32 + 7]) > 0
    assert torch.count_nonzero(arena.draft[5 * 32 + 8]) == 0
    assert torch.count_nonzero(arena.aux[6, 0]) > 0
    assert torch.count_nonzero(arena.aux[6, 1]) == 0
