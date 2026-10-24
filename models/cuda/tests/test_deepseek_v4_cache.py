from __future__ import annotations

import math

import numpy as np
import torch

from resin import BatchPhase, BatchPlan, CacheGroupPlan, CacheRetention
from resin.backend.plans import PagedCachePlan
from resin_models_cuda.deepseek_v4.cache import (
    C4_STATE_GROUP,
    CacheArena,
    CacheGeometry,
    plan_cache_capacity,
)


def _geometry(
    *, decode_input_tokens: int = 6, dspark_layers: int = 3, max_batch_size: int = 1
) -> CacheGeometry:
    return CacheGeometry.build(
        [0, 0, *([4, 128] * 20), 4, 0, 0, 0],
        target_layers=43,
        dspark_layers=dspark_layers,
        window_size=128,
        decode_input_tokens=decode_input_tokens,
        max_batch_size=max_batch_size,
    )


def test_deepseek_groups_use_tokenspeed_power_of_two_packing() -> None:
    geometry = _geometry()

    assert geometry.attention_layers == 46
    assert geometry.window_tokens == 134
    assert geometry.c4_state_tokens == 13
    assert geometry.packings == (1, 4, 8, 256, 8, 32)
    assert geometry.parent_rows == 5_888
    assert geometry.bytes_per_parent == 6_029_312

    groups = geometry.layout(100).groups
    assert tuple(group.block_tokens for group in groups) == (128, 128, 4, 128, 8, 4)
    assert tuple(group.retention for group in groups) == (
        CacheRetention.WINDOW,
        CacheRetention.HISTORY,
        CacheRetention.WINDOW,
        CacheRetention.HISTORY,
        CacheRetention.WINDOW,
        CacheRetention.WINDOW,
    )
    assert tuple(group.blocks_per_slab for group in groups) == geometry.packings
    assert groups[0].window_tokens == 134
    assert groups[0].window_resident_block_count == 3
    assert tuple(group.read_window_tokens for group in groups) == (
        129,
        None,
        8,
        None,
        128,
        8,
    )


def test_non_speculative_c4_tail_only_retains_its_logical_window() -> None:
    geometry = _geometry(decode_input_tokens=1, dspark_layers=0)

    assert geometry.window_tokens == 128
    assert geometry.c4_state_tokens == 8
    assert geometry.layout(10).groups[0].window_resident_block_count == 2
    assert geometry.layout(10).groups[C4_STATE_GROUP].window_tokens == 8


def test_prefill_guard_bands_preserve_history_beside_a_full_chunk() -> None:
    geometry = CacheGeometry.build(
        [0, 0, *([4, 128] * 20), 4, 0, 0, 0],
        target_layers=43,
        dspark_layers=3,
        window_size=128,
        decode_input_tokens=6,
        prefill_chunk_size=4096,
    )
    assert geometry.max_write_tokens == 4096
    assert geometry.layout(300).groups[0].window_resident_block_count == 34
    for group in geometry.layout(300).groups:
        if group.retention is CacheRetention.WINDOW:
            assert (
                group.window_tokens is not None and group.read_window_tokens is not None
            )
            assert group.window_tokens - group.read_window_tokens + 1 == 4096


def test_dspark_short_verify_lease_keeps_the_entire_preceding_anchor_window() -> None:
    window = 128
    for writes in (2, 6, 130, 4096):
        geometry = CacheGeometry.build(
            [0, 4, 128],
            target_layers=3,
            dspark_layers=1,
            window_size=window,
            decode_input_tokens=min(writes, 6),
            prefill_chunk_size=writes,
        )
        group = geometry.layout(1000).groups[0]
        assert group.window_resident_block_count is not None
        for context in range(2 * window, 3 * window):
            last_block = (context + writes - 1) // group.block_tokens
            first_block = max(0, last_block + 1 - group.window_resident_block_count)
            # DSpark reads [context-W, context) before writing its verify block;
            # even an unaligned anchor must retain that oldest physical page.
            assert first_block * group.block_tokens <= context - window


def test_post_load_vram_budget_includes_null_parent() -> None:
    geometry = _geometry()
    plan = plan_cache_capacity(192 << 30, 96 << 30, 0.95, geometry)
    graph_plan = plan_cache_capacity(
        192 << 30,
        96 << 30,
        0.95,
        geometry,
        reserved_parent_count=7,
    )

    expected_budget = (96 << 30) - math.ceil((192 << 30) * (1.0 - 0.95))
    assert plan.cache_budget_bytes == expected_budget
    assert plan.parent_capacity + 1 == expected_budget // geometry.bytes_per_parent
    assert (
        plan.allocation_bytes == (plan.parent_capacity + 1) * geometry.bytes_per_parent
    )
    assert graph_plan.parent_capacity + 7 == plan.parent_capacity + 1
    assert graph_plan.allocation_bytes == plan.allocation_bytes


def test_graph_capture_uses_group_private_scratch_parents() -> None:
    geometry = _geometry(max_batch_size=4)
    layout = geometry.layout(3)
    arena = CacheArena("cpu", layout, geometry, dtype=torch.bfloat16)

    inputs = arena.graph_shape_capture_inputs(4, 256)

    assert (
        arena.storage.shape[0]
        == layout.parent_capacity + 1 + geometry.graph_scratch_parents
    )
    seen_parents = set()
    for group in range(len(geometry.packings)):
        parent_ids = np.asarray(inputs[f"cache.{group}.parent_ids"]).reshape(4, -1)
        child_slots = np.asarray(inputs[f"cache.{group}.child_slots"]).reshape(4, -1)
        addresses = [arena._graph_padding_address(group, row) for row in range(4)]
        assert len(set(addresses)) == 4
        assert seen_parents.isdisjoint(parent for parent, _ in addresses)
        seen_parents.update(parent for parent, _ in addresses)
        for row, (parent, slot) in enumerate(addresses):
            assert (parent_ids[row] == parent).all()
            assert (child_slots[row] == slot).all()
    assert len(seen_parents) == geometry.graph_scratch_parents


def test_plan_preserves_structured_parent_and_child_addresses() -> None:
    layout = _geometry().layout(10)
    cache_groups = tuple(
        CacheGroupPlan(
            group_index=index,
            blocks_per_slab=descriptor.blocks_per_slab,
            block_indptr=np.array([0, 1], dtype=np.uint32),
            block_slab_ids=np.array([index + 1], dtype=np.uint32),
            block_slotnums=np.array(
                [min(index, descriptor.blocks_per_slab - 1)], dtype=np.uint32
            ),
            block_table_start_indices=np.array([0], dtype=np.uint32),
            append_slab_ids=np.array([index + 1], dtype=np.uint32),
            append_slotnums=np.array(
                [min(index, descriptor.blocks_per_slab - 1)], dtype=np.uint32
            ),
            append_block_offsets=np.array([0], dtype=np.uint32),
        )
        for index, descriptor in enumerate(layout.groups)
    )
    plan = BatchPlan(
        sequence_ids=np.array([1], dtype=np.uint64),
        token_ids=np.array([7], dtype=np.uint32),
        token_indptr=np.array([0, 1], dtype=np.uint32),
        context_lengths=np.array([0], dtype=np.uint32),
        sampling_mask=np.array([True]),
        cache_groups=cache_groups,
        phase=BatchPhase.PREFILL,
    )

    cache_plan = PagedCachePlan.from_plan(plan, layout, decode_lookahead=5)

    for index, (group, descriptor) in enumerate(
        zip(cache_plan.groups, layout.groups, strict=True)
    ):
        assert group.block_slabs(0).tolist() == [index + 1]
        assert group.block_slots(0).tolist() == [
            min(index, descriptor.blocks_per_slab - 1)
        ]
