from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from resin import BatchPhase, CacheRetention
from resin.backend.plans import PagedCachePlan
from resin_models_cuda.deepseek_v4.cache import (
    C4_STATE_GROUP,
    CacheArena,
    CacheGeometry,
    plan_cache_capacity,
)


def _geometry(*, decode_input_tokens: int = 6) -> CacheGeometry:
    return CacheGeometry.build(
        [0, 0, *([4, 128] * 20), 4, 0, 0, 0],
        target_layers=43,
        dspark_layers=3,
        window_size=128,
        decode_input_tokens=decode_input_tokens,
    )


def test_deepseek_groups_use_tokenspeed_power_of_two_packing() -> None:
    geometry = _geometry()

    assert geometry.attention_layers == 46
    assert geometry.window_tokens == 133
    assert geometry.c4_state_tokens == 12
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
    assert groups[0].window_tokens == 133
    assert groups[0].window_resident_block_count == 3


def test_non_speculative_c4_tail_only_retains_its_logical_window() -> None:
    geometry = _geometry(decode_input_tokens=1)

    assert geometry.window_tokens == 128
    assert geometry.c4_state_tokens == 8
    assert geometry.layout(10).groups[0].window_resident_block_count == 2
    assert geometry.layout(10).groups[C4_STATE_GROUP].window_tokens == 8


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
    geometry = _geometry()
    layout = geometry.layout(3)
    arena = CacheArena("cpu", layout, geometry, dtype=torch.bfloat16)

    inputs = arena.graph_shape_capture_inputs(2, 256)

    assert arena.storage.shape[0] == layout.parent_capacity + 7
    for group in range(len(geometry.packings)):
        expected_parent = layout.parent_capacity + group + 1
        assert set(inputs[f"cache.{group}.parent_ids"]) == {expected_parent}
        assert set(inputs[f"cache.{group}.child_slots"]) == {0}


def test_plan_preserves_structured_parent_and_child_addresses() -> None:
    layout = _geometry().layout(10)
    cache_groups = tuple(
        SimpleNamespace(
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
    plan = SimpleNamespace(
        num_sequences=1,
        sequence_ids=np.array([1], dtype=np.uint64),
        token_ids=np.array([7], dtype=np.uint32),
        token_indptr=np.array([0, 1], dtype=np.uint32),
        positions=np.array([0], dtype=np.uint32),
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
