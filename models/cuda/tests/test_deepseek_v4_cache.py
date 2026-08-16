from __future__ import annotations

import math

import torch

from resin import CacheRetention
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
    assert tuple(group.blocks_per_parent for group in groups) == geometry.packings


def test_non_speculative_c4_tail_only_retains_its_logical_window() -> None:
    geometry = _geometry(decode_input_tokens=1)

    assert geometry.c4_state_tokens == 8
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
    arena = CacheArena(torch, "cpu", layout, geometry, dtype=torch.bfloat16)

    inputs = arena.graph_shape_capture_inputs(2, 256)

    assert arena.storage.shape[0] == layout.parent_capacity + 7
    for group, packing in enumerate(geometry.packings):
        expected_page = (layout.parent_capacity + group) * packing + 1
        assert set(inputs[f"cache.{group}.pages"]) == {expected_page}
