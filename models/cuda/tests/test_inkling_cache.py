from __future__ import annotations

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupPlan,
    CacheRetention,
)
from resin_models_cuda.inkling.cache import (
    DRAFT_BLOCKS_PER_SLAB,
    GLOBAL_BLOCKS_PER_SLAB,
    LOCAL_BLOCKS_PER_SLAB,
    LOCAL_RETENTION_TOKENS,
    SLAB_UNITS,
    STATE_BLOCKS_PER_SLAB,
    CacheArena,
    CacheGeometry,
    build_cache_plan,
)


def _group(
    index: int,
    pages: list[tuple[int, int]],
    append: list[tuple[int, int]],
    offsets: list[int],
    *,
    packing: int,
) -> CacheGroupPlan:
    return CacheGroupPlan(
        index,
        [0, len(pages)],
        [parent for parent, _ in pages],
        [slot for _, slot in pages],
        [0],
        [parent for parent, _ in append],
        [slot for _, slot in append],
        offsets,
        blocks_per_slab=packing,
    )


def test_small_slab_packings_bound_fragmentation() -> None:
    geometry = CacheGeometry()
    assert geometry.bytes_per_slab == 35 * 1024 * 1024
    assert (
        geometry.local_blocks_per_slab,
        geometry.global_blocks_per_slab,
        geometry.draft_blocks_per_slab,
        geometry.state_blocks_per_slab,
    ) == (2, 10, 11, 14)
    layout = geometry.layout(100)
    assert layout.groups[1].retention is CacheRetention.WINDOW
    assert layout.groups[1].window_tokens == LOCAL_RETENTION_TOKENS
    assert layout.groups[3].family is CacheGroupFamily.STATE


def test_layer_page_ids_use_slab_units_not_combined_block_ids() -> None:
    assert CacheArena.unit_page(3, 1, 35, 7) == 3 * SLAB_UNITS + 35 + 7
    assert CacheArena.unit_page(3, 9, 7, 6) == 3 * SLAB_UNITS + 9 * 7 + 6
    assert CacheArena.unit_page(3, 10, 6, 5) == 3 * SLAB_UNITS + 10 * 6 + 5


def test_decode_lookahead_leases_the_second_attention_block() -> None:
    layout = CacheGeometry().layout(20)
    plan = BatchPlan(
        [9],
        [127],
        [42],
        [0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _group(
                0,
                [(1, 0), (1, 1)],
                [(1, 0)],
                [127],
                packing=GLOBAL_BLOCKS_PER_SLAB,
            ),
            _group(
                1,
                [(2, 0), (2, 1)],
                [(2, 0)],
                [127],
                packing=LOCAL_BLOCKS_PER_SLAB,
            ),
            _group(
                2,
                [(3, 0), (3, 1)],
                [(3, 0)],
                [127],
                packing=DRAFT_BLOCKS_PER_SLAB,
            ),
            _group(
                3,
                [(4, 0)],
                [(4, 0)],
                [127],
                packing=STATE_BLOCKS_PER_SLAB,
            ),
        ],
    )
    cache = build_cache_plan(plan, layout, decode_lookahead=5)
    assert cache.groups[0].address(0, 1) == (1, 1)
    assert cache.groups[1].address(0, 1) == (2, 1)
    assert cache.groups[2].address(0, 1) == (3, 1)
    assert cache.groups[3].address(0, 0) == (4, 0)
