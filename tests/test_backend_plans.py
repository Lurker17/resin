from __future__ import annotations

import numpy as np
import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
)
from resin.backend.paged_attention import (
    PagedAttentionGroupSpec,
    build_paged_attention_batch,
)
from resin.backend.plans import BatchView, CacheGroupView, PagedCachePlan


def _plan() -> BatchPlan:
    return BatchPlan(
        [10, 11],
        [0, 2],
        [1, 2, 3],
        [0, 2, 3],
        sampling_mask=[True, True],
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 1, 2],
                [1, 2],
                [0, 1],
                [0, 0],
                [1, 1, 2],
                [0, 0, 1],
                [0, 1, 0],
                blocks_per_slab=2,
            )
        ],
    )


def test_batch_view_derives_canonical_row_geometry() -> None:
    batch = BatchView.from_plan(_plan())
    assert batch.num_sequences == 2
    assert batch.num_tokens == 3
    assert batch.query_lengths.tolist() == [2, 1]
    assert batch.post_lengths.tolist() == [2, 3]
    assert batch.last_token_indices.tolist() == [1, 2]
    assert batch.token_ids[batch.token_slice(1)].tolist() == [3]


def test_cache_group_view_preserves_structured_and_packed_addresses() -> None:
    plan = _plan()
    batch = BatchView.from_plan(plan)
    group = CacheGroupView.from_plan(
        plan.cache_groups[0],
        batch,
        group_index=0,
        blocks_per_slab=2,
        parent_capacity=2,
        block_tokens=2,
    )
    assert group.block_slabs(0).tolist() == [1]
    assert group.block_slots(1).tolist() == [1]
    assert group.packed_block_ids.tolist() == [2, 5]
    assert group.packed_append_ids.tolist() == [2, 2, 5]
    assert group.append_block_offsets.dtype == np.uint32


def test_paged_attention_names_semantics_and_preserves_packed_addresses() -> None:
    plan = BatchPlan(
        [10],
        [0],
        [1, 2, 3],
        [0, 3],
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 2],
                [1, 2],
                [0, 1],
                [0],
                [1, 1, 2],
                [0, 0, 1],
                [0, 1, 0],
                blocks_per_slab=2,
            )
        ],
    )
    batch = build_paged_attention_batch(
        plan,
        CacheLayout(2, [CacheGroupLayout(2, blocks_per_slab=2)]),
        (PagedAttentionGroupSpec("history", 0, minimum_table_width=3),),
        max_query_tokens=3,
    )
    history = batch.group("history")

    assert batch.query_lengths == (3,)
    assert history.width == 3
    assert history.page_table.tolist() == [2, 5, 2]
    assert history.lengths == (3,)
    assert history.append_slots.tolist() == [4, 5, 10]


def test_paged_cache_plan_validates_exact_leased_tables() -> None:
    layout = CacheLayout(2, [CacheGroupLayout(2)])
    plan = BatchPlan(
        [10],
        [1],
        [3],
        [0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 2],
                [1, 2],
                [0, 0],
                [0],
                [1],
                [0],
                [1],
            )
        ],
    )
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=1)
    assert cache.rows == 1
    assert cache.groups[0].address(0, 1) == (2, 0)


def test_paged_cache_plan_accepts_the_two_checkpoint_state_ring() -> None:
    layout = CacheLayout(2, [CacheGroupLayout(4, family=CacheGroupFamily.STATE)])
    plan = BatchPlan(
        [10],
        [11],
        [3],
        [0, 1],
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 2],
                [1, 2],
                [0, 0],
                [1],
                [2],
                [0],
                [3],
            )
        ],
    )
    cache = PagedCachePlan.from_plan(plan, layout)
    assert cache.groups[0].address(0, 2) == (2, 0)


@pytest.mark.parametrize("decode", [False, True])
def test_source_free_window_plan_rejects_writes_that_discard_read_history(
    decode: bool,
) -> None:
    layout = CacheLayout(
        8,
        [CacheGroupLayout(2, retention=CacheRetention.WINDOW, window_tokens=4)],
    )
    token_count = 1 if decode else 6
    plan = BatchPlan(
        [10],
        [6],
        list(range(6, 6 + token_count)),
        [0, token_count],
        phase=BatchPhase.DECODE if decode else BatchPhase.PREFILL,
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 3],
                [1, 2, 3],
                [0, 0, 0],
                [3],
                [1 + index // 2 for index in range(token_count)],
                [0] * token_count,
                [index % 2 for index in range(token_count)],
            )
        ],
    )
    # The final suffix and every append address look valid, but the first
    # query still needs positions3..5, which this table has already discarded.
    with pytest.raises(ValueError, match="Window read/write capacity"):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=5 if decode else 0)
