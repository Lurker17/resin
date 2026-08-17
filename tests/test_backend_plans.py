from __future__ import annotations

import numpy as np

from resin import BatchPlan, CacheGroupPlan
from resin.backend.plans import BatchView, CacheGroupView


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
