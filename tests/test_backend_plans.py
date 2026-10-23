from __future__ import annotations

from types import SimpleNamespace
from typing import cast

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


def test_batch_view_rejects_noncontiguous_positions_after_a_row_boundary() -> None:
    native = _plan()
    plan = SimpleNamespace(
        **{name: getattr(native, name) for name in BatchView.__dataclass_fields__}
    )
    plan.positions = native.positions.copy()
    plan.positions[2] = 3
    with pytest.raises(
        ValueError, match="positions are not contiguous within each row"
    ):
        BatchView.from_plan(cast(BatchPlan, plan))


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


def _mixed_cache_plan(*, decode: bool) -> tuple[BatchPlan, CacheLayout]:
    layout = CacheLayout(
        64,
        [
            CacheGroupLayout(4, blocks_per_slab=2),
            CacheGroupLayout(
                4,
                blocks_per_slab=3,
                retention=CacheRetention.WINDOW,
                window_tokens=12,
                read_window_tokens=5,
            ),
            CacheGroupLayout(4, blocks_per_slab=4, family=CacheGroupFamily.STATE),
        ],
    )
    contexts, lengths = [0, 7, 14], [3, 1, 5]
    groups = []
    for index, descriptor in enumerate(layout.groups):
        indptr, slabs, slots, firsts = [0], [], [], []
        append_slabs, append_slots, offsets = [], [], []
        for row, (context, length) in enumerate(zip(contexts, lengths, strict=True)):
            last = (context + length + (3 if decode else 0) - 1) // 4
            if descriptor.family == CacheGroupFamily.STATE:
                first = max(0, last - 1)
            elif descriptor.window_resident_block_count is not None:
                first = max(0, last + 1 - descriptor.window_resident_block_count)
            else:
                first = 0
            count = last - first + 1
            row_slabs = list(range(1 + len(slabs), 1 + len(slabs) + count))
            slot = row % descriptor.blocks_per_slab
            slabs.extend(row_slabs)
            slots.extend([slot] * count)
            firsts.append(first)
            indptr.append(len(slabs))
            for position in range(context, context + length):
                # Older STATE writes can fall outside the retained checkpoint ring.
                append_slabs.append(row_slabs[max(0, position // 4 - first)])
                append_slots.append(slot)
                offsets.append(position % 4)
        groups.append(
            CacheGroupPlan(
                index,
                indptr,
                slabs,
                slots,
                firsts,
                append_slabs,
                append_slots,
                offsets,
                blocks_per_slab=descriptor.blocks_per_slab,
            )
        )
    plan = BatchPlan(
        [10, 11, 12],
        contexts,
        list(range(sum(lengths))),
        [0, 3, 4, 9],
        phase=BatchPhase.DECODE if decode else BatchPhase.PREFILL,
        cache_groups=groups,
    )
    return plan, layout


def _replace_plan_group(plan: BatchPlan, index: int, **fields: object) -> BatchPlan:
    """Corrupt native views to exercise validation after the Rust boundary."""

    group = SimpleNamespace(
        **{
            name: getattr(plan.cache_groups[index], name)
            for name in CacheGroupView.__dataclass_fields__
        }
    )
    vars(group).update(fields)
    groups = list(plan.cache_groups)
    groups[index] = cast(CacheGroupPlan, group)
    return cast(
        BatchPlan,
        SimpleNamespace(
            **{
                name: groups if name == "cache_groups" else getattr(plan, name)
                for name in BatchView.__dataclass_fields__
            }
        ),
    )


@pytest.mark.parametrize("decode", [False, True])
def test_paged_cache_plan_validates_heterogeneous_rows_across_retention_groups(
    decode: bool,
) -> None:
    plan, layout = _mixed_cache_plan(decode=decode)
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)
    assert cache.batch.query_lengths.tolist() == [3, 1, 5]
    assert cache.batch.positions.tolist() == [0, 1, 2, 7, 14, 15, 16, 17, 18]
    expected = (
        [[0, 0, 0], [0, 0, 2], [0, 1, 4]]
        if decode
        else [[0, 0, 0], [0, 0, 1], [0, 0, 3]]
    )
    assert [
        group.block_table_start_indices.tolist() for group in cache.groups
    ] == expected


@pytest.mark.parametrize("field", ["append_slab_ids", "append_slotnums"])
def test_paged_cache_plan_rejects_cross_row_append_targets(field: str) -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[0]
    values = getattr(group, field).copy()
    values[3] = values[4]  # Row 1 writes into row 2's valid physical address.
    mutated = _replace_plan_group(plan, 0, **{field: values})
    with pytest.raises(ValueError, match="cache group 0 row 1 append target"):
        PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


def test_paged_cache_plan_reports_the_first_invalid_row_before_later_ranges() -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[0]
    offsets = group.append_block_offsets.copy()
    offsets[0] = 1  # Earlier row has an invalid append, later row an invalid extent.
    firsts = group.block_table_start_indices.copy()
    firsts[1] = 1
    mutated = _replace_plan_group(
        plan, 0, append_block_offsets=offsets, block_table_start_indices=firsts
    )
    with pytest.raises(ValueError, match="cache group 0 row 0 append target"):
        PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


@pytest.mark.parametrize("earlier_bad_append", [False, True])
def test_paged_cache_plan_preserves_row_order_for_window_write_failures(
    earlier_bad_append: bool,
) -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    layout = CacheLayout(
        layout.parent_capacity,
        [
            layout.groups[0],
            CacheGroupLayout(
                4,
                blocks_per_slab=3,
                retention=CacheRetention.WINDOW,
                window_tokens=12,
                read_window_tokens=7,
            ),
            layout.groups[2],
        ],
    )
    if earlier_bad_append:
        offsets = plan.cache_groups[1].append_block_offsets.copy()
        offsets[3] = 0
        plan = _replace_plan_group(plan, 1, append_block_offsets=offsets)
    message = (
        "cache group 1 row 1 append target"
        if earlier_bad_append
        else "cache group 1 row 2 write span 8 exceeds 6-token"
    )
    with pytest.raises(ValueError, match=message):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)


def test_paged_cache_plan_skips_discarded_checkpoint_append_targets() -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[2]
    offsets = group.append_block_offsets.copy()
    offsets[4] = 0  # Position 14 is older than the retained ring beginning at 16.
    mutated = _replace_plan_group(plan, 2, append_block_offsets=offsets)
    PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


def test_paged_cache_plan_reports_empty_row_extents_without_indexing_them() -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    indptr = plan.cache_groups[0].block_indptr.copy()
    indptr[1] = 0
    mutated = _replace_plan_group(plan, 0, block_indptr=indptr)
    with pytest.raises(
        ValueError, match="cache group 0 row 0 is not its exact active range"
    ):
        PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


def test_paged_cache_plan_keeps_decode_leases_wider_than_u32() -> None:
    context = 2**32 - 5
    block_tokens = 2**31 - 1
    plan = BatchPlan(
        [10],
        [context],
        [1, 2, 3],
        [0, 3],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 3],
                [1, 2, 3],
                [0, 0, 0],
                [0],
                [2, 2, 2],
                [0, 0, 0],
                [block_tokens - 3, block_tokens - 2, block_tokens - 1],
            )
        ],
    )
    layout = CacheLayout(3, [CacheGroupLayout(block_tokens)])
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)
    assert cache.groups[0].address(0, 2) == (3, 0)
    with pytest.raises(ValueError, match="row 0 is not its exact active range"):
        PagedCachePlan.from_plan(plan, layout)


@pytest.mark.parametrize("lookahead", [2**63, 2**64, 10**100])
def test_paged_cache_plan_does_not_wrap_oversized_decode_lookahead(
    lookahead: int,
) -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    with pytest.raises(ValueError, match="row 0 is not its exact active range"):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=lookahead)
