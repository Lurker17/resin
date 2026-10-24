from __future__ import annotations

import numpy as np

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
)
from resin.models.gemma4.plan import (
    PAGE_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)
from resin.models.gemma4.plan import (
    build_gemma4_batch_plan as _build_gemma4_batch_plan,
)


def build_gemma4_batch_plan(
    plan,
    *,
    history_blocks_per_parent: int,
    parent_capacity: int,
    decode_lookahead: int = 0,
):
    layout = CacheLayout(
        parent_capacity=parent_capacity,
        groups=[
            CacheGroupLayout(PAGE_SIZE, blocks_per_slab=history_blocks_per_parent),
            CacheGroupLayout(
                PAGE_SIZE,
                retention=CacheRetention.WINDOW,
                window_tokens=WINDOW_RETENTION_TOKENS,
                read_window_tokens=SLIDING_WINDOW,
            ),
        ],
    )
    return _build_gemma4_batch_plan(plan, layout, decode_lookahead=decode_lookahead)


def _append_metadata(pages: list[int], token_count: int) -> tuple[list[int], list[int]]:
    return (
        [pages[position // PAGE_SIZE] for position in range(token_count)],
        [position % PAGE_SIZE for position in range(token_count)],
    )


def _cache_group(
    index: int,
    indptr: list[int],
    compact_pages: list[int],
    first_blocks: list[int],
    compact_append_pages: list[int],
    append_offsets: list[int],
    *,
    packing: int,
) -> CacheGroupPlan:
    def addresses(pages: list[int]) -> tuple[list[int], list[int]]:
        return (
            [(page - 1) // packing + 1 for page in pages],
            [(page - 1) % packing for page in pages],
        )

    parents, child_slots = addresses(compact_pages)
    append_parents, append_slotnums = addresses(compact_append_pages)
    return CacheGroupPlan(
        group_index=index,
        block_indptr=indptr,
        block_slab_ids=parents,
        block_slotnums=child_slots,
        block_table_start_indices=first_blocks,
        append_slab_ids=append_parents,
        append_slotnums=append_slotnums,
        append_block_offsets=append_offsets,
        blocks_per_slab=packing,
    )


def test_gemma4_plan_preserves_positive_ids_after_graph_null_page() -> None:
    tokens = list(range(70))
    history_pages = [11, 12]
    local_pages = [1, 2]
    history_append, offsets = _append_metadata(history_pages, len(tokens))
    local_append, _ = _append_metadata(local_pages, len(tokens))
    plan = BatchPlan(
        sequence_ids=[7],
        context_lengths=[0],
        token_ids=tokens,
        token_indptr=[0, len(tokens)],
        cache_groups=[
            _cache_group(
                0,
                [0, 2],
                history_pages,
                [0],
                history_append,
                offsets,
                packing=10,
            ),
            _cache_group(
                1,
                [0, 2],
                local_pages,
                [0],
                local_append,
                offsets,
                packing=1,
            ),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan, history_blocks_per_parent=10, parent_capacity=128
    )

    assert batch.global_.page_indices.tolist() == [20, 21]
    assert batch.local.page_indices.tolist() == [1, 2]
    assert batch.global_.append_slots[[0, 64, 69]].tolist() == [1280, 1344, 1349]
    assert batch.local.append_slots[[0, 64, 69]].tolist() == [64, 128, 133]
    assert batch.global_.last_page_lengths.tolist() == [6]
    assert batch.last_token_indices.tolist() == [69]


def test_decode_lookahead_expands_verify_but_not_assistant_context() -> None:
    plan = BatchPlan(
        sequence_ids=[9],
        context_lengths=[64],
        token_ids=[123],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _cache_group(0, [0, 2], [11, 12], [0], [12], [0], packing=10),
            _cache_group(1, [0, 2], [21, 22], [0], [22], [0], packing=1),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan,
        history_blocks_per_parent=10,
        parent_capacity=128,
        decode_lookahead=3,
    )
    assistant = gemma4_assistant_context(batch)
    verify = expand_gemma4_decode_batch(
        batch,
        np.array([[201, 202, 203]], dtype=np.int64),
    )

    assert assistant.global_.page_indices.tolist() == [20]
    assert assistant.local.page_indices.tolist() == [21]
    assert assistant.global_.last_page_lengths.tolist() == [64]
    assert assistant.global_.append_slots.tolist() == []
    assert verify.token_ids.tolist() == [123, 201, 202, 203]
    assert verify.positions.tolist() == [64, 65, 66, 67]
    assert verify.global_.page_indices.tolist() == [20, 21]
    assert verify.global_.append_slots.tolist() == [1344, 1345, 1346, 1347]
    assert verify.local.append_slots.tolist() == [1408, 1409, 1410, 1411]
    assert verify.global_.last_page_lengths.tolist() == [4]


def test_five_drafts_reserve_the_local_page_beyond_a_full_window() -> None:
    pages = list(range(1, 18))
    plan = BatchPlan(
        sequence_ids=[9],
        context_lengths=[1024],
        token_ids=[123],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _cache_group(0, [0, 17], pages, [0], [17], [0], packing=10),
            _cache_group(1, [0, 17], pages, [0], [17], [0], packing=1),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan,
        history_blocks_per_parent=10,
        parent_capacity=128,
        decode_lookahead=5,
    )
    assistant = gemma4_assistant_context(batch)
    verify = expand_gemma4_decode_batch(
        batch, np.array([[201, 202, 203, 204, 205]], dtype=np.int64)
    )

    assert len(assistant.local.page_indices) == 16
    assert assistant.local.last_page_lengths.tolist() == [64]
    assert len(verify.local.page_indices) == 17
    assert verify.local.kv_lengths == (1030,)
    assert verify.local.append_slots.tolist() == [1088, 1089, 1090, 1091, 1092, 1093]
