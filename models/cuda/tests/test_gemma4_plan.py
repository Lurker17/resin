from __future__ import annotations

from resin import BatchPhase, BatchPlan, CacheGroupPlan
from resin_models_cuda.gemma4.plan import (
    PAGE_SIZE,
    build_gemma4_batch_plan,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)


def _append_metadata(pages: list[int], token_count: int) -> tuple[list[int], list[int]]:
    return (
        [pages[position // PAGE_SIZE] for position in range(token_count)],
        [position % PAGE_SIZE for position in range(token_count)],
    )


def test_gemma4_plan_preserves_positive_ids_after_graph_null_page() -> None:
    tokens = list(range(70))
    history_pages = [11, 12]
    local_pages = [1, 2]
    history_append, offsets = _append_metadata(history_pages, len(tokens))
    local_append, _ = _append_metadata(local_pages, len(tokens))
    plan = BatchPlan(
        [7],
        [0],
        tokens,
        [0, len(tokens)],
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 2],
                history_pages,
                [0],
                history_append,
                offsets,
            ),
            CacheGroupPlan(
                1,
                [0, 2],
                local_pages,
                [0],
                local_append,
                offsets,
            ),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan, history_blocks_per_parent=10, parent_capacity=2
    )

    assert batch.global_.page_indices.tolist() == [11, 12]
    assert batch.local.page_indices.tolist() == [1, 2]
    assert batch.global_.append_slots[[0, 64, 69]].tolist() == [704, 768, 773]
    assert batch.local.append_slots[[0, 64, 69]].tolist() == [64, 128, 133]
    assert batch.global_.last_page_lengths.tolist() == [6]
    assert batch.last_token_indices.tolist() == [69]


def test_decode_lookahead_expands_verify_but_not_assistant_context() -> None:
    plan = BatchPlan(
        [9],
        [64],
        [123],
        [0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(0, [0, 2], [11, 12], [0], [12], [0]),
            CacheGroupPlan(1, [0, 2], [21, 22], [0], [22], [0]),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan,
        history_blocks_per_parent=10,
        parent_capacity=22,
        decode_lookahead=3,
    )
    assistant = gemma4_assistant_context(batch)
    verify = expand_gemma4_decode_batch(
        batch,
        [[201, 202, 203]],
    )

    assert assistant.global_.page_indices.tolist() == [11]
    assert assistant.local.page_indices.tolist() == [21]
    assert assistant.global_.last_page_lengths.tolist() == [64]
    assert assistant.global_.append_slots.tolist() == []
    assert verify.token_ids.tolist() == [123, 201, 202, 203]
    assert verify.positions.tolist() == [64, 65, 66, 67]
    assert verify.global_.page_indices.tolist() == [11, 12]
    assert verify.global_.append_slots.tolist() == [768, 769, 770, 771]
    assert verify.local.append_slots.tolist() == [1408, 1409, 1410, 1411]
    assert verify.global_.last_page_lengths.tolist() == [4]
