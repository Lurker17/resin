from __future__ import annotations

from resin import BatchPlan, CacheGroupPlan
from resin_models_cuda.gemma4.plan import PAGE_SIZE, build_gemma4_batch_plan


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
