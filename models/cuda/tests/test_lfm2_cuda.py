from __future__ import annotations

import inspect

from resin import BatchPhase, BatchPlan, CacheGroupPlan
from resin_models_cuda.lfm2.cuda import Lfm2CudaRunner
from resin_models_cuda.lfm2.plan import build_cache_metadata


def _group(index, indptr, pages, first, append_pages, append_offsets):
    return CacheGroupPlan(
        index,
        indptr,
        pages,
        first,
        append_pages,
        append_offsets,
    )


def _plan() -> BatchPlan:
    return BatchPlan(
        [10, 11],
        [0, 16],
        [1, 2, 3],
        [0, 2, 3],
        phase=BatchPhase.PREFILL,
        cache_groups=[
            _group(0, [0, 1, 3], [5, 3, 7], [0, 0], [5, 5, 7], [0, 1, 0]),
            _group(1, [0, 1, 2], [11, 13], [0, 0], [11, 11, 13], [0, 1, 16]),
        ],
    )


class _Cache:
    @staticmethod
    def check_pages(group, pages):
        assert group == 1
        assert all(page > 0 for page in pages)


def test_attention_metadata_uses_only_the_history_group() -> None:
    metadata = build_cache_metadata(_plan(), 16)
    assert metadata.block_ids.tolist() == [5, 3, 7]
    assert metadata.append_positions.tolist() == [0, 1, 16]
    assert metadata.prefill.dense_token_indices.tolist() == [0, 1]
    assert metadata.prefill.continuation_token_indices.tolist() == [2]


def test_recurrent_rows_are_derived_from_the_state_page_table() -> None:
    runner = Lfm2CudaRunner.__new__(Lfm2CudaRunner)
    runner._cache = _Cache()
    rows = runner._state_rows(_plan())
    assert [(row.input_page, row.output_page) for row in rows] == [
        (None, 11),
        (13, 13),
    ]


def test_prefix_boundary_uses_distinct_input_and_output_state_pages() -> None:
    plan = BatchPlan(
        [10],
        [128],
        [3],
        [0, 1],
        phase=BatchPhase.PREFILL,
        cache_groups=[
            _group(0, [0, 9], list(range(1, 10)), [0], [9], [0]),
            _group(1, [0, 2], [11, 12], [0], [12], [0]),
        ],
    )
    runner = Lfm2CudaRunner.__new__(Lfm2CudaRunner)
    runner._cache = _Cache()
    [row] = runner._state_rows(plan)
    assert (row.input_page, row.output_page) == (11, 12)


def test_lfm2_has_no_separate_state_pool_route() -> None:
    source = inspect.getsource(Lfm2CudaRunner)
    assert "StatePool" not in source
    assert "state_slots" not in source
    assert "MappedStateArena" not in source
