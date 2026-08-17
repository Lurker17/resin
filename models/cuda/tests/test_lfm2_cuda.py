from __future__ import annotations

import inspect

from resin import BatchPhase, BatchPlan, CacheGroupPlan
from resin_models_cuda.lfm2.cuda import Lfm2CudaRunner
from resin_models_cuda.lfm2.plan import build_cache_metadata


def _group(index, indptr, pages, first, append_pages, append_offsets, *, packing=1):
    parents = [(page - 1) // packing + 1 for page in pages]
    child_slots = [(page - 1) % packing for page in pages]
    append_parents = [(page - 1) // packing + 1 for page in append_pages]
    append_slotnums = [(page - 1) % packing for page in append_pages]
    return CacheGroupPlan(
        index,
        indptr,
        parents,
        child_slots,
        first,
        append_parents,
        append_slotnums,
        append_offsets,
        blocks_per_slab=packing,
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
            _group(
                1,
                [0, 1, 2],
                [11, 13],
                [0, 0],
                [11, 11, 13],
                [0, 1, 16],
                packing=2,
            ),
        ],
    )


class _Cache:
    state_packing = 2

    @staticmethod
    def check_addresses(group, parents, child_slots):
        assert group == 1
        assert all(parent > 0 for parent in parents)
        assert all(slot in (0, 1) for slot in child_slots)


def test_attention_metadata_uses_only_the_history_group() -> None:
    metadata = build_cache_metadata(_plan(), 16)
    assert metadata.page_indices.tolist() == [5, 3, 7]
    assert metadata.append_positions.tolist() == [0, 1, 16]
    assert metadata.prefill.dense_token_indices.tolist() == [0, 1]
    assert metadata.prefill.continuation_token_indices.tolist() == [2]


def test_recurrent_rows_are_derived_from_the_state_page_table() -> None:
    runner = Lfm2CudaRunner.__new__(Lfm2CudaRunner)
    runner._cache = _Cache()
    rows = runner._state_rows(_plan())
    assert [(row.input_address, row.output_address) for row in rows] == [
        (None, (6, 0)),
        ((7, 0), (7, 0)),
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
            _group(1, [0, 2], [11, 12], [0], [12], [0], packing=2),
        ],
    )
    runner = Lfm2CudaRunner.__new__(Lfm2CudaRunner)
    runner._cache = _Cache()
    [row] = runner._state_rows(plan)
    assert (row.input_address, row.output_address) == ((6, 0), (6, 1))


def test_lfm2_has_no_separate_state_pool_route() -> None:
    source = inspect.getsource(Lfm2CudaRunner)
    assert "StatePool" not in source
    assert "state_slots" not in source
    assert "MappedStateArena" not in source
