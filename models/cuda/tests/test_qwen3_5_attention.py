from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from resin.backend.cuda.paged_kv import SlabPagedKvCache
from resin.backend.plans import CacheGroupView
from resin_models_cuda.qwen3_5.attention import (
    GraphAttentionSpec,
    GraphPagedAttentionRuntime,
    PreparedPagedAttention,
)


@pytest.mark.parametrize("tokens", [1, 4])
def test_layer_pages_use_live_base_pages_and_stable_buffer(tokens):
    base = torch.tensor([9, 3, 15, 0], dtype=torch.int32)
    physical = torch.empty_like(base)
    storage = physical.data_ptr()
    seen = []

    class Wrapper:
        def run(self, query, cache):
            assert physical.data_ptr() == storage
            seen.append(physical.clone())
            keys, values = cache
            return query + keys[physical].sum() + values[physical].sum()

    cache = cast(
        SlabPagedKvCache,
        SimpleNamespace(keys=torch.arange(18), values=torch.arange(18) * 2),
    )
    attention = PreparedPagedAttention(Wrapper(), cache, base, physical, 3, tokens)
    query = torch.randn(2, tokens, 4, 8)
    for mutation in range(2):
        if mutation:
            base.copy_(torch.tensor([6, 12, 0, 3]))
        for layer in (0, 2, 1):
            expected = base + layer
            output = attention.run(query, layer)
            torch.testing.assert_close(seen[-1], expected, rtol=0, atol=0)
            torch.testing.assert_close(output, query + 3 * expected.sum())
    with pytest.raises(ValueError, match="outside"):
        attention.run(query, 3)


def _reference_host_pages(
    group,
    *,
    rows,
    graph_rows,
    graph_width,
    sequence_lengths,
    layer_count,
    slab_units,
    page_tokens,
):
    logical = np.zeros(graph_rows * graph_width, dtype=np.int32)
    indptr = np.zeros(graph_rows + 1, dtype=np.int32)
    last = np.ones(graph_rows, dtype=np.int32)
    cursor = 0
    pages_per_slab = slab_units // layer_count
    for row in range(graph_rows):
        if row < rows:
            retained = int(group.block_table_start_indices[row]) * page_tokens
            used = int(sequence_lengths[row]) - retained
            count = (used + page_tokens - 1) // page_tokens
            slabs = group.block_slabs(row)
            slots = group.block_slots(row)
            logical[cursor : cursor + count] = slabs[:count].astype(
                np.int64
            ) * pages_per_slab + slots[:count].astype(np.int64)
            last[row] = (used - 1) % page_tokens + 1
        else:
            count = 1
        cursor += count
        indptr[row + 1] = cursor
    return logical, indptr, last


@pytest.mark.parametrize("graph_rows", [5, 8])
@pytest.mark.parametrize("retained_pages", [(0, 0, 0, 0, 0), (1, 3, 0, 12, 1024)])
@pytest.mark.parametrize("extra_pages", [(0, 0, 0, 0, 0), (2, 1, 3, 0, 4)])
def test_host_page_tables_match_reference(graph_rows, retained_pages, extra_pages):
    rng = np.random.default_rng(0)
    rows, width, page_tokens = 5, 8, 128
    used = np.array([1, 128, 129, 256, 513], dtype=np.int64)
    counts = (used + page_tokens - 1) // page_tokens
    leased = counts + np.asarray(extra_pages)
    indptr = np.concatenate(([0], np.cumsum(leased))).astype(np.uint32)
    total = int(indptr[-1])
    # Sparse physical pages keep unused lease tails distinct from the next row.
    pages = rng.choice(200, total, replace=False).astype(np.uint32)
    slabs, slots = pages // 4 + 1, pages % 4
    append_indices = indptr[:-1].astype(np.int64) + counts - 1
    group = CacheGroupView(
        group_index=0,
        blocks_per_slab=4,
        block_indptr=indptr,
        block_slab_ids=slabs,
        block_slotnums=slots,
        block_table_start_indices=np.asarray(retained_pages, dtype=np.uint32),
        append_slab_ids=slabs[append_indices],
        append_slotnums=slots[append_indices],
        append_block_offsets=((used - 1) % page_tokens).astype(np.uint32),
    )
    options: dict[str, Any] = dict(
        rows=rows,
        graph_rows=graph_rows,
        graph_width=width,
        sequence_lengths=np.asarray(retained_pages, dtype=np.int64) * page_tokens
        + used,
        layer_count=8,
        slab_units=32,
        page_tokens=page_tokens,
    )
    actual = GraphPagedAttentionRuntime._host_pages(group, **options)
    expected = _reference_host_pages(group, **options)
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, reference)


def test_graph_plan_only_validates_the_immutable_lease():
    runtime = object.__new__(GraphPagedAttentionRuntime)
    runtime.cache = cast(
        SlabPagedKvCache,
        SimpleNamespace(
            groups=(SimpleNamespace(slab_ids=torch.zeros(4, 4)),),
            page_tokens=128,
        ),
    )
    runtime.calls = cast(
        Any,
        (SimpleNamespace(spec=GraphAttentionSpec(0, 3, 2, 2)),),
    )
    # Row zero owns future pages past both possible accepted positions. Nothing
    # in host planning may expose those pages or replan a captured wrapper.
    group = SimpleNamespace(
        block_table_start_indices=np.array([0, 0]),
        block_indptr=np.array([0, 3, 5]),
    )
    plan = SimpleNamespace(rows=2, groups=(group,))
    for positions in ([126, 253], [128, 254]):
        assert runtime.plan(plan, positions) == {}
        assert runtime.graph_inputs() == {}
    with pytest.raises(ValueError, match="exceeds its cache lease"):
        runtime.plan(plan, [383, 255])
    with pytest.raises(ValueError, match="requires retained queries"):
        runtime.plan(plan, [-1, 255])
