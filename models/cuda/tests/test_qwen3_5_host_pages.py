import numpy as np

from resin_models_cuda.qwen3_5.attention import GraphPagedAttentionRuntime


def _loop(
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


def test_vectorized_host_pages_matches_loop():
    rng = np.random.default_rng(0)
    rows, graph_rows, width, P = 5, 8, 40, 128
    blocks = rng.integers(3, 30, rows)
    indptr = np.concatenate(([0], np.cumsum(blocks))).astype(np.uint32)
    n = int(indptr[-1])

    class Group:
        block_indptr = indptr
        block_slab_ids = rng.integers(1, 50, n).astype(np.uint32)
        block_slotnums = rng.integers(0, 4, n).astype(np.uint32)
        block_table_start_indices = np.zeros(rows, dtype=np.uint32)

        def block_slabs(self, row):
            return self.block_slab_ids[indptr[row] : indptr[row + 1]]

        def block_slots(self, row):
            return self.block_slotnums[indptr[row] : indptr[row + 1]]

    group = Group()
    lengths = np.array(
        [int(b) * P - int(rng.integers(0, P)) for b in blocks], dtype=np.int64
    )
    kw = dict(
        rows=rows,
        graph_rows=graph_rows,
        graph_width=width,
        sequence_lengths=lengths,
        layer_count=8,
        slab_units=32,
        page_tokens=P,
    )
    for a, b in zip(
        GraphPagedAttentionRuntime._host_pages(group, **kw), _loop(group, **kw)
    ):
        np.testing.assert_array_equal(a, b)
