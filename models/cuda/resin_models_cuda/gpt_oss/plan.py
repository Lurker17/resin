"""Translate one validated native cache plan into FA4-shaped host metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase
from resin.backend.plans import CacheGroupView, PagedCachePlan

PAGE_SIZE = 128
SLIDING_WINDOW = 128
PREFILL_CHUNK_SIZE = 8192
# Paged prefill must keep the whole query plus the preceding attention window
# resident until the forward finishes. Decode still attends to only
# ``SLIDING_WINDOW`` tokens; this wider physical suffix permits one 8K prefill.
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1
WINDOW_TABLE_WIDTH = 1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE


@dataclass(frozen=True, slots=True)
class AttentionBatch:
    phase: BatchPhase
    token_indices: npt.NDArray[np.int64]
    sequence_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    full_width: int
    full_table: npt.NDArray[np.uint32]
    full_lengths: tuple[int, ...]
    sliding_width: int
    sliding_table: npt.NDArray[np.uint32]
    sliding_lengths: tuple[int, ...]
    full_append_slots: npt.NDArray[np.int64]
    sliding_append_slots: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        rows = len(self.sequence_indices)
        tokens = sum(self.query_lengths)
        if rows == 0 or len(set(self.sequence_indices)) != rows:
            raise ValueError("attention batch rows must be nonempty and unique")
        if len(self.query_lengths) != rows or any(
            length <= 0 or length > PREFILL_CHUNK_SIZE for length in self.query_lengths
        ):
            raise ValueError("attention query lengths are invalid")
        if len(self.token_indices) != tokens:
            raise ValueError("query lengths do not cover the selected tokens")
        if len(self.full_table) != rows * self.full_width:
            raise ValueError("full-attention table shape is invalid")
        if len(self.sliding_table) != rows * self.sliding_width:
            raise ValueError("sliding-attention table shape is invalid")
        if len(self.full_lengths) != rows or len(self.sliding_lengths) != rows:
            raise ValueError("key lengths do not cover every row")
        if (
            len(self.full_append_slots) != tokens
            or len(self.sliding_append_slots) != tokens
        ):
            raise ValueError("append slots do not cover every selected token")


def _page_table(
    rows: tuple[npt.NDArray[np.uint32], ...], minimum_width: int = 1
) -> tuple[int, npt.NDArray[np.uint32]]:
    if not rows or any(len(row) == 0 for row in rows):
        raise ValueError("cache page-table rows must be nonempty")
    width = max(minimum_width, *(len(row) for row in rows))
    table = np.empty((len(rows), width), dtype=np.uint32)
    for index, row in enumerate(rows):
        table[index, : len(row)] = row
        table[index, len(row) :] = row[0]
    return width, table.reshape(-1)


def _append_slots(
    group: CacheGroupView, token_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    return (
        group.append_slab_ids[token_indices].astype(np.int64) * PAGE_SIZE
        + group.append_block_offsets[token_indices]
    )


def build_attention_batches(
    plan, cache_layout
) -> tuple[tuple[AttentionBatch, ...], AttentionBatch | None]:
    """Return one prefill batch or one decode batch from a homogeneous plan."""

    cache = PagedCachePlan.from_plan(plan, cache_layout)
    native = cache.batch
    tokens = native.token_ids
    indptr = native.token_indptr
    row_count = native.num_sequences
    history, window = cache.groups
    if np.intersect1d(history.block_slab_ids, window.block_slab_ids).size:
        raise ValueError("History and Window cannot address one parent in a batch")

    post_lengths = [int(length) for length in native.post_lengths]
    for row in range(row_count):
        start, end = map(int, indptr[row : row + 2])
        query_length = end - start
        if native.phase == BatchPhase.PREFILL:
            if query_length > PREFILL_CHUNK_SIZE:
                raise ValueError(
                    f"prefill query length exceeds {PREFILL_CHUNK_SIZE} tokens"
                )
        elif query_length != 1:
            raise ValueError("decode rows must contain one token")

    rows = tuple(range(row_count))
    token_indices = np.arange(len(tokens), dtype=np.int64)

    def group_view(group: CacheGroupView):
        page_rows = tuple(group.block_slabs(row) for row in rows)
        lengths = tuple(
            post_lengths[row]
            - int(group.block_table_start_indices[row]) * PAGE_SIZE
            for row in rows
        )
        return page_rows, lengths

    full_rows, full_lengths = group_view(history)
    sliding_rows, sliding_lengths = group_view(window)
    full_width, full_table = _page_table(full_rows)
    sliding_width, sliding_table = _page_table(sliding_rows, WINDOW_TABLE_WIDTH)
    query_lengths = tuple(int(indptr[row + 1] - indptr[row]) for row in rows)
    batch = AttentionBatch(
        phase=native.phase,
        token_indices=token_indices,
        sequence_indices=rows,
        query_lengths=query_lengths,
        full_width=full_width,
        full_table=full_table,
        full_lengths=full_lengths,
        sliding_width=sliding_width,
        sliding_table=sliding_table,
        sliding_lengths=sliding_lengths,
        full_append_slots=_append_slots(history, token_indices),
        sliding_append_slots=_append_slots(window, token_indices),
    )
    return ((batch,), None) if native.phase == BatchPhase.PREFILL else ((), batch)
