"""Translate one validated native cache plan into FA4-shaped host metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase

PAGE_SIZE = 128
SLIDING_WINDOW = 128
PREFILL_CHUNK_SIZE = 8192
# Paged prefill must keep the whole query plus the preceding attention window
# resident until the forward finishes. Decode still attends to only
# ``SLIDING_WINDOW`` tokens; this wider physical suffix permits one 8K prefill.
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1
WINDOW_TABLE_WIDTH = 1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE


@dataclass(frozen=True, slots=True)
class CacheGroup:
    index: int
    indptr: npt.NDArray[np.uint32]
    pages: npt.NDArray[np.uint32]
    first_blocks: npt.NDArray[np.uint32]
    append_pages: npt.NDArray[np.uint32]
    append_offsets: npt.NDArray[np.uint32]

    @classmethod
    def from_native(
        cls, native, index: int, rows: int, tokens: int, capacity: int | None
    ) -> CacheGroup:
        if native.group_index != index:
            raise ValueError("cache groups must be ordered History then Window")
        if native.blocks_per_parent != 1:
            raise ValueError("GPT-OSS cache groups require one block per parent")
        block_parents = np.asarray(native.block_parent_ids, dtype=np.uint32)
        block_slots = np.asarray(native.block_child_slots, dtype=np.uint32)
        append_parents = np.asarray(native.append_parent_ids, dtype=np.uint32)
        append_slots = np.asarray(native.append_child_slots, dtype=np.uint32)
        group = cls(
            index,
            np.asarray(native.block_indptr, dtype=np.uint32),
            block_parents,
            np.asarray(native.first_logical_blocks, dtype=np.uint32),
            append_parents,
            np.asarray(native.append_block_offsets, dtype=np.uint32),
        )
        if (
            len(group.indptr) != rows + 1
            or group.indptr[0] != 0
            or group.indptr[-1] != len(group.pages)
            or np.any(group.indptr[1:] < group.indptr[:-1])
            or len(block_parents) != len(block_slots)
            or len(append_parents) != len(append_slots)
        ):
            raise ValueError("cache block_indptr is malformed")
        if len(group.first_blocks) != rows:
            raise ValueError("cache first_logical_blocks does not cover every row")
        if len(group.append_pages) != tokens or len(group.append_offsets) != tokens:
            raise ValueError("cache append metadata does not cover every token")
        if (
            np.any(group.pages == 0)
            or np.any(group.append_pages == 0)
            or np.any(block_slots != 0)
            or np.any(append_slots != 0)
        ):
            raise ValueError("cache physical addresses are malformed")
        if np.any(group.append_offsets >= PAGE_SIZE):
            raise ValueError("cache append offset is outside its page")
        if capacity is not None and (
            np.any(group.pages > capacity) or np.any(group.append_pages > capacity)
        ):
            raise ValueError("cache page is outside the bound allocation")
        return group

    def row(self, index: int) -> npt.NDArray[np.uint32]:
        start, end = map(int, self.indptr[index : index + 2])
        return self.pages[start:end]


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
    group: CacheGroup, token_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    return (
        group.append_pages[token_indices].astype(np.int64) * PAGE_SIZE
        + group.append_offsets[token_indices]
    )


def build_attention_batches(
    plan, *, parent_capacity: int | None = None
) -> tuple[tuple[AttentionBatch, ...], AttentionBatch | None]:
    """Return one prefill batch or one decode batch from a homogeneous plan."""

    tokens = np.asarray(plan.token_ids, dtype=np.uint32)
    positions = np.asarray(plan.positions, dtype=np.uint32)
    indptr = np.asarray(plan.token_indptr, dtype=np.uint32)
    contexts = np.asarray(plan.context_lengths, dtype=np.uint32)
    row_count = plan.num_sequences
    if row_count == 0:
        raise ValueError("GPT-OSS cannot execute an empty batch")
    if (
        len(positions) != len(tokens)
        or len(indptr) != row_count + 1
        or indptr[0] != 0
        or indptr[-1] != len(tokens)
        or np.any(indptr[1:] <= indptr[:-1])
        or len(contexts) != row_count
    ):
        raise ValueError("sequence metadata does not describe nonempty token rows")
    if plan.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("plan phase must be prefill or decode")

    try:
        native_groups = tuple(plan.cache_groups)
    except (AttributeError, TypeError) as error:
        raise ValueError("plan cache_groups must be iterable") from error
    if len(native_groups) != 2:
        raise ValueError("GPT-OSS requires History and Window cache groups")
    history, window = tuple(
        CacheGroup.from_native(group, index, row_count, len(tokens), parent_capacity)
        for index, group in enumerate(native_groups)
    )
    if np.intersect1d(history.pages, window.pages).size:
        raise ValueError("History and Window cannot address one parent in a batch")

    post_lengths = []
    for row in range(row_count):
        start, end = map(int, indptr[row : row + 2])
        query_length = end - start
        context = int(contexts[row])
        post_length = context + query_length
        post_lengths.append(post_length)
        last_block = (post_length - 1) // PAGE_SIZE
        expected_window_first = max(0, last_block + 1 - WINDOW_TABLE_WIDTH)
        if (
            int(history.first_blocks[row]) != 0
            or len(history.row(row)) != last_block + 1
        ):
            raise ValueError("History row is not the exact post-append prefix")
        if (
            int(window.first_blocks[row]) != expected_window_first
            or len(window.row(row)) != last_block - expected_window_first + 1
        ):
            raise ValueError("Window row is not the exact active suffix")
        for group in (history, window):
            pages = group.row(row)
            row_positions = positions[start:end]
            relative = row_positions // PAGE_SIZE - group.first_blocks[row]
            if (
                np.any(relative >= len(pages))
                or not np.array_equal(group.append_pages[start:end], pages[relative])
                or not np.array_equal(
                    group.append_offsets[start:end], row_positions % PAGE_SIZE
                )
            ):
                raise ValueError("cache append destinations do not match the row")
        if plan.phase == BatchPhase.PREFILL:
            if query_length > PREFILL_CHUNK_SIZE:
                raise ValueError(
                    f"prefill query length exceeds {PREFILL_CHUNK_SIZE} tokens"
                )
        elif query_length != 1:
            raise ValueError("decode rows must contain one token")

    rows = tuple(range(row_count))
    token_indices = np.arange(len(tokens), dtype=np.int64)

    def group_view(group: CacheGroup):
        page_rows = tuple(group.row(row) for row in rows)
        lengths = tuple(
            post_lengths[row] - int(group.first_blocks[row]) * PAGE_SIZE for row in rows
        )
        return page_rows, lengths

    full_rows, full_lengths = group_view(history)
    sliding_rows, sliding_lengths = group_view(window)
    full_width, full_table = _page_table(full_rows)
    sliding_width, sliding_table = _page_table(sliding_rows, WINDOW_TABLE_WIDTH)
    query_lengths = tuple(int(indptr[row + 1] - indptr[row]) for row in rows)
    batch = AttentionBatch(
        phase=plan.phase,
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
    return ((batch,), None) if plan.phase == BatchPhase.PREFILL else ((), batch)
