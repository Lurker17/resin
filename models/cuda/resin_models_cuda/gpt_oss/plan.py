"""Translate one validated native cache plan into FA4-shaped host metadata."""

from __future__ import annotations

from dataclasses import dataclass

from resin import BatchPhase

PAGE_SIZE = 128
SLIDING_WINDOW = 128
PREFILL_CHUNK_SIZE = 8192
# Paged prefill must keep the whole query plus the preceding attention window
# resident until the forward finishes. Decode still attends to only
# ``SLIDING_WINDOW`` tokens; this wider physical suffix permits one 8K prefill.
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1
WINDOW_TABLE_WIDTH = 1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE


def _integers(name: str, values, *, positive: bool = False) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error
    minimum = 1 if positive else 0
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value in result
    ):
        raise ValueError(f"{name} contains an invalid integer")
    return result


@dataclass(frozen=True, slots=True)
class CacheGroup:
    index: int
    indptr: tuple[int, ...]
    pages: tuple[int, ...]
    first_blocks: tuple[int, ...]
    append_pages: tuple[int, ...]
    append_offsets: tuple[int, ...]

    @classmethod
    def from_native(
        cls, native, index: int, rows: int, tokens: int, capacity: int | None
    ) -> CacheGroup:
        if native.group_index != index:
            raise ValueError("cache groups must be ordered History then Window")
        group = cls(
            index,
            _integers("block_indptr", native.block_indptr),
            _integers("block_ids", native.block_ids, positive=True),
            _integers("first_logical_blocks", native.first_logical_blocks),
            _integers("append_block_ids", native.append_block_ids, positive=True),
            _integers("append_block_offsets", native.append_block_offsets),
        )
        if (
            len(group.indptr) != rows + 1
            or group.indptr[0] != 0
            or group.indptr[-1] != len(group.pages)
            or any(left > right for left, right in zip(group.indptr, group.indptr[1:]))
        ):
            raise ValueError("cache block_indptr is malformed")
        if len(group.first_blocks) != rows:
            raise ValueError("cache first_logical_blocks does not cover every row")
        if len(group.append_pages) != tokens or len(group.append_offsets) != tokens:
            raise ValueError("cache append metadata does not cover every token")
        if any(offset >= PAGE_SIZE for offset in group.append_offsets):
            raise ValueError("cache append offset is outside its page")
        if capacity is not None and any(
            page > capacity for page in (*group.pages, *group.append_pages)
        ):
            raise ValueError("cache page is outside the bound allocation")
        return group

    def row(self, index: int) -> tuple[int, ...]:
        start, end = self.indptr[index : index + 2]
        return self.pages[start:end]


@dataclass(frozen=True, slots=True)
class AttentionBatch:
    phase: BatchPhase
    token_indices: tuple[int, ...]
    sequence_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    full_width: int
    full_table: tuple[int, ...]
    full_lengths: tuple[int, ...]
    sliding_width: int
    sliding_table: tuple[int, ...]
    sliding_lengths: tuple[int, ...]
    full_append_slots: tuple[int, ...]
    sliding_append_slots: tuple[int, ...]

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
    rows: tuple[tuple[int, ...], ...], minimum_width: int = 1
) -> tuple[int, tuple[int, ...]]:
    if not rows or any(not row for row in rows):
        raise ValueError("cache page-table rows must be nonempty")
    width = max(minimum_width, *(len(row) for row in rows))
    table = tuple(
        page for row in rows for page in (*row, *((row[0],) * (width - len(row))))
    )
    return width, table


def _append_slots(group: CacheGroup, token_indices: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(
            group.append_pages[index] * PAGE_SIZE + group.append_offsets[index]
            for index in token_indices
        )
    except IndexError as error:
        raise ValueError(
            "cache append metadata does not cover selected tokens"
        ) from error


def build_attention_batches(
    plan, *, parent_capacity: int | None = None
) -> tuple[tuple[AttentionBatch, ...], AttentionBatch | None]:
    """Return one prefill batch or one decode batch from a homogeneous plan."""

    sequence_ids = _integers("sequence_ids", plan.sequence_ids)
    tokens = _integers("token_ids", plan.token_ids)
    positions = _integers("positions", plan.positions)
    indptr = _integers("sequence_indptr", plan.sequence_indptr)
    contexts = _integers("context_lengths", plan.context_lengths)
    row_count = len(sequence_ids)
    if row_count == 0:
        raise ValueError("GPT-OSS cannot execute an empty batch")
    if (
        len(positions) != len(tokens)
        or len(indptr) != row_count + 1
        or indptr[0] != 0
        or indptr[-1] != len(tokens)
        or any(left >= right for left, right in zip(indptr, indptr[1:]))
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
    if set(history.pages) & set(window.pages):
        raise ValueError("History and Window cannot address one parent in a batch")

    post_lengths = []
    for row in range(row_count):
        start, end = indptr[row : row + 2]
        query_length = end - start
        context = contexts[row]
        if positions[start:end] != tuple(range(context, context + query_length)):
            raise ValueError("query positions do not continue their row context")
        post_length = context + query_length
        post_lengths.append(post_length)
        last_block = (post_length - 1) // PAGE_SIZE
        expected_window_first = max(0, last_block + 1 - WINDOW_TABLE_WIDTH)
        if history.first_blocks[row] != 0 or len(history.row(row)) != last_block + 1:
            raise ValueError("History row is not the exact post-append prefix")
        if (
            window.first_blocks[row] != expected_window_first
            or len(window.row(row)) != last_block - expected_window_first + 1
        ):
            raise ValueError("Window row is not the exact active suffix")
        for group in (history, window):
            pages = group.row(row)
            for token in range(start, end):
                relative = positions[token] // PAGE_SIZE - group.first_blocks[row]
                if (
                    relative < 0
                    or relative >= len(pages)
                    or group.append_pages[token] != pages[relative]
                    or group.append_offsets[token] != positions[token] % PAGE_SIZE
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
    token_indices = tuple(
        index for row in rows for index in range(indptr[row], indptr[row + 1])
    )

    def group_view(group: CacheGroup):
        page_rows = tuple(group.row(row) for row in rows)
        lengths = tuple(
            post_lengths[row] - group.first_blocks[row] * PAGE_SIZE for row in rows
        )
        return page_rows, lengths

    full_rows, full_lengths = group_view(history)
    sliding_rows, sliding_lengths = group_view(window)
    full_width, full_table = _page_table(full_rows)
    sliding_width, sliding_table = _page_table(sliding_rows, WINDOW_TABLE_WIDTH)
    query_lengths = tuple(indptr[row + 1] - indptr[row] for row in rows)
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
