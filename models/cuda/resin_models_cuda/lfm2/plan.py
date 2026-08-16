"""Translate one LFM2 History cache plan into FA4/FlashInfer host metadata."""

from __future__ import annotations

from dataclasses import dataclass

from resin import BatchPhase
from resin.backend.plans import plan_positions


@dataclass(frozen=True, slots=True)
class PrefillMetadata:
    dense_token_indices: tuple[int, ...]
    dense_cu_seqlens: tuple[int, ...]
    dense_max_q_len: int
    continuation_token_indices: tuple[int, ...]
    continuation_cu_seqlens: tuple[int, ...]
    continuation_max_q_len: int
    continuation_page_table: tuple[int, ...]
    continuation_page_table_width: int
    continuation_seqused_k: tuple[int, ...]
    continuation_min_seqlen_k: int
    continuation_max_seqlen_k: int


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    block_indptr: tuple[int, ...]
    block_ids: tuple[int, ...]
    last_block_lengths: tuple[int, ...]
    append_batch_indices: tuple[int, ...]
    append_positions: tuple[int, ...]
    decode_token_indices: tuple[int, ...]
    decode_block_indptr: tuple[int, ...]
    decode_block_ids: tuple[int, ...]
    decode_last_block_lengths: tuple[int, ...]
    prefill: PrefillMetadata


def build_cache_metadata(plan, page_size: int) -> CacheMetadata:
    """Derive kernel views from one homogeneous native History plan."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    rows = len(plan.sequence_ids)
    indptr = tuple(plan.token_indptr)
    contexts = tuple(plan.context_lengths)
    positions = tuple(plan_positions(plan))
    token_count = len(plan.token_ids)
    if (
        rows == 0
        or len(indptr) != rows + 1
        or indptr[0] != 0
        or indptr[-1] != token_count
        or any(left >= right for left, right in zip(indptr, indptr[1:]))
        or len(contexts) != rows
        or len(positions) != token_count
    ):
        raise ValueError("sequence metadata does not describe nonempty token rows")
    if plan.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("plan phase must be prefill or decode")
    groups = tuple(plan.cache_groups)
    if len(groups) != 2 or tuple(group.group_index for group in groups) != (0, 1):
        raise ValueError("LFM2 requires History and State cache groups")
    group = groups[0]
    block_indptr = tuple(group.block_indptr)
    block_ids = tuple(group.block_ids)
    first_blocks = tuple(group.first_logical_blocks)
    append_pages = tuple(group.append_block_ids)
    append_offsets = tuple(group.append_block_offsets)
    if (
        len(block_indptr) != rows + 1
        or block_indptr[0] != 0
        or block_indptr[-1] != len(block_ids)
        or any(left > right for left, right in zip(block_indptr, block_indptr[1:]))
        or len(first_blocks) != rows
        or len(append_pages) != token_count
        or len(append_offsets) != token_count
    ):
        raise ValueError("cache group metadata is malformed")

    dense_tokens: list[int] = []
    dense_cu = [0]
    continuation_tokens: list[int] = []
    continuation_cu = [0]
    continuation_rows: list[tuple[int, ...]] = []
    continuation_lengths: list[int] = []
    last_lengths: list[int] = []
    append_batches: list[int] = []
    append_positions: list[int] = []
    decode_tokens: list[int] = []
    decode_indptr = [0]
    decode_pages: list[int] = []
    decode_lengths: list[int] = []
    dense_max = continuation_max = 0

    for row in range(rows):
        start, end = indptr[row : row + 2]
        query_length = end - start
        context = contexts[row]
        if context < 0 or positions[start:end] != tuple(
            range(context, context + query_length)
        ):
            raise ValueError("query positions do not continue their row context")
        page_start, page_end = block_indptr[row : row + 2]
        pages = block_ids[page_start:page_end]
        post_length = context + query_length
        expected_pages = (post_length + page_size - 1) // page_size
        if first_blocks[row] != 0 or len(pages) != expected_pages:
            raise ValueError("History row is not the exact post-append prefix")
        if not pages or any(page <= 0 for page in pages):
            raise ValueError("cache page IDs must be positive")
        for token in range(start, end):
            position = positions[token]
            if (
                append_pages[token] != pages[position // page_size]
                or append_offsets[token] != position % page_size
            ):
                raise ValueError("cache append destinations do not match the row")

        last_length = (post_length - 1) % page_size + 1
        last_lengths.append(last_length)
        append_batches.extend((row,) * query_length)
        append_positions.extend(positions[start:end])
        if plan.phase == BatchPhase.DECODE:
            if query_length != 1:
                raise ValueError("decode rows must contain one token")
            decode_tokens.append(start)
            decode_pages.extend(pages)
            decode_indptr.append(len(decode_pages))
            decode_lengths.append(last_length)
        elif context == 0:
            dense_tokens.extend(range(start, end))
            dense_cu.append(len(dense_tokens))
            dense_max = max(dense_max, query_length)
        else:
            continuation_tokens.extend(range(start, end))
            continuation_cu.append(len(continuation_tokens))
            continuation_rows.append(pages)
            continuation_lengths.append(post_length)
            continuation_max = max(continuation_max, query_length)

    continuation_width = max(map(len, continuation_rows), default=0)
    continuation_table = tuple(
        page
        for pages in continuation_rows
        for page in (*pages, *((pages[0],) * (continuation_width - len(pages))))
    )
    return CacheMetadata(
        block_indptr,
        block_ids,
        tuple(last_lengths),
        tuple(append_batches),
        tuple(append_positions),
        tuple(decode_tokens),
        tuple(decode_indptr),
        tuple(decode_pages),
        tuple(decode_lengths),
        PrefillMetadata(
            tuple(dense_tokens),
            tuple(dense_cu),
            dense_max,
            tuple(continuation_tokens),
            tuple(continuation_cu),
            continuation_max,
            continuation_table,
            continuation_width,
            tuple(continuation_lengths),
            min(continuation_lengths, default=0),
            max(continuation_lengths, default=0),
        ),
    )
