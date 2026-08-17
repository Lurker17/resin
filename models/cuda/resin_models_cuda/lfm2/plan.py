"""Translate one LFM2 History cache plan into FA4/FlashInfer host metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase


@dataclass(frozen=True, slots=True)
class PrefillMetadata:
    dense_token_indices: npt.NDArray[np.uint32]
    dense_cu_seqlens: npt.NDArray[np.uint32]
    dense_max_q_len: int
    continuation_token_indices: npt.NDArray[np.uint32]
    continuation_cu_seqlens: npt.NDArray[np.uint32]
    continuation_max_q_len: int
    continuation_page_table: npt.NDArray[np.uint32]
    continuation_page_table_width: int
    continuation_seqused_k: npt.NDArray[np.uint32]
    continuation_min_seqlen_k: int
    continuation_max_seqlen_k: int


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    block_indptr: npt.NDArray[np.uint32]
    page_indices: npt.NDArray[np.uint32]
    last_block_lengths: npt.NDArray[np.uint32]
    append_batch_indices: npt.NDArray[np.uint32]
    append_positions: npt.NDArray[np.uint32]
    decode_token_indices: npt.NDArray[np.uint32]
    decode_block_indptr: npt.NDArray[np.uint32]
    decode_page_indices: npt.NDArray[np.uint32]
    decode_last_block_lengths: npt.NDArray[np.uint32]
    prefill: PrefillMetadata


def build_cache_metadata(plan, page_size: int) -> CacheMetadata:
    """Derive kernel views from one homogeneous native History plan."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    rows = plan.num_sequences
    indptr = plan.token_indptr
    contexts = plan.context_lengths
    positions = plan.positions
    token_count = plan.num_tokens
    if (
        rows == 0
        or len(indptr) != rows + 1
        or indptr[0] != 0
        or indptr[-1] != token_count
        or np.any(indptr[1:] <= indptr[:-1])
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
    if group.blocks_per_slab != 1:
        raise ValueError("LFM2 History requires one block per physical parent")
    block_indptr = group.block_indptr
    page_indices = group.block_slab_ids
    block_slotnums = group.block_slotnums
    first_blocks = group.block_table_start_indices
    append_pages = group.append_slab_ids
    append_slotnums = group.append_slotnums
    append_offsets = group.append_block_offsets
    if (
        len(block_indptr) != rows + 1
        or block_indptr[0] != 0
        or block_indptr[-1] != len(page_indices)
        or np.any(block_indptr[1:] < block_indptr[:-1])
        or len(first_blocks) != rows
        or len(append_pages) != token_count
        or len(append_offsets) != token_count
        or len(page_indices) != len(block_slotnums)
        or len(append_pages) != len(append_slotnums)
        or np.any(page_indices == 0)
        or np.any(append_pages == 0)
        or np.any(block_slotnums != 0)
        or np.any(append_slotnums != 0)
        or np.any(append_offsets >= page_size)
    ):
        raise ValueError("cache group metadata is malformed")

    dense_tokens: list[npt.NDArray[np.uint32]] = []
    dense_cu = [0]
    continuation_tokens: list[npt.NDArray[np.uint32]] = []
    continuation_cu = [0]
    continuation_rows: list[npt.NDArray[np.uint32]] = []
    continuation_lengths: list[int] = []
    last_lengths: list[int] = []
    decode_tokens: list[int] = []
    decode_indptr = [0]
    decode_pages: list[npt.NDArray[np.uint32]] = []
    decode_lengths: list[int] = []
    dense_max = continuation_max = 0

    for row in range(rows):
        start, end = map(int, indptr[row : row + 2])
        query_length = end - start
        context = int(contexts[row])
        page_start, page_end = map(int, block_indptr[row : row + 2])
        pages = page_indices[page_start:page_end]
        post_length = context + query_length
        expected_pages = (post_length + page_size - 1) // page_size
        if int(first_blocks[row]) != 0 or len(pages) != expected_pages:
            raise ValueError("History row is not the exact post-append prefix")
        if len(pages) == 0:
            raise ValueError("cache page IDs must be positive")
        row_positions = positions[start:end]
        if not np.array_equal(
            append_pages[start:end], pages[row_positions // page_size]
        ) or not np.array_equal(append_offsets[start:end], row_positions % page_size):
            raise ValueError("cache append destinations do not match the row")

        last_length = (post_length - 1) % page_size + 1
        last_lengths.append(last_length)
        if plan.phase == BatchPhase.DECODE:
            if query_length != 1:
                raise ValueError("decode rows must contain one token")
            decode_tokens.append(start)
            decode_pages.append(pages)
            decode_indptr.append(decode_indptr[-1] + len(pages))
            decode_lengths.append(last_length)
        elif context == 0:
            dense_tokens.append(np.arange(start, end, dtype=np.uint32))
            dense_cu.append(dense_cu[-1] + query_length)
            dense_max = max(dense_max, query_length)
        else:
            continuation_tokens.append(np.arange(start, end, dtype=np.uint32))
            continuation_cu.append(continuation_cu[-1] + query_length)
            continuation_rows.append(pages)
            continuation_lengths.append(post_length)
            continuation_max = max(continuation_max, query_length)

    continuation_width = max(map(len, continuation_rows), default=0)
    continuation_table = np.empty(
        (len(continuation_rows), continuation_width), dtype=np.uint32
    )
    for index, pages in enumerate(continuation_rows):
        continuation_table[index, : len(pages)] = pages
        continuation_table[index, len(pages) :] = pages[0]

    def concatenate(parts: list[npt.NDArray[np.uint32]]) -> npt.NDArray[np.uint32]:
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)

    return CacheMetadata(
        block_indptr,
        page_indices,
        np.asarray(last_lengths, dtype=np.uint32),
        np.repeat(np.arange(rows, dtype=np.uint32), np.diff(indptr).astype(np.intp)),
        positions,
        np.asarray(decode_tokens, dtype=np.uint32),
        np.asarray(decode_indptr, dtype=np.uint32),
        concatenate(decode_pages),
        np.asarray(decode_lengths, dtype=np.uint32),
        PrefillMetadata(
            concatenate(dense_tokens),
            np.asarray(dense_cu, dtype=np.uint32),
            dense_max,
            concatenate(continuation_tokens),
            np.asarray(continuation_cu, dtype=np.uint32),
            continuation_max,
            continuation_table.reshape(-1),
            continuation_width,
            np.asarray(continuation_lengths, dtype=np.uint32),
            min(continuation_lengths, default=0),
            max(continuation_lengths, default=0),
        ),
    )
