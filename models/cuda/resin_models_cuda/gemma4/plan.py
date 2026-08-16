"""Translate native flat-cache metadata to Gemma 4 paged-attention inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase

PAGE_SIZE = 64
PREFILL_CHUNK_SIZE = 4096
SLIDING_WINDOW = 1024
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1


@dataclass(frozen=True, slots=True)
class PagedAttentionPlan:
    qo_indptr: npt.NDArray[np.int32]
    kv_indptr: npt.NDArray[np.int32]
    page_indices: npt.NDArray[np.int32]
    last_page_lengths: npt.NDArray[np.int32]
    append_slots: npt.NDArray[np.int64]
    first_token_positions: tuple[int, ...]
    kv_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Gemma4BatchPlan:
    phase: BatchPhase
    token_ids: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    token_indptr: npt.NDArray[np.int32]
    local: PagedAttentionPlan
    global_: PagedAttentionPlan
    last_token_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _CacheGroup:
    index: int
    indptr: npt.NDArray[np.int64]
    pages: npt.NDArray[np.int64]
    first_blocks: npt.NDArray[np.int64]
    append_pages: npt.NDArray[np.int64]
    append_offsets: npt.NDArray[np.int64]
    packing: int
    parent_capacity: int | None

    @classmethod
    def from_native(
        cls,
        native,
        *,
        index: int,
        rows: int,
        tokens: int,
        packing: int,
        parent_capacity: int | None,
    ) -> _CacheGroup:
        group = cls(
            index=index,
            indptr=np.asarray(native.block_indptr, dtype=np.int64),
            pages=np.asarray(native.block_ids, dtype=np.int64),
            first_blocks=np.asarray(native.first_logical_blocks, dtype=np.int64),
            append_pages=np.asarray(native.append_block_ids, dtype=np.int64),
            append_offsets=np.asarray(native.append_block_offsets, dtype=np.int64),
            packing=packing,
            parent_capacity=parent_capacity,
        )
        if native.group_index != index:
            raise ValueError("Gemma 4 cache groups must be ordered global then local")
        if (
            len(group.indptr) != rows + 1
            or group.indptr[0] != 0
            or group.indptr[-1] != len(group.pages)
            or np.any(group.indptr[1:] < group.indptr[:-1])
            or len(group.first_blocks) != rows
        ):
            raise ValueError("cache group CSR metadata is malformed")
        if len(group.append_pages) != tokens or len(group.append_offsets) != tokens:
            raise ValueError("cache append metadata does not cover every token")
        maximum = None if parent_capacity is None else parent_capacity * packing
        if (
            np.any(group.pages <= 0)
            or np.any(group.append_pages <= 0)
            or np.any(group.append_offsets < 0)
            or np.any(group.append_offsets >= PAGE_SIZE)
            or (
                maximum is not None
                and (np.any(group.pages > maximum) or np.any(group.append_pages > maximum))
            )
        ):
            raise ValueError("cache group addresses outside its physical slab")
        return group

    def row(self, index: int) -> npt.NDArray[np.int64]:
        start, end = self.indptr[index : index + 2]
        return self.pages[int(start) : int(end)]


def _paged_plan(
    group: _CacheGroup,
    token_indptr: npt.NDArray[np.int32],
    positions: npt.NDArray[np.int64],
) -> PagedAttentionPlan:
    page_rows: list[npt.NDArray[np.int64]] = []
    last_page_lengths = []
    first_positions = []
    kv_lengths = []
    for row in range(len(token_indptr) - 1):
        start, end = map(int, token_indptr[row : row + 2])
        post_length = int(positions[end - 1]) + 1
        last_block = (post_length - 1) // PAGE_SIZE
        first_block = int(group.first_blocks[row])
        pages = group.row(row)
        if first_block + len(pages) != last_block + 1:
            raise ValueError("cache row is not the exact active logical range")
        row_positions = positions[start:end]
        relative = row_positions // PAGE_SIZE - first_block
        if (
            np.any(relative < 0)
            or np.any(relative >= len(pages))
            or not np.array_equal(group.append_pages[start:end], pages[relative])
            or not np.array_equal(
                group.append_offsets[start:end], row_positions % PAGE_SIZE
            )
        ):
            raise ValueError("cache append targets do not match scheduled positions")
        first_position = first_block * PAGE_SIZE
        kv_length = post_length - first_position
        page_rows.append(pages - 1)
        last_page_lengths.append((post_length - 1) % PAGE_SIZE + 1)
        first_positions.append(first_position)
        kv_lengths.append(kv_length)

    kv_indptr = np.zeros(len(page_rows) + 1, dtype=np.int32)
    kv_indptr[1:] = np.cumsum([len(row) for row in page_rows], dtype=np.int32)
    return PagedAttentionPlan(
        qo_indptr=token_indptr,
        kv_indptr=kv_indptr,
        page_indices=np.concatenate(page_rows).astype(np.int32, copy=False),
        last_page_lengths=np.asarray(last_page_lengths, dtype=np.int32),
        append_slots=(
            (group.append_pages - 1) * PAGE_SIZE + group.append_offsets
        ).astype(np.int64, copy=False),
        first_token_positions=tuple(first_positions),
        kv_lengths=tuple(kv_lengths),
    )


def build_gemma4_batch_plan(
    plan,
    *,
    history_blocks_per_parent: int,
    parent_capacity: int | None = None,
) -> Gemma4BatchPlan:
    token_ids = np.asarray(plan.token_ids, dtype=np.int64)
    positions = np.asarray(plan.positions, dtype=np.int64)
    token_indptr = np.asarray(plan.token_indptr, dtype=np.int32)
    contexts = np.asarray(plan.context_lengths, dtype=np.int64)
    rows = plan.num_sequences
    if (
        not rows
        or not len(token_ids)
        or len(positions) != len(token_ids)
        or len(token_indptr) != rows + 1
        or token_indptr[0] != 0
        or token_indptr[-1] != len(token_ids)
        or np.any(token_indptr[1:] <= token_indptr[:-1])
        or len(contexts) != rows
    ):
        raise ValueError("plan does not describe nonempty Gemma 4 token rows")
    if plan.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("plan phase must be prefill or decode")
    query_lengths = np.diff(token_indptr)
    if plan.phase == BatchPhase.PREFILL and np.any(query_lengths > PREFILL_CHUNK_SIZE):
        raise ValueError(f"prefill rows cannot exceed {PREFILL_CHUNK_SIZE} tokens")
    if plan.phase == BatchPhase.DECODE and np.any(query_lengths != 1):
        raise ValueError("decode rows must contain one token")
    for row, (start, end) in enumerate(zip(token_indptr[:-1], token_indptr[1:])):
        expected = np.arange(contexts[row], contexts[row] + end - start)
        if not np.array_equal(positions[start:end], expected):
            raise ValueError("plan positions are not contiguous within each row")

    groups = tuple(plan.cache_groups)
    if len(groups) != 2:
        raise ValueError("Gemma 4 requires global History and local Window groups")
    history = _CacheGroup.from_native(
        groups[0],
        index=0,
        rows=rows,
        tokens=len(token_ids),
        packing=history_blocks_per_parent,
        parent_capacity=parent_capacity,
    )
    local = _CacheGroup.from_native(
        groups[1],
        index=1,
        rows=rows,
        tokens=len(token_ids),
        packing=1,
        parent_capacity=parent_capacity,
    )
    for row in range(rows):
        post_length = int(positions[int(token_indptr[row + 1]) - 1]) + 1
        last_block = (post_length - 1) // PAGE_SIZE
        expected_local_first = max(
            0,
            last_block
            + 1
            - (1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE),
        )
        if history.first_blocks[row] != 0:
            raise ValueError("global History cache must begin at logical block zero")
        if local.first_blocks[row] != expected_local_first:
            raise ValueError("local Window cache does not expose the required suffix")

    return Gemma4BatchPlan(
        phase=plan.phase,
        token_ids=token_ids,
        positions=positions,
        token_indptr=token_indptr,
        local=_paged_plan(local, token_indptr, positions),
        global_=_paged_plan(history, token_indptr, positions),
        last_token_indices=(token_indptr[1:] - 1).astype(np.int64),
    )


__all__ = [
    "Gemma4BatchPlan",
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "PagedAttentionPlan",
    "SLIDING_WINDOW",
    "WINDOW_RETENTION_TOKENS",
    "build_gemma4_batch_plan",
]
