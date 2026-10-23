"""Small framework-neutral plan adapters for readable model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from resin._resin import BatchPlan, CacheGroupPlan, CacheLayout


@dataclass(frozen=True, slots=True)
class BatchView:
    """Validated NumPy view of the framework-neutral ``BatchPlan`` fields."""

    phase: Any
    sequence_ids: npt.NDArray[np.uint64]
    context_lengths: npt.NDArray[np.uint32]
    token_ids: npt.NDArray[np.uint32]
    token_indptr: npt.NDArray[np.uint32]
    positions: npt.NDArray[np.uint32]
    sampling_mask: npt.NDArray[np.bool_]
    cache_groups: tuple[CacheGroupPlan, ...]

    @classmethod
    def from_plan(cls, plan: BatchPlan) -> BatchView:
        """Read and validate the generic portion of one native plan."""

        sequence_ids = np.asarray(plan.sequence_ids, dtype=np.uint64)
        context_lengths = np.asarray(plan.context_lengths, dtype=np.uint32)
        token_ids = np.asarray(plan.token_ids, dtype=np.uint32)
        token_indptr = np.asarray(plan.token_indptr, dtype=np.uint32)
        positions = np.asarray(plan.positions, dtype=np.uint32)
        sampling_mask = np.asarray(plan.sampling_mask, dtype=np.bool_)
        rows = len(sequence_ids)
        tokens = len(token_ids)
        if (
            rows == 0
            or tokens == 0
            or len(context_lengths) != rows
            or len(sampling_mask) != rows
            or len(token_indptr) != rows + 1
            or int(token_indptr[0]) != 0
            or int(token_indptr[-1]) != tokens
            or np.any(token_indptr[1:] <= token_indptr[:-1])
            or len(positions) != tokens
        ):
            raise ValueError("BatchPlan does not describe nonempty token rows")
        query_lengths = np.diff(token_indptr)
        expected = np.arange(tokens, dtype=np.int64) + np.repeat(
            context_lengths.astype(np.int64) - token_indptr[:-1], query_lengths
        )
        if not np.array_equal(positions, expected):
            raise ValueError("BatchPlan positions are not contiguous within each row")
        try:
            cache_groups = tuple(plan.cache_groups)
        except (AttributeError, TypeError) as error:
            raise ValueError("BatchPlan cache_groups must be iterable") from error
        return cls(
            plan.phase,
            sequence_ids,
            context_lengths,
            token_ids,
            token_indptr,
            positions,
            sampling_mask,
            cache_groups,
        )

    @property
    def num_sequences(self) -> int:
        return len(self.sequence_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def query_lengths(self) -> npt.NDArray[np.uint32]:
        return np.diff(self.token_indptr)

    @property
    def post_lengths(self) -> npt.NDArray[np.uint32]:
        return self.context_lengths + self.query_lengths

    @property
    def last_token_indices(self) -> npt.NDArray[np.uint32]:
        return self.token_indptr[1:] - 1

    def token_slice(self, row: int) -> slice:
        start, end = map(int, self.token_indptr[row : row + 2])
        return slice(start, end)


@dataclass(frozen=True, slots=True)
class CacheGroupView:
    """Validated native cache addresses for one ordered cache group."""

    group_index: int
    blocks_per_slab: int
    block_indptr: npt.NDArray[np.uint32]
    block_slab_ids: npt.NDArray[np.uint32]
    block_slotnums: npt.NDArray[np.uint32]
    block_table_start_indices: npt.NDArray[np.uint32]
    append_slab_ids: npt.NDArray[np.uint32]
    append_slotnums: npt.NDArray[np.uint32]
    append_block_offsets: npt.NDArray[np.uint32]

    @classmethod
    def from_plan(
        cls,
        native: CacheGroupPlan,
        batch: BatchView,
        *,
        group_index: int,
        blocks_per_slab: int | None = None,
        parent_capacity: int | None = None,
        block_tokens: int | None = None,
    ) -> CacheGroupView:
        """Validate generic CSR shape, packing, bounds, and append coverage."""

        packing = int(native.blocks_per_slab)
        expected_packing = packing if blocks_per_slab is None else blocks_per_slab
        block_indptr = np.asarray(native.block_indptr, dtype=np.uint32)
        block_slab_ids = np.asarray(native.block_slab_ids, dtype=np.uint32)
        block_slotnums = np.asarray(native.block_slotnums, dtype=np.uint32)
        first_blocks = np.asarray(native.block_table_start_indices, dtype=np.uint32)
        append_slab_ids = np.asarray(native.append_slab_ids, dtype=np.uint32)
        append_slotnums = np.asarray(native.append_slotnums, dtype=np.uint32)
        append_offsets = np.asarray(native.append_block_offsets, dtype=np.uint32)
        if int(native.group_index) != group_index:
            raise ValueError(f"cache groups are not ordered at group {group_index}")
        if packing != expected_packing or packing <= 0:
            raise ValueError(f"cache group {group_index} has the wrong slab packing")
        if (
            len(block_indptr) != batch.num_sequences + 1
            or int(block_indptr[0]) != 0
            or int(block_indptr[-1]) != len(block_slab_ids)
            or np.any(block_indptr[1:] < block_indptr[:-1])
            or len(block_slotnums) != len(block_slab_ids)
            or len(first_blocks) != batch.num_sequences
        ):
            raise ValueError(f"cache group {group_index} has malformed CSR metadata")
        if (
            len(append_slab_ids) != batch.num_tokens
            or len(append_slotnums) != batch.num_tokens
            or len(append_offsets) != batch.num_tokens
        ):
            raise ValueError(
                f"cache group {group_index} append metadata does not cover every token"
            )
        if (
            np.any(block_slab_ids == 0)
            or np.any(append_slab_ids == 0)
            or np.any(block_slotnums >= packing)
            or np.any(append_slotnums >= packing)
            or (block_tokens is not None and np.any(append_offsets >= block_tokens))
            or (
                parent_capacity is not None
                and (
                    np.any(block_slab_ids > parent_capacity)
                    or np.any(append_slab_ids > parent_capacity)
                )
            )
        ):
            raise ValueError(
                f"cache group {group_index} has a physical address outside its layout"
            )
        return cls(
            group_index,
            packing,
            block_indptr,
            block_slab_ids,
            block_slotnums,
            first_blocks,
            append_slab_ids,
            append_slotnums,
            append_offsets,
        )

    def block_slice(self, row: int) -> slice:
        start, end = map(int, self.block_indptr[row : row + 2])
        return slice(start, end)

    def block_slabs(self, row: int) -> npt.NDArray[np.uint32]:
        return self.block_slab_ids[self.block_slice(row)]

    def block_slots(self, row: int) -> npt.NDArray[np.uint32]:
        return self.block_slotnums[self.block_slice(row)]

    def row_addresses(self, row: int) -> tuple[tuple[int, int], ...]:
        """Return the ordered physical addresses in one logical table row."""

        return tuple(
            zip(
                map(int, self.block_slabs(row)),
                map(int, self.block_slots(row)),
            )
        )

    def address(self, row: int, logical_block: int) -> tuple[int, int]:
        """Resolve one covered logical block to its physical slab and slot."""

        column = logical_block - int(self.block_table_start_indices[row])
        addresses = self.row_addresses(row)
        if not 0 <= column < len(addresses):
            raise ValueError("cache row does not cover the logical block")
        return addresses[column]

    @property
    def packed_block_ids(self) -> npt.NDArray[np.uint64]:
        return (
            self.block_slab_ids.astype(np.uint64) * self.blocks_per_slab
            + self.block_slotnums
        )

    @property
    def packed_append_ids(self) -> npt.NDArray[np.uint64]:
        return (
            self.append_slab_ids.astype(np.uint64) * self.blocks_per_slab
            + self.append_slotnums
        )


@dataclass(frozen=True, slots=True)
class PagedCachePlan:
    """Validated exact active cache tables for one native batch plan."""

    batch: BatchView
    layout: CacheLayout
    groups: tuple[CacheGroupView, ...]

    @classmethod
    def from_plan(
        cls,
        plan: BatchPlan,
        layout: CacheLayout,
        *,
        decode_lookahead: int = 0,
    ) -> PagedCachePlan:
        """Validate group order, physical bounds, extents, and append ownership."""

        from resin._resin import BatchPhase, CacheGroupFamily

        if decode_lookahead < 0:
            raise ValueError("decode_lookahead cannot be negative")
        batch = BatchView.from_plan(plan)
        if batch.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
            raise ValueError("BatchPlan phase must be prefill or decode")
        if len(batch.cache_groups) != len(layout.groups):
            raise ValueError("BatchPlan cache groups do not match CacheLayout")

        query_lengths = batch.query_lengths.astype(np.int64)
        row_ids = np.repeat(np.arange(batch.num_sequences), query_lengths)
        positions = batch.positions.astype(np.int64)
        lookahead = decode_lookahead if batch.phase == BatchPhase.DECODE else 0
        # Native fields are u32, but a leased decode span can extend beyond u32.
        # Keep even oversized caller-supplied lookahead from wrapping arithmetic.
        post_lengths = batch.context_lengths.astype(np.int64) + query_lengths
        arithmetic = (
            object
            if lookahead > np.iinfo(np.int64).max - int(post_lengths.max())
            else np.int64
        )
        write_tokens = query_lengths.astype(arithmetic) + lookahead
        leased = batch.context_lengths.astype(arithmetic) + write_tokens

        groups = []
        for index, (native, descriptor) in enumerate(
            zip(batch.cache_groups, layout.groups, strict=True)
        ):
            block_tokens = descriptor.block_tokens
            group = CacheGroupView.from_plan(
                native,
                batch,
                group_index=index,
                blocks_per_slab=descriptor.blocks_per_slab,
                parent_capacity=layout.parent_capacity,
                block_tokens=block_tokens,
            )
            window_blocks = descriptor.window_resident_block_count
            invalid_span = np.zeros(batch.num_sequences, dtype=np.bool_)
            maximum = 0
            if window_blocks is not None:
                assert descriptor.window_tokens is not None
                assert descriptor.read_window_tokens is not None
                maximum = descriptor.window_tokens - descriptor.read_window_tokens + 1
                invalid_span = write_tokens > maximum
            last_blocks = (leased - 1) // block_tokens
            if descriptor.family == CacheGroupFamily.STATE:
                expected_first = np.maximum(0, last_blocks - 1)
            elif window_blocks is not None:
                expected_first = np.maximum(0, last_blocks + 1 - window_blocks)
            else:
                expected_first = np.zeros(batch.num_sequences, dtype=np.int64)
            invalid_range = (group.block_table_start_indices != expected_first) | (
                np.diff(group.block_indptr) != last_blocks - expected_first + 1
            )

            logical_blocks = positions // block_tokens
            # Malformed extents, including empty rows, cannot be indexed safely.
            retained = (logical_blocks >= expected_first[row_ids]) & ~invalid_range[
                row_ids
            ]
            retained_rows = row_ids[retained]
            columns = (logical_blocks[retained] - expected_first[retained_rows]).astype(
                np.intp
            )
            addresses = group.block_indptr[:-1][retained_rows] + columns
            invalid_append = (
                group.append_slab_ids[retained] != group.block_slab_ids[addresses]
            ) | (group.append_slotnums[retained] != group.block_slotnums[addresses])
            invalid_append |= (
                group.append_block_offsets[retained]
                != positions[retained] % block_tokens
            )
            invalid_append_rows = np.zeros(batch.num_sequences, dtype=np.bool_)
            invalid_append_rows[retained_rows[invalid_append]] = True
            # Keep the original row-first failure order across all three checks.
            invalid_rows = np.flatnonzero(
                invalid_span | invalid_range | invalid_append_rows
            )
            if len(invalid_rows):
                row = int(invalid_rows[0])
                if invalid_span[row]:
                    raise ValueError(
                        f"cache group {index} row {row} write span {write_tokens[row]} "
                        f"exceeds {maximum}-token Window read/write capacity"
                    )
                if invalid_range[row]:
                    raise ValueError(
                        f"cache group {index} row {row} is not its exact active range"
                    )
                raise ValueError(
                    f"cache group {index} row {row} append target is not row-owned"
                )
            groups.append(group)
        return cls(batch, layout, tuple(groups))

    @property
    def phase(self) -> Any:
        return self.batch.phase

    @property
    def rows(self) -> int:
        return self.batch.num_sequences

    @property
    def token_ids(self) -> npt.NDArray[np.uint32]:
        return self.batch.token_ids

    @property
    def positions(self) -> npt.NDArray[np.uint32]:
        return self.batch.positions

    @property
    def token_indptr(self) -> npt.NDArray[np.uint32]:
        return self.batch.token_indptr

    @property
    def context_lengths(self) -> npt.NDArray[np.uint32]:
        return self.batch.context_lengths


@dataclass(frozen=True, slots=True)
class PlanRow:
    request_id: int
    token_ids: npt.NDArray[np.uint32]
    positions: npt.NDArray[np.uint32]
    cache: Any


def plan_rows(
    plan: BatchPlan, caches: dict[int, Any], vocab_size: int
) -> list[PlanRow]:
    """Validate and split one native plan into request-local rows."""

    batch = BatchView.from_plan(plan)
    rows = []
    for index, sequence_id in enumerate(batch.sequence_ids):
        token_slice = batch.token_slice(index)
        sequence_id = int(sequence_id)
        cache = caches.get(sequence_id)
        offset = 0 if cache is None else cache.length
        positions = batch.positions[token_slice]
        expected = np.arange(offset, offset + len(positions), dtype=np.uint32)
        if not np.array_equal(positions, expected):
            raise ValueError(
                f"non-contiguous positions for request {sequence_id}: "
                f"expected {expected.tolist()}, got {positions.tolist()}"
            )
        row_tokens = batch.token_ids[token_slice]
        if np.any(row_tokens >= vocab_size):
            raise ValueError(f"token id is out of range for request {sequence_id}")
        rows.append(PlanRow(sequence_id, row_tokens, positions, cache))
    return rows
