"""Backend-neutral lowering from native cache plans to named paged attention.

Models name the semantic cache groups they consume. Backends decide how those
tables become device tensors and which imported attention kernel receives them.
There is deliberately no kernel selection or registration in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt

from resin.backend.plans import PagedCachePlan


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PagedAttentionGroupSpec:
    """Name one native cache group and its minimum physical table width."""

    name: str
    group_index: int
    minimum_table_width: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("paged attention group name must be nonempty")
        if (
            isinstance(self.group_index, bool)
            or not isinstance(self.group_index, int)
            or self.group_index < 0
        ):
            raise ValueError("paged attention group index must be nonnegative")
        _positive_integer("minimum_table_width", self.minimum_table_width)


@dataclass(frozen=True, slots=True)
class PagedAttentionGroup:
    """One named dense page-table view derived from a sparse native group."""

    name: str
    page_size: int
    width: int
    page_table: npt.NDArray[np.uint64]
    lengths: tuple[int, ...]
    append_slots: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("paged attention group name must be nonempty")
        _positive_integer("page_size", self.page_size)
        _positive_integer("page table width", self.width)
        if len(self.page_table) != len(self.lengths) * self.width:
            raise ValueError(f"{self.name} page-table shape is invalid")
        if not self.lengths or any(length <= 0 for length in self.lengths):
            raise ValueError(f"{self.name} key lengths are invalid")


@dataclass(frozen=True, slots=True)
class PagedAttentionBatch:
    """Kernel-independent token rows and named paged cache groups."""

    phase: Any
    token_indices: npt.NDArray[np.int64]
    sequence_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    groups: tuple[PagedAttentionGroup, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        rows = len(self.sequence_indices)
        tokens = sum(self.query_lengths)
        if rows == 0 or len(set(self.sequence_indices)) != rows:
            raise ValueError("paged attention rows must be nonempty and unique")
        if len(self.query_lengths) != rows or any(
            length <= 0 for length in self.query_lengths
        ):
            raise ValueError("paged attention query lengths are invalid")
        if len(self.token_indices) != tokens:
            raise ValueError("query lengths do not cover the selected tokens")
        if any(not isinstance(group, PagedAttentionGroup) for group in groups):
            raise TypeError("groups must contain PagedAttentionGroup values")
        names = tuple(group.name for group in groups)
        if not groups or len(names) != len(set(names)):
            raise ValueError("paged attention group names must be nonempty and unique")
        for group in groups:
            if len(group.lengths) != rows or len(group.append_slots) != tokens:
                raise ValueError(
                    f"paged attention group {group.name!r} does not cover the batch"
                )
        object.__setattr__(self, "groups", groups)

    def group(self, name: str) -> PagedAttentionGroup:
        try:
            return next(group for group in self.groups if group.name == name)
        except StopIteration as error:
            raise KeyError(f"unknown paged attention group: {name!r}") from error


def _page_table(
    rows: tuple[npt.NDArray[np.uint64], ...], minimum_width: int
) -> tuple[int, npt.NDArray[np.uint64]]:
    if not rows or any(len(row) == 0 for row in rows):
        raise ValueError("cache page-table rows must be nonempty")
    width = max(minimum_width, *(len(row) for row in rows))
    table = np.empty((len(rows), width), dtype=np.uint64)
    for index, row in enumerate(rows):
        table[index, : len(row)] = row
        table[index, len(row) :] = row[0]
    return width, table.reshape(-1)


def build_paged_attention_batch(
    plan,
    cache_layout,
    groups: tuple[PagedAttentionGroupSpec, ...],
    *,
    max_query_tokens: int,
    decode_lookahead: int = 0,
) -> PagedAttentionBatch:
    """Validate and lower one native plan using explicit semantic group names."""

    _positive_integer("max_query_tokens", max_query_tokens)
    specs = tuple(groups)
    if not specs or any(
        not isinstance(spec, PagedAttentionGroupSpec) for spec in specs
    ):
        raise TypeError("groups must contain PagedAttentionGroupSpec values")
    names = tuple(spec.name for spec in specs)
    indices = tuple(spec.group_index for spec in specs)
    if len(names) != len(set(names)) or len(indices) != len(set(indices)):
        raise ValueError("paged attention group names and indices must be unique")

    cache = PagedCachePlan.from_plan(
        plan,
        cache_layout,
        decode_lookahead=decode_lookahead,
    )
    native = cache.batch
    from resin import BatchPhase

    if any(index >= len(cache.groups) for index in indices):
        raise ValueError("paged attention group index is outside the cache layout")

    query_lengths = tuple(map(int, native.query_lengths))
    if any(length > max_query_tokens for length in query_lengths):
        raise ValueError(f"attention query length exceeds {max_query_tokens} tokens")
    if native.phase == BatchPhase.DECODE and any(
        length != 1 for length in query_lengths
    ):
        raise ValueError("decode rows must contain one token")

    selected = tuple(cache.groups[spec.group_index] for spec in specs)
    for left, right in combinations(selected, 2):
        if np.intersect1d(left.block_slab_ids, right.block_slab_ids).size:
            raise ValueError("attention cache groups cannot address one parent")

    rows = tuple(range(native.num_sequences))
    token_indices = np.arange(native.num_tokens, dtype=np.int64)
    lowered = []
    for spec, group in zip(specs, selected, strict=True):
        descriptor = cache_layout.groups[spec.group_index]
        page_size = int(descriptor.block_tokens)
        page_rows = tuple(
            group.packed_block_ids[group.block_slice(row)] for row in rows
        )
        width, page_table = _page_table(page_rows, spec.minimum_table_width)
        lengths = tuple(
            int(native.post_lengths[row])
            - int(group.block_table_start_indices[row]) * page_size
            for row in rows
        )
        append_slots = group.packed_append_ids.astype(
            np.int64, copy=False
        ) * page_size + group.append_block_offsets.astype(np.int64, copy=False)
        lowered.append(
            PagedAttentionGroup(
                name=spec.name,
                page_size=page_size,
                width=width,
                page_table=page_table,
                lengths=lengths,
                append_slots=append_slots,
            )
        )
    return PagedAttentionBatch(
        phase=native.phase,
        token_indices=token_indices,
        sequence_indices=rows,
        query_lengths=query_lengths,
        groups=tuple(lowered),
    )


__all__ = [
    "PagedAttentionBatch",
    "PagedAttentionGroup",
    "PagedAttentionGroupSpec",
    "build_paged_attention_batch",
]
