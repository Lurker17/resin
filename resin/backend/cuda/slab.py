"""Shared device page-table mechanics for backend-owned CUDA cache arenas."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from resin import CacheGroupFamily, CacheRetention

_SLAB_NAMES = ("slabs", "slots", "first")
_PARENT_NAMES = ("parent_ids", "child_slots", "first_blocks")


@dataclass(frozen=True, slots=True)
class DevicePageGroup:
    slab_ids: Any
    slotnums: Any
    first_blocks: Any

    parent_ids = property(lambda self: self.slab_ids)
    child_slots = property(lambda self: self.slotnums)


def slice_page_groups(groups: Sequence[DevicePageGroup], row: int):
    return tuple(
        DevicePageGroup(
            group.slab_ids[row : row + 1],
            group.slotnums[row : row + 1],
            group.first_blocks[row : row + 1],
        )
        for group in groups
    )


def slab_locations(groups, group_index, positions, block_tokens):
    group = groups[group_index]
    logical = positions.div(block_tokens, rounding_mode="floor")
    first = group.first_blocks.view((-1,) + (1,) * (positions.ndim - 1))
    columns = logical - first
    width = group.slab_ids.shape[1]
    gather = columns.flatten(1).clamp(0, width - 1).long()
    slabs = torch.gather(group.slab_ids, 1, gather).view_as(columns)
    slots = torch.gather(group.slotnums, 1, gather).view_as(columns)
    valid = (columns >= 0) & (columns < width)
    return (
        torch.where(valid, slabs, 0),
        torch.where(valid, slots, 0),
        torch.where(valid, positions.remainder(block_tokens), 0),
    )


class SlabDeviceCache:
    """Common addressing for model-specific views over a fungible slab."""

    page_tokens = 128

    @property
    def slab_units(self) -> int:
        return self.geometry.slab_units

    def row(self, index: int):
        return replace(self, groups=slice_page_groups(self.groups, index))

    def locations(self, group_index: int, positions, *, block_tokens: int):
        return slab_locations(self.groups, group_index, positions, block_tokens)

    def pages(self, group_index: int, layer: int, layer_count: int):
        if not 0 <= layer < layer_count:
            raise ValueError("cache layer lies outside its cache group")
        group = self.groups[group_index]
        return (
            group.slab_ids * self.slab_units + group.slotnums * layer_count + layer
        ).to(torch.int32)

    def rows(self, group_index: int, layer: int, layer_count: int, positions):
        slabs, slots, offsets = self.locations(
            group_index, positions, block_tokens=self.page_tokens
        )
        pages = slabs * self.slab_units + slots * layer_count + layer
        return (pages * self.page_tokens + offsets).long()


def stage_page_groups(
    plan,
    staging,
    *,
    names=_SLAB_NAMES,
    width: Callable[[int], int] = lambda value: value,
):
    groups = []
    for index, source in enumerate(plan.groups):
        columns = width(max(len(source.block_slabs(row)) for row in range(plan.rows)))
        ids = np.zeros((plan.rows, columns), dtype=np.uint32)
        slots = np.zeros_like(ids)
        for row in range(plan.rows):
            row_ids, row_slots = source.block_slabs(row), source.block_slots(row)
            ids[row, : len(row_ids)], slots[row, : len(row_slots)] = row_ids, row_slots
        prefix = f"cache.{index}"
        groups.append(
            DevicePageGroup(
                staging.device(f"{prefix}.{names[0]}", ids, torch.int32),
                staging.device(f"{prefix}.{names[1]}", slots, torch.int32),
                staging.device(
                    f"{prefix}.{names[2]}",
                    source.block_table_start_indices,
                    torch.int32,
                ),
            )
        )
    return tuple(groups)


def graph_page_widths(layout, sequence_capacity: int) -> tuple[int, ...]:
    if sequence_capacity <= 0:
        raise ValueError("graph sequence capacity must be positive")
    widths = []
    for group in layout.groups:
        blocks = math.ceil(sequence_capacity / group.block_tokens)
        if group.retention is CacheRetention.WINDOW:
            blocks = min(blocks, group.window_resident_block_count)
        elif group.family is CacheGroupFamily.STATE:
            blocks = min(blocks, 2)
        widths.append(max(blocks, 1))
    return tuple(widths)


def empty_page_groups(device, batch_size: int, widths):
    def empty(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    return tuple(
        DevicePageGroup(
            empty(batch_size, width), empty(batch_size, width), empty(batch_size)
        )
        for width in widths
    )


def page_group_device_inputs(groups, *, names=_SLAB_NAMES) -> dict[str, object]:
    return {
        f"cache.{index}.{name}": value
        for index, group in enumerate(groups)
        for name, value in zip(
            names, (group.slab_ids, group.slotnums, group.first_blocks), strict=True
        )
    }


def page_group_capture_inputs(
    widths,
    batch_size: int,
    *,
    names=_SLAB_NAMES,
    padding_ids: Callable[[int], int] = lambda _index: 0,
) -> dict[str, object]:
    inputs = {}
    for index, width in enumerate(widths):
        inputs[f"cache.{index}.{names[0]}"] = (padding_ids(index),) * (
            batch_size * width
        )
        inputs[f"cache.{index}.{names[1]}"] = (0,) * (batch_size * width)
        inputs[f"cache.{index}.{names[2]}"] = (0,) * batch_size
    return inputs


def page_group_plan_inputs(
    groups,
    plan,
    *,
    label: str,
    names=_SLAB_NAMES,
    padding_ids: Callable[[int], int] = lambda _index: 0,
) -> dict[str, object]:
    inputs = {}
    for index, (target, source) in enumerate(zip(groups, plan.groups, strict=True)):
        shape = tuple(target.slab_ids.shape)
        if plan.rows > shape[0]:
            raise ValueError(f"{label} cache batch exceeds CUDA graph capacity")
        ids, slots = np.zeros(shape, dtype=np.uint32), np.zeros(shape, dtype=np.uint32)
        # Padded requests may still write dummy tokens during graph replay.
        # Use capture's scratch storage without making absent live history valid.
        ids[plan.rows :] = padding_ids(index)
        for row in range(plan.rows):
            row_ids, row_slots = source.block_slabs(row), source.block_slots(row)
            if len(row_ids) > shape[1]:
                raise ValueError(f"{label} cache table exceeds CUDA graph capacity")
            ids[row, : len(row_ids)], slots[row, : len(row_slots)] = row_ids, row_slots
        first = np.zeros(shape[0], dtype=np.uint32)
        first[: plan.rows] = source.block_table_start_indices
        for name, value in zip(names, (ids, slots, first)):
            inputs[f"cache.{index}.{name}"] = value
    return inputs


PARENT_PAGE_NAMES = _PARENT_NAMES


class GraphPageTableArena:
    """Graph-buffer lifecycle shared by model-owned paged cache arenas."""

    graph_input_names = _SLAB_NAMES
    graph_label = "CUDA model"

    def _make_graph_cache(self, groups):
        raise NotImplementedError

    def _graph_padding_id(self, _index: int) -> int:
        return 0

    def _graph_widths(self, sequence_capacity: int) -> tuple[int, ...]:
        return graph_page_widths(self.layout, sequence_capacity)

    def graph_cache(self, batch_size: int, sequence_capacity: int):
        groups = empty_page_groups(
            self.device,
            batch_size,
            self._graph_widths(sequence_capacity),
        )
        return self._make_graph_cache(groups)

    def graph_device_inputs(self, cache) -> dict[str, object]:
        return page_group_device_inputs(cache.groups, names=self.graph_input_names)

    def graph_shape_capture_inputs(
        self, batch_size: int, sequence_capacity: int
    ) -> dict[str, object]:
        return page_group_capture_inputs(
            self._graph_widths(sequence_capacity),
            batch_size,
            names=self.graph_input_names,
            padding_ids=self._graph_padding_id,
        )

    def graph_plan_inputs(self, cache, plan) -> dict[str, object]:
        return page_group_plan_inputs(
            cache.groups,
            plan,
            label=self.graph_label,
            names=self.graph_input_names,
            padding_ids=self._graph_padding_id,
        )


__all__ = [
    "DevicePageGroup",
    "GraphPageTableArena",
    "PARENT_PAGE_NAMES",
    "SlabDeviceCache",
    "empty_page_groups",
    "graph_page_widths",
    "page_group_capture_inputs",
    "page_group_device_inputs",
    "page_group_plan_inputs",
    "slab_locations",
    "slice_page_groups",
    "stage_page_groups",
]
