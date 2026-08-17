"""Fungible Inkling target, draft, and convolution-state cache geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from resin import (
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
)
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding

if TYPE_CHECKING:
    from torch import Tensor

PAGE_TOKENS = 128
PREFILL_CHUNK_SIZE = 4096
GLOBAL_GROUP = 0
LOCAL_GROUP = 1
DRAFT_GROUP = 2
STATE_GROUP = 3
GROUP_COUNT = 4

GLOBAL_LAYERS = 7
LOCAL_LAYERS = 35
DRAFT_LAYERS = 6
KV_HEADS = 8
HEAD_DIM = 128
SLAB_UNITS = 70
GLOBAL_BLOCKS_PER_SLAB = 10
LOCAL_BLOCKS_PER_SLAB = 2
DRAFT_BLOCKS_PER_SLAB = 11
STATE_BLOCKS_PER_SLAB = 14
STATE_WIDTH_PER_LAYER = 1024 + 1024 + 4096 + 4096
STATE_LENGTH = 3
STATE_ELEMENTS = 42 * STATE_WIDTH_PER_LAYER * STATE_LENGTH
UNIT_ELEMENTS = 2 * PAGE_TOKENS * KV_HEADS * HEAD_DIM
SLAB_ELEMENTS = SLAB_UNITS * UNIT_ELEMENTS
LOCAL_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + 512 - 1


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """A 35 MiB slab split into small single-layer KV address units."""

    slab_units: int = SLAB_UNITS
    global_blocks_per_slab: int = GLOBAL_BLOCKS_PER_SLAB
    local_blocks_per_slab: int = LOCAL_BLOCKS_PER_SLAB
    draft_blocks_per_slab: int = DRAFT_BLOCKS_PER_SLAB
    state_blocks_per_slab: int = STATE_BLOCKS_PER_SLAB

    def __post_init__(self) -> None:
        usages = {
            "global": self.global_blocks_per_slab * GLOBAL_LAYERS * UNIT_ELEMENTS,
            "local": self.local_blocks_per_slab * LOCAL_LAYERS * UNIT_ELEMENTS,
            "draft": self.draft_blocks_per_slab * DRAFT_LAYERS * UNIT_ELEMENTS,
            "state": self.state_blocks_per_slab * STATE_ELEMENTS,
        }
        if any(elements > self.slab_elements for elements in usages.values()):
            raise ValueError(f"Inkling cache block packing exceeds a slab: {usages}")

    @property
    def slab_elements(self) -> int:
        return self.slab_units * UNIT_ELEMENTS

    @property
    def bytes_per_slab(self) -> int:
        return self.slab_elements * 2

    def layout(self, slab_capacity: int) -> CacheLayout:
        return CacheLayout(
            slab_capacity,
            (
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.global_blocks_per_slab
                ),
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.local_blocks_per_slab,
                    retention=CacheRetention.WINDOW,
                    window_tokens=LOCAL_RETENTION_TOKENS,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.draft_blocks_per_slab
                ),
                CacheGroupLayout(
                    PREFILL_CHUNK_SIZE,
                    blocks_per_slab=self.state_blocks_per_slab,
                    family=CacheGroupFamily.STATE,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceGroup:
    slab_ids: Tensor
    slotnums: Tensor
    first_blocks: Tensor


@dataclass(frozen=True, slots=True)
class DeviceCache:
    """Graph-stable cache tables over one slab arena."""

    torch: Any
    page_tokens: int
    slab_units: int
    groups: tuple[DeviceGroup, DeviceGroup, DeviceGroup, DeviceGroup]
    keys: Tensor
    values: Tensor
    states: Tensor

    def row(self, index: int) -> DeviceCache:
        groups = tuple(
            DeviceGroup(
                group.slab_ids[index : index + 1],
                group.slotnums[index : index + 1],
                group.first_blocks[index : index + 1],
            )
            for group in self.groups
        )
        return DeviceCache(
            self.torch,
            self.page_tokens,
            self.slab_units,
            groups,  # type: ignore[arg-type]
            self.keys,
            self.values,
            self.states,
        )

    def pages(self, group_index: int, layer: int, layer_count: int) -> Tensor:
        if not 0 <= layer < layer_count:
            raise ValueError("Inkling cache layer lies outside its cache group")
        group = self.groups[group_index]
        return (
            group.slab_ids * self.slab_units + group.slotnums * layer_count + layer
        ).to(self.torch.int32)

    def locations(
        self, group_index: int, positions: Tensor, *, block_tokens: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        group = self.groups[group_index]
        logical = positions.div(block_tokens, rounding_mode="floor")
        first = group.first_blocks.view((-1,) + (1,) * (positions.ndim - 1))
        columns = logical - first
        width = group.slab_ids.shape[1]
        gather = columns.flatten(1).clamp(0, width - 1).long()
        slabs = self.torch.gather(group.slab_ids, 1, gather).view_as(columns)
        slots = self.torch.gather(group.slotnums, 1, gather).view_as(columns)
        valid = (columns >= 0) & (columns < width)
        slabs = self.torch.where(valid, slabs, 0)
        slots = self.torch.where(valid, slots, 0)
        offsets = self.torch.where(valid, positions.remainder(block_tokens), 0)
        return slabs, slots, offsets

    def rows(
        self,
        group_index: int,
        layer: int,
        layer_count: int,
        positions: Tensor,
    ) -> Tensor:
        slabs, slots, offsets = self.locations(
            group_index, positions, block_tokens=self.page_tokens
        )
        pages = slabs * self.slab_units + slots * layer_count + layer
        return (pages * self.page_tokens + offsets).long()

    def state_at(self, positions: Tensor) -> Tensor:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        return self.states[slabs.long(), slots.long()]

    def store_state(self, positions: Tensor, state: Tensor) -> None:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        self.states[slabs.long(), slots.long()] = state


class CacheArena:
    """Layer-addressable views over one Engine-owned fungible slab allocation."""

    def __init__(self, torch: Any, device: Any, layout: CacheLayout) -> None:
        self.torch = torch
        self.device = device
        self.layout = layout
        self.geometry = CacheGeometry()
        self.binding = CachePoolBinding(layout, resource="Inkling cache")
        physical = layout.parent_capacity + 1
        self.storage = torch.zeros(
            (physical, SLAB_ELEMENTS), dtype=torch.bfloat16, device=device
        )
        flat = self.storage.flatten()
        page_elements = PAGE_TOKENS * KV_HEADS * HEAD_DIM
        page_shape = (physical * SLAB_UNITS, PAGE_TOKENS, KV_HEADS, HEAD_DIM)
        page_stride = (UNIT_ELEMENTS, KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
        self.keys = torch.as_strided(flat, page_shape, page_stride)
        self.values = torch.as_strided(
            flat, page_shape, page_stride, storage_offset=page_elements
        )
        self.states = torch.as_strided(
            flat,
            (
                physical,
                STATE_BLOCKS_PER_SLAB,
                42,
                STATE_LENGTH,
                STATE_WIDTH_PER_LAYER,
            ),
            (
                SLAB_ELEMENTS,
                STATE_ELEMENTS,
                STATE_LENGTH * STATE_WIDTH_PER_LAYER,
                STATE_WIDTH_PER_LAYER,
                1,
            ),
        )

    @staticmethod
    def unit_page(slab_id: int, slotnum: int, layers: int, layer: int) -> int:
        if not 0 <= layer < layers:
            raise ValueError("cache layer offset is outside its block")
        return slab_id * SLAB_UNITS + slotnum * layers + layer

    def state(self, slab_id: int, slotnum: int) -> Tensor:
        if (
            slab_id < 0
            or slab_id > self.layout.parent_capacity
            or slotnum < 0
            or slotnum >= STATE_BLOCKS_PER_SLAB
        ):
            raise ValueError("Inkling state address lies outside the cache arena")
        return self.states[slab_id, slotnum]

    def reset_null(self) -> None:
        self.storage[0].zero_()

    def tensorize(self, plan: PagedCachePlan, staging: Any) -> DeviceCache:
        groups = []
        for index, group in enumerate(plan.groups):
            rows = plan.rows
            width = max(
                len(group.block_slabs(row))
                for row in range(rows)
            )
            slabs = np.zeros((rows, width), dtype=np.uint32)
            slots = np.zeros_like(slabs)
            for row in range(rows):
                row_slabs = group.block_slabs(row)
                row_slots = group.block_slots(row)
                slabs[row, : len(row_slabs)] = row_slabs
                slots[row, : len(row_slots)] = row_slots
            groups.append(
                DeviceGroup(
                    staging.device(f"cache.{index}.slabs", slabs, self.torch.int32),
                    staging.device(f"cache.{index}.slots", slots, self.torch.int32),
                    staging.device(
                        f"cache.{index}.first",
                        group.block_table_start_indices,
                        self.torch.int32,
                    ),
                )
            )
        return DeviceCache(
            self.torch,
            PAGE_TOKENS,
            self.geometry.slab_units,
            groups,  # type: ignore[arg-type]
            self.keys,
            self.values,
            self.states,
        )

    def _graph_group_widths(self, sequence_capacity: int) -> tuple[int, ...]:
        if sequence_capacity <= 0:
            raise ValueError("Inkling graph sequence capacity must be positive")
        widths = []
        for descriptor in self.layout.groups:
            total = math.ceil(sequence_capacity / descriptor.block_tokens)
            if descriptor.family is CacheGroupFamily.STATE:
                total = min(total, 2)
            elif descriptor.retention is CacheRetention.WINDOW:
                total = min(total, descriptor.window_resident_block_count)
            widths.append(max(total, 1))
        return tuple(widths)

    def graph_cache(self, batch_size: int, sequence_capacity: int) -> DeviceCache:
        """Allocate fixed cache tables for one padded CUDA graph bucket."""

        groups = tuple(
            DeviceGroup(
                self.torch.empty(
                    (batch_size, width),
                    dtype=self.torch.int32,
                    device=self.device,
                ),
                self.torch.empty(
                    (batch_size, width),
                    dtype=self.torch.int32,
                    device=self.device,
                ),
                self.torch.empty(
                    batch_size,
                    dtype=self.torch.int32,
                    device=self.device,
                ),
            )
            for width in self._graph_group_widths(sequence_capacity)
        )
        return DeviceCache(
            self.torch,
            PAGE_TOKENS,
            self.geometry.slab_units,
            groups,  # type: ignore[arg-type]
            self.keys,
            self.values,
            self.states,
        )

    @staticmethod
    def graph_device_inputs(cache: DeviceCache) -> dict[str, object]:
        inputs: dict[str, object] = {}
        for index, group in enumerate(cache.groups):
            inputs[f"cache.{index}.slabs"] = group.slab_ids
            inputs[f"cache.{index}.slots"] = group.slotnums
            inputs[f"cache.{index}.first"] = group.first_blocks
        return inputs

    def graph_shape_capture_inputs(
        self, batch_size: int, sequence_capacity: int
    ) -> dict[str, object]:
        """Return null-slab tables used only to warm up and capture a graph."""

        inputs: dict[str, object] = {}
        for index, width in enumerate(self._graph_group_widths(sequence_capacity)):
            inputs[f"cache.{index}.slabs"] = (0,) * (batch_size * width)
            inputs[f"cache.{index}.slots"] = (0,) * (batch_size * width)
            inputs[f"cache.{index}.first"] = (0,) * batch_size
        return inputs

    @staticmethod
    def graph_plan_inputs(
        cache: DeviceCache, plan: PagedCachePlan
    ) -> dict[str, object]:
        """Pad a current Engine cache plan into fixed graph-owned tables."""

        inputs: dict[str, object] = {}
        for index, (target, source) in enumerate(
            zip(cache.groups, plan.groups, strict=True)
        ):
            shape = tuple(target.slab_ids.shape)
            slabs = np.zeros(shape, dtype=np.uint32)
            slots = np.zeros(shape, dtype=np.uint32)
            if plan.rows > shape[0]:
                raise ValueError("Inkling cache batch exceeds CUDA graph capacity")
            for row in range(plan.rows):
                row_slabs = source.block_slabs(row)
                row_slots = source.block_slots(row)
                if len(row_slabs) > shape[1]:
                    raise ValueError("Inkling cache table exceeds CUDA graph capacity")
                slabs[row, : len(row_slabs)] = row_slabs
                slots[row, : len(row_slots)] = row_slots
            first = np.zeros(shape[0], dtype=np.uint32)
            first[: plan.rows] = source.block_table_start_indices
            inputs[f"cache.{index}.slabs"] = slabs
            inputs[f"cache.{index}.slots"] = slots
            inputs[f"cache.{index}.first"] = first
        return inputs


__all__ = [
    "CacheArena",
    "CacheGeometry",
    "DeviceCache",
    "DeviceGroup",
    "DRAFT_GROUP",
    "GLOBAL_GROUP",
    "GROUP_COUNT",
    "LOCAL_GROUP",
    "LOCAL_RETENTION_TOKENS",
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "SLAB_ELEMENTS",
    "SLAB_UNITS",
    "STATE_ELEMENTS",
    "STATE_GROUP",
    "STATE_LENGTH",
    "STATE_WIDTH_PER_LAYER",
    "UNIT_ELEMENTS",
]
