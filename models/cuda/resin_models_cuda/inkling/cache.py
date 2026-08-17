"""Fungible Inkling target, draft, and convolution-state cache geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from resin import (
    BatchPhase,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
)
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
class CacheCapacityPlan:
    total_memory_bytes: int
    free_memory_bytes: int
    gpu_memory_utilization: float
    protected_free_bytes: int
    cache_budget_bytes: int
    bytes_per_slab: int
    slab_capacity: int
    allocation_bytes: int


def plan_cache_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    geometry: CacheGeometry,
    *,
    reserved_slabs: int = 1,
) -> CacheCapacityPlan:
    if total_memory_bytes <= 0 or not 0 < free_memory_bytes <= total_memory_bytes:
        raise ValueError("CUDA memory totals must be positive and internally valid")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if reserved_slabs <= 0:
        raise ValueError("reserved_slabs must include the null slab")
    protected = math.ceil(total_memory_bytes * (1.0 - gpu_memory_utilization))
    budget = free_memory_bytes - protected
    capacity = budget // geometry.bytes_per_slab - reserved_slabs
    if capacity <= 0:
        raise MemoryError(
            "Inkling cache budget leaves no usable slabs "
            f"(total={total_memory_bytes}, free={free_memory_bytes}, "
            f"protected={protected}, slab={geometry.bytes_per_slab})"
        )
    allocation = (capacity + reserved_slabs) * geometry.bytes_per_slab
    return CacheCapacityPlan(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        protected,
        budget,
        geometry.bytes_per_slab,
        capacity,
        allocation,
    )


@dataclass(frozen=True, slots=True)
class GroupPlan:
    block_indptr: npt.NDArray[np.uint32]
    block_slab_ids: npt.NDArray[np.uint32]
    block_slotnums: npt.NDArray[np.uint32]
    first_blocks: npt.NDArray[np.uint32]
    append_slab_ids: npt.NDArray[np.uint32]
    append_slotnums: npt.NDArray[np.uint32]
    append_offsets: npt.NDArray[np.uint32]
    blocks_per_slab: int

    def row_addresses(self, row: int) -> tuple[tuple[int, int], ...]:
        start, end = map(int, self.block_indptr[row : row + 2])
        return tuple(
            zip(
                map(int, self.block_slab_ids[start:end]),
                map(int, self.block_slotnums[start:end]),
            )
        )

    def address(self, row: int, logical_block: int) -> tuple[int, int]:
        addresses = self.row_addresses(row)
        column = logical_block - int(self.first_blocks[row])
        if column < 0 or column >= len(addresses):
            raise ValueError("Inkling cache row does not cover a logical block")
        return addresses[column]


@dataclass(frozen=True, slots=True)
class CachePlan:
    phase: BatchPhase
    token_ids: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    token_indptr: npt.NDArray[np.int32]
    context_lengths: npt.NDArray[np.int64]
    groups: tuple[GroupPlan, GroupPlan, GroupPlan, GroupPlan]

    @property
    def rows(self) -> int:
        return len(self.token_indptr) - 1


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


def _group_plan(
    native: Any,
    *,
    index: int,
    rows: int,
    tokens: int,
    expected_packing: int,
    slab_capacity: int,
) -> GroupPlan:
    if native.group_index != index or native.blocks_per_slab != expected_packing:
        raise ValueError("Inkling cache groups do not match the declared layout")
    values = GroupPlan(
        np.asarray(native.block_indptr, dtype=np.uint32),
        np.asarray(native.block_slab_ids, dtype=np.uint32),
        np.asarray(native.block_slotnums, dtype=np.uint32),
        np.asarray(native.block_table_start_indices, dtype=np.uint32),
        np.asarray(native.append_slab_ids, dtype=np.uint32),
        np.asarray(native.append_slotnums, dtype=np.uint32),
        np.asarray(native.append_block_offsets, dtype=np.uint32),
        expected_packing,
    )
    if (
        len(values.block_indptr) != rows + 1
        or values.block_indptr[0] != 0
        or values.block_indptr[-1] != len(values.block_slab_ids)
        or np.any(values.block_indptr[1:] < values.block_indptr[:-1])
        or len(values.block_slab_ids) != len(values.block_slotnums)
        or len(values.first_blocks) != rows
        or len(values.append_slab_ids) != tokens
        or len(values.append_slotnums) != tokens
        or len(values.append_offsets) != tokens
    ):
        raise ValueError("Inkling cache group CSR metadata is malformed")
    if (
        np.any(values.block_slab_ids == 0)
        or np.any(values.append_slab_ids == 0)
        or np.any(values.block_slab_ids > slab_capacity)
        or np.any(values.append_slab_ids > slab_capacity)
        or np.any(values.block_slotnums >= expected_packing)
        or np.any(values.append_slotnums >= expected_packing)
    ):
        raise ValueError("Inkling cache address lies outside the physical arena")
    return values


def build_cache_plan(
    plan: Any,
    layout: CacheLayout,
    *,
    decode_lookahead: int = 0,
) -> CachePlan:
    rows = plan.num_sequences
    token_ids = np.asarray(plan.token_ids, dtype=np.int64)
    positions = np.asarray(plan.positions, dtype=np.int64)
    indptr = np.asarray(plan.token_indptr, dtype=np.int32)
    contexts = np.asarray(plan.context_lengths, dtype=np.int64)
    if (
        rows <= 0
        or len(indptr) != rows + 1
        or indptr[0] != 0
        or indptr[-1] != len(token_ids)
        or np.any(indptr[1:] <= indptr[:-1])
        or len(positions) != len(token_ids)
        or len(contexts) != rows
    ):
        raise ValueError("Inkling batch metadata is malformed")
    if plan.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("Inkling batch phase must be prefill or decode")
    if decode_lookahead < 0 or (plan.phase != BatchPhase.DECODE and decode_lookahead):
        raise ValueError("Inkling decode lookahead is only valid for decode")
    for row, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        expected = np.arange(int(contexts[row]), int(contexts[row]) + end - start)
        if not np.array_equal(positions[start:end], expected):
            raise ValueError("Inkling positions must continue the row context")
        if plan.phase == BatchPhase.DECODE and end - start != 1:
            raise ValueError("Inkling decode rows contain one anchor token")

    native_groups = tuple(plan.cache_groups)
    if len(native_groups) != GROUP_COUNT:
        raise ValueError("Inkling requires global, local, draft, and state groups")
    packings = (
        GLOBAL_BLOCKS_PER_SLAB,
        LOCAL_BLOCKS_PER_SLAB,
        DRAFT_BLOCKS_PER_SLAB,
        STATE_BLOCKS_PER_SLAB,
    )
    groups = tuple(
        _group_plan(
            native,
            index=index,
            rows=rows,
            tokens=len(token_ids),
            expected_packing=packings[index],
            slab_capacity=layout.parent_capacity,
        )
        for index, native in enumerate(native_groups)
    )
    for group_index, (group, descriptor) in enumerate(
        zip(groups, layout.groups, strict=True)
    ):
        block_tokens = descriptor.block_tokens
        for row, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
            post = int(contexts[row]) + int(end - start)
            leased = post + (decode_lookahead if plan.phase == BatchPhase.DECODE else 0)
            first = int(group.first_blocks[row])
            addresses = group.row_addresses(row)
            last = (leased - 1) // block_tokens
            expected_first = (
                max(0, last + 1 - descriptor.window_resident_block_count)
                if descriptor.retention is CacheRetention.WINDOW
                else 0
            )
            if first != expected_first or first + len(addresses) != last + 1:
                raise ValueError("Inkling cache row is not the exact active range")
            logical = positions[start:end] // block_tokens
            retained = logical >= first
            for flat, logical_block, keep in zip(
                range(start, end), logical, retained, strict=True
            ):
                if not keep:
                    continue
                address = group.address(row, int(logical_block))
                if address != (
                    int(group.append_slab_ids[flat]),
                    int(group.append_slotnums[flat]),
                ) or int(group.append_offsets[flat]) != int(
                    positions[flat] % block_tokens
                ):
                    raise ValueError("Inkling append address disagrees with its table")
        if group_index in (GLOBAL_GROUP, DRAFT_GROUP) and np.any(group.first_blocks):
            raise ValueError("Inkling full-history caches must begin at block zero")
    return CachePlan(
        plan.phase,
        token_ids,
        positions,
        indptr,
        contexts,
        groups,  # type: ignore[arg-type]
    )


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

    def bind(self, pool_id: int, layout: CacheLayout) -> int:
        return self.binding.bind(pool_id, layout)

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

    def tensorize(self, plan: CachePlan, staging: Any) -> DeviceCache:
        groups = []
        for index, group in enumerate(plan.groups):
            rows = plan.rows
            width = max(
                int(group.block_indptr[row + 1] - group.block_indptr[row])
                for row in range(rows)
            )
            slabs = np.zeros((rows, width), dtype=np.uint32)
            slots = np.zeros_like(slabs)
            for row in range(rows):
                start, end = map(int, group.block_indptr[row : row + 2])
                count = end - start
                slabs[row, :count] = group.block_slab_ids[start:end]
                slots[row, :count] = group.block_slotnums[start:end]
            groups.append(
                DeviceGroup(
                    staging.device(f"cache.{index}.slabs", slabs, self.torch.int32),
                    staging.device(f"cache.{index}.slots", slots, self.torch.int32),
                    staging.device(
                        f"cache.{index}.first",
                        group.first_blocks,
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
    def graph_plan_inputs(cache: DeviceCache, plan: CachePlan) -> dict[str, object]:
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
                start, end = map(int, source.block_indptr[row : row + 2])
                count = end - start
                if count > shape[1]:
                    raise ValueError("Inkling cache table exceeds CUDA graph capacity")
                slabs[row, :count] = source.block_slab_ids[start:end]
                slots[row, :count] = source.block_slotnums[start:end]
            first = np.zeros(shape[0], dtype=np.uint32)
            first[: plan.rows] = source.first_blocks
            inputs[f"cache.{index}.slabs"] = slabs
            inputs[f"cache.{index}.slots"] = slots
            inputs[f"cache.{index}.first"] = first
        return inputs


__all__ = [
    "CacheArena",
    "CacheCapacityPlan",
    "CacheGeometry",
    "CachePlan",
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
    "build_cache_plan",
    "plan_cache_capacity",
]
