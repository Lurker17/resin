"""Fungible paged target and EAGLE3 MLA cache for the Kimi K2.5 family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from resin import BatchPhase, CacheGroupLayout, CacheLayout
from resin.backend.resources import CachePoolBinding

if TYPE_CHECKING:
    from torch import Tensor

PAGE_TOKENS = 64
TARGET_GROUP = 0
DRAFT_GROUP = 1
AUX_GROUP = 2
GROUP_COUNT = 3
GRAPH_SCRATCH_PARENTS = GROUP_COUNT
PAGE_TABLE_ALIGNMENT = 128 // PAGE_TOKENS


def _aligned_page_table_width(width: int) -> int:
    return math.ceil(width / PAGE_TABLE_ALIGNMENT) * PAGE_TABLE_ALIGNMENT


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Byte-compatible target-FP8 and draft-BF16 MLA page geometry."""

    target_layers: int
    latent_width: int = 576
    hidden_size: int = 7168
    parent_units: int = 0
    draft_blocks_per_parent: int = 0
    aux_blocks_per_parent: int = 0

    @classmethod
    def build(
        cls,
        target_layers: int,
        *,
        latent_width: int = 576,
        hidden_size: int = 7168,
    ) -> CacheGeometry:
        if target_layers <= 0 or latent_width <= 0 or hidden_size <= 0:
            raise ValueError("Kimi cache geometry must be positive")
        # One unit is one FP8 target layer. A BF16 draft block consumes two
        # units. Round the parent to a power of two so both physical views are
        # contiguous and page IDs remain simple integer arithmetic.
        parent_units = 1 << (target_layers - 1).bit_length()
        draft_blocks = parent_units // 2
        aux_ratio = parent_units * latent_width // (2 * hidden_size)
        if aux_ratio <= 0:
            raise ValueError("Kimi cache parent cannot hold an EAGLE state block")
        aux_blocks = 1 << (aux_ratio.bit_length() - 1)
        return cls(
            target_layers,
            latent_width,
            hidden_size,
            parent_units,
            draft_blocks,
            aux_blocks,
        )

    @property
    def bytes_per_parent(self) -> int:
        return self.parent_units * PAGE_TOKENS * self.latent_width

    def layout(self, parent_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity,
            (
                CacheGroupLayout(PAGE_TOKENS),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.draft_blocks_per_parent
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.aux_blocks_per_parent
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class CacheCapacityPlan:
    """Post-load Kimi cache budget on one TP rank."""

    total_memory_bytes: int
    free_memory_bytes: int
    gpu_memory_utilization: float
    protected_free_bytes: int
    cache_budget_bytes: int
    bytes_per_parent: int
    reserved_parent_count: int
    parent_capacity: int
    allocation_bytes: int


def plan_cache_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    geometry: CacheGeometry,
    *,
    reserved_parent_count: int = 1,
) -> CacheCapacityPlan:
    """Fill post-load VRAM up to the requested memory utilization."""

    if total_memory_bytes <= 0 or not 0 < free_memory_bytes <= total_memory_bytes:
        raise ValueError("CUDA memory totals must be positive and internally valid")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if reserved_parent_count <= 0:
        raise ValueError("reserved_parent_count must include the null parent")
    protected = math.ceil(total_memory_bytes * (1.0 - gpu_memory_utilization))
    budget = free_memory_bytes - protected
    parent_capacity = budget // geometry.bytes_per_parent - reserved_parent_count
    if parent_capacity <= 0:
        raise MemoryError("Kimi cache budget leaves no usable parent pages")
    allocation = (parent_capacity + reserved_parent_count) * geometry.bytes_per_parent
    return CacheCapacityPlan(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        protected,
        budget,
        geometry.bytes_per_parent,
        reserved_parent_count,
        parent_capacity,
        allocation,
    )


@dataclass(frozen=True, slots=True)
class GroupPlan:
    parent_ids: tuple[npt.NDArray[np.uint32], ...]
    child_slots: tuple[npt.NDArray[np.uint32], ...]
    first_blocks: npt.NDArray[np.uint32]
    append_slab_ids: npt.NDArray[np.uint32]
    append_slotnums: npt.NDArray[np.uint32]
    append_offsets: npt.NDArray[np.uint32]


@dataclass(frozen=True, slots=True)
class CachePlan:
    groups: tuple[GroupPlan, GroupPlan, GroupPlan]


@dataclass(frozen=True, slots=True)
class DeviceGroup:
    parent_ids: Tensor
    child_slots: Tensor
    first_blocks: Tensor
    append_slab_ids: Tensor
    append_slotnums: Tensor
    append_offsets: Tensor


@dataclass(frozen=True, slots=True)
class DeviceCache:
    """Graph-stable page tables and aliased target/draft cache tensors."""

    torch: object
    geometry: CacheGeometry
    groups: tuple[DeviceGroup, DeviceGroup, DeviceGroup]
    target: Tensor
    draft: Tensor
    aux: Tensor

    def row(self, index: int, token_start: int, token_end: int) -> DeviceCache:
        """Return one sequence and its scheduled token slice."""

        groups = tuple(
            DeviceGroup(
                group.parent_ids[index : index + 1],
                group.child_slots[index : index + 1],
                group.first_blocks[index : index + 1],
                group.append_slab_ids[token_start:token_end],
                group.append_slotnums[token_start:token_end],
                group.append_offsets[token_start:token_end],
            )
            for group in self.groups
        )
        return DeviceCache(
            self.torch,
            self.geometry,
            groups,
            self.target,
            self.draft,
            self.aux,
        )

    def target_pages(self, layer: int) -> Tensor:
        """Return physical FP8 page IDs for one target layer."""

        if not 0 <= layer < self.geometry.target_layers:
            raise ValueError("target layer is outside the Kimi cache")
        parents = self.groups[TARGET_GROUP].parent_ids
        return parents * self.geometry.parent_units + layer

    def draft_pages(self) -> Tensor:
        """Return physical BF16 page IDs for the EAGLE3 layer."""

        group = self.groups[DRAFT_GROUP]
        return (
            group.parent_ids * self.geometry.draft_blocks_per_parent + group.child_slots
        )

    def target_append_rows(self, layer: int) -> Tensor:
        group = self.groups[TARGET_GROUP]
        pages = group.append_slab_ids * self.geometry.parent_units + layer
        return pages * PAGE_TOKENS + group.append_offsets

    def draft_append_rows(self) -> Tensor:
        group = self.groups[DRAFT_GROUP]
        pages = (
            group.append_slab_ids * self.geometry.draft_blocks_per_parent
            + group.append_slotnums
        )
        return pages * PAGE_TOKENS + group.append_offsets

    def _locations(
        self, group_index: int, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        group = self.groups[group_index]
        logical_blocks = positions.div(PAGE_TOKENS, rounding_mode="floor")
        first = group.first_blocks.view((-1,) + (1,) * (positions.ndim - 1))
        columns = logical_blocks - first
        flat = columns.flatten(1).clamp(0, group.parent_ids.shape[1] - 1).long()
        parents = self.torch.gather(group.parent_ids, 1, flat).view_as(columns)
        slots = self.torch.gather(group.child_slots, 1, flat).view_as(columns)
        return parents, slots, positions.remainder(PAGE_TOKENS)

    def target_rows(self, layer: int, positions: Tensor) -> Tensor:
        """Map logical target positions to flat layer-specific cache rows."""

        parents, _, offsets = self._locations(TARGET_GROUP, positions)
        pages = parents * self.geometry.parent_units + layer
        return pages * PAGE_TOKENS + offsets

    def draft_rows(self, positions: Tensor) -> Tensor:
        """Map speculative positions to flat BF16 draft-cache rows."""

        parents, slots, offsets = self._locations(DRAFT_GROUP, positions)
        pages = parents * self.geometry.draft_blocks_per_parent + slots
        return pages * PAGE_TOKENS + offsets

    def store_aux(self, hidden: Tensor, positions: Tensor) -> None:
        """Store target-projected EAGLE state for every scheduled token."""

        parents, slots, offsets = self._locations(AUX_GROUP, positions)
        self.aux[parents.long(), slots.long(), offsets.long()] = hidden.reshape(
            *parents.shape, self.geometry.hidden_size
        )

    def aux_at(self, positions: Tensor) -> Tensor:
        """Load target-projected EAGLE state at one position per sequence."""

        parents, slots, offsets = self._locations(AUX_GROUP, positions[:, None])
        return self.aux[parents.long(), slots.long(), offsets.long()][:, 0]


def build_cache_plan(plan, layout: CacheLayout, *, decode_lookahead: int) -> CachePlan:
    """Validate Engine-owned full-history pages before a Kimi launch."""

    rows = plan.num_sequences
    token_indptr = np.asarray(plan.token_indptr, dtype=np.uint32)
    positions = np.asarray(plan.positions, dtype=np.uint32)
    contexts = np.asarray(plan.context_lengths, dtype=np.uint32)
    if (
        rows == 0
        or len(token_indptr) != rows + 1
        or token_indptr[0] != 0
        or token_indptr[-1] != len(positions)
        or np.any(token_indptr[1:] <= token_indptr[:-1])
        or len(contexts) != rows
    ):
        raise ValueError("Kimi sequence metadata is malformed")
    native = tuple(plan.cache_groups)
    if len(native) != GROUP_COUNT:
        raise ValueError(f"Kimi requires exactly {GROUP_COUNT} cache groups")

    groups = []
    for group_index, (source, descriptor) in enumerate(
        zip(native, layout.groups, strict=True)
    ):
        block_indptr = np.asarray(source.block_indptr, dtype=np.uint32)
        parent_ids = np.asarray(source.block_slab_ids, dtype=np.uint32)
        child_slots = np.asarray(source.block_slotnums, dtype=np.uint32)
        first = np.asarray(source.block_table_start_indices, dtype=np.uint32)
        append_parents = np.asarray(source.append_slab_ids, dtype=np.uint32)
        append_slots = np.asarray(source.append_slotnums, dtype=np.uint32)
        append_offsets = np.asarray(source.append_block_offsets, dtype=np.uint32)
        packing = descriptor.blocks_per_slab
        if (
            source.group_index != group_index
            or source.blocks_per_slab != packing
            or len(block_indptr) != rows + 1
            or block_indptr[0] != 0
            or block_indptr[-1] != len(parent_ids)
            or len(child_slots) != len(parent_ids)
            or len(first) != rows
            or len(append_parents) != len(positions)
            or len(append_slots) != len(positions)
            or len(append_offsets) != len(positions)
            or np.any(parent_ids == 0)
            or np.any(append_parents == 0)
            or np.any(parent_ids > layout.parent_capacity)
            or np.any(append_parents > layout.parent_capacity)
            or np.any(child_slots >= packing)
            or np.any(append_slots >= packing)
            or np.any(append_offsets >= PAGE_TOKENS)
        ):
            raise ValueError(f"Kimi cache group {group_index} is malformed")
        parent_rows = []
        slot_rows = []
        for row in range(rows):
            token_start, token_end = map(int, token_indptr[row : row + 2])
            context = int(contexts[row])
            leased = context + token_end - token_start
            if plan.phase == BatchPhase.DECODE:
                leased += decode_lookahead
            last_block = (leased - 1) // PAGE_TOKENS
            page_start, page_end = map(int, block_indptr[row : row + 2])
            row_parents = parent_ids[page_start:page_end]
            row_slots = child_slots[page_start:page_end]
            if int(first[row]) != 0 or len(row_parents) != last_block + 1:
                raise ValueError("Kimi cache row has the wrong logical extent")
            row_positions = positions[token_start:token_end]
            logical = row_positions // PAGE_TOKENS
            if (
                not np.array_equal(
                    append_parents[token_start:token_end], row_parents[logical]
                )
                or not np.array_equal(
                    append_slots[token_start:token_end], row_slots[logical]
                )
                or not np.array_equal(
                    append_offsets[token_start:token_end],
                    row_positions % PAGE_TOKENS,
                )
            ):
                raise ValueError("Kimi cache append target is not row-owned")
            parent_rows.append(row_parents)
            slot_rows.append(row_slots)
        groups.append(
            GroupPlan(
                tuple(parent_rows),
                tuple(slot_rows),
                first,
                append_parents,
                append_slots,
                append_offsets,
            )
        )
    return CachePlan((groups[0], groups[1], groups[2]))


class CacheArena:
    """One byte allocation with contiguous FP8 target and BF16 draft views."""

    def __init__(self, torch, device, layout: CacheLayout, geometry: CacheGeometry):
        self.torch = torch
        self.binding = CachePoolBinding(layout, resource="Kimi K2.5-family MLA KV")
        self.geometry = geometry
        physical_parents = layout.parent_capacity + 1 + GRAPH_SCRATCH_PARENTS
        self.storage = torch.empty(
            (physical_parents, geometry.bytes_per_parent),
            dtype=torch.uint8,
            device=device,
        )
        self.target = self.storage.view(torch.float8_e4m3fn).reshape(
            physical_parents * geometry.parent_units,
            PAGE_TOKENS,
            geometry.latent_width,
        )
        self.draft = self.storage.view(torch.bfloat16).reshape(
            physical_parents * geometry.draft_blocks_per_parent,
            PAGE_TOKENS,
            geometry.latent_width,
        )
        parent_elements = geometry.bytes_per_parent // 2
        self.aux = torch.as_strided(
            self.storage.view(torch.bfloat16).flatten(),
            (
                physical_parents,
                geometry.aux_blocks_per_parent,
                PAGE_TOKENS,
                geometry.hidden_size,
            ),
            (
                parent_elements,
                PAGE_TOKENS * geometry.hidden_size,
                geometry.hidden_size,
                1,
            ),
        )
        self.reset_scratch()

    def bind(self, pool_id: int, layout: CacheLayout) -> int:
        return self.binding.bind(pool_id, layout)

    def reset_scratch(self) -> None:
        self.storage[0].zero_()
        self.storage[-GRAPH_SCRATCH_PARENTS:].zero_()

    def prepare_step(
        self, plan: CachePlan, context_lengths: npt.NDArray[np.integer]
    ) -> None:
        """Clear every leased page beyond each sequence's resident context."""

        new_blocks = []
        for group in plan.groups:
            addresses = []
            for parents, slots, context in zip(
                group.parent_ids,
                group.child_slots,
                context_lengths,
                strict=True,
            ):
                first_new = (int(context) + PAGE_TOKENS - 1) // PAGE_TOKENS
                if first_new < len(parents):
                    addresses.append(
                        np.stack((parents[first_new:], slots[first_new:]), axis=1)
                    )
            if not addresses:
                new_blocks.append(np.empty((0, 2), dtype=np.uint32))
            else:
                new_blocks.append(np.unique(np.concatenate(addresses), axis=0))

        target_blocks, draft_blocks, aux_blocks = new_blocks
        if len(target_blocks):
            parents = self.torch.as_tensor(
                target_blocks[:, 0].astype(np.int64),
                dtype=self.torch.long,
                device=self.storage.device,
            )
            self.storage.index_fill_(0, parents, 0)
        if len(draft_blocks):
            pages = (
                draft_blocks[:, 0].astype(np.int64)
                * self.geometry.draft_blocks_per_parent
                + draft_blocks[:, 1]
            )
            page_ids = self.torch.as_tensor(
                pages, dtype=self.torch.long, device=self.storage.device
            )
            self.draft.index_fill_(0, page_ids, 0)
        if len(aux_blocks):
            parents = self.torch.as_tensor(
                aux_blocks[:, 0].astype(np.int64),
                dtype=self.torch.long,
                device=self.storage.device,
            )
            slots = self.torch.as_tensor(
                aux_blocks[:, 1].astype(np.int64),
                dtype=self.torch.long,
                device=self.storage.device,
            )
            self.aux[parents, slots] = 0

    def tensorize(self, plan: CachePlan, staging) -> DeviceCache:
        groups = []
        for index, group in enumerate(plan.groups):
            width = _aligned_page_table_width(max(map(len, group.parent_ids)))
            parents = np.zeros((len(group.parent_ids), width), dtype=np.uint32)
            slots = np.zeros_like(parents)
            for row, (row_parents, row_slots) in enumerate(
                zip(group.parent_ids, group.child_slots, strict=True)
            ):
                parents[row, : len(row_parents)] = row_parents
                slots[row, : len(row_slots)] = row_slots
            groups.append(
                DeviceGroup(
                    staging.device(f"cache.{index}.parents", parents, self.torch.int32),
                    staging.device(f"cache.{index}.slots", slots, self.torch.int32),
                    staging.device(
                        f"cache.{index}.first", group.first_blocks, self.torch.int32
                    ),
                    staging.device(
                        f"cache.{index}.append_parents",
                        group.append_slab_ids,
                        self.torch.int32,
                    ),
                    staging.device(
                        f"cache.{index}.append_slots",
                        group.append_slotnums,
                        self.torch.int32,
                    ),
                    staging.device(
                        f"cache.{index}.append_offsets",
                        group.append_offsets,
                        self.torch.int32,
                    ),
                )
            )
        return DeviceCache(
            self.torch,
            self.geometry,
            (groups[0], groups[1], groups[2]),
            self.target,
            self.draft,
            self.aux,
        )

    def _graph_width(self, sequence_capacity: int) -> int:
        return _aligned_page_table_width(math.ceil(sequence_capacity / PAGE_TOKENS))

    def graph_cache(self, batch_size: int, sequence_capacity: int) -> DeviceCache:
        """Allocate fixed page-table tensors owned by one CUDA graph."""

        width = self._graph_width(sequence_capacity)
        groups = []
        for _ in range(GROUP_COUNT):
            groups.append(
                DeviceGroup(
                    self.torch.empty(
                        (batch_size, width),
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                    self.torch.empty(
                        (batch_size, width),
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                    self.torch.zeros(
                        batch_size,
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                    self.torch.empty(
                        batch_size,
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                    self.torch.empty(
                        batch_size,
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                    self.torch.empty(
                        batch_size,
                        dtype=self.torch.int32,
                        device=self.storage.device,
                    ),
                )
            )
        return DeviceCache(
            self.torch,
            self.geometry,
            (groups[0], groups[1], groups[2]),
            self.target,
            self.draft,
            self.aux,
        )

    @staticmethod
    def graph_device_inputs(cache: DeviceCache) -> dict[str, object]:
        inputs = {}
        for index, group in enumerate(cache.groups):
            inputs[f"cache.{index}.parents"] = group.parent_ids
            inputs[f"cache.{index}.slots"] = group.child_slots
        return inputs

    def graph_shape_capture_inputs(
        self, batch_size: int, sequence_capacity: int
    ) -> dict[str, object]:
        width = self._graph_width(sequence_capacity)
        inputs = {}
        for index, descriptor in enumerate(self.binding.layout.groups):
            parent = self.binding.layout.parent_capacity + index + 1
            inputs[f"cache.{index}.parents"] = (parent,) * (batch_size * width)
            inputs[f"cache.{index}.slots"] = (0,) * (batch_size * width)
            if descriptor.blocks_per_slab < 1:
                raise RuntimeError("Kimi graph cache has invalid page packing")
        return inputs

    @staticmethod
    def graph_plan_inputs(cache: DeviceCache, plan: CachePlan) -> dict[str, object]:
        inputs = {}
        for index, (target, source) in enumerate(zip(cache.groups, plan.groups)):
            width = target.parent_ids.shape[1]
            parents = np.zeros(target.parent_ids.shape, dtype=np.uint32)
            slots = np.zeros_like(parents)
            if len(source.parent_ids) > target.parent_ids.shape[0]:
                raise ValueError("Kimi cache batch exceeds CUDA graph capacity")
            for row, (row_parents, row_slots) in enumerate(
                zip(source.parent_ids, source.child_slots, strict=True)
            ):
                if len(row_parents) > width:
                    raise ValueError("Kimi cache table exceeds CUDA graph capacity")
                parents[row, : len(row_parents)] = row_parents
                slots[row, : len(row_slots)] = row_slots
            inputs[f"cache.{index}.parents"] = parents
            inputs[f"cache.{index}.slots"] = slots
        return inputs


__all__ = [
    "AUX_GROUP",
    "DRAFT_GROUP",
    "GRAPH_SCRATCH_PARENTS",
    "GROUP_COUNT",
    "PAGE_TOKENS",
    "TARGET_GROUP",
    "CacheArena",
    "CacheCapacityPlan",
    "CacheGeometry",
    "CachePlan",
    "DeviceCache",
    "DeviceGroup",
    "GroupPlan",
    "build_cache_plan",
    "plan_cache_capacity",
]
