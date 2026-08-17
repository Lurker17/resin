"""Paged DeepSeek-V4 history geometry and TokenSpeed-style VRAM sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from resin import BatchPhase, CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.capacity import CacheCapacityPlan
from resin.backend.capacity import plan_cache_capacity as plan_capacity
from resin.backend.plans import BatchView, CacheGroupView
from resin.backend.resources import CachePoolBinding

if TYPE_CHECKING:
    from torch import Tensor

PAGE_TOKENS = 128
WINDOW_GROUP = 0
C4_KV_GROUP = 1
C4_STATE_GROUP = 2
C128_KV_GROUP = 3
C128_STATE_GROUP = 4
INDEX_STATE_GROUP = 5
GROUP_COUNT = 6
GRAPH_SCRATCH_PARENTS = GROUP_COUNT
_GROUP_BLOCK_TOKENS = (128, 128, 4, 128, 8, 4)


def _power_of_two_packing(largest: int, size: int) -> int:
    """Pack smaller fields without introducing large LCM padding."""

    return 1 << max(0, (largest // size).bit_length() - 1)


def _ints(name, values, *, positive=False) -> tuple[int, ...]:
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error
    minimum = 1 if positive else 0
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value in values
    ):
        raise ValueError(f"{name} contains an invalid integer")
    return values


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Offsets for all layers bundled into each 128-token cache block."""

    attention_layers: int
    window_tokens: int
    c4_state_tokens: int
    index_base_rows: int
    compressed_offsets: tuple[int | None, ...]
    index_offsets: tuple[int | None, ...]
    state_value_offsets: tuple[int | None, ...]
    state_score_offsets: tuple[int | None, ...]
    index_state_value_offsets: tuple[int | None, ...]
    index_state_score_offsets: tuple[int | None, ...]
    state_widths: tuple[int, ...]
    packings: tuple[int, ...]
    parent_rows: int

    @classmethod
    def build(
        cls,
        compress_ratios,
        *,
        target_layers: int,
        dspark_layers: int,
        window_size: int,
        decode_input_tokens: int = 1,
    ) -> CacheGeometry:
        ratios = _ints("compress_ratios", compress_ratios)
        if (
            isinstance(decode_input_tokens, bool)
            or not isinstance(decode_input_tokens, int)
            or decode_input_tokens <= 0
        ):
            raise ValueError("decode_input_tokens must be a positive integer")
        if len(ratios) < target_layers or any(
            ratio not in (0, 4, 128) for ratio in ratios[:target_layers]
        ):
            raise ValueError("DeepSeek cache supports compression ratios 0, 4, and 128")
        compressed_offsets: list[int | None] = []
        index_offsets: list[int | None] = []
        state_value_offsets: list[int | None] = []
        state_score_offsets: list[int | None] = []
        index_state_value_offsets: list[int | None] = []
        index_state_score_offsets: list[int | None] = []
        compressed_cursors = {4: 0, 128: 0}
        state_cursors = {4: 0, 128: 0}
        index_cursor = index_state_cursor = 0
        ratio_counts = {4: 0, 128: 0}
        for ratio in ratios[:target_layers]:
            if ratio:
                compressed_offsets.append(compressed_cursors[ratio])
                compressed_cursors[ratio] += PAGE_TOKENS // ratio
                width = 1024 if ratio == 4 else 512
                state_value_offsets.append(state_cursors[ratio])
                state_cursors[ratio] += (
                    _GROUP_BLOCK_TOKENS[
                        C4_STATE_GROUP if ratio == 4 else C128_STATE_GROUP
                    ]
                    * width
                )
                state_score_offsets.append(state_cursors[ratio])
                state_cursors[ratio] += (
                    _GROUP_BLOCK_TOKENS[
                        C4_STATE_GROUP if ratio == 4 else C128_STATE_GROUP
                    ]
                    * width
                )
                ratio_counts[ratio] += 1
            else:
                compressed_offsets.append(None)
                state_value_offsets.append(None)
                state_score_offsets.append(None)
            if ratio == 4:
                # Indexer keys share c4's history group after the 512D main KV.
                index_offsets.append(index_cursor)
                index_cursor += PAGE_TOKENS // ratio
                index_state_value_offsets.append(index_state_cursor)
                index_state_cursor += _GROUP_BLOCK_TOKENS[INDEX_STATE_GROUP] * 256
                index_state_score_offsets.append(index_state_cursor)
                index_state_cursor += _GROUP_BLOCK_TOKENS[INDEX_STATE_GROUP] * 256
            else:
                index_offsets.append(None)
                index_state_value_offsets.append(None)
                index_state_score_offsets.append(None)
        compressed_offsets.extend((None,) * dspark_layers)
        index_offsets.extend((None,) * dspark_layers)
        state_value_offsets.extend((None,) * dspark_layers)
        state_score_offsets.extend((None,) * dspark_layers)
        index_state_value_offsets.extend((None,) * dspark_layers)
        index_state_score_offsets.extend((None,) * dspark_layers)
        attention_layers = target_layers + dspark_layers
        c4_state_tokens = (
            math.ceil(
                max(8, 4 + decode_input_tokens) / _GROUP_BLOCK_TOKENS[C4_STATE_GROUP]
            )
            * _GROUP_BLOCK_TOKENS[C4_STATE_GROUP]
        )
        # Express every group in equivalent 512D BF16 rows.  The physical
        # parent is a union: at runtime it belongs to exactly one group, whose
        # smaller logical blocks are packed into the same bytes.  TokenSpeed's
        # power-of-two rule avoids pathological LCM padding while retaining
        # naturally aligned child strides.
        raw_rows = (
            attention_layers * PAGE_TOKENS,
            # c4 main 512D KV followed by 128D indexer keys.
            compressed_cursors[4] + math.ceil(index_cursor / 4),
            math.ceil(state_cursors[4] * 4 / (512 * 2)),
            compressed_cursors[128],
            math.ceil(state_cursors[128] * 4 / (512 * 2)),
            math.ceil(index_state_cursor * 4 / (512 * 2)),
        )
        if any(rows <= 0 for rows in raw_rows):
            raise ValueError("DeepSeek cache geometry has an empty cache group")
        largest = max(raw_rows)
        packings = tuple(_power_of_two_packing(largest, rows) for rows in raw_rows)
        alignment = math.lcm(*packings)
        parent_rows = (
            math.ceil(
                max(rows * packing for rows, packing in zip(raw_rows, packings))
                / alignment
            )
            * alignment
        )
        # Verification writes the anchor plus every draft before acceptance is
        # known. Keep the ordinary attention window and that uncommitted tail
        # resident together, even when their union straddles three pages.
        window_tokens = window_size + decode_input_tokens - 1
        return cls(
            attention_layers,
            window_tokens,
            c4_state_tokens,
            compressed_cursors[4],
            tuple(compressed_offsets),
            tuple(index_offsets),
            tuple(state_value_offsets),
            tuple(state_score_offsets),
            tuple(index_state_value_offsets),
            tuple(index_state_score_offsets),
            tuple(
                1024 if ratio == 4 else 512 if ratio == 128 else 0
                for ratio in ratios[:target_layers]
            )
            + (0,) * dspark_layers,
            packings,
            parent_rows,
        )

    def layout(self, parent_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity,
            (
                CacheGroupLayout(
                    PAGE_TOKENS,
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.window_tokens,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[C4_KV_GROUP]
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[C4_STATE_GROUP],
                    blocks_per_slab=self.packings[C4_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.c4_state_tokens,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[C128_KV_GROUP]
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[C128_STATE_GROUP],
                    blocks_per_slab=self.packings[C128_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=128,
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[INDEX_STATE_GROUP],
                    blocks_per_slab=self.packings[INDEX_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=12,
                ),
            ),
        )

    @property
    def bytes_per_parent(self) -> int:
        """BF16 bytes in one physical union parent on one TP rank."""

        return self.parent_rows * 512 * 2

def plan_cache_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    geometry: CacheGeometry,
    *,
    reserved_parent_count: int = 1,
) -> CacheCapacityPlan:
    """Apply TokenSpeed's post-load free-memory budget to Resin's parent pool."""

    return plan_capacity(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        geometry.bytes_per_parent,
        reserved_parent_count=reserved_parent_count,
    )


@dataclass(frozen=True, slots=True)
class GroupPlan:
    parent_ids: tuple[npt.NDArray[np.uint32], ...]
    child_slots: tuple[npt.NDArray[np.uint32], ...]
    first_blocks: npt.NDArray[np.uint32]


@dataclass(frozen=True, slots=True)
class CachePlan:
    groups: tuple[GroupPlan, ...]


@dataclass(frozen=True, slots=True)
class DeviceGroup:
    parent_ids: object
    child_slots: object
    first_blocks: object


@dataclass(frozen=True, slots=True)
class DeviceMapping:
    """Raw page-table geometry consumed directly by CUDA kernels."""

    parent_ids: object
    child_slots: object
    first_blocks: object
    block_tokens: int
    divisor: int
    parent_rows: int
    packing: int
    row_scale: int
    layer_offset: int


@dataclass(frozen=True, slots=True)
class DeviceCache:
    """Device page tables plus graph-safe logical-to-physical mapping."""

    torch: object
    geometry: CacheGeometry
    groups: tuple[DeviceGroup, ...]
    latents: object
    index_keys: object

    def row(self, index: int):
        """Return a one-sequence metadata view over the same physical pools."""

        return DeviceCache(
            self.torch,
            self.geometry,
            tuple(
                DeviceGroup(
                    group.parent_ids[index : index + 1],
                    group.child_slots[index : index + 1],
                    group.first_blocks[index : index + 1],
                )
                for group in self.groups
            ),
            self.latents,
            self.index_keys,
        )

    def _rows(
        self,
        group: int,
        positions,
        *,
        block_tokens: int,
        divisor: int,
        layer_offset: int,
        row_scale: int = 1,
    ):
        if positions.is_cuda:
            from resin_models_cuda.deepseek_v4.cutile import paged_cache_rows

            mapping = self.mapping(
                group,
                block_tokens=block_tokens,
                divisor=divisor,
                layer_offset=layer_offset,
                row_scale=row_scale,
            )
            return paged_cache_rows(
                mapping.parent_ids,
                mapping.child_slots,
                mapping.first_blocks,
                positions,
                block_tokens=mapping.block_tokens,
                divisor=mapping.divisor,
                parent_rows=mapping.parent_rows,
                packing=mapping.packing,
                row_scale=mapping.row_scale,
                layer_offset=mapping.layer_offset,
            )
        table = self.groups[group]
        logical_blocks = positions.div(block_tokens, rounding_mode="floor")
        columns = logical_blocks - table.first_blocks.view(
            (-1,) + (1,) * (positions.ndim - 1)
        )
        flat = columns.flatten(1).clamp(0, table.parent_ids.shape[1] - 1)
        parents = self.torch.gather(table.parent_ids, 1, flat).view_as(columns)
        children = self.torch.gather(table.child_slots, 1, flat).view_as(columns)
        packing = self.geometry.packings[group]
        child_rows = self.geometry.parent_rows // packing
        offsets = positions.remainder(block_tokens).div(divisor, rounding_mode="floor")
        physical = (
            parents * self.geometry.parent_rows + children * child_rows
        ) * row_scale
        physical = physical + layer_offset + offsets
        in_table = (columns >= 0) & (columns < table.parent_ids.shape[1])
        # Parent/page zero is a real, permanently zeroed graph-padding target.
        # It maps to physical row zero; out-of-table logical positions remain
        # sparse-index sentinels.
        physical = self.torch.where(parents > 0, physical, 0)
        return self.torch.where(in_table, physical, -1).to(self.torch.int32)

    def mapping(
        self,
        group: int,
        *,
        block_tokens: int,
        divisor: int,
        layer_offset: int,
        row_scale: int = 1,
    ) -> DeviceMapping:
        """Describe a mapping for fusion into an existing compute kernel."""

        table = self.groups[group]
        return DeviceMapping(
            table.parent_ids,
            table.child_slots,
            table.first_blocks,
            block_tokens,
            divisor,
            self.geometry.parent_rows,
            self.geometry.packings[group],
            row_scale,
            layer_offset,
        )

    def window_mapping(self, layer: int) -> DeviceMapping:
        return self.mapping(
            WINDOW_GROUP,
            block_tokens=PAGE_TOKENS,
            divisor=1,
            layer_offset=layer * PAGE_TOKENS,
        )

    def compressed_mapping(self, layer: int, ratio: int) -> DeviceMapping:
        offset = self.geometry.compressed_offsets[layer]
        if offset is None:
            raise ValueError("layer has no compressed cache")
        return self.mapping(
            C4_KV_GROUP if ratio == 4 else C128_KV_GROUP,
            block_tokens=PAGE_TOKENS,
            divisor=ratio,
            layer_offset=offset,
        )

    def index_mapping(self, layer: int) -> DeviceMapping:
        offset = self.geometry.index_offsets[layer]
        if offset is None:
            raise ValueError("layer has no index cache")
        return self.mapping(
            C4_KV_GROUP,
            block_tokens=PAGE_TOKENS,
            divisor=4,
            layer_offset=self.geometry.index_base_rows * 4 + offset,
            row_scale=4,
        )

    def state_mapping(self, group: int) -> DeviceMapping:
        if group not in (C4_STATE_GROUP, C128_STATE_GROUP, INDEX_STATE_GROUP):
            raise ValueError("requested group is not recurrent state")
        return self.mapping(
            group,
            block_tokens=_GROUP_BLOCK_TOKENS[group],
            divisor=1,
            layer_offset=0,
        )

    def window_rows(self, layer: int, positions):
        return self._rows(
            WINDOW_GROUP,
            positions,
            block_tokens=PAGE_TOKENS,
            divisor=1,
            layer_offset=layer * PAGE_TOKENS,
        )

    def compressed_rows(self, layer: int, ratio: int, positions):
        offset = self.geometry.compressed_offsets[layer]
        if offset is None:
            raise ValueError("layer has no compressed cache")
        return self._rows(
            C4_KV_GROUP if ratio == 4 else C128_KV_GROUP,
            positions,
            block_tokens=PAGE_TOKENS,
            divisor=ratio,
            layer_offset=offset,
        )

    def index_rows(self, layer: int, positions):
        offset = self.geometry.index_offsets[layer]
        if offset is None:
            raise ValueError("layer has no index cache")
        # Convert 512D parent rows into the index pool's 128D row numbering.
        return self._rows(
            C4_KV_GROUP,
            positions,
            block_tokens=PAGE_TOKENS,
            divisor=4,
            layer_offset=self.geometry.index_base_rows * 4 + offset,
            row_scale=4,
        )

    def state_locations(self, group: int, positions):
        """Return physical child IDs and within-child token rows."""

        block_tokens = _GROUP_BLOCK_TOKENS[group]
        table = self.groups[group]
        if positions.is_cuda:
            from resin_models_cuda.deepseek_v4.cutile import paged_state_locations

            return paged_state_locations(
                table.parent_ids,
                table.child_slots,
                table.first_blocks,
                positions,
                block_tokens=block_tokens,
                packing=self.geometry.packings[group],
            )
        logical_blocks = positions.div(block_tokens, rounding_mode="floor")
        columns = logical_blocks - table.first_blocks.view(
            (-1,) + (1,) * (positions.ndim - 1)
        )
        flat = columns.flatten(1).clamp(0, table.parent_ids.shape[1] - 1)
        parents = self.torch.gather(table.parent_ids, 1, flat).view_as(columns)
        children = self.torch.gather(table.child_slots, 1, flat).view_as(columns)
        packing = self.geometry.packings[group]
        physical_children = self.torch.where(
            parents > 0, parents * packing + children, 0
        )
        valid = (columns >= 0) & (columns < table.parent_ids.shape[1])
        physical_children = self.torch.where(valid, physical_children, -1)
        offsets = self.torch.where(valid, positions.remainder(block_tokens), 0)
        return physical_children.to(self.torch.int32), offsets.to(self.torch.int32)


def build_cache_plan(plan, layout: CacheLayout, *, decode_lookahead: int) -> CachePlan:
    """Validate the six Engine-issued page tables before a CUDA launch."""

    batch = BatchView.from_plan(plan)
    rows = batch.num_sequences
    indptr = batch.token_indptr
    positions = batch.positions
    contexts = batch.context_lengths
    native = batch.cache_groups
    if len(native) != GROUP_COUNT:
        raise ValueError(f"DeepSeek requires exactly {GROUP_COUNT} cache groups")

    groups = []
    for group_index, (source, descriptor) in enumerate(zip(native, layout.groups)):
        block_tokens = descriptor.block_tokens
        packing = descriptor.blocks_per_slab
        group = CacheGroupView.from_plan(
            source,
            batch,
            group_index=group_index,
            blocks_per_slab=packing,
            parent_capacity=layout.parent_capacity,
            block_tokens=block_tokens,
        )
        first = group.block_table_start_indices
        append_parents = group.append_slab_ids
        append_slots = group.append_slotnums
        append_offsets = group.append_block_offsets
        parent_rows = []
        slot_rows = []
        for row in range(rows):
            start, end = indptr[row : row + 2]
            start = int(start)
            end = int(end)
            context = int(contexts[row])
            leased = context + end - start
            if batch.phase == BatchPhase.DECODE:
                leased += decode_lookahead
            last = (leased - 1) // block_tokens
            if descriptor.retention is CacheRetention.WINDOW:
                expected_first = max(
                    0, last + 1 - descriptor.window_resident_block_count
                )
            else:
                expected_first = 0
            row_parents = group.block_slabs(row)
            row_slots = group.block_slots(row)
            if (
                int(first[row]) != expected_first
                or len(row_parents) != last - expected_first + 1
            ):
                raise ValueError("DeepSeek cache row has the wrong logical extent")
            row_positions = positions[start:end]
            logical = row_positions // block_tokens
            retained = logical >= expected_first
            # Sliding state/history groups intentionally ignore the old part of
            # a large prefill; only their retained tail is materialized.
            relative = logical[retained] - expected_first
            if not np.array_equal(
                append_parents[start:end][retained], row_parents[relative]
            ) or not np.array_equal(
                append_slots[start:end][retained], row_slots[relative]
            ) or not np.array_equal(
                append_offsets[start:end][retained],
                row_positions[retained] % block_tokens,
            ):
                raise ValueError("DeepSeek cache append target is not row-owned")
            parent_rows.append(row_parents)
            slot_rows.append(row_slots)
        groups.append(GroupPlan(tuple(parent_rows), tuple(slot_rows), first))
    return CachePlan(tuple(groups))


class CacheArena:
    """One allocation shared by six mutually exclusive cache groups."""

    def __init__(
        self, torch, device, layout, geometry: CacheGeometry, *, dtype
    ) -> None:
        self.torch = torch
        self.binding = CachePoolBinding(layout, resource="DeepSeek-V4 KV")
        self.geometry = geometry
        self.storage = torch.empty(
            (
                layout.parent_capacity + 1 + GRAPH_SCRATCH_PARENTS,
                geometry.parent_rows,
                512,
            ),
            dtype=dtype,
            device=device,
        )
        self.latents = self.storage.view(-1, PAGE_TOKENS, 512)
        self.index_keys = self.storage.view(-1, PAGE_TOKENS, 128)
        self.reset_scratch()

    def bind(self, pool_id: int, layout) -> int:
        return self.binding.bind(pool_id, layout)

    def reset_scratch(self) -> None:
        self.storage[0].zero_()
        self.storage[-GRAPH_SCRATCH_PARENTS:].zero_()

    def state_tensors(
        self, layer: int, *, indexer: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return strided FP32 value/score views for one recurrent field."""

        if indexer:
            group = INDEX_STATE_GROUP
            value_offset = self.geometry.index_state_value_offsets[layer]
            score_offset = self.geometry.index_state_score_offsets[layer]
            width = 256
        else:
            width = self.geometry.state_widths[layer]
            if width == 1024:
                group = C4_STATE_GROUP
            elif width == 512:
                group = C128_STATE_GROUP
            else:
                raise ValueError("layer has no recurrent compressor state")
            value_offset = self.geometry.state_value_offsets[layer]
            score_offset = self.geometry.state_score_offsets[layer]
        if value_offset is None or score_offset is None:
            raise ValueError("layer has no requested recurrent state field")
        packing = self.geometry.packings[group]
        block_tokens = _GROUP_BLOCK_TOKENS[group]
        child_floats = self.geometry.parent_rows * 512 * 2 // packing // 4
        child_count = self.storage.shape[0] * packing
        flat = self.storage.view(self.torch.float32).flatten()

        def field(offset: int) -> Tensor:
            return self.torch.as_strided(
                flat,
                (child_count, block_tokens, width),
                (child_floats, width, 1),
                storage_offset=offset,
            )

        return field(value_offset), field(score_offset)

    def initialize_state_sentinels(self) -> None:
        """Restore the null parent after graph capture writes."""

        # Every group aliases parent zero, so it must remain bytewise zero.
        # Invalid recurrent history uses page -1 and cuTile padding values;
        # storing -inf into a score view here would corrupt null KV aliases.
        self.reset_scratch()

    def tensorize(self, plan: CachePlan, staging):
        torch = self.torch
        groups = []
        for index, group in enumerate(plan.groups):
            width = max(map(len, group.parent_ids))
            # Keep the dynamic width in one cuTile divisibility class. Merely
            # rounding to 16 is insufficient: 128, 192, and 256 produce three
            # progressively stronger JIT assumptions. Every value here is 16
            # modulo 32, so its exact power-of-two divisor is always 16.
            width = (width + 15) // 32 * 32 + 16
            parents = np.zeros((len(group.parent_ids), width), dtype=np.uint32)
            slots = np.zeros_like(parents)
            for row_index, (row_parents, row_slots) in enumerate(
                zip(group.parent_ids, group.child_slots, strict=True)
            ):
                parents[row_index, : len(row_parents)] = row_parents
                slots[row_index, : len(row_slots)] = row_slots
            groups.append(
                DeviceGroup(
                    staging.device(
                        f"cache.{index}.parent_ids", parents, torch.int32
                    ),
                    staging.device(
                        f"cache.{index}.child_slots", slots, torch.int32
                    ),
                    staging.device(
                        f"cache.{index}.first_blocks",
                        group.first_blocks,
                        torch.int32,
                    ),
                )
            )
        return DeviceCache(
            torch,
            self.geometry,
            tuple(groups),
            self.latents,
            self.index_keys,
        )

    def _graph_widths(self, sequence_capacity: int) -> tuple[int, ...]:
        return tuple(
            descriptor.window_resident_block_count
            if descriptor.retention is CacheRetention.WINDOW
            else math.ceil(sequence_capacity / descriptor.block_tokens)
            for descriptor in self.binding.layout.groups
        )

    def graph_cache(self, batch_size: int, sequence_capacity: int):
        """Allocate fixed page-table tensors owned by one CUDA graph shape."""

        groups = []
        for width in self._graph_widths(sequence_capacity):
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
            tuple(groups),
            self.latents,
            self.index_keys,
        )

    @staticmethod
    def graph_device_inputs(cache: DeviceCache) -> dict[str, object]:
        inputs = {}
        for index, group in enumerate(cache.groups):
            inputs[f"cache.{index}.parent_ids"] = group.parent_ids
            inputs[f"cache.{index}.child_slots"] = group.child_slots
            inputs[f"cache.{index}.first_blocks"] = group.first_blocks
        return inputs

    def graph_shape_capture_inputs(
        self, batch_size: int, sequence_capacity: int
    ) -> dict[str, object]:
        inputs = {}
        for index, width in enumerate(self._graph_widths(sequence_capacity)):
            parent = self.binding.layout.parent_capacity + index + 1
            inputs[f"cache.{index}.parent_ids"] = (parent,) * (
                batch_size * width
            )
            inputs[f"cache.{index}.child_slots"] = (0,) * (batch_size * width)
            inputs[f"cache.{index}.first_blocks"] = (0,) * batch_size
        return inputs

    @staticmethod
    def graph_plan_inputs(cache: DeviceCache, plan: CachePlan) -> dict[str, object]:
        inputs = {}
        for index, (target, source) in enumerate(zip(cache.groups, plan.groups)):
            width = target.parent_ids.shape[1]
            padding_rows = target.parent_ids.shape[0] - len(source.parent_ids)
            if padding_rows < 0:
                raise ValueError("DeepSeek cache batch exceeds CUDA graph capacity")
            if any(len(row) > width for row in source.parent_ids):
                raise ValueError("DeepSeek cache table exceeds CUDA graph capacity")
            parents = np.zeros(target.parent_ids.shape, dtype=np.uint32)
            slots = np.zeros_like(parents)
            for row_index, (row_parents, row_slots) in enumerate(
                zip(source.parent_ids, source.child_slots, strict=True)
            ):
                parents[row_index, : len(row_parents)] = row_parents
                slots[row_index, : len(row_slots)] = row_slots
            inputs[f"cache.{index}.parent_ids"] = parents
            inputs[f"cache.{index}.child_slots"] = slots
            inputs[f"cache.{index}.first_blocks"] = np.pad(
                source.first_blocks, (0, padding_rows)
            )
        return inputs


__all__ = [
    "C4_KV_GROUP",
    "C4_STATE_GROUP",
    "C128_KV_GROUP",
    "C128_STATE_GROUP",
    "GROUP_COUNT",
    "INDEX_STATE_GROUP",
    "PAGE_TOKENS",
    "WINDOW_GROUP",
    "CacheArena",
    "CacheCapacityPlan",
    "CacheGeometry",
    "CachePlan",
    "DeviceCache",
    "DeviceMapping",
    "DeviceGroup",
    "GroupPlan",
    "build_cache_plan",
    "plan_cache_capacity",
]
