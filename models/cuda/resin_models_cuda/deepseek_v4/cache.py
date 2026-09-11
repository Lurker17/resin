"""Paged DeepSeek-V4 history geometry and TokenSpeed-style VRAM sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from resin import CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.capacity import CacheCapacityPlan
from resin.backend.capacity import plan_cache_capacity as plan_capacity
from resin.backend.cuda.slab import (
    PARENT_PAGE_NAMES,
    GraphPageTableArena,
    slice_page_groups,
    stage_page_groups,
)
from resin.backend.cuda.slab import (
    DevicePageGroup as DeviceGroup,
)
from resin.backend.plans import PagedCachePlan
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
    read_window_tokens: int
    max_write_tokens: int
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
        prefill_chunk_size: int = 1,
    ) -> CacheGeometry:
        ratios = _ints("compress_ratios", compress_ratios)
        if (
            isinstance(decode_input_tokens, bool)
            or not isinstance(decode_input_tokens, int)
            or decode_input_tokens <= 0
        ):
            raise ValueError("decode_input_tokens must be a positive integer")
        if (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer")
        max_write_tokens = max(prefill_chunk_size, decode_input_tokens)
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
        c4_state_tokens = 8 + max_write_tokens - 1
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
        # Prefill/verification writes the entire block before reading attention
        # or compressor history. The earliest query's read window and all new
        # rows must remain addressable together, including after a prefix hit.
        window_tokens = window_size + max_write_tokens - 1
        return cls(
            attention_layers,
            window_size,
            max_write_tokens,
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
                    read_window_tokens=self.read_window_tokens,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[C4_KV_GROUP]
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[C4_STATE_GROUP],
                    blocks_per_slab=self.packings[C4_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.c4_state_tokens,
                    read_window_tokens=8,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[C128_KV_GROUP]
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[C128_STATE_GROUP],
                    blocks_per_slab=self.packings[C128_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=128 + self.max_write_tokens - 1,
                    read_window_tokens=128,
                ),
                CacheGroupLayout(
                    _GROUP_BLOCK_TOKENS[INDEX_STATE_GROUP],
                    blocks_per_slab=self.packings[INDEX_STATE_GROUP],
                    retention=CacheRetention.WINDOW,
                    window_tokens=8 + self.max_write_tokens - 1,
                    read_window_tokens=8,
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

    geometry: CacheGeometry
    groups: tuple[DeviceGroup, ...]
    latents: object
    index_keys: object

    def row(self, index: int):
        """Return a one-sequence metadata view over the same physical pools."""

        return DeviceCache(
            self.geometry,
            slice_page_groups(self.groups, index),
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
        parents = torch.gather(table.parent_ids, 1, flat).view_as(columns)
        children = torch.gather(table.child_slots, 1, flat).view_as(columns)
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
        physical = torch.where(parents > 0, physical, 0)
        return torch.where(in_table, physical, -1).to(torch.int32)

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
            from resin_models_cuda.deepseek_v4.cutile import (
                paged_state_locations,
            )

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
        parents = torch.gather(table.parent_ids, 1, flat).view_as(columns)
        children = torch.gather(table.child_slots, 1, flat).view_as(columns)
        packing = self.geometry.packings[group]
        physical_children = torch.where(parents > 0, parents * packing + children, 0)
        valid = (columns >= 0) & (columns < table.parent_ids.shape[1])
        physical_children = torch.where(valid, physical_children, -1)
        offsets = torch.where(valid, positions.remainder(block_tokens), 0)
        return physical_children.to(torch.int32), offsets.to(torch.int32)


class CacheArena(GraphPageTableArena):
    """One allocation shared by six mutually exclusive cache groups."""

    graph_input_names = PARENT_PAGE_NAMES
    graph_label = "DeepSeek"

    def __init__(self, device, layout, geometry: CacheGeometry, *, dtype) -> None:
        self.device = device
        self.layout = layout
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
        flat = self.storage.view(torch.float32).flatten()

        def field(offset: int) -> Tensor:
            return torch.as_strided(
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

    def tensorize(self, plan: PagedCachePlan, staging):
        groups = stage_page_groups(
            plan,
            staging,
            names=PARENT_PAGE_NAMES,
            # Preserve one cuTile divisibility class for dynamic widths.
            width=lambda value: (value + 15) // 32 * 32 + 16,
        )
        return DeviceCache(
            self.geometry,
            tuple(groups),
            self.latents,
            self.index_keys,
        )

    def _make_graph_cache(self, groups):
        return DeviceCache(
            self.geometry,
            groups,
            self.latents,
            self.index_keys,
        )

    def _graph_padding_id(self, index: int) -> int:
        return self.layout.parent_capacity + index + 1


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
    "DeviceCache",
    "DeviceMapping",
    "DeviceGroup",
    "plan_cache_capacity",
]
