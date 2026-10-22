"""Paged shared KV, index keys, compressor rows and Engram token history.

Only source layers allocate compressed storage. Consumers map to those owners;
all four logical groups share the Engine's fungible physical slab pool.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from resin import CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.cuda.slab import (
    PARENT_PAGE_NAMES,
    GraphPageTableArena,
    graph_scratch_address,
    graph_scratch_parent_count,
    slab_locations,
    slice_page_groups,
    stage_page_groups,
)
from resin.backend.resources import CachePoolBinding
from resin_models_cuda.deepseek_v4.cache import DeviceCache as BaseDeviceCache

PAGE_TOKENS = 128
WINDOW, COMPRESSED, COMPRESSOR, TOKENS = range(4)


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    attention_layers: int
    kv_sources: tuple[int, ...]
    ratios: tuple[int, ...]
    compressed_offsets: tuple[int | None, ...]
    index_offsets: tuple[int | None, ...]
    index_base_rows: int
    compressor_sources: tuple[int, ...]
    packings: tuple[int, ...]
    parent_rows: int
    max_write_tokens: int
    window_size: int
    head_dim: int = 512
    max_batch_size: int = 1

    @classmethod
    def build(cls, args, *, prefill_chunk_size: int, max_batch_size: int = 1, **_):
        count = args.n_layers + args.n_mtp_layers
        ratios = tuple(args.compress_ratios)
        owners, compressed, indices = [], [], []
        kv_cursor = index_cursor = 0
        for layer in range(count):
            owner = next(
                (x for x in reversed(args.kv_source_layers) if x <= layer), None
            )
            if layer >= args.n_layers or not ratios[layer]:
                owner = None
            owners.append(owner)
            if layer in args.kv_source_layers:
                compressed.append(kv_cursor)
                indices.append(index_cursor)
                kv_cursor += PAGE_TOKENS // ratios[layer]
                index_cursor += PAGE_TOKENS // ratios[layer]
            else:
                compressed.append(None)
                indices.append(None)
        for layer, owner in enumerate(owners):
            if owner is not None:
                compressed[layer], indices[layer] = compressed[owner], indices[owner]
        state_sources = tuple(x for x in args.kv_source_layers if ratios[x] > 1)
        raw_rows = (
            count * PAGE_TOKENS,
            kv_cursor + math.ceil(index_cursor / 4),
            max(1, len(state_sources)) * PAGE_TOKENS * 4,
            1,  # 128 int64 compressed token IDs fit one 512D BF16 row.
        )
        largest = max(raw_rows)
        packings = tuple(1 << max(0, (largest // x).bit_length() - 1) for x in raw_rows)
        # Every child starts at a whole 128D index page and a pair of BF16
        # rows, permitting FP32 and int64 aliases without misaligned storage.
        alignment = math.lcm(128, *packings)
        parent_rows = (
            math.ceil(max(x * p for x, p in zip(raw_rows, packings)) / alignment)
            * alignment
        )
        return cls(
            count,
            tuple(args.kv_source_layers),
            ratios,
            tuple(compressed),
            tuple(indices),
            kv_cursor,
            state_sources,
            packings,
            parent_rows,
            max(prefill_chunk_size, args.dspark_block_size + 1),
            args.window_size,
            args.head_dim,
            max_batch_size,
        )

    @property
    def graph_scratch_parents(self) -> int:
        return graph_scratch_parent_count(self.packings, self.max_batch_size)

    def layout(self, parent_capacity: int) -> CacheLayout:
        def window(read):
            return dict(
                retention=CacheRetention.WINDOW,
                read_window_tokens=read,
                window_tokens=read + self.max_write_tokens - 1,
            )

        return CacheLayout(
            parent_capacity,
            (
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.packings[WINDOW],
                    **window(self.window_size + 1),
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[COMPRESSED]
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[COMPRESSOR], **window(2)
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.packings[TOKENS], **window(4)
                ),
            ),
        )

    @property
    def bytes_per_parent(self) -> int:
        return self.parent_rows * self.head_dim * 2


@dataclass(frozen=True, slots=True)
class DeviceCache(BaseDeviceCache):
    geometry: Any
    storage: torch.Tensor

    def row(self, index):
        return replace(self, groups=slice_page_groups(self.groups, index))

    def compressed_rows(self, layer, ratio, positions):
        offset = self.geometry.compressed_offsets[layer]
        if offset is None:
            raise ValueError("layer does not consume shared compressed KV")
        return self._rows(
            COMPRESSED, positions, block_tokens=128, divisor=ratio, layer_offset=offset
        )

    def index_rows(self, layer, positions):
        offset = self.geometry.index_offsets[layer]
        if offset is None:
            raise ValueError("layer does not consume shared index keys")
        ratio = self.geometry.ratios[layer]
        return self._rows(
            COMPRESSED,
            positions,
            block_tokens=128,
            divisor=ratio,
            row_scale=4,
            layer_offset=self.geometry.index_base_rows * 4 + offset,
        )

    def compressor_rows(self, layer, positions):
        owner = self.geometry.compressor_sources.index(layer)
        rows = self._rows(
            COMPRESSOR, positions, block_tokens=128, divisor=1, layer_offset=0
        ).long()
        base = rows - positions.remainder(128)
        value = (
            base.div(2, rounding_mode="floor")
            + owner * 256
            + positions.remainder(128) * 2
        )
        return value, value + 1

    def token_rows(self, positions):
        # Token history aliases the whole slab in int64 units. Unlike KV row
        # IDs these offsets exceed int32 once the physical pool passes 16 GiB.
        parents, slots, offsets = slab_locations(self.groups, TOKENS, positions, 128)
        units = self.geometry.parent_rows * 128
        return (
            parents.long() * units
            + slots.long() * (units // self.geometry.packings[TOKENS])
            + offsets.long()
        )

    @property
    def compressor_values(self):
        return self.storage.view(torch.float32).reshape(-1, 512)

    @property
    def token_values(self):
        return self.storage.view(torch.int64).flatten()


class CacheArena(GraphPageTableArena):
    graph_input_names = PARENT_PAGE_NAMES
    graph_label = "DeepSeek-V4.1"

    def __init__(self, device, layout, geometry, *, dtype):
        self.device, self.layout, self.geometry = device, layout, geometry
        self.binding = CachePoolBinding(layout, resource="DeepSeek-V4.1 KV")
        self.storage = torch.empty(
            (
                layout.parent_capacity + 1 + geometry.graph_scratch_parents,
                geometry.parent_rows,
                512,
            ),
            dtype=dtype,
            device=device,
        )
        self.latents = self.storage.view(-1, 128, 512)
        self.index_keys = self.storage.view(-1, 128, 128)
        self.reset_scratch()

    def reset_scratch(self):
        self.storage[0].zero_()
        self.storage[-self.geometry.graph_scratch_parents :].zero_()

    initialize_state_sentinels = reset_scratch

    def tensorize(self, plan, staging):
        return self._make_graph_cache(
            stage_page_groups(plan, staging, names=PARENT_PAGE_NAMES)
        )

    def _make_graph_cache(self, groups):
        return DeviceCache(
            self.geometry, tuple(groups), self.latents, self.index_keys, self.storage
        )

    def _graph_padding_address(self, index, row):
        return graph_scratch_address(
            self.layout.parent_capacity,
            self.geometry.packings,
            self.geometry.max_batch_size,
            index,
            row,
        )
