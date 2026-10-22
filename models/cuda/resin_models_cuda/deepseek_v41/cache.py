"""Packed FlashMLA KV and typed state views over one fungible byte slab pool."""

import math
from dataclasses import dataclass, replace

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

PAGE_TOKENS = 128
WINDOW, COMPRESSED, COMPRESSOR, TOKENS = range(4)
FP8_BYTES, FP4_BYTES = 528, 288


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    attention_layers: int
    kv_sources: tuple[int, ...]
    ratios: tuple[int, ...]
    compressed_offsets: tuple[int | None, ...]
    index_offsets: tuple[int | None, ...]
    compressor_sources: tuple[int, ...]
    packings: tuple[int, ...]
    parent_bytes: int
    max_write_tokens: int
    window_size: int
    draft_tokens: int
    max_batch_size: int = 1

    @classmethod
    def build(cls, args, *, prefill_chunk_size: int, max_batch_size: int = 1, **_):
        count = args.n_layers + args.n_mtp_layers
        ratios = tuple(args.compress_ratios)
        cursor, compressed, indices = 0, [None] * count, [None] * count
        for layer in args.kv_source_layers:
            compressed[layer] = cursor
            cursor += PAGE_TOKENS // ratios[layer] * FP4_BYTES
        for layer in args.kv_source_layers:
            indices[layer] = cursor
            cursor += PAGE_TOKENS // ratios[layer] * 128 * 2
        for layer in range(args.n_layers):
            if ratios[layer]:
                owner = next(x for x in reversed(args.kv_source_layers) if x <= layer)
                compressed[layer], indices[layer] = compressed[owner], indices[owner]
        sources = tuple(x for x in args.kv_source_layers if ratios[x] > 1)
        sizes = (
            count * PAGE_TOKENS * FP8_BYTES,
            cursor,
            max(1, len(sources)) * PAGE_TOKENS * 2 * 512 * 4,
            PAGE_TOKENS * 8,
        )
        largest = max(sizes)
        packings = tuple(1 << ((largest // size).bit_length() - 1) for size in sizes)
        # FlashMLA's flattened TMA coordinates divide each page stride by its
        # encoded row width. State/index/token aliases retain natural alignment.
        alignment = math.lcm(
            packings[WINDOW] * FP8_BYTES,
            packings[COMPRESSED] * math.lcm(FP4_BYTES, 256),
            packings[COMPRESSOR] * 2048,
            packings[TOKENS] * 8,
        )
        parent_bytes = (
            math.ceil(max(s * p for s, p in zip(sizes, packings)) / alignment)
            * alignment
        )
        return cls(
            count,
            tuple(args.kv_source_layers),
            ratios,
            tuple(compressed),
            tuple(indices),
            sources,
            packings,
            parent_bytes,
            max(prefill_chunk_size, args.dspark_block_size + 1),
            args.window_size,
            math.ceil(args.dspark_block_size / 8) * 8,
            max_batch_size,
        )

    @property
    def graph_scratch_parents(self):
        # A trailing mapped guard also covers the TMA span of layer-offset views.
        return (
            graph_scratch_parent_count(self.packings, self.max_batch_size)
            + self.guard_parents
        )

    @property
    def guard_parents(self):
        return max(
            1,
            math.ceil(
                self.max_batch_size * self.draft_tokens * FP8_BYTES / self.parent_bytes
            ),
        )

    @property
    def bytes_per_parent(self):
        return self.parent_bytes

    def layout(self, parent_capacity):
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


@dataclass(frozen=True, slots=True)
class DeviceCache:
    geometry: CacheGeometry
    groups: tuple
    storage: torch.Tensor

    def row(self, index):
        return replace(self, groups=slice_page_groups(self.groups, index))

    def _locations(self, group, positions):
        return slab_locations(self.groups, group, positions, PAGE_TOKENS)

    def _bytes(self, group, positions):
        parents, slots, offsets = self._locations(group, positions)
        return (
            parents.long() * self.geometry.parent_bytes
            + slots.long()
            * (self.geometry.parent_bytes // self.geometry.packings[group])
        ), offsets

    def _rows(self, group, positions, ratio):
        parents, slots, offsets = self._locations(group, positions)
        pages = parents * self.geometry.packings[group] + slots
        return (pages * (PAGE_TOKENS // ratio) + offsets // ratio).int()

    def window_rows(self, layer, positions):
        return self._rows(WINDOW, positions, 1)

    def compressed_rows(self, layer, ratio, positions):
        if self.geometry.compressed_offsets[layer] is None:
            raise ValueError("layer does not consume shared compressed KV")
        return self._rows(COMPRESSED, positions, ratio)

    def _view(self, group, offset, tokens, width):
        packing = self.geometry.packings[group]
        return self.storage.as_strided(
            (
                (self.storage.shape[0] - self.geometry.guard_parents) * packing,
                tokens,
                1,
                width,
            ),
            (self.geometry.parent_bytes // packing, width, width, 1),
            offset,
        )

    def window_cache(self, layer):
        return self._view(
            WINDOW, layer * PAGE_TOKENS * FP8_BYTES, PAGE_TOKENS, FP8_BYTES
        )

    def compressed_cache(self, layer):
        offset = self.geometry.compressed_offsets[layer]
        if offset is None:
            raise ValueError("layer does not consume shared compressed KV")
        return self._view(
            COMPRESSED, offset, PAGE_TOKENS // self.geometry.ratios[layer], FP4_BYTES
        )

    @property
    def draft_cache(self):
        tokens = self.geometry.draft_tokens
        return self.storage.as_strided(
            (self.geometry.max_batch_size, tokens, 1, FP8_BYTES),
            (tokens * FP8_BYTES, FP8_BYTES, FP8_BYTES, 1),
            (self.storage.shape[0] - self.geometry.guard_parents)
            * self.geometry.parent_bytes,
        )

    def index_rows(self, layer, positions):
        base, offsets = self._bytes(COMPRESSED, positions)
        offset = self.geometry.index_offsets[layer]
        if offset is None:
            raise ValueError("layer does not consume shared index keys")
        return ((base + offset) // 256 + offsets // self.geometry.ratios[layer]).long()

    @property
    def index_keys(self):
        return self.storage.view(torch.bfloat16).reshape(-1, 128)

    def compressor_rows(self, layer, positions):
        base, offsets = self._bytes(COMPRESSOR, positions)
        owner = self.geometry.compressor_sources.index(layer)
        values = base // 2048 + owner * PAGE_TOKENS * 2 + offsets.long() * 2
        return values, values + 1

    @property
    def compressor_values(self):
        return self.storage.view(torch.float32).reshape(-1, 512)

    def token_rows(self, positions):
        base, offsets = self._bytes(TOKENS, positions)
        return base // 8 + offsets.long()

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
                geometry.parent_bytes,
            ),
            dtype=torch.uint8,
            device=device,
        )
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
        return DeviceCache(self.geometry, tuple(groups), self.storage)

    def _graph_padding_address(self, index, row):
        return graph_scratch_address(
            self.layout.parent_capacity,
            self.geometry.packings,
            self.geometry.max_batch_size,
            index,
            row,
        )
