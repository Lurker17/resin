"""Expanded Kimi image tokens and their CPU-resident vision inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from resin.protocols import MediaTokenSpan


def _owned_cpu_tensor(value: Any, name: str, *, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    return value.detach().contiguous().clone()


@dataclass(frozen=True, slots=True, eq=False)
class KimiK25Media:
    """One scheduler-expanded prompt plus MoonViT patch inputs."""

    token_ids: tuple[int, ...]
    image_spans: tuple[MediaTokenSpan, ...]
    pixel_values: Tensor
    grid_thws: Tensor
    image_hashes: tuple[str, ...]
    cache_key: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokens = tuple(int(token) for token in self.token_ids)
        spans = tuple(
            MediaTokenSpan(int(span.position), int(span.length))
            for span in self.image_spans
        )
        pixels = _owned_cpu_tensor(self.pixel_values, "pixel_values", ndim=4)
        grids = _owned_cpu_tensor(self.grid_thws, "grid_thws", ndim=2).to(torch.int64)
        hashes = tuple(str(value) for value in self.image_hashes)
        if not tokens:
            raise ValueError("Kimi media token IDs must not be empty")
        if grids.shape != (len(spans), 3) or len(hashes) != len(spans):
            raise ValueError("Kimi image metadata must have one row per image")
        if sum(int(t * h * w) for t, h, w in grids.tolist()) != len(pixels):
            raise ValueError("Kimi pixel patches do not match grid_thws")
        cursor = 0
        for span, (_, height, width) in zip(spans, grids.tolist(), strict=True):
            if height % 2 or width % 2:
                raise ValueError("Kimi image grids must be divisible by the 2x2 merger")
            expected = height * width // 4
            if span.length != expected or span.position < cursor:
                raise ValueError("Kimi expanded image span has the wrong extent")
            cursor = span.position + span.length
        if cursor > len(tokens):
            raise ValueError("Kimi image span exceeds the prompt")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "image_spans", spans)
        object.__setattr__(self, "pixel_values", pixels)
        object.__setattr__(self, "grid_thws", grids)
        object.__setattr__(self, "image_hashes", hashes)
        object.__setattr__(self, "cache_key", self._cache_identity())

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]:
        """Kimi image embeddings use ordinary causal language attention."""

        return ()

    def _cache_identity(self) -> tuple[int, ...]:
        identity = list(self.token_ids)
        for image_index, (span, image_hash) in enumerate(
            zip(self.image_spans, self.image_hashes, strict=True)
        ):
            digest = hashlib.shake_256(
                b"resin-kimi-k25-media-v1\0"
                + image_index.to_bytes(4, "little")
                + image_hash.encode()
            ).digest(4 * span.length)
            for offset in range(span.length):
                start = 4 * offset
                identity[span.position + offset] = int.from_bytes(
                    digest[start : start + 4], "little"
                )
        return tuple(identity)

    @classmethod
    def from_rendered(cls, rendered: Any) -> KimiK25Media:
        """Expand each renderer placeholder to its projected patch count."""

        data = rendered.multi_modal_data
        if data is None:
            raise ValueError("Kimi rendered tokens contain no image data")
        placeholders = list(data.mm_placeholders.get("image") or ())
        items = list(data.mm_items.get("image") or ())
        hashes = tuple(data.mm_hashes.get("image") or ())
        if not placeholders or not (len(placeholders) == len(items) == len(hashes)):
            raise ValueError("Kimi renderer image metadata is incomplete")

        source = tuple(int(token) for token in rendered.token_ids)
        expanded: list[int] = []
        spans = []
        pixels = []
        grids = []
        source_cursor = 0
        for placeholder, item in zip(placeholders, items, strict=True):
            offset = int(placeholder.offset)
            if int(placeholder.length) != 1 or not source_cursor <= offset < len(
                source
            ):
                raise ValueError("Kimi renderer placeholder is malformed")
            grid = _owned_cpu_tensor(item["grid_thws"], "grid_thws", ndim=2).to(
                torch.int64
            )
            if grid.shape[0] != 1:
                raise ValueError("each Kimi renderer item must contain one image")
            _, height, width = map(int, grid[0].tolist())
            if height % 2 or width % 2:
                raise ValueError("Kimi image grid must be divisible by two")
            feature_count = height * width // 4
            expanded.extend(source[source_cursor:offset])
            start = len(expanded)
            expanded.extend((source[offset],) * feature_count)
            spans.append(MediaTokenSpan(start, feature_count))
            pixels.append(
                _owned_cpu_tensor(item["pixel_values"], "pixel_values", ndim=4)
            )
            grids.append(grid)
            source_cursor = offset + 1
        expanded.extend(source[source_cursor:])
        return cls(
            tuple(expanded),
            tuple(spans),
            torch.cat(pixels),
            torch.cat(grids),
            hashes,
        )


__all__ = ["KimiK25Media"]
