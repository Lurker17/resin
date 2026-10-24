"""Backend-neutral Qwen3.5 image inputs and multimodal rotary positions."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

import resin
from resin._recipe_media import (
    digest_tensors,
    normalize_image,
    owned_cpu_tensor,
    salt_token_positions,
    single_row,
    token_id,
)
from resin.protocols import MediaTokenSpan


def _mrope_positions(
    token_types: tuple[int, ...],
    image_grid_thw: Tensor,
    *,
    spatial_merge_size: int,
) -> tuple[Tensor, int]:
    positions = []
    current = 0
    image_index = 0
    for token_type, values in itertools.groupby(token_types):
        length = sum(1 for _ in values)
        if token_type == 0:
            positions.append(
                torch.arange(current, current + length, dtype=torch.int64)
                .view(1, -1)
                .expand(3, -1)
            )
            current += length
            continue
        if token_type != 1 or image_index >= len(image_grid_thw):
            raise ValueError("Qwen media token types do not match image grids")
        grid_t, grid_h, grid_w = map(int, image_grid_thw[image_index])
        image_index += 1
        grid_h //= spatial_merge_size
        grid_w //= spatial_merge_size
        if min(grid_t, grid_h, grid_w) <= 0 or grid_t * grid_h * grid_w != length:
            raise ValueError("Qwen image grid does not match its placeholder range")
        temporal = torch.arange(grid_t, dtype=torch.int64) + current
        height = torch.arange(grid_h, dtype=torch.int64) + current
        width = torch.arange(grid_w, dtype=torch.int64) + current
        axes = torch.meshgrid(temporal, height, width, indexing="ij")
        positions.append(torch.stack(axes).reshape(3, -1))
        current += max(grid_h, grid_w)
    if image_index != len(image_grid_thw):
        raise ValueError("Qwen image grids outnumber placeholder ranges")
    result = torch.cat(positions, dim=1)
    return result, int(result.max()) + 1 - len(token_types)


@dataclass(frozen=True, slots=True, eq=False)
class Qwen35Media:
    """One rendered Qwen prompt plus its processor-owned image tensors."""

    token_ids: tuple[int, ...]
    image_positions: tuple[int, ...]
    pixel_values: Tensor
    image_grid_thw: Tensor
    spatial_merge_size: int
    image_token_id: int
    position_ids: Tensor = field(init=False, repr=False)
    rope_delta: int = field(init=False)
    digest: str = field(init=False, repr=False)
    cache_key: tuple[int, ...] = field(init=False, repr=False)
    token_spans: tuple[MediaTokenSpan, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        token_ids = tuple(int(token) for token in self.token_ids)
        image_positions = tuple(int(position) for position in self.image_positions)
        if not token_ids or not image_positions:
            raise ValueError("Qwen image media requires tokens and placeholders")
        if any(
            position < 0
            or position >= len(token_ids)
            or token_ids[position] != self.image_token_id
            for position in image_positions
        ):
            raise ValueError("Qwen image placeholders do not match image token IDs")
        pixels = owned_cpu_tensor(self.pixel_values, "pixel_values", ndim=2).float()
        grids = owned_cpu_tensor(self.image_grid_thw, "image_grid_thw", ndim=2).to(
            torch.int64
        )
        if grids.shape[1] != 3 or self.spatial_merge_size <= 0:
            raise ValueError("Qwen image grids or spatial merge size are invalid")
        token_types = [0] * len(token_ids)
        for position in image_positions:
            token_types[position] = 1
        position_ids, rope_delta = _mrope_positions(
            tuple(token_types),
            grids,
            spatial_merge_size=self.spatial_merge_size,
        )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "image_positions", image_positions)
        object.__setattr__(self, "pixel_values", pixels)
        object.__setattr__(self, "image_grid_thw", grids)
        object.__setattr__(self, "position_ids", position_ids)
        object.__setattr__(self, "rope_delta", rope_delta)
        digest = _media_digest(pixels, grids)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(
            self,
            "cache_key",
            salt_token_positions(token_ids, image_positions, digest),
        )

    @classmethod
    def from_processor_output(
        cls,
        output: Any,
        *,
        spatial_merge_size: int,
        image_token_id: int,
    ) -> Qwen35Media:
        if output.get("pixel_values_videos") is not None:
            raise ValueError("Qwen35Media currently supports image input only")
        token_ids = tuple(single_row(output["input_ids"], "input_ids").tolist())
        image_positions = tuple(
            index for index, token in enumerate(token_ids) if token == image_token_id
        )
        return cls(
            token_ids=token_ids,
            image_positions=image_positions,
            pixel_values=output["pixel_values"],
            image_grid_thw=output["image_grid_thw"],
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
        )


def _media_digest(pixel_values: Tensor, image_grid_thw: Tensor) -> str:
    return digest_tensors(
        b"resin-qwen35-media-v1\0",
        (("pixel_values", pixel_values), ("image_grid_thw", image_grid_thw)),
    )


class Qwen35MediaProcessor:
    """Prepare image tokens, pixels and prompt-relative multimodal RoPE."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def process(
        self,
        media: Sequence[resin.recipe.MediaBlock],
    ) -> resin.recipe.ProcessedMedia:
        if any(not isinstance(item, resin.recipe.Image) for item in media):
            raise ValueError("Qwen does not support video or audio input")
        processor = self.processor
        if not hasattr(processor, "tokenizer"):
            raise ValueError("media input requires a pixel processor")
        payload = processor.image_processor(
            images=[normalize_image(item.source) for item in media],
            do_rescale=False,
            return_tensors="pt",
        )
        marker = token_id(processor.tokenizer, "<|image_pad|>")
        merge = processor.image_processor.merge_size**2
        tokens = [
            resin.recipe.MediaTokens(
                "<|image_pad|>", [marker] * (int(grid.prod()) // merge)
            )
            for grid in payload["image_grid_thw"]
        ]
        return resin.recipe.ProcessedMedia(tokens, payload)

    def bind(
        self,
        processed: resin.recipe.ProcessedMedia,
        token_ids: Sequence[int],
        spans: Sequence[MediaTokenSpan],
    ) -> Qwen35Media:
        payload = processed.payload
        return Qwen35Media(
            tuple(token_ids),
            tuple(
                position
                for span in spans
                for position in range(span.position, span.position + span.length)
            ),
            payload["pixel_values"],
            payload["image_grid_thw"],
            int(self.processor.image_processor.merge_size),
            token_id(self.processor.tokenizer, "<|image_pad|>"),
        )


def make_media_processor(processor: Any) -> Qwen35MediaProcessor:
    return Qwen35MediaProcessor(processor)


__all__ = ["Qwen35Media", "Qwen35MediaProcessor"]
