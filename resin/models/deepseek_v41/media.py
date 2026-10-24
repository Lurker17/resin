"""Checkpoint image geometry and owned inputs for the V4.1 vision tower.

Resize, gray padding, BF16 normalization and patch order follow the published
Flash-1017 ``inference/image_processor.py``. The language model remains causal,
so an image's embeddings may be consumed across ordinary prefill chunks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

import resin
from resin._recipe_media import (
    canonical_rgb_image,
    digest_tensors,
    normalize_image,
    owned_cpu_tensor,
    salt_token_positions,
    token_id,
)
from resin.protocols import MediaTokenSpan

IMAGE_TOKEN = "<｜deepseek_image｜>"
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass(frozen=True, slots=True, eq=False)
class DeepSeekV41Image:
    """One image's patch grid and absolute language-model token positions."""

    start: int
    patches: Tensor
    n_vit_h: int
    n_vit_w: int
    types: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("image start must be a nonnegative integer")
        if any(type(n) is not int or n < 1 for n in (self.n_vit_h, self.n_vit_w)):
            raise ValueError("image patch grid must be positive integers")
        patches = owned_cpu_tensor(self.patches, "patches", ndim=4)
        if patches.dtype != torch.bfloat16 or patches.shape != (
            self.n_vit_h * self.n_vit_w,
            3,
            14,
            14,
        ):
            raise ValueError("image patches must be BF16 [grid_h * grid_w, 3, 14, 14]")
        types = tuple(self.types)
        expected = image_token_types(
            math.ceil(self.n_vit_h / 3), math.ceil(self.n_vit_w / 3)
        )
        if types != expected:
            raise ValueError("image token types do not match the 3x3 aligner grid")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "types", types)


@dataclass(frozen=True, slots=True, eq=False)
class DeepSeekV41Media:
    """Tokenized prompt with image payloads opaque to the shared scheduler."""

    token_ids: tuple[int, ...]
    images: tuple[DeepSeekV41Image, ...]
    image_token_id: int = 129264
    digest: str = field(init=False, repr=False)
    cache_key: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokens, images = tuple(self.token_ids), tuple(self.images)
        if not tokens or not images:
            raise ValueError("image media requires a nonempty prompt and images")
        positions = []
        end = 0
        for image in images:
            if image.start < end or image.start + len(image.types) > len(tokens):
                raise ValueError(
                    "image spans must be ordered, disjoint and inside the prompt"
                )
            end = image.start + len(image.types)
            span = range(image.start, end)
            if any(tokens[index] != self.image_token_id for index in span):
                raise ValueError("image span contains a different placeholder token ID")
            positions.extend(span)
        if tokens.count(self.image_token_id) != len(positions):
            raise ValueError("image placeholder token appears outside an image span")
        digest = digest_tensors(
            b"resin.deepseek_v41.images.v1",
            ((f"image.{index}", image.patches) for index, image in enumerate(images)),
            metadata=(
                str(
                    (
                        self.image_token_id,
                        tuple(
                            (image.start, image.n_vit_h, image.n_vit_w)
                            for image in images
                        ),
                    )
                ).encode(),
            ),
        )
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(
            self, "cache_key", salt_token_positions(tokens, tuple(positions), digest)
        )

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]:
        """Causal image embeddings need no atomic/bidirectional prefill spans."""
        return ()


def image_token_types(height: int, width: int) -> tuple[int, ...]:
    return (IMAGE_START, *((IMAGE,) * width + (IMAGE_NEW_LINE,)) * height, IMAGE_END)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekV41ImageProcessor:
    """The released checkpoint's any-resolution preprocessing parameters."""

    patch_size: int = 14
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 544 * 544
    max_wh_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.patch_size != 14 or self.downsample_ratio != 3:
            raise ValueError(
                "DeepSeek V4.1 vision requires patch_size=14 and downsample_ratio=3"
            )
        if type(self.max_image_tokens) is not int or self.max_image_tokens < 4:
            raise ValueError("max_image_tokens must be an integer >= 4")
        if type(self.min_pixels) is not int or self.min_pixels < 0:
            raise ValueError("min_pixels must be a nonnegative integer")
        if self.max_wh_ratio is not None and (
            not math.isfinite(self.max_wh_ratio) or self.max_wh_ratio < 1
        ):
            raise ValueError("max_wh_ratio must be null or finite and >= 1")

    def grid(self, width: int | float, height: int | float) -> tuple[int, int]:
        """Plan the padded pixel grid before decoding or allocating patches."""
        if min(width, height) < 1:
            raise ValueError("image dimensions must be positive")
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = height * self.max_wh_ratio
        if width * height < self.min_pixels:
            scale = math.sqrt(self.min_pixels / (width * height))
            width, height = int(width * scale), int(height * scale)
        patch, cell = self.patch_size, self.patch_size * self.downsample_ratio
        best_h, best_w = (
            math.ceil(height / patch) * patch,
            math.ceil(width / patch) * patch,
        )
        if _num_tokens(best_h, best_w, cell) > self.max_image_tokens:
            ratio = height / width
            max_w = math.sqrt((self.max_image_tokens - 2) / ratio + 0.25) - 0.5
            max_h = max_w * ratio
            if max_w < 1:
                best_h, best_w = (self.max_image_tokens - 2) // 2 * cell, cell
            elif max_h < 1:
                best_h, best_w = cell, (self.max_image_tokens - 3) * cell
            else:
                scale = min(
                    math.floor(max_w) * cell / width, math.floor(max_h) * cell / height
                )
                best_h, best_w = (
                    math.floor(height * scale / patch) * patch,
                    math.floor(width * scale / patch) * patch,
                )
        return best_h // patch, best_w // patch

    def prepare(self, image: np.ndarray, *, start: int = 0) -> DeepSeekV41Image:
        patches, height, width, types = self._patches(canonical_rgb_image(image))
        return DeepSeekV41Image(start, patches, height, width, types)

    def _patches(self, image: np.ndarray) -> tuple[Tensor, int, int, tuple[int, ...]]:
        from PIL import Image, ImageOps

        # Shared data-URL decoding produces exact uint8/255 values. Round the
        # canonical float input back to RGB bytes before Pillow resampling.
        source = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        height, width = self.grid(source.width, source.height)
        size = width * self.patch_size, height * self.patch_size
        if (
            self.max_wh_ratio is not None
            and source.width >= self.max_wh_ratio * source.height
        ):
            resized = source.resize(size)
        else:
            resized = ImageOps.pad(source, size, color=(127, 127, 127))
        values = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1)
            / 255
        )
        values = ((values - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            values.reshape(3, height, 14, width, 14)
            .permute(1, 3, 0, 2, 4)
            .reshape(height * width, 3, 14, 14)
        )
        return (
            patches,
            height,
            width,
            image_token_types(math.ceil(height / 3), math.ceil(width / 3)),
        )

    def token_count(self, width: int, height: int) -> int:
        height, width = self.grid(width, height)
        return math.ceil(height / 3) * (math.ceil(width / 3) + 1) + 2


def _num_tokens(height: int, width: int, cell: int) -> int:
    return math.ceil(height / cell) * (math.ceil(width / cell) + 1) + 2


class DeepSeekV41MediaProcessor:
    """Prepare V4.1 image patches and their causal language-model tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        vision_config: Mapping[str, Any] | None = None,
        image_token_id: int | None = None,
    ) -> None:
        self.marker = token_id(tokenizer, IMAGE_TOKEN)
        if image_token_id is not None and self.marker != image_token_id:
            raise ValueError(
                "tokenizer image token does not match checkpoint image_token_id"
            )
        self.processor = DeepSeekV41ImageProcessor(
            **{
                key: value
                for key, value in (vision_config or {}).items()
                if key in DeepSeekV41ImageProcessor.__dataclass_fields__
            }
        )

    def process(
        self,
        media: Sequence[resin.recipe.MediaBlock],
    ) -> resin.recipe.ProcessedMedia:
        if any(not isinstance(item, resin.recipe.Image) for item in media):
            raise ValueError("DeepSeek V4.1 supports image input only")
        patches = [
            self.processor._patches(normalize_image(item.source)) for item in media
        ]
        tokens = [
            resin.recipe.MediaTokens(IMAGE_TOKEN, [self.marker] * len(types))
            for _, _, _, types in patches
        ]
        return resin.recipe.ProcessedMedia(tokens, patches)

    def bind(
        self,
        processed: resin.recipe.ProcessedMedia,
        token_ids: Sequence[int],
        spans: Sequence[MediaTokenSpan],
    ) -> DeepSeekV41Media:
        images = tuple(
            DeepSeekV41Image(span.position, pixels, height, width, types)
            for (pixels, height, width, types), span in zip(
                processed.payload, spans, strict=True
            )
        )
        return DeepSeekV41Media(tuple(token_ids), images, image_token_id=self.marker)


def make_media_processor(
    tokenizer: Any,
    *,
    vision_config: Mapping[str, Any] | None = None,
    image_token_id: int | None = None,
) -> DeepSeekV41MediaProcessor:
    return DeepSeekV41MediaProcessor(
        tokenizer, vision_config=vision_config, image_token_id=image_token_id
    )


__all__ = [
    "DeepSeekV41Image",
    "DeepSeekV41ImageProcessor",
    "DeepSeekV41Media",
    "DeepSeekV41MediaProcessor",
]
