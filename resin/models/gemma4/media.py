"""Backend-neutral Gemma 4 media inputs kept outside the token scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from resin.protocols import MediaTokenSpan
from resin.server.media import canonical_rgb_image, decode_image
from resin.server.media_inputs import (
    digest_tensors,
    owned_cpu_tensor,
    salt_token_positions,
    single_row,
)


@dataclass(frozen=True, slots=True, eq=False)
class Gemma4Media:
    """One tokenized prompt plus its preprocessed image/video tensors.

    ``Engine.submit(media=...)`` reads ``token_ids`` and otherwise treats the
    object as opaque. The Gemma runner validates placeholder IDs and maps the
    remaining fields to that request's absolute positions and KV pages.
    """

    token_ids: tuple[int, ...]
    mm_token_type_ids: tuple[int, ...]
    pixel_values: Tensor | None = None
    image_position_ids: Tensor | None = None
    pixel_values_videos: Tensor | None = None
    video_position_ids: Tensor | None = None
    digest: str = field(init=False, repr=False)
    cache_key: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        token_ids = tuple(int(token) for token in self.token_ids)
        token_types = tuple(int(token_type) for token_type in self.mm_token_type_ids)
        if not token_ids or len(token_ids) != len(token_types):
            raise ValueError(
                "media token IDs and token types must have one nonempty row"
            )
        if unsupported := set(token_types) - {0, 1, 2}:
            raise ValueError(
                f"unsupported Gemma 4 media token types: {sorted(unsupported)}"
            )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "mm_token_type_ids", token_types)

        image_pair = (self.pixel_values is None, self.image_position_ids is None)
        if image_pair[0] != image_pair[1]:
            raise ValueError(
                "pixel_values and image_position_ids must be supplied together"
            )
        video_pair = (
            self.pixel_values_videos is None,
            self.video_position_ids is None,
        )
        if video_pair[0] != video_pair[1]:
            raise ValueError(
                "pixel_values_videos and video_position_ids must be supplied together"
            )
        if 1 in token_types and self.pixel_values is None:
            raise ValueError("image placeholders require pixel_values")
        if 2 in token_types and self.pixel_values_videos is None:
            raise ValueError("video placeholders require pixel_values_videos")
        if self.pixel_values is not None:
            object.__setattr__(
                self,
                "pixel_values",
                owned_cpu_tensor(self.pixel_values, "pixel_values", ndim=3),
            )
            object.__setattr__(
                self,
                "image_position_ids",
                owned_cpu_tensor(
                    self.image_position_ids, "image_position_ids", ndim=3
                ).to(torch.int64),
            )
        if self.pixel_values_videos is not None:
            object.__setattr__(
                self,
                "pixel_values_videos",
                owned_cpu_tensor(
                    self.pixel_values_videos, "pixel_values_videos", ndim=4
                ),
            )
            object.__setattr__(
                self,
                "video_position_ids",
                owned_cpu_tensor(
                    self.video_position_ids, "video_position_ids", ndim=4
                ).to(torch.int64),
            )
        digest = _media_digest(self)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "cache_key", _cache_identity(self, digest))

    @property
    def image_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, token_type in enumerate(self.mm_token_type_ids)
            if token_type == 1
        )

    @property
    def video_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, token_type in enumerate(self.mm_token_type_ids)
            if token_type == 2
        )

    @property
    def block_ids(self) -> tuple[int, ...]:
        """Contiguous vision block identity used by local bidirectional attention."""

        result = []
        block = -1
        previous_vision = False
        for token_type in self.mm_token_type_ids:
            vision = token_type in {1, 2}
            if vision and not previous_vision:
                block += 1
            result.append(block if vision else -1)
            previous_vision = vision
        return tuple(result)

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]:
        """Vision ranges that require one bidirectional prefill row."""

        spans = []
        start = None
        for index, token_type in enumerate((*self.mm_token_type_ids, 0)):
            if token_type in {1, 2} and start is None:
                start = index
            elif token_type not in {1, 2} and start is not None:
                spans.append(MediaTokenSpan(position=start, length=index - start))
                start = None
        return tuple(spans)

    def validate_placeholder_tokens(
        self, *, image_token_id: int, video_token_id: int
    ) -> None:
        for index, (token, token_type) in enumerate(
            zip(self.token_ids, self.mm_token_type_ids, strict=True)
        ):
            expected = (
                image_token_id
                if token_type == 1
                else video_token_id
                if token_type == 2
                else None
            )
            if expected is not None and token != expected:
                raise ValueError(
                    f"media token type at position {index} expects token {expected}, got {token}"
                )

    @classmethod
    def from_processor_output(cls, output: Any) -> Gemma4Media:
        token_ids = single_row(output["input_ids"], "input_ids")
        token_types = single_row(output["mm_token_type_ids"], "mm_token_type_ids")
        return cls(
            token_ids=tuple(token_ids.tolist()),
            mm_token_type_ids=tuple(token_types.tolist()),
            pixel_values=output.get("pixel_values"),
            image_position_ids=output.get("image_position_ids"),
            pixel_values_videos=output.get("pixel_values_videos"),
            video_position_ids=output.get("video_position_ids"),
        )

    @classmethod
    def from_messages(
        cls,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> Gemma4Media:
        template_kwargs = {}
        if processor_kwargs is not None:
            template_kwargs["processor_kwargs"] = processor_kwargs
        output = processor.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        return cls.from_processor_output(output)

    @classmethod
    def from_images(
        cls,
        processor: Any,
        images: list[np.ndarray],
        *,
        prompt: str,
    ) -> Gemma4Media:
        """Build media from canonical RGB/HWC float32 arrays in ``[0, 1]``."""

        owned = [
            canonical_rgb_image(image, index=index)
            for index, image in enumerate(images)
        ]
        if not owned:
            raise ValueError("images must not be empty")
        content = [{"type": "image", "image": image} for image in owned]
        content.append({"type": "text", "text": prompt})
        return cls.from_messages(
            processor,
            [{"role": "user", "content": content}],
            processor_kwargs={"do_rescale": False},
        )

    @classmethod
    def from_bytes(
        cls,
        processor: Any,
        data: bytes,
        *,
        media_type: str,
        prompt: str,
    ) -> Gemma4Media:
        image = decode_image(data, media_type)
        return cls.from_messages(
            processor,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )


def _media_digest(media: Gemma4Media) -> str:
    """SHA-256 of canonical model inputs, independent of file codecs."""

    names = (
        "pixel_values",
        "image_position_ids",
        "pixel_values_videos",
        "video_position_ids",
    )
    return digest_tensors(
        b"resin-gemma4-media-v1\0",
        ((name, getattr(media, name)) for name in names),
        metadata=(np.asarray(media.mm_token_type_ids, dtype=np.uint8).tobytes(),),
    )


def _cache_identity(media: Gemma4Media, digest: str) -> tuple[int, ...]:
    """Build a same-length token identity salted at vision placeholders."""

    media_positions = media.image_positions + media.video_positions
    return salt_token_positions(media.token_ids, media_positions, digest)


__all__ = ["Gemma4Media", "canonical_rgb_image", "decode_image"]
