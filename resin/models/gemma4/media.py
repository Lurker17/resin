"""Backend-neutral Gemma 4 media inputs kept outside the token scheduler."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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


class Gemma4MediaProcessor:
    """Prepare Gemma pixels and the token spans that consume them."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def process(
        self,
        media: Sequence[resin.recipe.MediaBlock],
    ) -> resin.recipe.ProcessedMedia:
        images, videos = [], []
        for item in media:
            if isinstance(item, resin.recipe.Image):
                images.append(normalize_image(item.source))
            elif isinstance(item, resin.recipe.Video):
                if item.source is None or isinstance(item.source, (str, bytes)):
                    raise ValueError("video content must contain decoded frames")
                videos.append(copy.deepcopy(item.source))
            else:
                raise ValueError("Gemma 4 does not support audio input")
        processor = self.processor
        if not hasattr(processor, "tokenizer"):
            raise ValueError("media input requires a pixel processor")
        payload: dict[str, Any] = {}
        expansions: dict[type, list[resin.recipe.MediaTokens]] = {}
        kwargs = {"do_rescale": False, "return_tensors": "pt"}
        if images:
            prepared = processor.image_processor(images=images, **kwargs)
            expansions[resin.recipe.Image] = [
                resin.recipe.MediaTokens(
                    processor.image_token,
                    self._encode(
                        processor.replace_image_token(prepared, image_idx=index)
                    ),
                )
                for index in range(len(images))
            ]
            payload.update(prepared)
        if videos:
            prepared = processor.video_processor(
                videos=videos, return_metadata=True, **kwargs
            )
            expanded = []
            for index in range(len(videos)):
                text = processor.replace_video_token(prepared, video_idx=index)
                first_image = text.index(processor.boi_token)
                # Keep the leading timestamp beside adjacent prompt text for BPE.
                expanded.append(
                    resin.recipe.MediaTokens(
                        processor.video_token,
                        self._encode(text[first_image:]),
                        prefix=text[:first_image],
                    )
                )
            expansions[resin.recipe.Video] = expanded
            payload.update(prepared)
        payload.pop("video_metadata", None)
        pending = {kind: iter(spans) for kind, spans in expansions.items()}
        return resin.recipe.ProcessedMedia(
            [next(pending[type(item)]) for item in media], payload
        )

    def _encode(self, text: str) -> Sequence[int]:
        tokenizer = self.processor.tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
        encoded = backend.encode(text, add_special_tokens=False)
        return encoded.ids if hasattr(encoded, "ids") else encoded

    def bind(
        self,
        processed: resin.recipe.ProcessedMedia,
        token_ids: Sequence[int],
        spans: Sequence[MediaTokenSpan],
    ) -> Gemma4Media:
        kinds = {token_id(self.processor.tokenizer, self.processor.image_token): 1}
        if hasattr(self.processor, "video_token"):
            kinds[token_id(self.processor.tokenizer, self.processor.video_token)] = 2
        token_types = [0] * len(token_ids)
        for span in spans:
            for index in range(span.position, span.position + span.length):
                token_types[index] = kinds.get(token_ids[index], 0)
        payload = processed.payload
        return Gemma4Media(
            tuple(token_ids),
            tuple(token_types),
            pixel_values=payload.get("pixel_values"),
            image_position_ids=payload.get("image_position_ids"),
            pixel_values_videos=payload.get("pixel_values_videos"),
            video_position_ids=payload.get("video_position_ids"),
        )


def make_media_processor(processor: Any) -> Gemma4MediaProcessor:
    return Gemma4MediaProcessor(processor)


__all__ = ["Gemma4Media", "Gemma4MediaProcessor"]
