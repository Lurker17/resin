"""Prepared Inkling image inputs kept outside the token scheduler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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


def _single_row(value: Any, name: str) -> Tensor:
    tensor = _owned_cpu_tensor(value, name, ndim=2)
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one prompt")
    return tensor[0].to(torch.int64)


@dataclass(frozen=True, slots=True, eq=False)
class InklingMedia:
    """One tokenized prompt and its normalized image patch tensors."""

    token_ids: tuple[int, ...]
    image_token_id: int
    pixel_values: Tensor | None = None
    digest: str = field(init=False, repr=False)
    cache_key: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        token_ids = tuple(int(token) for token in self.token_ids)
        if not token_ids:
            raise ValueError("Inkling media token IDs must not be empty")
        object.__setattr__(self, "token_ids", token_ids)
        image_token_id = int(self.image_token_id)
        object.__setattr__(self, "image_token_id", image_token_id)
        positions = self.image_positions
        if self.pixel_values is None:
            if positions:
                raise ValueError("Inkling image placeholders require pixel_values")
        else:
            pixels = _owned_cpu_tensor(self.pixel_values, "pixel_values", ndim=5)
            if pixels.shape[0] != len(positions):
                raise ValueError(
                    "Inkling image placeholder count does not match pixel patches"
                )
            object.__setattr__(self, "pixel_values", pixels)
        digest = self._digest()
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "cache_key", self._cache_identity(digest))

    @property
    def image_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, token_id in enumerate(self.token_ids)
            if token_id == self.image_token_id
        )

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]:
        return ()

    def _digest(self) -> str:
        digest = hashlib.sha256(b"resin-inkling-media-v1\0")
        pixels = self.pixel_values
        if pixels is not None:
            digest.update(str(pixels.dtype).encode())
            digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
            digest.update(pixels.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _cache_identity(self, digest: str) -> tuple[int, ...]:
        identity = list(self.token_ids)
        positions = self.image_positions
        stream = hashlib.shake_256(bytes.fromhex(digest)).digest(4 * len(positions))
        for index, position in enumerate(positions):
            identity[position] = int.from_bytes(
                stream[4 * index : 4 * index + 4], "little"
            )
        return tuple(identity)

    @classmethod
    def from_processor_output(cls, processor: Any, output: Any) -> InklingMedia:
        token_ids = _single_row(output["input_ids"], "input_ids")
        return cls(
            token_ids=tuple(token_ids.tolist()),
            image_token_id=int(processor.image_token_id),
            pixel_values=output.get("pixel_values"),
        )

    @classmethod
    def from_messages(
        cls,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> InklingMedia:
        template_kwargs: dict[str, Any] = {
            "tools": tools,
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {"do_rescale": False},
        }
        if reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort
        output = processor.apply_chat_template(messages, **template_kwargs)
        return cls.from_processor_output(processor, output)


__all__ = ["InklingMedia"]
