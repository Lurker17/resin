"""OpenAI-wire adapter for canonical Gemma 4 media arrays."""

from __future__ import annotations

import base64
import copy
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from resin_models_cuda.gemma4.media import (
    Gemma4Media,
    canonical_rgb_image,
    decode_image,
)

_DATA_URL = re.compile(
    r"^data:(?P<media_type>[-\w.+]+/[-\w.+]+);base64,(?P<payload>.*)$",
    re.DOTALL,
)


def normalized_rgb_from_data_url(value: str) -> np.ndarray:
    """Decode a data URL at the HTTP edge into canonical float32 RGB/HWC."""

    match = _DATA_URL.fullmatch(value)
    if match is None:
        raise ValueError(
            "Gemma 4 image URLs must be base64 data URLs; remote fetching is disabled"
        )
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except ValueError as error:
        raise ValueError("image data URL contains invalid base64") from error
    image = decode_image(payload, match.group("media_type"))
    return canonical_rgb_image(
        np.asarray(image, dtype=np.float32) / np.float32(255.0)
    )


def _image_value(block: dict[str, Any]) -> Any:
    value = block.get("image")
    if value is not None:
        return value
    value = block.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return value


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise TypeError("message content blocks must be objects")
            kind = block.get("type")
            if kind in {"image", "image_url", "input_image"}:
                value = _image_value(block)
                if isinstance(value, str):
                    value = normalized_rgb_from_data_url(value)
                if not isinstance(value, np.ndarray):
                    raise TypeError("image content must contain a NumPy array or data URL")
                blocks.append({"type": "image", "image": canonical_rgb_image(value)})
            elif kind in {"text", "input_text"}:
                text = block.get("text")
                if not isinstance(text, str):
                    raise TypeError("text content blocks must contain text")
                blocks.append({"type": "text", "text": text})
            else:
                blocks.append(block)
        message["content"] = blocks
    return normalized


class Gemma4Codec:
    """Render HTTP conversations into one opaque ``Gemma4Media`` request."""

    def __init__(self, processor: Any, *, eos_token_ids: Sequence[int]) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.eos_token_ids = tuple(int(token) for token in eos_token_ids)

    def render(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(request["messages"])
        media = Gemma4Media.from_messages(
            self.processor,
            messages,
            tools=request.get("tools") or None,
            processor_kwargs={"do_rescale": False},
        )
        stop_token_ids = list(self.eos_token_ids)
        for stop in request.get("stop") or []:
            encoded = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                    "multi-token stop sequences are not supported yet"
                )
            stop_token_ids.append(int(encoded[0]))
        return {
            "token_ids": list(media.token_ids),
            "stop_token_ids": list(dict.fromkeys(stop_token_ids)),
            "media": media,
        }

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> Gemma4ResponseParser:
        del request, prompt_ids
        return Gemma4ResponseParser(self.tokenizer)


class Gemma4ResponseParser:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.text = ""

    def feed(self, token_ids: Sequence[int]) -> list[dict[str, str]]:
        self.token_ids.extend(int(token) for token in token_ids)
        text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        text = text.rstrip("\N{REPLACEMENT CHARACTER}")
        if not text.startswith(self.text):
            raise ValueError("Gemma 4 decoded output is not prefix-stable")
        delta = text[len(self.text) :]
        self.text = text
        return [] if not delta else [{"type": "text_delta", "delta": delta}]


__all__ = ["Gemma4Codec", "Gemma4ResponseParser", "normalized_rgb_from_data_url"]
