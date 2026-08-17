"""Optional image conveniences for server codecs, demos, and tests.

Resin's Engine and media protocol do not depend on this module. Applications
may bypass it entirely. Security-sensitive deployments should apply their own
input-size, MIME, codec, and fetching policy before constructing model media.
"""

from __future__ import annotations

import base64
import copy
import io
import re
from typing import Any

import numpy as np

_IMAGE_MIME_TYPES = {
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}
_DATA_URL = re.compile(
    r"^data:(?P<media_type>[-\w.+]+/[-\w.+]+);base64,(?P<payload>.*)$",
    re.DOTALL,
)


def decode_image(data: bytes, media_type: str):
    """Decode explicitly typed bytes into an owned RGB PIL image.

    This convenience whitelist is not a security boundary. Pillow is imported
    lazily so servers that use their own decoder do not acquire a runtime
    dependency merely by importing :mod:`resin.server`.
    """

    if media_type.lower() not in _IMAGE_MIME_TYPES:
        raise ValueError(f"unsupported image media type: {media_type!r}")
    if not data:
        raise ValueError("image bytes must not be empty")
    from PIL import Image

    with Image.open(io.BytesIO(data)) as source:
        source.verify()
    with Image.open(io.BytesIO(data)) as source:
        return source.convert("RGB").copy()


def canonical_rgb_image(image: np.ndarray, *, index: int = 0) -> np.ndarray:
    """Validate and copy one float32 RGB/HWC image in ``[0, 1]``."""

    if not isinstance(image, np.ndarray):
        raise TypeError(f"images[{index}] must be a numpy array")
    if image.dtype != np.float32:
        raise TypeError(f"images[{index}] must have dtype float32")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"images[{index}] must have RGB/HWC shape [height, width, 3]")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"images[{index}] height and width must be positive")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"images[{index}] values must be finite and in [0, 1]")
    return np.ascontiguousarray(image).copy()


def normalized_rgb_from_data_url(value: str) -> np.ndarray:
    """Decode a base64 image data URL into owned float32 RGB/HWC data.

    Remote fetching is deliberately out of scope. A caller that accepts remote
    URLs should fetch and validate them under its own network and size policy.
    """

    match = _DATA_URL.fullmatch(value)
    if match is None:
        raise ValueError(
            "image URLs must be base64 data URLs; remote fetching is disabled"
        )
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except ValueError as error:
        raise ValueError("image data URL contains invalid base64") from error
    image = decode_image(payload, match.group("media_type"))
    return canonical_rgb_image(np.asarray(image, dtype=np.float32) / np.float32(255.0))


def _image_value(block: dict[str, Any]) -> Any:
    value = block.get("image")
    if value is not None:
        return value
    value = block.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return value


def normalize_image_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy chat messages and normalize image blocks for model processors.

    ``image``, ``image_url``, and ``input_image`` blocks become ``image``
    blocks containing canonical NumPy data. Text block spellings are likewise
    normalized. Unknown block types pass through unchanged.
    """

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
                    raise TypeError(
                        "image content must contain a NumPy array or data URL"
                    )
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


__all__ = [
    "canonical_rgb_image",
    "decode_image",
    "normalize_image_messages",
    "normalized_rgb_from_data_url",
]
