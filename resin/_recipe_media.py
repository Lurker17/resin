"""Source normalization and cache-identity helpers for model media processors.

Importing this module does not import Torch.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


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
    """Decode explicitly typed bytes into an owned RGB image."""
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
    """Validate and own one float32 RGB/HWC image in ``[0, 1]``."""
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
    return np.array(image, copy=True, order="C")


def normalized_rgb_from_data_url(value: str) -> np.ndarray:
    """Decode a base64 image; remote fetching belongs to the application."""
    match = _DATA_URL.fullmatch(value)
    if match is None:
        raise ValueError(
            "image URLs must be base64 data URLs; remote fetching is disabled"
        )
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except ValueError as error:
        raise ValueError("image data URL contains invalid base64") from error
    return _rgb_from_bytes(payload, match.group("media_type"))


def _rgb_from_bytes(data: bytes, media_type: str) -> np.ndarray:
    # The conversion already owns the pixel storage; no second copy is needed.
    return np.asarray(decode_image(data, media_type), dtype=np.float32) / np.float32(
        255
    )


def normalize_image(source: Any) -> np.ndarray:
    if isinstance(source, str):
        return normalized_rgb_from_data_url(source)
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[0], bytes)
        and isinstance(source[1], str)
    ):
        return _rgb_from_bytes(*source)
    return canonical_rgb_image(source)


def owned_cpu_tensor(value: Any, name: str, *, ndim: int) -> Tensor:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {tensor.ndim}")
    return tensor.detach().contiguous().clone()


def single_row(value: Any, name: str) -> Tensor:
    import torch

    tensor = owned_cpu_tensor(value, name, ndim=2)
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one prompt")
    return tensor[0].to(torch.int64)


def salt_token_positions(
    token_ids: tuple[int, ...], positions: tuple[int, ...], digest: str
) -> tuple[int, ...]:
    identity = list(token_ids)
    stream = hashlib.shake_256(bytes.fromhex(digest)).digest(4 * len(positions))
    for index, position in enumerate(positions):
        identity[position] = int.from_bytes(stream[4 * index : 4 * index + 4], "little")
    return tuple(identity)


def digest_tensors(
    namespace: bytes,
    tensors: Iterable[tuple[str, Tensor | None]],
    *,
    metadata: Iterable[bytes] = (),
) -> str:
    """Hash model inputs without copying their storage into Python bytes."""
    import torch

    digest = hashlib.sha256(namespace)
    for value in metadata:
        digest.update(value)
    for name, tensor in tensors:
        if tensor is not None:
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(bytes(str(tuple(tensor.shape)), "ascii"))
            storage = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            digest.update(memoryview(storage.numpy()))
    return digest.hexdigest()


def token_id(tokenizer: Any, spelling: str) -> int:
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    lookup = getattr(backend, "token_to_id", None)
    token = lookup(spelling) if lookup is not None else None
    if token is None:
        lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
        token = lookup(spelling) if lookup is not None else None
    if token is None:
        raise ValueError(f"tokenizer must define image token {spelling}")
    return int(token)
