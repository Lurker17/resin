"""Internal pixel preparation for native conversation recipes.

Recipes own message ordering, control-token expansion and response parsing.
These strategies only prepare model pixels and bind their geometry to the
rendered token positions. Importing this module does not import Torch.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from resin.protocols import MediaInput

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


def _token_id(tokenizer: Any, spelling: str) -> int:
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    lookup = getattr(backend, "token_to_id", None)
    token = lookup(spelling) if lookup is not None else None
    if token is None:
        lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
        token = lookup(spelling) if lookup is not None else None
    if token is None:
        raise ValueError(f"tokenizer must define image token {spelling}")
    return int(token)


class _ProcessorPreparer:
    def __init__(self, model: str, processor_or_tokenizer: Any) -> None:
        self.model = model
        self.processor = (
            processor_or_tokenizer
            if hasattr(processor_or_tokenizer, "tokenizer")
            else None
        )

    def prepare(
        self, sources: Sequence[tuple[str, Any]]
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        if not sources:
            return {}, {}
        allowed = {"image", "video"} if self.model == "gemma4" else {"image"}
        for kind, _ in sources:
            if kind not in allowed:
                raise ValueError(f"{self.model} does not support {kind} input")
        processor = self.processor
        if processor is None:
            raise ValueError(f"{self.model} media input requires a pixel processor")
        images, videos = [], []
        for kind, source in sources:
            if kind == "image":
                images.append(normalize_image(source))
            else:
                if source is None or isinstance(source, (str, bytes)):
                    raise ValueError("video content must contain decoded frames")
                # Only the actual frames need ownership, not the conversation.
                videos.append(copy.deepcopy(source))
        output: dict[str, Any] = {}
        replacements: dict[str, list[str]] = {}
        kwargs = {"do_rescale": False, "return_tensors": "pt"}
        if images:
            prepared = processor.image_processor(images=images, **kwargs)
            replacements[processor.image_token] = [
                processor.replace_image_token(prepared, image_idx=index)
                for index in range(len(images))
            ]
            output.update(prepared)
        if videos:
            prepared = processor.video_processor(
                videos=videos, return_metadata=True, **kwargs
            )
            replacements[processor.video_token] = [
                processor.replace_video_token(prepared, video_idx=index)
                for index in range(len(videos))
            ]
            output.update(prepared)
        output.pop("video_metadata", None)
        return replacements, output

    def finalize(
        self, state: dict[str, Any], token_ids: list[int]
    ) -> MediaInput | None:
        tokens = tuple(token_ids)
        processor = self.processor
        if self.model == "gemma4":
            from resin.models.gemma4.media import Gemma4Media

            token_types = (
                processor.create_mm_token_type_ids([token_ids])[0]
                if processor is not None
                else [0] * len(tokens)
            )
            return Gemma4Media(
                tokens,
                tuple(token_types),
                pixel_values=state.get("pixel_values"),
                image_position_ids=state.get("image_position_ids"),
                pixel_values_videos=state.get("pixel_values_videos"),
                video_position_ids=state.get("video_position_ids"),
            )
        if state.get("pixel_values") is None:
            return None
        if processor is None:
            raise ValueError("prepared pixels require a pixel processor")
        if self.model in {"qwen3_5", "qwen3_8"}:
            from resin.models.qwen3_5.media import Qwen35Media

            marker = _token_id(processor.tokenizer, "<|image_pad|>")
            return Qwen35Media(
                tokens,
                tuple(index for index, token in enumerate(tokens) if token == marker),
                state["pixel_values"],
                state["image_grid_thw"],
                int(processor.image_processor.merge_size),
                marker,
            )
        from resin.models.inkling.media import InklingMedia

        return InklingMedia(
            tokens, int(processor.image_token_id), state["pixel_values"]
        )


class _DeepSeekPreparer:
    def __init__(
        self,
        tokenizer: Any,
        vision_config: Mapping[str, Any] | None,
        image_token_id: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.vision_config = dict(vision_config or {})

    def prepare(
        self, sources: Sequence[tuple[str, Any]]
    ) -> tuple[dict[str, list[str]], tuple[Any, int, list[np.ndarray], list[int]]]:
        from resin.models.deepseek_v41.media import (
            IMAGE_TOKEN,
            DeepSeekV41ImageProcessor,
        )

        if any(kind != "image" for kind, _ in sources):
            raise ValueError("DeepSeek V4.1 supports image input only")
        marker = _token_id(self.tokenizer, IMAGE_TOKEN)
        if self.image_token_id is not None and marker != self.image_token_id:
            raise ValueError(
                "tokenizer image token does not match checkpoint image_token_id"
            )
        processor = DeepSeekV41ImageProcessor(
            **{
                key: value
                for key, value in self.vision_config.items()
                if key in DeepSeekV41ImageProcessor.__dataclass_fields__
            }
        )
        images = [normalize_image(source) for _, source in sources]
        counts = [
            processor.token_count(image.shape[1], image.shape[0]) for image in images
        ]
        return {IMAGE_TOKEN: [IMAGE_TOKEN * count for count in counts]}, (
            processor,
            marker,
            images,
            counts,
        )

    def finalize(
        self,
        state: tuple[Any, int, list[np.ndarray], list[int]],
        token_ids: list[int],
    ) -> MediaInput:
        from resin.models.deepseek_v41.media import DeepSeekV41Media

        processor, marker, images, counts = state
        positions = iter(
            index for index, token in enumerate(token_ids) if token == marker
        )
        placed = []
        for image, count in zip(images, counts, strict=True):
            start = next(positions, None)
            if start is None:
                raise ValueError("missing expanded image token span")
            for expected in range(start + 1, start + count):
                if next(positions, None) != expected:
                    raise ValueError("image token span must be contiguous")
            placed.append(processor.prepare(image, start=start))
        if next(positions, None) is not None:
            raise ValueError("unexpected image placeholder outside image content")
        return DeepSeekV41Media(tuple(token_ids), tuple(placed), image_token_id=marker)


def make_preparer(
    model: str,
    tokenizer_or_processor: Any,
    *,
    eos_token_ids: Sequence[int] = (),
    vision_config: Mapping[str, Any] | None = None,
    image_token_id: int | None = None,
) -> _ProcessorPreparer | _DeepSeekPreparer:
    if model == "deepseek_v41":
        return _DeepSeekPreparer(tokenizer_or_processor, vision_config, image_token_id)
    if model not in {"gemma4", "qwen3_5", "qwen3_8", "inkling"}:
        raise ValueError(f"no media processor for {model}")
    return _ProcessorPreparer(model, tokenizer_or_processor)
