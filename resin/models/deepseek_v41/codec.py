"""The 1017 V4.1 recipe, distinct from the earlier V4 checkpoint grammar."""

from collections.abc import Mapping, Sequence
from typing import Any

from resin.models.deepseek_v41.media import (
    IMAGE_TOKEN,
    DeepSeekV41ImageProcessor,
    DeepSeekV41Media,
)
from resin.protocols import RenderedPrompt
from resin.recipe import Recipe
from resin.server.media import normalize_image_messages


class DeepSeekV41Codec(Recipe):
    def __init__(
        self,
        tokenizer: Any,
        *,
        vision_config: Mapping[str, Any] | None = None,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__(tokenizer, model="deepseek_v41")
        self.image_processor = DeepSeekV41ImageProcessor(
            **{
                key: value
                for key, value in (vision_config or {}).items()
                if key in DeepSeekV41ImageProcessor.__dataclass_fields__
            }
        )
        self.image_token_id = image_token_id

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        if not _has_images(request):
            return super().render(request)
        marker = self._image_token_id()
        messages = normalize_image_messages(request["messages"])
        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for index, block in enumerate(content):
                if block.get("type") == "image":
                    if message.get("role") != "user":
                        raise ValueError("DeepSeek V4.1 images belong in user messages")
                    images.append(block["image"])
                    content[index] = {"type": "image"}
        counts = [
            self.image_processor.token_count(image.shape[1], image.shape[0])
            for image in images
        ]
        ids = self.render_media_ids(
            {**request, "messages": messages},
            {IMAGE_TOKEN: [IMAGE_TOKEN * count for count in counts]},
        )
        # Only trusted recipe slots encode to the marker ID. Literal spellings
        # in user text pass through the native encoder's ordinary-token path.
        positions = iter(index for index, token in enumerate(ids) if token == marker)
        placed = []
        for image, count in zip(images, counts, strict=True):
            start = next(positions, None)
            if start is None:
                raise ValueError("missing expanded image token span")
            for expected in range(start + 1, start + count):
                if next(positions, None) != expected:
                    raise ValueError("image token span must be contiguous")
            placed.append(self.image_processor.prepare(image, start=start))
        if next(positions, None) is not None:
            raise ValueError("unexpected image placeholder outside image content")
        media = DeepSeekV41Media(tuple(ids), tuple(placed), image_token_id=marker)
        return {
            "token_ids": ids,
            "stop_token_ids": self.get_stop_token_ids(),
            "media": media,
        }

    def render_ids(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        return self.render(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "add_generation_prompt": add_generation_prompt,
            }
        )["token_ids"]

    def _image_token_id(self) -> int:
        backend = getattr(self.tokenizer, "backend_tokenizer", self.tokenizer)
        lookup = getattr(backend, "token_to_id", None)
        marker = lookup(IMAGE_TOKEN) if lookup is not None else None
        if marker is None:
            lookup = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            marker = lookup(IMAGE_TOKEN) if lookup is not None else None
        if marker is None:
            raise ValueError(f"tokenizer must define image token {IMAGE_TOKEN}")
        if self.image_token_id is not None and marker != self.image_token_id:
            raise ValueError(
                "tokenizer image token does not match checkpoint image_token_id"
            )
        return int(marker)


def _has_images(request: dict[str, Any]) -> bool:
    return any(
        isinstance(content := message.get("content"), list)
        and any(
            isinstance(block, dict)
            and block.get("type") in {"image", "image_url", "input_image"}
            for block in content
        )
        for message in request.get("messages", ())
    )


__all__ = ["DeepSeekV41Codec"]
