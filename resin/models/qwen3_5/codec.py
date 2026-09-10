"""OpenAI conversation adapter for Qwen3.5 image prompts."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from resin.models.qwen3_5.media import Qwen35Media
from resin.protocols import RenderedPrompt, ResponseParser
from resin.server import RendererCodec
from resin.server.media import normalize_image_messages


class Qwen35Codec:
    """Render Qwen text/images while retaining shared reasoning/tool parsing."""

    def __init__(self, renderer: Any, processor: Any) -> None:
        self.renderer = renderer
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.spatial_merge_size = int(processor.image_processor.merge_size)
        self._base = RendererCodec(
            renderer,
            tokenizer=self.tokenizer,
            include_generation_prefix_in_parser=True,
            generation_prefix_parser_anchor_token_id=int(
                self.tokenizer.convert_tokens_to_ids("<think>")
            ),
        )

    @staticmethod
    def _request_with_pil_images(request: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(request)
        messages = normalize_image_messages(request["messages"])
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "image":
                    continue
                image = block["image"]
                pixels = np.rint(image * np.float32(255.0)).astype(np.uint8)
                block["image"] = Image.fromarray(pixels, mode="RGB")
        normalized["messages"] = messages
        return normalized

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        normalized = self._request_with_pil_images(request)
        rendered = self.renderer.render(
            normalized["messages"],
            tools=normalized.get("tools") or None,
            add_generation_prompt=True,
        )
        stop_token_ids = list(self.renderer.get_stop_token_ids())
        for stop in normalized.get("stop") or []:
            encoded = list(self.tokenizer.encode(stop, add_special_tokens=False))
            if len(encoded) != 1:
                raise ValueError(
                    f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                    "multi-token stop sequences are not supported yet"
                )
            stop_token_ids.append(int(encoded[0]))
        result: RenderedPrompt = {
            "token_ids": list(rendered.token_ids),
            "stop_token_ids": list(dict.fromkeys(stop_token_ids)),
        }
        if rendered.multi_modal_data is not None:
            result["media"] = Qwen35Media.from_rendered(
                rendered,
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.image_token_id,
            )
        return result

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> ResponseParser:
        return self._base.create_parser(
            self._request_with_pil_images(request), prompt_ids
        )


__all__ = ["Qwen35Codec"]
