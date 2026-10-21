"""Qwen3.5's native conversation recipe with Python image preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from resin.models.qwen3_5.media import Qwen35Media
from resin.protocols import FinalizingResponseParser, RenderedPrompt
from resin.recipe import Recipe
from resin.server import encode_stop_tokens
from resin.server.media import prepare_recipe_inputs


class Qwen35Codec:
    """Render Qwen text/images and incrementally parse reasoning and tools."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.recipe = Recipe(self.tokenizer, model="qwen3_5")
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.spatial_merge_size = int(processor.image_processor.merge_size)

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        output = prepare_recipe_inputs(self.processor, self.recipe, request)
        if output.get("pixel_values_videos") is not None:
            raise ValueError("Qwen35Media currently supports image input only")
        token_ids = output["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if len(token_ids) != 1:
            raise ValueError("Qwen codec expects one conversation")
        result: RenderedPrompt = {
            "token_ids": [int(token) for token in token_ids[0]],
            "stop_token_ids": encode_stop_tokens(
                self.tokenizer,
                self.recipe.get_stop_token_ids(),
                request.get("stop") or [],
            ),
        }
        if output.get("pixel_values") is not None:
            result["media"] = Qwen35Media.from_processor_output(
                output,
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.image_token_id,
            )
        return result

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> FinalizingResponseParser:
        return self.recipe.create_parser(request, prompt_ids)


__all__ = ["Qwen35Codec"]
