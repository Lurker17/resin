"""Inkling's native conversation recipe with Python image preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from resin.models.inkling.media import InklingMedia
from resin.protocols import FinalizingResponseParser, RenderedPrompt
from resin.recipe import Recipe
from resin.server.media import prepare_recipe_inputs


class InklingCodec:
    """Render Inkling requests and parse reasoning, text, and tool-call blocks."""

    def __init__(self, processor_or_tokenizer: Any) -> None:
        self.processor = (
            processor_or_tokenizer
            if hasattr(processor_or_tokenizer, "tokenizer")
            else None
        )
        self.tokenizer = (
            processor_or_tokenizer.tokenizer
            if self.processor is not None
            else processor_or_tokenizer
        )
        self.recipe = Recipe(self.tokenizer, model="inkling")

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        if self.processor is None:
            return self.recipe.render(request)
        media = InklingMedia.from_processor_output(
            self.processor, prepare_recipe_inputs(self.processor, self.recipe, request)
        )
        rendered: RenderedPrompt = {
            "token_ids": list(media.token_ids),
            "stop_token_ids": self.recipe.get_stop_token_ids(),
        }
        if media.pixel_values is not None:
            rendered["media"] = media
        return rendered

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> FinalizingResponseParser:
        return self.recipe.create_parser(request, prompt_ids)


__all__ = ["InklingCodec"]
