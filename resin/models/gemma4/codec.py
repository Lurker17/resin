"""Gemma 4's native conversation recipe with Python media preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from resin.models.gemma4.media import Gemma4Media
from resin.protocols import FinalizingResponseParser, RenderedPrompt
from resin.recipe import Recipe
from resin.server import encode_stop_tokens
from resin.server.media import prepare_recipe_inputs


class Gemma4Codec:
    """Render HTTP conversations into one opaque ``Gemma4Media`` request."""

    def __init__(self, processor: Any, *, eos_token_ids: Sequence[int]) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.eos_token_ids = tuple(int(token) for token in eos_token_ids)
        self.recipe = Recipe(self.tokenizer, model="gemma4")

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        media = Gemma4Media.from_processor_output(
            prepare_recipe_inputs(self.processor, self.recipe, request)
        )
        return {
            "token_ids": list(media.token_ids),
            "stop_token_ids": encode_stop_tokens(
                self.tokenizer, self.eos_token_ids, request.get("stop") or []
            ),
            "media": media,
        }

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> FinalizingResponseParser:
        return self.recipe.create_parser(request, prompt_ids)


__all__ = ["Gemma4Codec"]
