"""Native model conversation formats and incremental response parsers.

Formatting and parsing live in Rust. Python owns processor integration and the
structural codec protocol, so applications may still supply their own codec.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from resin._resin import InklingResponseParser, ModelRecipe, RecipeResponseParser
from resin._resin import RecipeTokenizer as Tokenizer
from resin.protocols import RenderedPrompt


class Recipe:
    """Prepare a model format and tokenizer once, then render and parse requests."""

    def __init__(
        self, tokenizer: Any, *, model: str, options: Mapping[str, Any] | None = None
    ) -> None:
        self.tokenizer = tokenizer
        defaults = dict(options or {})
        if model == "gpt_oss":
            defaults.setdefault("conversation_start_date", date.today().isoformat())
        name = str(getattr(tokenizer, "name_or_path", "")).lower()
        if model == "qwen3_5" and "qwen3.5" in name:
            defaults.setdefault("recipe_variant", "qwen3_5")
            if re.search(r"qwen3\.5-(?:0\.8b|2b)(?:-|$)", name):
                defaults.setdefault("enable_thinking", False)
        self._native = ModelRecipe(Tokenizer(tokenizer), model=model, options=defaults)

    def render_text(self, request: dict[str, Any]) -> str:
        return self._native.render_text(request)

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        return {
            "token_ids": self._native.render_ids(request),
            "stop_token_ids": self._native.get_stop_token_ids(),
        }

    def render_ids(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        return self._native.render_ids(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "add_generation_prompt": add_generation_prompt,
            }
        )

    def get_stop_token_ids(self) -> list[int]:
        return self._native.get_stop_token_ids()

    def render_media_ids(
        self, request: dict[str, Any], replacements: dict[str, list[str]]
    ) -> list[int]:
        """Expand explicit media slots while keeping ordinary prompt text literal."""
        return self._native.render_ids(request, media_replacements=replacements)

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> RecipeResponseParser:
        # Media payloads belong to processors, not response parser state.
        controls = {
            key: value
            for key, value in request.items()
            if key in {"tools", "reasoning_effort", "extra"}
        }
        return self._native.create_parser(controls, list(prompt_ids))

    def parse_response(
        self,
        token_ids: Sequence[int],
        request: dict[str, Any] | None = None,
        *,
        prompt_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Collect the same semantic events used by the streaming server."""
        request = request or {}
        if prompt_ids is None:
            prompt_ids = (
                self.render(request)["token_ids"] if "messages" in request else []
            )
        parser = self.create_parser(request, prompt_ids)
        events = [*parser.feed(token_ids), *parser.finish("stop")]
        result: dict[str, Any] = {
            "content": "",
            "reasoning_content": "",
            "tool_calls": [],
        }
        tools: dict[int, dict[str, Any]] = {}
        for event in events:
            match event["type"]:
                case "text_delta":
                    result["content"] += event["delta"]
                case "reasoning_delta":
                    result["reasoning_content"] += event["delta"]
                case "tool_call_start":
                    tool = {"id": event["id"], "name": event["name"], "arguments": ""}
                    tools[event["index"]] = tool
                    result["tool_calls"].append(tool)
                case "tool_call_arguments_delta":
                    tools[event["index"]]["arguments"] += event["delta"]
        return result


__all__ = ["InklingResponseParser", "Recipe", "RecipeResponseParser", "Tokenizer"]
