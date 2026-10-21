"""OpenAI conversation codec for Inkling's typed response blocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from renderers import DefaultRendererConfig, create_renderer

from resin.models.inkling.media import InklingMedia
from resin.protocols import RenderedPrompt
from resin.recipe import InklingResponseParser, Tokenizer
from resin.server import encode_stop_tokens
from resin.server.media import normalize_image_messages

MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
END_MESSAGE = "<|end_message|>"

_TOKEN_NAMES = (
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_TOOL_ERROR,
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
)
_EFFORT_ALIASES = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "none": "none",
}


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Inkling special token {token!r} has no integer ID")
    if tokenizer.convert_ids_to_tokens(token_id) != token:
        raise ValueError(f"Inkling special token {token!r} did not round-trip")
    return token_id


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
        self.token_ids = {
            token: _token_id(self.tokenizer, token) for token in _TOKEN_NAMES
        }
        if len(set(self.token_ids.values())) != len(self.token_ids):
            raise ValueError("Inkling response markers must have distinct token IDs")
        self._recipe_tokenizer = Tokenizer(self.tokenizer)

    @staticmethod
    def _reasoning_effort(request: dict[str, Any]) -> str | None:
        effort = request.get("reasoning_effort")
        if effort is None:
            return None
        if not isinstance(effort, str):
            raise ValueError("reasoning_effort must be a string")
        try:
            return _EFFORT_ALIASES[effort]
        except KeyError as error:
            raise ValueError(f"unsupported reasoning_effort: {effort!r}") from error

    def _renderer(self, request: dict[str, Any]):
        effort = self._reasoning_effort(request)
        config = (
            DefaultRendererConfig()
            if effort is None
            else DefaultRendererConfig(reasoning_effort=effort)
        )
        return create_renderer(self.tokenizer, config)

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        effort = self._reasoning_effort(request)
        rendered: RenderedPrompt
        if self.processor is not None:
            messages = normalize_image_messages(request["messages"])
            media = InklingMedia.from_messages(
                self.processor,
                messages,
                tools=request.get("tools") or None,
                reasoning_effort=effort,
            )
            rendered = {
                "token_ids": list(media.token_ids),
                "stop_token_ids": [self.token_ids[CONTENT_MODEL_END_SAMPLING]],
            }
            if media.pixel_values is not None:
                rendered["media"] = media
        else:
            renderer = self._renderer(request)
            token_ids = renderer.render_ids(
                request["messages"],
                tools=request.get("tools") or None,
                add_generation_prompt=True,
            )
            rendered = {
                "token_ids": list(token_ids),
                "stop_token_ids": [self.token_ids[CONTENT_MODEL_END_SAMPLING]],
            }
        rendered["stop_token_ids"] = encode_stop_tokens(
            self.tokenizer, rendered["stop_token_ids"], request.get("stop") or []
        )
        return rendered

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> InklingResponseParser:
        return InklingResponseParser(
            self._recipe_tokenizer,
            self.token_ids,
            prompt_ids=prompt_ids,
        )


__all__ = ["InklingCodec", "InklingResponseParser"]
