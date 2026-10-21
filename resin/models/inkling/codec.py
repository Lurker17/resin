"""OpenAI conversation codec for Inkling's typed response blocks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from renderers import DefaultRendererConfig, create_renderer

from resin.models.inkling.media import InklingMedia
from resin.protocols import ParserEvent, RenderedPrompt
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
            self.tokenizer,
            self.token_ids,
            request.get("tools") or None,
            prompt_ids=prompt_ids,
        )


BlockState = Literal["content", "header", "reasoning", "tool"]


class InklingResponseParser:
    """Incrementally parse Inkling's self-describing output block grammar."""

    def __init__(
        self,
        tokenizer: Any,
        token_ids: dict[str, int],
        tools: list[dict[str, Any]] | None,
        *,
        prompt_ids: Sequence[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._ids = token_ids
        self._tools = tools or []
        self._state: BlockState = self._initial_state(prompt_ids)
        self._active_ids: list[int] = []
        self._active_text = ""
        self._tool_index = 0

    def _initial_state(self, prompt_ids: Sequence[int]) -> BlockState:
        relevant = set(self._ids.values())
        for token_id in reversed(prompt_ids):
            if token_id == self._ids[CONTENT_THINKING]:
                return "reasoning"
            if token_id == self._ids[CONTENT_TEXT]:
                return "content"
            if token_id == self._ids[MESSAGE_MODEL]:
                return "header"
            if token_id in relevant:
                break
        return "header"

    def feed(self, token_ids: Sequence[int]) -> list[ParserEvent]:
        events: list[ParserEvent] = []
        for token_id in token_ids:
            if token_id == self._ids[MESSAGE_MODEL]:
                self._finish_block(events)
                self._start("header")
            elif token_id == self._ids[CONTENT_THINKING]:
                self._finish_block(events)
                self._start("reasoning")
            elif token_id == self._ids[CONTENT_TEXT]:
                self._finish_block(events)
                self._start("content")
            elif token_id == self._ids[CONTENT_INVOKE_TOOL_JSON]:
                self._finish_block(events)
                self._start("tool")
            elif token_id in (
                self._ids[CONTENT_INVOKE_TOOL_TEXT],
                self._ids[CONTENT_TOOL_ERROR],
            ):
                self._finish_block(events)
                self._start("content")
            elif token_id in (
                self._ids[END_MESSAGE],
                self._ids[CONTENT_MODEL_END_SAMPLING],
            ):
                self._finish_block(events)
                self._start("content")
            elif self._state != "header":
                self._active_ids.append(int(token_id))
        self._emit_text(events)
        return events

    def _start(self, state: BlockState) -> None:
        self._state = state
        self._active_ids.clear()
        self._active_text = ""

    def _finish_block(self, events: list[ParserEvent]) -> None:
        if self._state == "tool" and self._active_ids:
            self._emit_tool(events)
        else:
            self._emit_text(events)

    @staticmethod
    def _stable_text(value: str) -> str:
        return value.rstrip("\N{REPLACEMENT CHARACTER}")

    def _decode_active(self) -> str:
        return self._stable_text(
            self._tokenizer.decode(self._active_ids, skip_special_tokens=False)
        )

    def _emit_text(self, events: list[ParserEvent]) -> None:
        if self._state not in ("content", "reasoning") or not self._active_ids:
            return
        current = self._decode_active()
        if not current.startswith(self._active_text):
            raise ValueError("Inkling response text is not prefix-stable")
        delta = current[len(self._active_text) :]
        if delta:
            events.append(
                {
                    "type": (
                        "reasoning_delta"
                        if self._state == "reasoning"
                        else "text_delta"
                    ),
                    "delta": delta,
                }
            )
        self._active_text = current

    def _emit_tool(self, events: list[ParserEvent]) -> None:
        raw = self._decode_active()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Inkling emitted an invalid JSON tool call") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ValueError("Inkling tool call must contain a string name")
        arguments = payload.get("args", payload.get("arguments", {}))
        if not isinstance(arguments, dict):
            raise ValueError("Inkling tool call arguments must be an object")
        name = payload["name"]
        index = self._tool_index
        events.append(
            {
                "type": "tool_call_start",
                "index": index,
                "id": f"call_{index}",
                "name": name,
            }
        )
        events.append(
            {
                "type": "tool_call_arguments_delta",
                "index": index,
                "delta": json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        self._tool_index += 1


__all__ = ["InklingCodec", "InklingResponseParser"]
