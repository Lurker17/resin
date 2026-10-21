"""HTTP serving for a token-native Resin engine.

The networking, scheduling bridge, wire protocols, and streaming adapters run
in Rust. This module supplies the small Python-side model-format adapter needed
by renderers whose tokenizer and response parser live in Python.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from resin._resin import Engine, _HttpServer
from resin.protocols import (
    ConversationCodec,
    ParserEvent,
    RenderedPrompt,
    ResponseParser,
)

if TYPE_CHECKING:
    from renderers import Renderer


def encode_stop_tokens(
    tokenizer: Any, defaults: Sequence[int], stop_strings: Sequence[str]
) -> list[int]:
    """Combine model stops with tokenized user stops, preserving their order."""

    token_ids = list(defaults)
    for stop in stop_strings:
        encoded = list(tokenizer.encode(stop, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                "multi-token stop sequences are not supported yet"
            )
        token_ids.append(int(encoded[0]))
    return list(dict.fromkeys(token_ids))


class RendererCodec(ConversationCodec):
    """Adapt a ``renderers.Renderer`` to Resin's streaming server contract.

    ``renderers`` 0.1.9 exposes whole-response parsing only. This adapter
    reparses accumulated tokens on each feed and emits newly available fields;
    long responses can therefore require quadratic parsing work on the engine
    actor. A codec with its own incremental ``ResponseParser`` can avoid that
    cost. There is no upstream incremental parser or final-flush API to delegate
    to in the pinned dependency.
    """

    def __init__(
        self,
        renderer: Renderer,
        *,
        tokenizer: Any | None = None,
        include_generation_prefix_in_parser: bool = False,
        generation_prefix_parser_anchor_token_id: int | None = None,
    ) -> None:
        self.renderer = renderer
        self.tokenizer = tokenizer or getattr(renderer, "_tokenizer", None)
        self.include_generation_prefix_in_parser = include_generation_prefix_in_parser
        self.generation_prefix_parser_anchor_token_id = (
            generation_prefix_parser_anchor_token_id
        )

    def render(self, request: dict[str, Any]) -> RenderedPrompt:
        tools = request.get("tools") or None
        token_ids = self.renderer.render_ids(
            request["messages"], tools=tools, add_generation_prompt=True
        )
        stop_strings = request.get("stop") or []
        if stop_strings and self.tokenizer is None:
            raise ValueError(
                "string stop sequences require RendererCodec(tokenizer=...)"
            )
        return {
            "token_ids": list(token_ids),
            "stop_token_ids": encode_stop_tokens(
                self.tokenizer, self.renderer.get_stop_token_ids(), stop_strings
            ),
        }

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> ResponseParser:
        tools = request.get("tools") or None
        prefix_ids: list[int] = []
        if self.include_generation_prefix_in_parser:
            conversation_ids = list(
                self.renderer.render_ids(
                    request["messages"], tools=tools, add_generation_prompt=False
                )
            )
            if list(prompt_ids[: len(conversation_ids)]) != conversation_ids:
                raise ValueError(
                    "renderer generation prompt is not a conversation suffix"
                )
            prefix_ids = list(prompt_ids[len(conversation_ids) :])
            anchor = self.generation_prefix_parser_anchor_token_id
            if anchor is not None:
                try:
                    prefix_ids = prefix_ids[prefix_ids.index(anchor) :]
                except ValueError as error:
                    raise ValueError(
                        "renderer generation prompt is missing its parser anchor"
                    ) from error
        return _RendererResponseParser(self.renderer, tools, prefix_ids=prefix_ids)


class _RendererResponseParser:
    def __init__(
        self,
        renderer: Any,
        tools: list[dict[str, Any]] | None,
        *,
        prefix_ids: Sequence[int] = (),
    ) -> None:
        self._renderer = renderer
        self._tools = tools
        self._token_ids = list(prefix_ids)
        self._reasoning = ""
        self._content = ""
        self._tool_arguments: list[str] = []
        self._started_tools: set[int] = set()

    def feed(self, token_ids: Sequence[int]) -> list[ParserEvent]:
        self._token_ids.extend(token_ids)
        parsed = self._renderer.parse_response(self._token_ids, tools=self._tools)
        events: list[ParserEvent] = []
        reasoning = self._stable_text(parsed.reasoning_content or "")
        content = self._stable_text(parsed.content or "")
        self._append_delta(events, "reasoning_delta", self._reasoning, reasoning)
        self._append_delta(events, "text_delta", self._content, content)
        self._reasoning = reasoning
        self._content = content

        for index, tool in enumerate(parsed.tool_calls):
            status = getattr(getattr(tool, "status", None), "name", None)
            if status != "OK" or not tool.name:
                continue
            if index not in self._started_tools:
                events.append(
                    {
                        "type": "tool_call_start",
                        "index": index,
                        "id": tool.id or f"call_{index}",
                        "name": tool.name,
                    }
                )
                self._started_tools.add(index)
            arguments = (
                tool.arguments
                if isinstance(tool.arguments, str)
                else json.dumps(
                    tool.arguments, ensure_ascii=False, separators=(",", ":")
                )
            )
            arguments = self._stable_text(arguments)
            previous = (
                self._tool_arguments[index] if index < len(self._tool_arguments) else ""
            )
            while len(self._tool_arguments) <= index:
                self._tool_arguments.append("")
            self._append_delta(
                events,
                "tool_call_arguments_delta",
                previous,
                arguments,
                index=index,
            )
            self._tool_arguments[index] = arguments
        return events

    @staticmethod
    def _stable_text(value: str) -> str:
        # Byte-fallback tokenizers can temporarily decode an incomplete UTF-8
        # sequence as U+FFFD, then replace it once the next token arrives.
        return value.rstrip("\N{REPLACEMENT CHARACTER}")

    @staticmethod
    def _append_delta(
        events: list[ParserEvent],
        event_type: str,
        previous: str,
        current: str,
        *,
        index: int | None = None,
    ) -> None:
        if current == previous:
            return
        if not current.startswith(previous):
            raise ValueError(
                f"response parser field {event_type!r} is not prefix-stable"
            )
        event: ParserEvent = {"type": event_type, "delta": current[len(previous) :]}
        if index is not None:
            event["index"] = index
        events.append(event)


class HttpServer:
    """Serve one Resin Engine through OpenAI, Anthropic, and demo routes."""

    def __init__(
        self,
        engine: Engine,
        *,
        renderer: Renderer | None = None,
        codec: ConversationCodec | None = None,
        model: str | None = None,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
    ) -> None:
        if (renderer is None) == (codec is None):
            raise TypeError("pass exactly one of renderer or codec")
        if codec is None:
            assert renderer is not None
            codec = RendererCodec(renderer)
        if model is None:
            tokenizer = getattr(renderer, "_tokenizer", None)
            model = getattr(tokenizer, "name_or_path", None)
        if not model:
            raise ValueError(
                "model is required when it cannot be inferred from the renderer"
            )
        self._native = _HttpServer(
            engine,
            codec,
            model=model,
            enable_openai=enable_openai,
            enable_anthropic=enable_anthropic,
            enable_demo=enable_demo,
        )

    @property
    def model(self) -> str:
        return self._native.model

    @property
    def running(self) -> bool:
        return self._native.running

    def serve(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Block while serving, until Ctrl-C or another thread calls shutdown."""

        self._native.serve(host=host, port=port)

    def shutdown(self) -> None:
        """Request shutdown after the current inference step finishes.

        This does not block the caller. Join the thread running ``serve()`` to
        wait for shutdown; active requests are cancelled and the Engine closes.
        """

        self._native.shutdown()


__all__ = [
    "ConversationCodec",
    "encode_stop_tokens",
    "HttpServer",
    "ParserEvent",
    "RenderedPrompt",
    "RendererCodec",
    "ResponseParser",
]
