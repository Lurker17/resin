"""Backend-neutral DeepSeek-V4 rendering and response parsing."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any


def _thinking_settings(request: dict[str, Any]) -> tuple[str, str]:
    extra = request.get("extra") or {}
    configured_mode = extra.get("thinking_mode")
    anthropic_thinking = extra.get("thinking") or {}
    reasoning_effort = request.get("reasoning_effort")
    if configured_mode is None:
        configured_mode = (
            "thinking"
            if reasoning_effort or anthropic_thinking.get("type") == "enabled"
            else "chat"
        )
    if configured_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")

    effort_aliases = {
        None: "low",
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
    try:
        effort = effort_aliases[reasoning_effort]
    except KeyError as error:
        raise ValueError(
            f"unsupported reasoning_effort: {reasoning_effort!r}"
        ) from error
    return configured_mode, effort


class DeepSeekV4Codec:
    """Adapt DeepSeek's pinned encoding module to ``HttpServer`` requests."""

    def __init__(self, tokenizer: Any, encoding: Any) -> None:
        self._tokenizer = tokenizer
        self._encoding = encoding

    def render(self, request: dict[str, Any]) -> dict[str, list[int]]:
        messages = copy.deepcopy(request["messages"])
        tools = request.get("tools") or []
        extra = request.get("extra") or {}
        if tools or extra.get("response_format"):
            if messages and messages[0].get("role") == "system":
                system = messages[0]
            else:
                system = {"role": "system", "content": ""}
                messages.insert(0, system)
            if tools:
                system["tools"] = [
                    {"type": "function", "function": tool} for tool in tools
                ]
            if response_format := extra.get("response_format"):
                system["response_format"] = response_format

        thinking_mode, reasoning_effort = _thinking_settings(request)
        rendered = self._encoding.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        token_ids = [int(token_id) for token_id in self._tokenizer.encode(rendered)]
        if not token_ids:
            raise ValueError("conversation produced no tokens")

        eos_token_id = self._tokenizer.convert_tokens_to_ids(self._encoding.eos_token)
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise ValueError("DeepSeek-V4 EOS token did not resolve to one token ID")
        stop_token_ids = [eos_token_id]
        for stop in request.get("stop") or []:
            encoded = list(self._tokenizer.encode(stop, add_special_tokens=False))
            if len(encoded) != 1:
                raise ValueError(
                    f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                    "multi-token stop sequences are not supported yet"
                )
            stop_token_ids.append(int(encoded[0]))
        return {
            "token_ids": token_ids,
            "stop_token_ids": list(dict.fromkeys(stop_token_ids)),
        }

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> DeepSeekV4ResponseParser:
        del prompt_ids
        thinking_mode, _ = _thinking_settings(request)
        return DeepSeekV4ResponseParser(
            self._tokenizer,
            self._encoding,
            thinking_mode=thinking_mode,
        )


class DeepSeekV4ResponseParser:
    """Incrementally parse text, reasoning, and complete DSML tool calls."""

    def __init__(self, tokenizer: Any, encoding: Any, *, thinking_mode: str) -> None:
        self._tokenizer = tokenizer
        self._encoding = encoding
        self._thinking_mode = thinking_mode
        self._token_ids: list[int] = []
        self._reasoning = ""
        self._content = ""
        self._tools_emitted = False

    def feed(self, token_ids: Sequence[int]) -> list[dict[str, Any]]:
        self._token_ids.extend(token_ids)
        text = self._tokenizer.decode(self._token_ids, skip_special_tokens=False)
        events: list[dict[str, Any]] = []
        if self._encoding.eos_token in text:
            parsed = self._encoding.parse_message_from_completion_text(
                text, thinking_mode=self._thinking_mode
            )
            reasoning = self._stable_text(parsed.get("reasoning_content") or "")
            content = self._stable_text(parsed.get("content") or "")
            self._append_delta(events, "reasoning_delta", self._reasoning, reasoning)
            self._append_delta(events, "text_delta", self._content, content)
            self._reasoning = reasoning
            self._content = content
            if not self._tools_emitted:
                events.extend(self._tool_events(parsed.get("tool_calls") or []))
                self._tools_emitted = True
            return events

        remainder = text
        reasoning = ""
        if self._thinking_mode == "thinking":
            end = remainder.find(self._encoding.thinking_end_token)
            if end < 0:
                reasoning = self._stable_before_markers(
                    remainder, (self._encoding.thinking_end_token,)
                )
                remainder = ""
            else:
                reasoning = remainder[:end]
                remainder = remainder[end + len(self._encoding.thinking_end_token) :]

        tool_calls_start = (
            f"\n\n<{self._encoding.dsml_token}{self._encoding.tool_calls_block_name}"
        )
        content = self._stable_before_markers(
            remainder,
            (tool_calls_start, self._encoding.eos_token),
        )
        reasoning = self._stable_text(reasoning)
        content = self._stable_text(content)
        self._append_delta(events, "reasoning_delta", self._reasoning, reasoning)
        self._append_delta(events, "text_delta", self._content, content)
        self._reasoning = reasoning
        self._content = content
        return events

    @staticmethod
    def _stable_text(value: str) -> str:
        return value.rstrip("\N{REPLACEMENT CHARACTER}")

    @staticmethod
    def _stable_before_markers(value: str, markers: tuple[str, ...]) -> str:
        positions = [
            position for marker in markers if (position := value.find(marker)) >= 0
        ]
        if positions:
            return value[: min(positions)]
        held = max(
            (
                length
                for marker in markers
                for length in range(1, min(len(marker), len(value)) + 1)
                if value.endswith(marker[:length])
            ),
            default=0,
        )
        return value[:-held] if held else value

    @staticmethod
    def _append_delta(
        events: list[dict[str, Any]],
        event_type: str,
        previous: str,
        current: str,
    ) -> None:
        if current == previous:
            return
        if not current.startswith(previous):
            raise ValueError(f"DeepSeek-V4 {event_type} output is not prefix-stable")
        events.append({"type": event_type, "delta": current[len(previous) :]})

    @staticmethod
    def _tool_events(tool_calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        events = []
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            events.append(
                {
                    "type": "tool_call_start",
                    "index": index,
                    "id": tool_call.get("id") or f"call_{index}",
                    "name": name,
                }
            )
            arguments = function.get("arguments") or ""
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )
            if arguments:
                events.append(
                    {
                        "type": "tool_call_arguments_delta",
                        "index": index,
                        "delta": arguments,
                    }
                )
        return events


__all__ = ["DeepSeekV4Codec", "DeepSeekV4ResponseParser"]
