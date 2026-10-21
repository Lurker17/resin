"""Small shared semantic oracle and streaming driver for format adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from resin.protocols import FinalizingResponseParser, ParserEvent


@dataclass(frozen=True)
class Part:
    kind: Literal["text", "reasoning", "tool"]
    value: str | dict[str, Any]


@dataclass(frozen=True)
class Case:
    parts: tuple[Part, ...]


def normalize(parts: Sequence[Part]) -> list[Part]:
    """Ignore text chunk boundaries, preserving part order and JSON values."""
    result: list[Part] = []
    for part in parts:
        if part.kind != "tool" and not part.value:
            continue
        if result and part.kind != "tool" and result[-1].kind == part.kind:
            previous = result.pop()
            assert isinstance(previous.value, str) and isinstance(part.value, str)
            part = Part(part.kind, previous.value + part.value)
        result.append(part)
    return result


def semantic_events(events: Sequence[ParserEvent]) -> list[Part]:
    """Tool IDs are local identities; compare tool names and parsed JSON."""
    parts: list[Part] = []
    tools: dict[int, dict[str, Any]] = {}
    for event in events:
        match event["type"]:
            case "text_delta" | "reasoning_delta":
                kind = "text" if event["type"] == "text_delta" else "reasoning"
                parts.append(Part(kind, event["delta"]))
            case "tool_call_start":
                tool = {"name": event["name"], "args": ""}
                assert event["index"] not in tools
                tools[event["index"]] = tool
                parts.append(Part("tool", tool))
            case "tool_call_arguments_delta":
                tools[event["index"]]["args"] += event["delta"]
            case _:
                raise AssertionError(f"unexpected parser event: {event}")
    for tool in tools.values():
        tool["args"] = json.loads(tool["args"])
    return normalize(parts)


def assert_streams(
    create_parser: Callable[[], FinalizingResponseParser],
    token_ids: Sequence[int],
    expected: Sequence[Part],
    cuts: Sequence[int],
    *,
    reason: Literal["stop", "length"] = "stop",
) -> None:
    """Compare whole, token-at-a-time, and generated chunking to one oracle."""
    for schedule in ([len(token_ids)], range(len(token_ids) + 1), cuts):
        parser = create_parser()
        events = parser.feed([])
        previous = 0
        for end in sorted({*schedule, len(token_ids)}):
            end = min(max(end, previous), len(token_ids))
            events.extend(parser.feed(token_ids[previous:end]))
            previous = end
        events.extend(parser.feed([]))
        events.extend(parser.finish(reason))
        assert semantic_events(events) == normalize(expected)
        assert parser.finish(reason) == []
