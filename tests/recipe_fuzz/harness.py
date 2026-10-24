"""Small shared semantic oracle and streaming driver for format adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import resin
from resin.protocols import FinalizingResponseParser
from resin.recipe import ParserEvent


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


def semantic_events(
    events: Sequence[ParserEvent], *, parse_arguments: bool = True
) -> list[Part]:
    """Tool IDs are local identities; compare tool names and parsed JSON."""
    parts: list[Part] = []
    tools: dict[int, dict[str, Any]] = {}
    for event in events:
        match event:
            case resin.recipe.TextDelta(delta=delta):
                parts.append(Part("text", delta))
            case resin.recipe.ReasoningDelta(delta=delta):
                parts.append(Part("reasoning", delta))
            case resin.recipe.ToolCallStart(index=index, name=name):
                tool = {"name": name, "args": ""}
                assert index not in tools
                tools[index] = tool
                parts.append(Part("tool", tool))
            case resin.recipe.ToolCallArgumentsDelta(index=index, delta=delta):
                tools[index]["args"] += delta
            case _:
                raise AssertionError(f"unexpected parser event: {event}")
    if parse_arguments:
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
    float_ulps: int = 0,
) -> None:
    """Compare whole, token-at-a-time, and generated chunking to one oracle."""
    for schedule in ([len(token_ids)], range(len(token_ids) + 1), cuts):
        parser = create_parser()
        events = list(parser.feed([]))
        previous = 0
        for end in sorted({*schedule, len(token_ids)}):
            end = min(max(end, previous), len(token_ids))
            events.extend(parser.feed(token_ids[previous:end]))
            previous = end
        events.extend(parser.feed([]))
        events.extend(parser.finish(reason))
        assert_semantics(
            semantic_events(events), normalize(expected), float_ulps=float_ulps
        )
        assert parser.finish(reason) == []


def assert_chunk_invariance(create_parser, token_ids, cuts, *, reason="length"):
    """Partial tool arguments may be incomplete JSON, but cannot depend on chunks."""
    expected = None
    for schedule in ([len(token_ids)], range(len(token_ids) + 1), cuts):
        parser = create_parser()
        events = []
        previous = 0
        for end in sorted({*schedule, len(token_ids)}):
            end = min(max(end, previous), len(token_ids))
            events.extend(parser.feed(token_ids[previous:end]))
            previous = end
        events.extend(parser.finish(reason))
        actual = semantic_events(events, parse_arguments=False)
        if expected is None:
            expected = actual
        else:
            assert actual == expected
        assert parser.finish(reason) == []


def assert_semantics(actual, expected, *, float_ulps=0):
    """Primary oracles are exact; TML's serde float decoding may differ by 1 ULP."""

    def equal(a, b):
        if a == b:
            return True
        if float_ulps and isinstance(a, float) and isinstance(b, float):
            return abs(a - b) <= float_ulps * max(math.ulp(a), math.ulp(b))
        if isinstance(a, Part) and isinstance(b, Part):
            return a.kind == b.kind and equal(a.value, b.value)
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(
                equal(x, y) for x, y in zip(a, b, strict=True)
            )
        return False

    assert equal(actual, expected), (actual, expected)
