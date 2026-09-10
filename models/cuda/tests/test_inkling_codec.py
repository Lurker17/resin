from __future__ import annotations

from typing import Any

import pytest

import resin.models.inkling.codec as codec_module
from resin.models.inkling.codec import InklingCodec

SPECIAL_TOKENS = {
    "<|message_model|>": 200001,
    "<|content_text|>": 200004,
    "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008,
    "<|end_message|>": 200010,
    "<|content_tool_error|>": 200022,
    "<|content_invoke_tool_json|>": 200049,
    "<|content_invoke_tool_text|>": 200057,
}


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIAL_TOKENS[token]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {value: key for key, value in SPECIAL_TOKENS.items()}[token_id]

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return _text_ids(text)

    @staticmethod
    def decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _text_ids(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_render_forwards_reasoning_effort_and_uses_inkling_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Renderer:
        @staticmethod
        def render_ids(messages, *, tools, add_generation_prompt):
            seen.update(
                messages=messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
            )
            return [4, 5]

    def create_renderer(tokenizer, config):
        seen["tokenizer"] = tokenizer
        seen["config"] = config
        return Renderer()

    monkeypatch.setattr(codec_module, "create_renderer", create_renderer)
    tokenizer = FakeTokenizer()
    codec = InklingCodec(tokenizer)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "weather"}}]

    rendered = codec.render(
        {
            "messages": messages,
            "tools": tools,
            "reasoning_effort": "low",
            "stop": ["!"],
        }
    )

    assert seen["tokenizer"] is tokenizer
    assert seen["config"].model_extra == {"reasoning_effort": "low"}
    assert seen["messages"] == messages
    assert seen["tools"] == tools
    assert seen["add_generation_prompt"] is True
    assert rendered == {"token_ids": [4, 5], "stop_token_ids": [200006, 33]}


def test_render_maps_openai_xhigh_to_inkling_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Renderer:
        @staticmethod
        def render_ids(messages, *, tools, add_generation_prompt):
            del messages, tools, add_generation_prompt
            return []

    def create_renderer(tokenizer, config):
        del tokenizer
        seen["config"] = config
        return Renderer()

    monkeypatch.setattr(codec_module, "create_renderer", create_renderer)
    InklingCodec(FakeTokenizer()).render(
        {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "xhigh"}
    )

    assert seen["config"].model_extra == {"reasoning_effort": "max"}


def test_render_rejects_unknown_reasoning_effort() -> None:
    codec = InklingCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="unsupported reasoning_effort"):
        codec.render(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "turbo",
            }
        )


def test_parser_separates_repeated_reasoning_and_visible_text_blocks() -> None:
    codec = InklingCodec(FakeTokenizer())
    parser = codec.create_parser({}, [200001])

    assert parser.feed([200008, *_text_ids("think")]) == [
        {"type": "reasoning_delta", "delta": "think"}
    ]
    assert parser.feed([200010, 200001, 200004, *_text_ids("Hello")]) == [
        {"type": "text_delta", "delta": "Hello"}
    ]
    assert parser.feed(
        [
            200010,
            200001,
            200008,
            *_text_ids(" again"),
            200010,
            200001,
            200004,
            *_text_ids("!"),
            200010,
            200006,
        ]
    ) == [
        {"type": "reasoning_delta", "delta": " again"},
        {"type": "text_delta", "delta": "!"},
    ]


def test_parser_uses_prompt_tail_to_continue_reasoning() -> None:
    codec = InklingCodec(FakeTokenizer())
    parser = codec.create_parser({}, [200001, 200008])

    assert parser.feed(_text_ids("continued")) == [
        {"type": "reasoning_delta", "delta": "continued"}
    ]


def test_parser_extracts_json_tool_call_and_hides_header_metadata() -> None:
    codec = InklingCodec(FakeTokenizer())
    parser = codec.create_parser({}, [200001])
    payload = '{"name":"weather","args":{"city":"SF"}}'

    assert parser.feed(
        [
            *_text_ids("weather"),
            200049,
            *_text_ids(payload),
            200010,
            200006,
        ]
    ) == [
        {
            "type": "tool_call_start",
            "index": 0,
            "id": "call_0",
            "name": "weather",
        },
        {
            "type": "tool_call_arguments_delta",
            "index": 0,
            "delta": '{"city":"SF"}',
        },
    ]


def test_raw_tool_and_tool_error_blocks_remain_visible() -> None:
    codec = InklingCodec(FakeTokenizer())
    parser = codec.create_parser({}, [200001])

    assert parser.feed([200057, *_text_ids("raw"), 200010]) == [
        {"type": "text_delta", "delta": "raw"}
    ]
    assert parser.feed([200001, 200022, *_text_ids("error"), 200010]) == [
        {"type": "text_delta", "delta": "error"}
    ]
