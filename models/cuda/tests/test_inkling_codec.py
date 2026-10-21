from __future__ import annotations

from typing import Any, Literal

import pytest

import resin.models.inkling.codec as codec_module
from resin.models.inkling.codec import InklingCodec
from resin.protocols import ConversationCodec, FinalizingResponseParser

SPECIAL_TOKENS = {
    "<|message_model|>": 200001,
    "<|message_user|>": 200000,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
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


def test_render_forwards_request_to_native_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class RecordingRecipe:
        def __init__(self, tokenizer, *, model):
            seen.update(tokenizer=tokenizer, model=model)

        def render(self, request):
            seen["request"] = request
            return {"token_ids": [4, 5], "stop_token_ids": [200006, 33]}

    monkeypatch.setattr(codec_module, "Recipe", RecordingRecipe)
    tokenizer = FakeTokenizer()
    codec = InklingCodec(tokenizer)
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "reasoning_effort": "low",
        "stop": ["!"],
    }
    assert codec.render(request) == {
        "token_ids": [4, 5],
        "stop_token_ids": [200006, 33],
    }
    assert seen == {"tokenizer": tokenizer, "model": "inkling", "request": request}


def test_render_maps_openai_xhigh_to_inkling_max() -> None:
    codec = InklingCodec(FakeTokenizer())
    request = {"messages": [{"role": "user", "content": "hello"}]}
    assert codec.render({**request, "reasoning_effort": "xhigh"}) == codec.render(
        {**request, "reasoning_effort": "max"}
    )


def test_processor_recipe_receives_all_format_controls(monkeypatch: pytest.MonkeyPatch):
    class Processor:
        tokenizer = FakeTokenizer()
        image_token_id = 200054

    seen = []

    def prepare(processor, recipe, request):
        seen.append(request)
        return {"input_ids": [[1, 2]]}

    monkeypatch.setattr(codec_module, "prepare_recipe_inputs", prepare)
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": None,
        "extra": {"reasoning_effort": "low"},
    }
    assert InklingCodec(Processor()).render(request)["token_ids"] == [1, 2]
    assert seen == [request]


def test_render_rejects_unknown_reasoning_effort() -> None:
    codec = InklingCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="reasoning_effort"):
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


def test_native_codec_implements_python_protocols() -> None:
    codec = InklingCodec(FakeTokenizer())
    compatible_codec: ConversationCodec = codec
    parser: FinalizingResponseParser = codec.create_parser({}, [200001, 200004])

    assert compatible_codec is codec
    assert parser.feed(_text_ids("hello")) == [{"type": "text_delta", "delta": "hello"}]
    assert parser.finish("stop") == []


def test_codec_prepares_one_native_tokenizer_for_multiple_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = codec_module.Recipe
    calls: list[Any] = []

    def prepare_recipe(tokenizer, *, model):
        calls.append(tokenizer)
        return prepare(tokenizer, model=model)

    monkeypatch.setattr(codec_module, "Recipe", prepare_recipe)
    tokenizer = FakeTokenizer()
    codec = InklingCodec(tokenizer)
    first = codec.create_parser({}, [200001, 200004])
    second = codec.create_parser({}, [200001, 200008])

    assert calls == [tokenizer]
    assert first.feed(_text_ids("visible")) == [
        {"type": "text_delta", "delta": "visible"}
    ]
    assert second.feed(_text_ids("private")) == [
        {"type": "reasoning_delta", "delta": "private"}
    ]


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_parser_discards_unterminated_tool_at_finish(
    reason: Literal["stop", "length"],
) -> None:
    parser = InklingCodec(FakeTokenizer()).create_parser({}, [200001])

    assert parser.feed([200049, *_text_ids('{"name":"weather","args":')]) == []
    assert parser.finish(reason) == []
    assert parser.finish(reason) == []
    with pytest.raises(ValueError, match="finish"):
        parser.feed(_text_ids("late"))


@pytest.mark.parametrize(
    "payload",
    ["{bad json", "[]", '{"args":{}}', '{"name":"weather","args":"SF"}'],
)
def test_parser_rejects_invalid_closed_tool_block(payload: str) -> None:
    parser = InklingCodec(FakeTokenizer()).create_parser({}, [200001])

    with pytest.raises(ValueError):
        parser.feed([200049, *_text_ids(payload), 200010])
