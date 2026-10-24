from __future__ import annotations

from typing import Literal

import pytest

import resin
from resin.protocols import FinalizingResponseParser
from resin.recipe import InklingRecipe

SPECIAL_TOKENS = {
    "<|message_model|>": 200001,
    "<|message_user|>": 200000,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
    "<|content_text|>": 200004,
    "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008,
    "<|end_message|>": 200010,
    "<|content_xml|>": 200011,
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


def test_native_recipe_renders_typed_messages_and_tools() -> None:
    model_recipe = InklingRecipe(FakeTokenizer())
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hello")])
    ]
    rendered = model_recipe.render(
        messages,
        tools=[resin.recipe.Tool("weather")],
        options=resin.recipe.RenderOptions(reasoning_effort="low"),
    )
    assert isinstance(rendered, resin.recipe.RenderedPrompt)
    assert rendered.token_ids
    assert rendered.stop_token_ids == [200006]
    assert model_recipe.render_ids(messages) == model_recipe.render(messages).token_ids


def test_render_maps_openai_xhigh_to_inkling_max() -> None:
    model_recipe = InklingRecipe(FakeTokenizer())
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hello")])
    ]
    assert model_recipe.render_ids(
        messages, options=resin.recipe.RenderOptions(reasoning_effort="xhigh")
    ) == model_recipe.render_ids(
        messages, options=resin.recipe.RenderOptions(reasoning_effort="max")
    )


def test_processor_recipe_preserves_explicit_format_controls():
    class Processor:
        tokenizer = FakeTokenizer()
        image_token_id = 200054

    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hello")])
    ]
    options = resin.recipe.RenderOptions(reasoning_effort="low")
    expected = InklingRecipe(Processor.tokenizer).render_ids(messages, options=options)
    assert InklingRecipe(Processor()).render_ids(messages, options=options) == expected


def test_render_rejects_unknown_reasoning_effort() -> None:
    model_recipe = InklingRecipe(FakeTokenizer())
    with pytest.raises(ValueError, match="reasoning_effort"):
        model_recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER, [resin.recipe.Text("hello")]
                )
            ],
            options=resin.recipe.RenderOptions(reasoning_effort="turbo"),
        )


def _parser(model_recipe, *prefix):
    prompt = model_recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    )
    parser = model_recipe.create_parser(prompt)
    assert parser.feed(prefix) == []
    return parser


def test_parser_separates_repeated_reasoning_and_visible_text_blocks() -> None:
    codec = InklingRecipe(FakeTokenizer())
    parser = _parser(codec)

    assert parser.feed([200008, *_text_ids("think")]) == [
        resin.recipe.ReasoningDelta("think")
    ]
    assert parser.feed([200010, 200001, 200004, *_text_ids("Hello")]) == [
        resin.recipe.TextDelta("Hello")
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
        resin.recipe.ReasoningDelta(" again"),
        resin.recipe.TextDelta("!"),
    ]


def test_parser_uses_prompt_tail_to_continue_reasoning() -> None:
    codec = InklingRecipe(FakeTokenizer())
    parser = _parser(codec, 200008)

    assert parser.feed(_text_ids("continued")) == [
        resin.recipe.ReasoningDelta("continued")
    ]


def test_parser_extracts_json_tool_call_and_hides_header_metadata() -> None:
    codec = InklingRecipe(FakeTokenizer())
    parser = _parser(codec)
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
        resin.recipe.ToolCallStart(0, "call_0", "weather"),
        resin.recipe.ToolCallArgumentsDelta(0, '{"city":"SF"}'),
    ]


def test_raw_tool_and_tool_error_blocks_remain_visible() -> None:
    codec = InklingRecipe(FakeTokenizer())
    parser = _parser(codec)

    assert parser.feed([200057, *_text_ids("raw"), 200010]) == [
        resin.recipe.TextDelta("raw")
    ]
    assert parser.feed([200001, 200022, *_text_ids("error"), 200010]) == [
        resin.recipe.TextDelta("error")
    ]


def test_native_codec_implements_python_protocols() -> None:
    codec = InklingRecipe(FakeTokenizer())
    compatible_codec: resin.recipe.Recipe = codec
    parser: FinalizingResponseParser = _parser(codec, 200004)

    assert compatible_codec is codec
    assert parser.feed(_text_ids("hello")) == [resin.recipe.TextDelta("hello")]
    assert parser.finish("stop") == []


def test_recipe_prepares_one_tokenizer_for_independent_parsers() -> None:
    class CountingTokenizer(FakeTokenizer):
        reads = 0

        @property
        def decode(self):
            self.reads += 1
            return super().decode

    tokenizer = CountingTokenizer()
    model_recipe = InklingRecipe(tokenizer)
    first = _parser(model_recipe, 200004)
    second = _parser(model_recipe, 200008)
    assert tokenizer.reads == 1
    assert first.feed(_text_ids("visible")) == [resin.recipe.TextDelta("visible")]
    assert second.feed(_text_ids("private")) == [resin.recipe.ReasoningDelta("private")]


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_parser_discards_unterminated_tool_at_finish(
    reason: Literal["stop", "length"],
) -> None:
    parser = _parser(InklingRecipe(FakeTokenizer()))

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
    parser = _parser(InklingRecipe(FakeTokenizer()))

    with pytest.raises(ValueError):
        parser.feed([200049, *_text_ids(payload), 200010])
