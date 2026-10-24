"""Ownership and state boundaries of the native conversation interface."""

from __future__ import annotations

import gc
import json
import weakref
from types import SimpleNamespace

import pytest
from test_http_lifecycle import post, running_server
from test_recipe import _byte_tokenizer

import resin


def tokenizer():
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|message_user|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_image|>",
            "<|unused_200054|>",
        ]
    )
    return backend


def test_typed_inputs_own_json_values_and_copy_out_fields():
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}}
    tool = resin.recipe.Tool("lookup", parameters=parameters)
    call = resin.recipe.ToolCall("lookup", {"n": 3}, id="call-1")
    message = resin.recipe.Message(resin.recipe.Role.ASSISTANT, [call])
    options = resin.recipe.RenderOptions(response_format={"type": "json_object"})
    parameters["properties"]["n"]["type"] = "string"
    tool.parameters["properties"]["n"]["type"] = "boolean"
    call.arguments["n"] = 99
    options.response_format["type"] = "text"
    assert tool.parameters["properties"]["n"]["type"] == "integer"
    saved_call = message.content[0]
    assert isinstance(saved_call, resin.recipe.ToolCall)
    assert saved_call.arguments == {"n": 3}
    assert options.response_format == {"type": "json_object"}
    assert isinstance(message.content, tuple)
    with pytest.raises(AttributeError):
        setattr(message, "role", resin.recipe.Role.USER)
    with pytest.raises(AttributeError):
        setattr(saved_call, "name", "changed")


def test_native_prompt_is_owned_bound_to_one_recipe_and_parsers_are_fresh():
    backend = tokenizer()
    first = resin.recipe.GptOssRecipe(backend)
    second = resin.recipe.GptOssRecipe(backend)
    prompt = first.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    )
    original = prompt.token_ids
    changed = prompt.token_ids
    changed[0] = 99
    prompt.stop_token_ids.clear()
    assert prompt.token_ids == original
    assert prompt.stop_token_ids
    with pytest.raises(ValueError, match="this recipe"):
        second.create_parser(prompt)
    with pytest.raises(ValueError, match="this recipe"):
        first.create_parser(resin.recipe.RenderedPrompt(original))
    ids = backend.encode("thought", add_special_tokens=False).ids
    a, b = first.create_parser(prompt), first.create_parser(prompt)
    events = a.feed(ids)
    assert events == b.feed(ids)
    assert events
    assert all(isinstance(event, resin.recipe.ReasoningDelta) for event in events)
    assert all(isinstance(event, resin.recipe.ParserEvent) for event in events)
    assert a is not b
    assert first.parse_response(ids, prompt=prompt)["reasoning_content"] == "thought"


def test_native_events_preserve_variants_values_and_owned_provenance():
    spans = [(0, 0, 2, 0, 2)]
    text = resin.recipe.TextDelta("é", token_spans=spans)
    reasoning = resin.recipe.ReasoningDelta("é", token_spans=spans)
    start = resin.recipe.ToolCallStart(2, "call-2", "lookup", token_spans=spans)
    arguments = resin.recipe.ToolCallArgumentsDelta(2, '{"x":', token_spans=spans)
    spans.clear()
    assert text.type == "text_delta" and text.delta == "é"
    assert reasoning.type == "reasoning_delta" and reasoning.delta == "é"
    assert start.type == "tool_call_start"
    assert (start.index, start.id, start.name) == (2, "call-2", "lookup")
    assert arguments.type == "tool_call_arguments_delta"
    assert (arguments.index, arguments.delta) == (2, '{"x":')
    assert text == resin.recipe.TextDelta("é", token_spans=[(0, 0, 2, 0, 2)])
    assert text != reasoning
    assert text != resin.recipe.TextDelta("é")
    assert text != {"type": "text_delta", "delta": "é"}
    for event in (text, reasoning, start, arguments):
        assert isinstance(event, resin.recipe.ParserEvent)
        assert event.token_spans == ((0, 0, 2, 0, 2),)
        assert isinstance(event.token_spans, tuple)
        with pytest.raises(AttributeError):
            setattr(event, "token_spans", ())
    with pytest.raises(AttributeError):
        setattr(text, "delta", "changed")
    with pytest.raises(AttributeError):
        setattr(start, "index", 0)
    assert resin.recipe.TextDelta("synthetic").token_spans == ()


@pytest.mark.parametrize(
    ("spans", "error"),
    [
        ([(0, 0, 1, 0)], ValueError),
        ([(-1, 0, 1, 0, 1)], OverflowError),
        ([(1 << 32, 0, 1, 0, 1)], OverflowError),
    ],
)
def test_native_event_constructor_validates_span_storage(spans, error):
    with pytest.raises(error):
        resin.recipe.TextDelta("x", token_spans=spans)


def test_parser_does_not_retain_media_or_the_rendered_prompt(monkeypatch):
    import resin._recipe_media as media

    class Payload:
        pass

    class Preparer:
        def prepare(self, sources):
            return {}, SimpleNamespace(payload=sources[0][1])

        def finalize(self, state, token_ids):
            return state

    monkeypatch.setattr(media, "make_preparer", lambda *_args, **_kwargs: Preparer())
    native = resin.recipe.InklingRecipe(tokenizer())
    payload = Payload()
    reference = weakref.ref(payload)
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Image(payload)])
    ]
    prompt = native.render(messages)
    parser = native.create_parser(prompt)
    del payload, messages, prompt
    gc.collect()
    assert reference() is None
    assert parser.feed([]) == []


def test_public_order_and_per_message_controls_are_not_silently_dropped():
    native = resin.recipe.GptOssRecipe(tokenizer())
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.ASSISTANT,
            [
                resin.recipe.ToolCall("f", {}),
                resin.recipe.Text("later text"),
            ],
        )
    ]
    with pytest.raises(ValueError, match="block order"):
        native.render_text(messages)
    with pytest.raises(ValueError, match="per-message"):
        native.render_text(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER, [resin.recipe.Text("hi")], task="action"
                )
            ]
        )
    with pytest.raises(ValueError, match="initial"):
        native.render_text(
            [
                resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")]),
                resin.recipe.Message(
                    resin.recipe.Role.SYSTEM, [], tools=[resin.recipe.Tool("f")]
                ),
            ]
        )
    with pytest.raises(ValueError, match="does not support rendering option"):
        native.render_text(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
            options=resin.recipe.RenderOptions(add_vision_id=True),
        )
    with pytest.raises(ValueError, match="latest_reminder"):
        native.render_text(
            [
                resin.recipe.Message(
                    resin.recipe.Role.LATEST_REMINDER, [resin.recipe.Text("Remember")]
                )
            ]
        )


def test_raw_tool_arguments_survive_without_json_reserialization():
    native = resin.recipe.GptOssRecipe(
        tokenizer(), options=resin.recipe.RenderOptions(use_system_prompt=False)
    )
    raw = '{ "n": 1.00, "nested": [true, null] }'
    text = native.render_text(
        [
            resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("run")]),
            resin.recipe.Message(
                resin.recipe.Role.ASSISTANT, [resin.recipe.ToolCall("f", raw)]
            ),
        ]
    )
    assert raw in text


def test_native_http_lowering_matches_public_typed_messages():
    backend = tokenizer()
    backend.add_special_tokens(
        [
            "<bos>",
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            '<|"|>',
        ]
    )
    native = resin.recipe.Gemma4Recipe(backend)
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Go")]),
        resin.recipe.Message(
            resin.recipe.Role.ASSISTANT,
            [
                resin.recipe.Thinking("Use the tool"),
                resin.recipe.ToolCall("f", {"n": 3}, id="c1"),
                resin.recipe.ToolResult({"answer": 4}, name="f"),
                resin.recipe.Text("Here"),
            ],
        ),
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Next")]),
    ]
    reconstructed = [
        resin.recipe.Message(message.role, message.content) for message in messages
    ]
    expected = native.render(messages).token_ids
    assert native.render(reconstructed).token_ids == expected
    seen = []

    class Runner:
        def forward(self, plan):
            seen.extend(plan.token_ids.tolist())
            return [backend.token_to_id("<eos>")]

    wire_messages = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": "Here",
            "reasoning_content": "Use the tool",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"n": 3}},
                }
            ],
            "tool_responses": [{"name": "f", "response": {"answer": 4}}],
        },
        {"role": "user", "content": "Next"},
    ]
    with running_server(Runner(), native) as (_, _, _, errors, base, _):
        with post(
            base,
            json.dumps(
                {"model": "test", "messages": wire_messages, "max_tokens": 1}
            ).encode(),
        ) as response:
            assert response.status == 200
        assert not errors
    assert seen == expected
