"""GLM's native binding, pinned template, and streaming behavior."""

import json
from pathlib import Path

import pytest
from test_recipe import _byte_tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

from resin.recipe import (
    Message,
    ReasoningDelta,
    RenderOptions,
    Role,
    Text,
    Thinking,
    Tool,
    ToolCall,
    ToolResult,
    get_native_recipe,
)

MARKERS = (
    "[gMASK]",
    "<sop>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|begin_of_image|>",
    "<|image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|video|>",
    "<|end_of_video|>",
)


@pytest.fixture
def backend():
    tokenizer = _byte_tokenizer()
    tokenizer.add_special_tokens(list(MARKERS))
    return tokenizer


@pytest.fixture
def template():
    return _compile_jinja_template(
        (Path(__file__).parent / "recipe_fuzz/templates/glm5_3_flash.jinja").read_text()
    )


def test_pinned_fixture_is_reproducible_from_original_template(template):
    fixtures = json.loads(
        (
            Path(__file__).parents[1] / "crates/resin-recipe/tests/fixtures/glm_hf.json"
        ).read_text()
    )
    for case in fixtures["cases"]:
        expected = template.render(
            **{"tools": [], "add_generation_prompt": True, **case["request"]}
        )
        assert expected == case["text"], case["name"]


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_native_render_matches_template_and_tokens(backend, template, clear, effort):
    recipe = get_native_recipe("glm5_3_flash", backend)
    messages = [
        Message(Role.SYSTEM, [Text("Be precise.")]),
        Message(Role.USER, [Text("Hello 🦀")]),
        Message(Role.ASSISTANT, [Thinking("earlier"), Text("  A response  ")]),
        Message(Role.USER, [Text("Continue")]),
    ]
    options = RenderOptions(clear_thinking=clear, reasoning_effort=effort)
    expected = template.render(
        messages=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello 🦀"},
            {
                "role": "assistant",
                "reasoning_content": "earlier",
                "content": "  A response  ",
            },
            {"role": "user", "content": "Continue"},
        ],
        tools=[],
        add_generation_prompt=True,
        clear_thinking=clear,
        reasoning_effort=effort,
    )
    assert recipe.debug_render_text(messages, options=options) == expected
    assert (
        recipe.render(messages, options=options).token_ids.tolist()
        == backend.encode(expected, add_special_tokens=False).ids
    )
    assert recipe.stop_token_ids == [
        backend.token_to_id(marker)
        for marker in ("<|endoftext|>", "<|user|>", "<|observation|>")
    ]


def test_calls_results_and_parser_share_tool_schema(backend, template):
    recipe = get_native_recipe("glm5_3_flash", backend)
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}}
    tool = Tool("lookup", parameters=parameters)
    messages = [
        Message(Role.USER, [Text("Find 001")]),
        Message(Role.ASSISTANT, [ToolCall("lookup", {"code": "001"}, id="a")]),
        Message(Role.TOOL, [ToolResult("found", call_id="a")]),
    ]
    expected = template.render(
        messages=[
            {"role": "user", "content": "Find 001"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "lookup", "arguments": {"code": "001"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "found"},
        ],
        tools=[{"name": "lookup", "parameters": parameters}],
        add_generation_prompt=True,
    )
    prompt = recipe.render(messages, tools=[tool])
    assert recipe.debug_render_text(messages, tools=[tool]) == expected
    assert (
        prompt.token_ids.tolist()
        == backend.encode(expected, add_special_tokens=False).ids
    )
    output = backend.encode(
        "why 🦀</think>yes<tool_call>lookup<arg_key>code</arg_key>"
        "<arg_value>001</arg_value></tool_call><|observation|>",
        add_special_tokens=False,
    ).ids
    assert recipe.parse_response(output, prompt=prompt) == {
        "content": "yes",
        "reasoning_content": "why 🦀",
        "tool_calls": [
            {"id": "call_0", "name": "lookup", "arguments": '{"code":"001"}'}
        ],
    }
    parser = recipe.create_parser(prompt)
    reasoning = []
    for token in output:
        reasoning.extend(
            event for event in parser.feed([token]) if isinstance(event, ReasoningDelta)
        )
    assert "".join(event.delta for event in reasoning) == "why 🦀"
    assert all(event.token_spans for event in reasoning)


def test_unknown_controls_are_explicitly_rejected(backend):
    recipe = get_native_recipe("glm5_3_flash", backend)
    messages = [Message(Role.USER, [Text("hi")])]
    with pytest.raises(
        ValueError, match="does not support rendering option enable_thinking"
    ):
        recipe.render(messages, options=RenderOptions(enable_thinking=False))
    with pytest.raises(ValueError, match="role"):
        recipe.render([Message(Role.DEVELOPER, [Text("hi")])])
