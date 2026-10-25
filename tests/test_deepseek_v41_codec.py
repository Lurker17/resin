"""The DeepSeek V4.1 recipe preserves the 1017 prompt and tool grammar."""

import pytest
from recipe_fuzz.deepseek import DeepSeekV41Adapter
from recipe_fuzz.harness import Case, Part

import resin
from resin.recipe import get_native_recipe


@pytest.fixture(scope="module")
def adapter():
    return DeepSeekV41Adapter()


def test_exported_recipe_uses_system_token_and_numeric_native_options(adapter):
    model_recipe = get_native_recipe("deepseek_v41", adapter.recipe.tokenizer)
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.SYSTEM, [resin.recipe.Text("literal <｜System｜>")]
        ),
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")]),
    ]
    options = resin.recipe.RenderOptions(thinking_mode="thinking", reasoning_effort=37)
    rendered = model_recipe.render(messages, options=options)
    assert rendered.token_ids.tolist().count(adapter.ids["<｜System｜>"]) == 1
    assert "Reasoning Effort: 37 (range 1-100," in model_recipe.debug_render_text(
        messages, options=options
    )
    assert model_recipe.stop_token_ids == [adapter.ids["<｜end▁of▁sentence｜>"]]
    assert model_recipe.debug_render_text(messages, options=options).endswith(
        "<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", [0, 101, -1, True, 1.5, "medium"])
def test_numeric_budget_validates_before_rendering(adapter, effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        adapter.recipe.render(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
            options=resin.recipe.RenderOptions(reasoning_effort=effort),
        )


def test_v41_tool_grammar_preserves_namespaces_and_arguments(adapter):
    case = Case(
        (
            Part("reasoning", "plan"),
            Part("text", "answer"),
            Part("tool", {"name": "ns::f", "args": {"n": 1}}),
        )
    )
    ids = adapter.encode(case)
    prompt = adapter.recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
        options=resin.recipe.RenderOptions(thinking_mode="thinking"),
    )
    parsed = adapter.recipe.parse_response(ids, prompt=prompt)
    assert parsed["reasoning_content"] == "R:plan."
    assert parsed["content"] == "C:answer."
    assert parsed["tool_calls"] == [
        {"id": "call_0", "name": "ns::f", "arguments": '{"n": 1}'}
    ]
