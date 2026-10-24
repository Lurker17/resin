"""The 1017 codec selects an explicit format over the shared native grammar."""

import pytest
from recipe_fuzz.deepseek import DeepSeekV41Adapter
from recipe_fuzz.harness import Case, Part

import resin
from resin.recipe import DeepSeekV4Recipe, DeepSeekV41Recipe


@pytest.fixture(scope="module")
def adapter():
    return DeepSeekV41Adapter()


def test_exported_recipe_uses_system_token_and_numeric_native_options(adapter):
    model_recipe = DeepSeekV41Recipe(adapter.codec.tokenizer)
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.SYSTEM, [resin.recipe.Text("literal <｜System｜>")]
        ),
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")]),
    ]
    options = resin.recipe.RenderOptions(thinking_mode="thinking", reasoning_effort=37)
    rendered = model_recipe.render(messages, options=options)
    assert rendered.token_ids.count(adapter.ids["<｜System｜>"]) == 1
    assert "Reasoning Effort: 37 (range 1-100," in model_recipe.render_text(
        messages, options=options
    )
    assert rendered.stop_token_ids == [adapter.ids["<｜end▁of▁sentence｜>"]]
    assert model_recipe.render_text(messages, options=options).endswith(
        "<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", [0, 101, -1, True, 1.5, "medium"])
def test_numeric_budget_validates_before_rendering(adapter, effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        adapter.codec.render(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
            options=resin.recipe.RenderOptions(reasoning_effort=effort),
        )


def test_v41_and_v4_select_different_tool_grammars(adapter):
    case = Case(
        (
            Part("reasoning", "plan"),
            Part("text", "answer"),
            Part("tool", {"name": "ns::f", "args": {"n": 1}}),
        )
    )
    ids = adapter.encode(case)
    prompt = adapter.codec.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
        options=resin.recipe.RenderOptions(thinking_mode="thinking"),
    )
    parsed = adapter.codec.parse_response(ids, prompt=prompt)
    assert parsed["reasoning_content"] == "R:plan."
    assert parsed["content"] == "C:answer."
    assert parsed["tool_calls"] == [
        {"id": "call_0", "name": "ns::f", "arguments": '{"n": 1}'}
    ]
    legacy_recipe = DeepSeekV4Recipe(adapter.codec.tokenizer)
    legacy = legacy_recipe.parse_response(
        ids,
        prompt=legacy_recipe.render(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
            options=resin.recipe.RenderOptions(thinking_mode="thinking"),
        ),
    )
    assert legacy["tool_calls"] == []
    assert "<｜DSML｜ calls>" in legacy["content"]
