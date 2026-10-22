"""The 1017 codec selects an explicit format over the shared native grammar."""

import pytest
from recipe_fuzz.deepseek import DeepSeekV41Adapter
from recipe_fuzz.harness import Case, Part

from resin.models.deepseek_v41 import DeepSeekV41Codec
from resin.recipe import Recipe


@pytest.fixture(scope="module")
def adapter():
    return DeepSeekV41Adapter()


def test_exported_codec_uses_system_token_and_numeric_native_options(adapter):
    codec = DeepSeekV41Codec(adapter.codec.tokenizer)
    request = {
        "messages": [
            {"role": "system", "content": "literal <｜System｜>"},
            {"role": "user", "content": "hi"},
        ],
        "extra": {"thinking_mode": "thinking", "reasoning_effort": 37},
    }
    rendered = codec.render(request)
    assert rendered["token_ids"].count(adapter.ids["<｜System｜>"]) == 1
    assert "Reasoning Effort: 37 (range 1-100," in codec.render_text(request)
    assert rendered["stop_token_ids"] == [adapter.ids["<｜end▁of▁sentence｜>"]]
    assert codec.render_text(request).endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("effort", [0, 101, -1, True, 1.5, "medium"])
def test_numeric_budget_validates_before_rendering(adapter, effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        adapter.codec.render(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            }
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
    request = {"extra": {"thinking_mode": "thinking"}}
    parsed = adapter.codec.parse_response(ids, request)
    assert parsed["reasoning_content"] == "R:plan."
    assert parsed["content"] == "C:answer."
    assert parsed["tool_calls"] == [
        {"id": "call_0", "name": "ns::f", "arguments": '{"n": 1}'}
    ]
    legacy = Recipe(adapter.codec.tokenizer, model="deepseek_v4").parse_response(
        ids, request
    )
    assert legacy["tool_calls"] == []
    assert "<｜DSML｜ calls>" in legacy["content"]
