"""Exercise each stock native parser through the actual HTTP actor."""

from __future__ import annotations

import json
from typing import Any

import pytest
from output_helpers import forward_output
from recipe_fuzz.harness import Case, Part, normalize
from recipe_fuzz.test_differential import ADAPTERS
from test_http_lifecycle import post, running_server

import resin


@pytest.fixture(scope="module", params=ADAPTERS)
def recipe_adapter(request):
    return request.param, ADAPTERS[request.param]()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completion_reason", ["stop", "length"])
def test_stock_recipe_http_events(recipe_adapter, stream, completion_reason):
    name, adapter = recipe_adapter
    parts = [Part("reasoning", "plan"), Part("text", "answer")]
    if completion_reason == "stop":
        parts.append(Part("tool", {"name": "weather", "args": {"n": 1}}))
    case = getattr(adapter, "case", lambda value: value)(Case(tuple(parts)))
    output = adapter.encode(case, terminal=completion_reason == "stop")
    expected = normalize(adapter.oracles[0][1](case))
    # DeepSeek's generated continuation begins in its rendered thinking prefix.
    request = {"messages": [{"role": "user", "content": "hello"}]}
    if name.startswith("deepseek_"):
        request["reasoning_effort"] = "high"

    class Runner:
        step = 0

        def forward(self, plan):
            assert plan.num_sequences == 1
            token = output[self.step]
            self.step += 1
            return forward_output([token])

    with running_server(Runner(), adapter.recipe) as (_, engine, _, errors, base, _):
        body = json.dumps(
            {
                **request,
                "model": "test",
                "stream": stream,
                "max_tokens": len(output) + (completion_reason == "stop"),
            }
        ).encode()
        with post(base, body) as response:
            if stream:
                payloads = [
                    line[6:]
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
                assert payloads[-1] == "[DONE]"
                choices = [
                    json.loads(payload)["choices"][0] for payload in payloads[:-1]
                ]
                message: dict[str, Any] = {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                }
                tools = {}
                for choice in choices:
                    delta = choice["delta"]
                    for key in ("content", "reasoning_content"):
                        message[key] += delta.get(key) or ""
                    for item in delta.get("tool_calls", []):
                        tool = tools.setdefault(
                            item["index"], {"function": {"name": "", "arguments": ""}}
                        )
                        for key in ("name", "arguments"):
                            tool["function"][key] += (
                                item.get("function", {}).get(key) or ""
                            )
                message["tool_calls"] = list(tools.values())
                reason = choices[-1]["finish_reason"]
            else:
                choice = json.load(response)["choices"][0]
                message, reason = choice["message"], choice["finish_reason"]
        assert message.get("content", "") == "".join(
            part.value
            for part in expected
            if part.kind == "text" and isinstance(part.value, str)
        )
        assert message.get("reasoning_content", "") == "".join(
            part.value
            for part in expected
            if part.kind == "reasoning" and isinstance(part.value, str)
        )
        actual_tools = [
            {
                "name": tool["function"]["name"],
                "args": json.loads(tool["function"]["arguments"]),
            }
            for tool in message.get("tool_calls", [])
        ]
        assert actual_tools == [part.value for part in expected if part.kind == "tool"]
        assert reason == ("tool_calls" if actual_tools else completion_reason)
        assert not engine.poisoned
        assert errors == []


def test_native_prompt_provenance_and_stop_encoding(recipe_adapter):
    _, adapter = recipe_adapter
    marker, marker_id = next(iter(adapter.ids.items()))
    baseline = adapter.recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hello")])]
    )
    rendered = adapter.recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(marker)])]
    )
    assert rendered.token_ids.tolist().count(
        marker_id
    ) == baseline.token_ids.tolist().count(marker_id)
    # Stop strings belong to HTTP serving; native recipes supply model stops.
    assert adapter.recipe.stop_token_ids


@pytest.mark.parametrize("stream", [False, True])
def test_stock_recipe_stops_match_decoded_output(recipe_adapter, stream):
    name, adapter = recipe_adapter
    case = getattr(adapter, "case", lambda value: value)(
        Case((Part("text", "prefix STOP suffix"),))
    )
    output = adapter.encode(case)
    expected = "".join(
        part.value
        for part in normalize(adapter.oracles[0][1](case))
        if part.kind == "text" and isinstance(part.value, str)
    ).split("STOP")[0]
    request = {"messages": [{"role": "user", "content": "hello"}]}
    if name.startswith("deepseek_"):
        request["reasoning_effort"] = "high"

    class Runner:
        position = 0

        def forward(self, plan):
            token = output[self.position]
            self.position += 1
            return forward_output([token])

    with running_server(Runner(), adapter.recipe) as (_, engine, _, errors, base, _):
        with post(
            base,
            json.dumps(
                {
                    "model": "test",
                    **request,
                    "stream": stream,
                    "max_tokens": len(output) + 1,
                    "stop": ["STOP"],
                }
            ).encode(),
        ) as response:
            if stream:
                choices = [
                    json.loads(line[6:])["choices"][0]
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                text = "".join(
                    choice["delta"].get("content") or "" for choice in choices
                )
                reason = choices[-1]["finish_reason"]
            else:
                choice = json.load(response)["choices"][0]
                text, reason = choice["message"]["content"], choice["finish_reason"]
        assert text == expected
        assert reason == "stop"
        assert not engine.has_work()
        assert not engine.poisoned
        assert errors == []
