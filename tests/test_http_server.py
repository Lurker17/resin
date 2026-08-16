from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from deploy.http import TensorParallelServerRunner, follow_tensor_parallel_server
from resin import Engine
from resin.server import HttpServer, RendererCodec


class IncrementingRunner:
    @staticmethod
    def forward(plan):
        return [
            plan.token_ids[end - 1] + 1
            for _, end in map(plan.token_range, range(plan.num_sequences))
        ]


class RecordingDistributed:
    def __init__(self, incoming=None) -> None:
        self.commands = [] if incoming is None else list(incoming)
        self.broadcasts = []

    def broadcast_object_list(self, objects, *, src, group) -> None:
        assert src == 0
        assert group == "control"
        if self.commands:
            objects[0] = self.commands.pop(0)
        else:
            self.broadcasts.append(objects[0])


def test_tensor_parallel_server_broadcasts_resource_binding() -> None:
    layout = object()
    binding = SimpleNamespace(
        cache_pool_id=41,
        cache_layout=layout,
        state_pool_id=None,
        state_capacity=None,
        required_state_arena_id=None,
    )

    class Runner:
        @staticmethod
        def bind_resources(bound):
            assert bound is binding
            return 41

    distributed = RecordingDistributed()
    runner = TensorParallelServerRunner(Runner(), "control", distributed)

    assert runner.bind_resources(binding) == 41
    assert distributed.broadcasts == [
        {
            "operation": "bind_resources",
            "cache_pool_id": 41,
            "state_pool_id": None,
            "state_capacity": None,
            "required_state_arena_id": None,
        }
    ]


def test_tensor_parallel_follower_applies_resource_binding() -> None:
    layout = object()

    class Runner:
        runner_config = SimpleNamespace(cache_layout=layout, state_mapping=None)
        calls = []

        @classmethod
        def bind_resources(cls, binding):
            cls.calls.append(binding)

    distributed = RecordingDistributed(
        [
            {
                "operation": "bind_resources",
                "cache_pool_id": 41,
                "state_pool_id": None,
                "state_capacity": None,
                "required_state_arena_id": None,
            },
            {"operation": "close"},
        ]
    )

    follow_tensor_parallel_server(Runner(), "control", distributed)

    assert len(Runner.calls) == 1
    assert Runner.calls[0].cache_pool_id == 41
    assert Runner.calls[0].cache_layout is layout
    assert Runner.calls[0].state_mapping is None


@dataclass
class Parsed:
    content: str
    reasoning_content: str | None = None
    tool_calls: list[object] | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = []

    def render_ids(self, messages, *, tools, add_generation_prompt):
        self.calls.append((messages, tools, add_generation_prompt))
        return [1, 2]

    @staticmethod
    def get_stop_token_ids():
        return [9]

    @staticmethod
    def parse_response(token_ids, *, tools):
        del tools
        text = "".join(chr(96 + token) for token in token_ids)
        return Parsed(content=text)


def test_renderer_codec_renders_and_incrementally_parses() -> None:
    renderer = FakeRenderer()
    codec = RendererCodec(renderer)
    request = {"messages": [{"role": "user", "content": "hi"}], "tools": []}

    assert codec.render(request) == {"token_ids": [1, 2], "stop_token_ids": [9]}
    parser = codec.create_parser(request, [1, 2])
    assert parser.feed([3]) == [{"type": "text_delta", "delta": "c"}]
    assert parser.feed([4]) == [{"type": "text_delta", "delta": "d"}]
    assert renderer.calls[0][2] is True


def test_renderer_codec_emits_complete_tool_call_when_parseable() -> None:
    tool = SimpleNamespace(
        status=SimpleNamespace(name="OK"),
        id=None,
        name="weather",
        arguments={"city": "SF"},
    )

    class ToolRenderer(FakeRenderer):
        @staticmethod
        def parse_response(token_ids, *, tools):
            del token_ids, tools
            return Parsed(content="", tool_calls=[tool])

    parser = RendererCodec(ToolRenderer()).create_parser({"tools": []}, [])
    assert parser.feed([1]) == [
        {"type": "tool_call_start", "index": 0, "id": "call_0", "name": "weather"},
        {"type": "tool_call_arguments_delta", "delta": '{"city":"SF"}', "index": 0},
    ]


def test_renderer_codec_holds_incomplete_utf8_decoder_sentinel() -> None:
    class UnicodeRenderer(FakeRenderer):
        @staticmethod
        def parse_response(token_ids, *, tools):
            del tools
            content = (
                "hello \N{REPLACEMENT CHARACTER}" if len(token_ids) == 1 else "hello 😀"
            )
            return Parsed(content=content)

    parser = RendererCodec(UnicodeRenderer()).create_parser({"tools": []}, [])

    assert parser.feed([1]) == [{"type": "text_delta", "delta": "hello "}]
    assert parser.feed([2]) == [{"type": "text_delta", "delta": "😀"}]


def test_renderer_codec_can_restore_generation_prefix_for_parser() -> None:
    class PrefixRenderer(FakeRenderer):
        @staticmethod
        def render_ids(messages, *, tools, add_generation_prompt):
            del messages, tools
            return [1, 2, 10] if add_generation_prompt else [1, 2]

        @staticmethod
        def parse_response(token_ids, *, tools):
            del tools
            assert token_ids[0] == 10
            return Parsed(content="".join(chr(96 + token) for token in token_ids[1:]))

    codec = RendererCodec(PrefixRenderer(), include_generation_prefix_in_parser=True)
    request = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
    rendered = codec.render(request)
    parser = codec.create_parser(request, rendered["token_ids"])

    assert parser.feed([3]) == [{"type": "text_delta", "delta": "c"}]


def test_http_server_validates_route_configuration() -> None:
    with pytest.raises(ValueError, match="requires enable_openai"):
        HttpServer(
            Engine(IncrementingRunner()),
            FakeRenderer(),
            model="test",
            enable_openai=False,
            enable_demo=True,
        )
