from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from resin import BatchPhase, BatchPlan, CacheGroupPlan, Engine, SamplingParams
from resin.backend.parallel import (
    TensorParallelRunner,
    batch_plan_from_payload,
    batch_plan_payload,
    follow_tensor_parallel_runner,
)
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
    )

    class Runner:
        @staticmethod
        def bind_resources(bound):
            assert bound is binding

    distributed = RecordingDistributed()
    runner = TensorParallelRunner(Runner(), "control", distributed)

    assert runner.bind_resources(binding) is None
    assert distributed.broadcasts == [
        {
            "operation": "bind_resources",
            "cache_pool_id": 41,
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
            },
            {"operation": "close"},
        ]
    )

    follow_tensor_parallel_runner(Runner(), "control", distributed)

    assert len(Runner.calls) == 1
    assert Runner.calls[0].cache_pool_id == 41
    assert Runner.calls[0].cache_layout is layout


def test_tensor_parallel_plan_round_trip_preserves_native_contract() -> None:
    plan = BatchPlan(
        [11, 22],
        [0, 32],
        [1, 2, 3],
        [0, 2, 3],
        phase=BatchPhase.PREFILL,
        batch_id=7,
        batch_key=9,
        cache_groups=[
            CacheGroupPlan(
                0,
                [0, 1, 2],
                [3, 4],
                [0, 0],
                [0, 2],
                [3, 3, 4],
                [0, 0, 0],
                [0, 1, 0],
            )
        ],
        sampling_mask=[False, True],
        sampling_params=[
            SamplingParams(temperature=0.5, seed=31),
            SamplingParams(top_p=0.9, seed=32),
        ],
    )

    restored = batch_plan_from_payload(batch_plan_payload(plan))

    assert restored.sequence_ids.tolist() == plan.sequence_ids.tolist()
    assert restored.context_lengths.tolist() == plan.context_lengths.tolist()
    assert restored.token_ids.tolist() == plan.token_ids.tolist()
    assert restored.token_indptr.tolist() == plan.token_indptr.tolist()
    assert restored.phase == plan.phase
    assert restored.batch_id == plan.batch_id
    assert restored.batch_key == plan.batch_key
    assert restored.sampling_mask.tolist() == [False, True]
    assert restored.cache_groups[0].block_slab_ids.tolist() == [3, 4]
    assert restored.cache_groups[0].block_slotnums.tolist() == [0, 0]
    assert restored.sampling_params[0].temperature == 0.5
    assert restored.sampling_params[1].top_p == 0.9


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
            renderer=FakeRenderer(),
            model="test",
            enable_openai=False,
            enable_demo=True,
        )
