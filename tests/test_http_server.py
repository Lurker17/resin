from __future__ import annotations

from types import SimpleNamespace

import pytest

import resin
from resin import BatchPhase, BatchPlan, CacheGroupPlan, Engine, SamplingParams
from resin.backend.parallel import (
    TensorParallelRunner,
    batch_plan_from_payload,
    batch_plan_payload,
    follow_tensor_parallel_runner,
)
from resin.server import HttpServer


class IncrementingRunner:
    @staticmethod
    def forward(plan):
        return [
            plan.token_ids[end - 1] + 1
            for _, end in map(plan.token_range, range(plan.num_sequences))
        ]


class _Codec:
    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1], stop_token_ids=[])

    def create_parser(self, prompt):
        return SimpleNamespace(feed=lambda ids: [])


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


def test_http_server_validates_route_configuration() -> None:
    with pytest.raises(ValueError, match="requires enable_openai"):
        HttpServer(
            Engine(IncrementingRunner()),
            recipe=_Codec(),
            model="test",
            enable_openai=False,
            enable_demo=True,
        )


@pytest.mark.parametrize("setting", ["stream_interval", "request_queue_capacity"])
@pytest.mark.parametrize("value", [0, -1, 2**32, True, False, 1.5, "2", None])
def test_http_server_requires_positive_integer_settings(setting, value) -> None:
    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match=f"{setting}.*positive integer"):
        HttpServer(engine, recipe=_Codec(), model="test", **{setting: value})
    assert not engine.closed
    engine.close()


@pytest.mark.parametrize(
    "setting,default", [("stream_interval", 1), ("request_queue_capacity", 1024)]
)
@pytest.mark.parametrize("value", [None, 4, 2**32 - 1])
def test_http_server_forwards_default_and_custom_settings(
    setting, default, value, monkeypatch
):
    import resin.server as serving

    created = []

    def native(*args, **kwargs):
        created.append(kwargs)
        return SimpleNamespace(running=False)

    monkeypatch.setattr(serving, "_HttpServer", native)
    engine = Engine(IncrementingRunner())
    options = {} if value is None else {setting: value}
    server = HttpServer(engine, recipe=_Codec(), model="test", **options)
    assert server.model == "test"
    assert created[0][setting] == (default if value is None else value)
    engine.close()


@pytest.mark.parametrize("setting", ["stream_interval", "request_queue_capacity"])
def test_native_http_server_rejects_zero_settings(setting):
    from resin._resin import _HttpServer

    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match=f"{setting}.*positive"):
        _HttpServer(
            engine,
            _Codec(),
            model="test",
            stream_interval=0 if setting == "stream_interval" else 1,
            request_queue_capacity=0 if setting == "request_queue_capacity" else 1024,
        )
    engine.close()
