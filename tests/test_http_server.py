from __future__ import annotations

from types import SimpleNamespace

import pytest
from output_helpers import forward_output

import resin
from resin import Engine
from resin.server import HttpServer


class IncrementingRunner:
    @staticmethod
    def forward(plan):
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


class _Recipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1])

    def create_parser(self, prompt):
        return SimpleNamespace(feed=lambda ids: [])


def test_http_server_validates_route_configuration() -> None:
    with pytest.raises(ValueError, match="requires enable_openai"):
        HttpServer(
            Engine(IncrementingRunner()),
            recipe=_Recipe(),
            model="test",
            enable_openai=False,
            enable_demo=True,
        )


@pytest.mark.parametrize("setting", ["stream_interval", "request_queue_capacity"])
@pytest.mark.parametrize("value", [0, -1, 2**32, True, False, 1.5, "2", None])
def test_http_server_requires_positive_integer_settings(setting, value) -> None:
    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match=f"{setting}.*positive integer"):
        HttpServer(engine, recipe=_Recipe(), model="test", **{setting: value})
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
    server = HttpServer(engine, recipe=_Recipe(), model="test", **options)
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
            _Recipe(),
            model="test",
            stream_interval=0 if setting == "stream_interval" else 1,
            request_queue_capacity=0 if setting == "request_queue_capacity" else 1024,
        )
    engine.close()
