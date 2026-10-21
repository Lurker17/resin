from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from resin import BatchPhase, Engine
from resin.backend.cuda.failures import PostAcceptCudaError
from resin.server import HttpServer


class Codec:
    def render(self, request):
        slow = request["messages"][0]["content"] == "slow"
        return {"token_ids": [1 if slow else 2], "stop_token_ids": []}

    def create_parser(self, request, prompt_ids):
        delta = "x" * 65_536 if prompt_ids == [1] else "ok"

        class Parser:
            def feed(self, token_ids):
                return [{"type": "text_delta", "delta": delta}]

        return Parser()


class Runner:
    def __init__(self):
        self.counts = {}
        self.slow_id = None
        self.started = threading.Event()
        self.slow_released = threading.Event()

    def forward(self, plan):
        for row, sequence_id in enumerate(plan.sequence_ids):
            self.counts[int(sequence_id)] = self.counts.get(int(sequence_id), 0) + 1
            if (
                plan.phase == BatchPhase.PREFILL
                and plan.token_ids[plan.token_indptr[row]] == 1
            ):
                self.slow_id = int(sequence_id)
                self.started.set()
        return [3] * plan.num_sequences

    def release(self, sequence_ids):
        if self.slow_id in sequence_ids:
            self.slow_released.set()


def request_body(content="fast", *, stream=False, max_tokens=2):
    return json.dumps(
        {
            "model": "test",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
            "max_tokens": max_tokens,
        }
    ).encode()


@contextmanager
def running_server(runner, codec=None):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = Engine(runner)
    server = HttpServer(engine, codec=codec or Codec(), model="test")
    errors = []

    def serve():
        try:
            server.serve(port=port)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while True:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2) as response:
                assert response.status == 200
                break
        except (OSError, urllib.error.URLError):
            if errors or time.monotonic() >= deadline:
                server.shutdown()
                thread.join(timeout=2)
                raise AssertionError(f"server did not become ready: {errors}")
            time.sleep(0.01)
    try:
        yield server, engine, thread, errors, base, port
    finally:
        server.shutdown()
        thread.join(timeout=7)
        assert not thread.is_alive(), (
            "shutdown left the inference actor or listener running"
        )


def post(base, body):
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=5)


def test_programmatic_shutdown_stops_idle_server_and_closes_engine():
    with running_server(Runner()) as (server, engine, thread, errors, _base, port):
        server.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not server.running
        assert engine.closed
        assert errors == []
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
        server.shutdown()  # Shutdown remains harmless after serving has stopped.


def test_actor_failure_stops_listener_and_propagates_from_serve():
    class FailingRunner(Runner):
        def forward(self, plan):
            raise PostAcceptCudaError("runner failed after updating state")

    with running_server(FailingRunner()) as (
        server,
        engine,
        thread,
        errors,
        base,
        port,
    ):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(base, request_body())
        assert caught.value.code == 500
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not server.running
        assert engine.poisoned
        assert len(errors) == 1
        assert "runner failed after updating state" in str(errors[0])
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)


def test_slow_client_is_cancelled_without_blocking_other_clients():
    runner = Runner()
    with running_server(runner) as (_server, _engine, _thread, errors, base, port):
        with socket.socket() as slow:
            slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            slow.settimeout(2)
            slow.connect(("127.0.0.1", port))
            body = request_body("slow", stream=True, max_tokens=2000)
            slow.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            assert runner.started.wait(timeout=3)
            # The slow socket is deliberately never read. Another client must
            # finish while the actor cancels the blocked stream's generation.
            with post(base, request_body()) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "okok"
            assert runner.slow_released.wait(timeout=3)
            assert runner.counts[runner.slow_id] < 2000
            assert errors == []


def test_admission_burst_yields_to_inference_before_draining_queue():
    class GatedCodec(Codec):
        def __init__(self):
            self.rendering = threading.Event()
            self.resume = threading.Event()
            self.admitted = 0

        def render(self, request):
            if not self.admitted:
                self.rendering.set()
                assert self.resume.wait(timeout=3)
            self.admitted += 1
            return super().render(request)

    codec = GatedCodec()
    first_step = threading.Event()
    counts_at_step = []

    class ObservingRunner(Runner):
        def forward(self, plan):
            counts_at_step.append(codec.admitted)
            first_step.set()
            return super().forward(plan)

    connections = []
    with running_server(ObservingRunner(), codec) as (
        _server,
        _engine,
        _thread,
        errors,
        _base,
        port,
    ):
        try:
            body = request_body(stream=True, max_tokens=1)
            wire = (
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            for index in range(13):
                connection = socket.create_connection(("127.0.0.1", port), timeout=2)
                connections.append(connection)
                connection.sendall(wire)
                if index == 0:
                    assert codec.rendering.wait(timeout=2)
                # A response header acknowledges the request was queued, even
                # while its codec has not yet been admitted by the actor.
                assert connection.recv(1024).startswith(b"HTTP/1.1 200")
            codec.resume.set()
            assert first_step.wait(timeout=3)
            assert counts_at_step[0] == 4
            assert errors == []
        finally:
            codec.resume.set()
            for connection in connections:
                connection.close()
