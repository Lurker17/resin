"""Real Gloo ranks behind one HTTP listener, including empty participants."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin import Parallelism, RunnerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.sampling import format_output, prompt_scores
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.server import HttpServer


class RankCodec:
    def __init__(self, rank, case):
        if case == "initialize" and rank == 1:
            raise RuntimeError("injected codec initialization")

    def render(self, request):
        content = request["messages"][0]["content"]
        if content == "bad-render":
            raise ValueError("injected request rendering")
        return {
            "token_ids": [99]
            if content == "explode"
            else ([4] if content == "cancelme" else [1, 2, 3]),
            "stop_token_ids": [],
        }

    def create_parser(self, request, prompt_ids):
        class Parser:
            def feed(self, ids):
                return [
                    {
                        "type": "text_delta",
                        "delta": "".join(str(token - 10) for token in ids),
                    }
                ]

        return Parser()


class RankRunner:
    runner_config = RunnerConfig(continuation_prefill=True)
    parallelism_capabilities = ParallelismCapabilities(
        attn_dp=True, attn_tp=True, ep=True, expert_tp=True
    )

    def __init__(self, groups, directory):
        self.groups = groups
        self.directory = directory
        self.calls = []
        self.released = []
        self.cancel_requests = set()

    def release(self, ids):
        self.released.extend(map(int, ids))
        if self.cancel_requests.intersection(map(int, ids)):
            Path(self.directory, f"canceled-{self.groups.rank}").touch()

    def forward_distributed(self, plan, context):
        # Every replica and follower participates, even when another replica
        # owns all requests. A rank-stamped output proves actual placement.
        marker = torch.tensor(1)
        dist.all_reduce(marker, group=self.groups.device_domain.handle)
        self.calls.append(plan is None)
        if plan is None:
            return None
        for row, (start, end) in enumerate(
            zip(plan.token_indptr[:-1], plan.token_indptr[1:], strict=True)
        ):
            if 4 in plan.token_ids[start:end] and plan.context_lengths[row] == 0:
                self.cancel_requests.add(int(plan.sequence_ids[row]))
        if 99 in plan.token_ids:
            raise RuntimeError("injected remote forward failure")
        replica = self.groups.topology.coordinates(self.groups.rank)[0]
        logits = torch.zeros(plan.num_sequences, 32)
        logits[:, 10 + replica] = 2
        all_logits = torch.zeros(len(plan.token_ids), 32)
        return format_output(
            logits,
            logits.argmax(-1),
            plan,
            prompt_logprobs=prompt_scores(plan, all_logits),
        )


def _serve_rank(rank, world_size, directory, port, case):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=25),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=world_size, attn_dp=2, attn_tp=world_size // 2, ep=2),
        device_backend="gloo",
        timeout=timedelta(seconds=25),
    )
    runner = RankRunner(groups, directory)
    worker = DistributedWorker(runner, groups)
    engine = None
    error = None
    try:
        if not groups.is_driver:
            worker.serve()
        else:
            assert groups.drivers is not None
            engine = DistributedEngine(worker, context_length=2048)
            codec = RankCodec(rank, "normal")
            if case == "initialize" and groups.drivers.local_rank == 1:
                # Native session initialization snapshots an optional tokenizer.
                class BrokenCodec(RankCodec):
                    @property
                    def tokenizer(self):
                        raise RuntimeError("injected codec initialization")

                codec = BrokenCodec(rank, "normal")
            model = "rank-test"
            if case == "whitespace" and groups.drivers.local_rank == 0:
                model = " "
            if case == "infer" and groups.drivers.local_rank == 1:

                class InferenceFailureCodec(RankCodec):
                    @property
                    def tokenizer(self):
                        raise RuntimeError("injected model inference")

                codec = InferenceFailureCodec(rank, "normal")
                model = None
            if case == "frontend" and groups.drivers.local_rank == 0:
                import resin.server as serving

                def fail(*args, **kwargs):
                    raise RuntimeError("injected frontend construction")

                setattr(serving, "_HttpServer", fail)
            server = HttpServer(
                engine,
                codec=codec,
                model=model,
                enable_openai=not (case == "flags" and groups.drivers.local_rank == 0),
                stream_interval=(
                    3
                    if case == "normal"
                    else 2
                    if case == "stream_interval" and groups.drivers.local_rank == 1
                    else 0
                    if case == "invalid_stream_interval"
                    and groups.drivers.local_rank == 1
                    else 1
                ),
            )
            if groups.drivers.local_rank == 0:

                def stop():
                    while not Path(directory, "stop").exists():
                        time.sleep(0.01)
                    server.shutdown()

                threading.Thread(target=stop, daemon=True).start()
            server.serve(port=port)
    except BaseException as exc:
        error = str(exc)
    finally:
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "error": error,
                    "closed": worker.closed,
                    "calls": runner.calls,
                    "released": runner.released,
                    "engine_closed": None if engine is None else engine.closed,
                }
            )
        )
        groups.close()
        dist.destroy_process_group()


def _post(base, rank=None, *, content="hello", headers=None, **options):
    body = {
        "model": "rank-test",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2,
        **options,
    }
    if rank is not None:
        body["routed_dp_rank"] = rank
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        return urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        error.add_note(error.read().decode())
        raise


def _wait_ready(base, directory):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2):
                return
        except (OSError, urllib.error.URLError):
            failed = list(Path(directory).glob("rank-*.json"))
            if failed:
                raise AssertionError([path.read_text() for path in failed])
            time.sleep(0.02)
    raise AssertionError("distributed HTTP listener did not start")


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize(
    "case,world_size",
    [
        ("normal", 4),
        ("fatal", 2),
        ("initialize", 2),
        ("bind", 2),
        ("whitespace", 2),
        ("flags", 2),
        ("infer", 2),
        ("frontend", 2),
        ("stream_interval", 2),
        ("invalid_stream_interval", 2),
    ],
)
def test_distributed_http_routes_real_replica_engines(tmp_path, case, world_size):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    if case != "bind":
        probe.close()
    else:
        probe.listen()
    context = mp.spawn(
        _serve_rank,
        args=(world_size, str(tmp_path), port, case),
        nprocs=world_size,
        join=False,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        if case in ("normal", "fatal"):
            _wait_ready(base, tmp_path)
        if case == "fatal":
            with pytest.raises(urllib.error.HTTPError) as failure:
                _post(base, 1, content="explode")
            assert failure.value.code == 500
        elif case == "normal":
            for rank in (0, 1):
                with _post(base) as response:
                    assert (
                        json.load(response)["choices"][0]["message"]["content"]
                        == str(rank) * 2
                    )
            with _post(base, 1, logprobs=True, top_logprobs=2) as response:
                output = json.load(response)
            assert output["choices"][0]["message"]["content"] == "11"
            # Custom parsers cannot prove a token's semantic channel; raw
            # scores still cross replicas, while content-only scores stay empty.
            assert output["choices"][0]["logprobs"]["content"] == []
            assert len(output["resin"]["generation_logprobs"]) == 2
            with _post(base, 1, stream=True, max_tokens=5, logprobs=True) as response:
                chunks = [
                    json.loads(line[6:])
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
            assert [
                content
                for chunk in chunks
                for choice in chunk["choices"]
                if (content := choice["delta"].get("content"))
            ] == ["1", "111", "1"]
            assert [
                score["token_index"]
                for chunk in chunks
                for score in chunk.get("resin", {}).get("generation_logprobs", [])
            ] == list(range(5))
            with _post(base, headers={"resin-routed-dp-rank": "1"}) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "11"
            with _post(base, 999, headers={"resin-routed-dp-rank": "1"}) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "11"
            with pytest.raises(urllib.error.HTTPError) as failure:
                _post(base, 0, headers={"resin-routed-dp-rank": "bad"})
            assert failure.value.code == 400
            with _post(base, 1, max_tokens=0, prompt_logprobs=True) as response:
                output = json.load(response)
            assert output["choices"][0]["message"]["content"] == ""
            for rank in (2, -1):
                with pytest.raises(urllib.error.HTTPError) as failure:
                    _post(base, rank)
                assert failure.value.code == 400
            with pytest.raises(urllib.error.HTTPError):
                _post(base, 1, content="bad-render")
            with _post(base, 0) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "00"
            # Drop a streaming client while rank one remains active.
            with _post(
                base, 1, content="cancelme", stream=True, max_tokens=1000
            ) as response:
                response.readline()
            deadline = time.monotonic() + 5
            while not Path(tmp_path, f"canceled-{world_size // 2}").exists():
                assert time.monotonic() < deadline, (
                    "remote cancellation did not release its lease"
                )
                time.sleep(0.01)
    finally:
        Path(tmp_path, "stop").touch()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if context.join(timeout=0.2):
                break
        else:
            probe.close()
            for process in context.processes:
                process.kill()
            raise AssertionError(
                "distributed HTTP shutdown left a driver/follower running"
            )
        probe.close()
    records = [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert all(record["closed"] for record in records)
    if case == "normal":
        assert all(record["error"] is None for record in records)
        assert all(record["calls"] for record in records)
        assert any(any(record["calls"]) for record in records)
        assert all(len(record["calls"]) < 1000 for record in records)
    elif case == "bind":
        assert "bind" in records[0]["error"]
        assert records[1]["error"] is None
    elif case in (
        "whitespace",
        "flags",
        "infer",
        "frontend",
        "initialize",
        "stream_interval",
        "invalid_stream_interval",
    ):
        assert len({record["error"] for record in records}) == 1
        assert records[0]["error"] is not None
        assert all(record["engine_closed"] for record in records)
        if case == "stream_interval":
            assert "configuration must agree" in records[0]["error"]
        elif case == "invalid_stream_interval":
            assert "stream_interval must be a positive integer" in records[0]["error"]
    else:
        assert all("injected" in record["error"] for record in records)
