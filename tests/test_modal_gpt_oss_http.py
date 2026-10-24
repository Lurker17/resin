"""GPT-OSS HTTP ranks use the shared worker lifecycle and explicit model mesh."""

from inspect import signature
from types import SimpleNamespace
from typing import Any

import pytest

from deploy import modal_gpt_oss as deployment
from resin import Engine


@pytest.mark.parametrize(
    "rank,size,failure",
    [(0, 1, None), (0, 2, None), (1, 2, None), (0, 2, "warmup"), (0, 2, "serve")],
)
def test_http_rank_uses_shared_worker_and_closes_after_warmup_or_server_failure(
    monkeypatch, tmp_path, rank, size, failure
):
    calls = []
    runner_factory = deployment._new_cuda_runner
    groups = SimpleNamespace(is_driver=rank == 0)
    tensor_parallel = SimpleNamespace(rank=rank, size=size, enabled=size > 1)
    runner = SimpleNamespace(
        device=f"cuda:{rank}",
        eos_token_id=99,
        cache_parent_capacity=32,
        cache_capacity_plan=SimpleNamespace(
            bytes_per_parent=1024,
            allocation_bytes=32768,
            memory_utilization=0.9,
        ),
    )
    recipe = object()

    def initialize(actual_rank, actual_size, port):
        assert (actual_rank, actual_size, port) == (rank, size, 12345)
        return tensor_parallel

    def make_groups(topology, *, timeout):
        assert topology.world_size == topology.attn_tp == size
        assert topology.attn_dp == topology.cp == topology.ep == 1
        calls.append("groups")
        return groups

    def make_runner(model, capacity, parallel, **options):
        signature(runner_factory).bind(model, capacity, parallel, **options)
        assert model == tmp_path and capacity is None and parallel is tensor_parallel
        assert options["parallel_groups"] is (groups if size > 1 else None)
        calls.append("runner")
        return runner

    class Worker:
        def __init__(self, local_runner, rank_groups):
            assert local_runner is runner and rank_groups is groups
            self.groups = rank_groups
            calls.append("worker")

        def serve(self):
            calls.append("follower")

    class RequestEngine:
        def __init__(self, owner, **options):
            signature(Engine).bind(owner, **options)
            assert isinstance(owner, Worker) if size > 1 else owner is runner
            assert options["context_length"] == 131072
            assert options["max_running_requests"] == deployment.DEFAULT_MAX_NUM_SEQS
            assert options["eos_token_id"] == 99 and options["prefix_cache"]
            self.closed = False
            self.remaining = 2
            calls.append("engine")

        def submit(self, tokens, **options):
            assert tokens == [1, 2] and options["max_new_tokens"] == 2
            assert options["sampling_params"].temperature == 0
            calls.append("submit")
            return SimpleNamespace(take_result=lambda: calls.append("warmup_done"))

        def has_work(self):
            return self.remaining > 0

        def step(self):
            if failure == "warmup":
                raise RuntimeError("injected warmup failure")
            self.remaining -= 1
            calls.append("step")

        def close(self):
            assert not self.closed
            self.closed = True
            calls.append("close")

    class Server:
        def __init__(self, engine, *, recipe, model):
            assert engine.remaining == 0 and recipe is not None
            assert model == deployment.MODEL_SPECS["120b"]["repo_id"]
            self.engine = engine
            calls.append("http")

        def serve(self, *, host, port):
            assert (host, port) == ("0.0.0.0", 8000)
            calls.append("serve")
            if failure == "serve":
                raise RuntimeError("injected serve failure")
            self.engine.close()

    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setattr(deployment, "_initialize_tensor_parallel", initialize)
    monkeypatch.setattr(deployment.ParallelGroups, "create", make_groups)
    monkeypatch.setattr(deployment, "_new_cuda_runner", make_runner)
    monkeypatch.setattr(deployment, "install_flashinfer_hash_cache", lambda: None)
    monkeypatch.setattr(
        deployment, "capture_cuda_graphs", lambda _: calls.append("capture")
    )
    monkeypatch.setattr(deployment, "DistributedWorker", Worker)
    monkeypatch.setattr(deployment, "DistributedEngine", RequestEngine)
    monkeypatch.setattr(deployment, "Engine", RequestEngine)
    monkeypatch.setattr(deployment, "_load_recipe", lambda _: (None, recipe))
    monkeypatch.setattr(deployment, "_encode_prompt", lambda *_: [1, 2])
    monkeypatch.setattr(deployment, "HttpServer", Server)
    monkeypatch.setattr(deployment.torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(deployment.os, "_exit", exit_process)
    if failure:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            deployment._serve_http_rank(rank, 12345, "120b", tmp_path, size)
    else:
        with pytest.raises(SystemExit) as exited:
            deployment._serve_http_rank(rank, 12345, "120b", tmp_path, size)
        assert exited.value.code == 0
    if rank == 1:
        assert calls == ["groups", "runner", "capture", "worker", "follower"]
    else:
        assert calls.count("close") == 1
        assert calls.index("capture") < calls.index("engine")
        assert ("worker" in calls) == (size > 1)
        assert ("http" in calls) == (failure != "warmup")


@pytest.mark.parametrize("explicit_groups", [False, True])
def test_cuda_runner_factory_selects_one_parallel_configuration(
    monkeypatch, tmp_path, explicit_groups
):
    context: Any = SimpleNamespace(rank=1, size=2, barrier=lambda: None)
    groups: Any = object() if explicit_groups else None

    class Runner:
        def __init__(self, model, **options):
            assert model == tmp_path
            assert options["parallel_groups"] is groups
            assert options["tensor_parallel"] is (None if explicit_groups else context)
            self.device = options["device"]

    monkeypatch.setattr(deployment, "GptOssCudaRunner", Runner, raising=False)
    monkeypatch.setattr(
        deployment, "FlashInferSM90MXFP4Kernel", lambda: object(), raising=False
    )
    monkeypatch.setattr(deployment.torch.cuda, "synchronize", lambda _: None)
    result = deployment._new_cuda_runner(
        tmp_path, None, context, parallel_groups=groups
    )
    assert result.device == "cuda:1"
