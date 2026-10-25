"""Reject invalid benchmark topology and undersized resident cache budgets."""

import json
from types import SimpleNamespace

import pytest

from resin import Parallelism
from scripts import bench


@pytest.mark.parametrize(
    "model,topology,message",
    [
        ("deepseek-v4", Parallelism(), "unknown model"),
        ("gpt-oss-20b", Parallelism(), "currently requires V4.1"),
        ("deepseek-v41", Parallelism(), "at least two ranks"),
        (
            "deepseek-v41",
            Parallelism(world_size=8, attn_dp=8, ep=4),
            "ADP=EP=world_size",
        ),
        (
            "deepseek-v41",
            Parallelism(world_size=8, cp=2, attn_dp=4, ep=8),
            "ADP=EP=world_size",
        ),
    ],
)
def test_deepgemm_rejects_wrong_models_and_meshes_before_cuda(
    monkeypatch, model, topology, message
):
    def cannot_inspect_cuda():
        raise AssertionError("invalid options must fail before CUDA inspection")

    monkeypatch.setattr(bench.torch.cuda, "device_count", cannot_inspect_cuda)
    with pytest.raises(ValueError, match=message):
        bench.run_benchmark(
            model, "unused", [], parallelism=topology, expert_backend="deepgemm"
        )


def test_cache_preflight_rejects_decode_growth_before_prefill_or_capture(capsys):
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    layout = CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)])
    raw = SimpleNamespace(
        cache_layout=layout,
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    options = dict(warmup_steps=0, measure_steps=1, draft_tokens=0)
    bench._cache_preflight(
        raw, [BenchCase(phase="prefill", context_length=4, batch_size=4)], options
    )
    with pytest.raises(
        ValueError, match="decode ctx=4 batch=4 needs 8 cache parents, capacity=4"
    ):
        bench._cache_preflight(
            raw, [BenchCase(phase="decode", context_length=4, batch_size=4)], options
        )
    rows = [
        json.loads(line.split(" ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert rows[0]["cases"][0]["required_parents"] == 4
    assert rows[1]["cases"][0]["peak_context"] == 6
    assert rows[1]["capacity"]["bytes_per_parent"] == 10
    assert rows[1]["capacity"]["reserved_parent_count"] == 1


def test_cache_preflight_propagates_a_peers_capacity_failure():
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    class Distributed:
        @staticmethod
        def all_gather_object(values, local, *, group):
            assert not local  # This rank fits; a peer does not.
            values[:] = [[], ["rank 1 requires more cache parents"]]

    groups = SimpleNamespace(
        rank=0,
        domain=SimpleNamespace(size=2, handle=object()),
        distributed=Distributed(),
    )
    raw = SimpleNamespace(
        cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)]),
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    with pytest.raises(ValueError, match="rank 1 requires more cache parents"):
        bench._cache_preflight(
            raw,
            [BenchCase(phase="prefill", context_length=4, batch_size=4)],
            dict(warmup_steps=0, measure_steps=1, draft_tokens=0),
            groups,
        )


def test_cache_preflight_credits_only_the_configured_shared_history():
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    raw = SimpleNamespace(
        cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)]),
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    options = dict(warmup_steps=0, measure_steps=1, draft_tokens=0)
    bench._cache_preflight(
        raw,
        [
            BenchCase(
                phase="prefill", context_length=12, batch_size=2, shared_prefix_tokens=8
            )
        ],
        options,
    )
    with pytest.raises(ValueError, match="needs 6 cache parents, capacity=4"):
        bench._cache_preflight(
            raw, [BenchCase(phase="prefill", context_length=12, batch_size=2)], options
        )
