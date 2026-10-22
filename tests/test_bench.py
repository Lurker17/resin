from __future__ import annotations

import pytest
import torch

from resin import BatchPhase, CacheGroupLayout, CacheLayout, Engine, RunnerConfig
from resin.backend.speculative import SpeculativeStats
from resin.backend.torch import benchmark_expert_indices
from resin.bench import BenchCase, BenchmarkRunner, distributed_reducer, run_sweep


class Runner:
    def __init__(self, *, chunked=False, drafts=0):
        self.runner_config = RunnerConfig(
            max_batch_tokens=5 if chunked else None,
            prefill_chunk_size=3 if chunked else None,
            continuation_prefill=chunked,
            cache_write_lookahead_tokens=drafts,
            cache_layout=CacheLayout(256, [CacheGroupLayout(2)]) if drafts else None,
        )
        self.speculative_tokens = drafts
        self._speculation = SpeculativeStats(drafts)
        self.plans = []
        self.processed_counts = []

    def bind_resources(self, resources):
        pass

    def forward(self, plan):
        self.plans.append(plan)
        if plan.phase == BatchPhase.DECODE and self.speculative_tokens:
            drafts = [[2] * self.speculative_tokens for _ in plan.sequence_ids]
            targets = [[9] * (self.speculative_tokens + 1) for _ in plan.sequence_ids]
            acceptance = self._speculation.accept(drafts, targets)
            self.processed_counts.append(acceptance.processed_counts)
            return acceptance.token_rows
        return [0] * plan.num_sequences


def test_prefill_measures_actual_chunked_inputs_and_keeps_engine_resident():
    raw = Runner(chunked=True)
    runner = BenchmarkRunner(raw)
    engine = Engine(
        runner, max_running_requests=2, context_length=64, prefix_cache=True
    )
    records = run_sweep(
        engine,
        [BenchCase(phase="prefill", context_length=7, batch_size=2)],
        runner=runner,
        prompt_tokens=[1, 2, 3],
        warmup_steps=1,
        measure_steps=1,
        metadata={"model": "test"},
    )
    record = records[0]
    assert record["input_tokens"] == 14
    assert record["prefill_tokens"] == 14 / record["measure_steps"]
    assert record["measure_steps"] > 1  # A whole cohort spans several steps.
    assert record["warmup_steps"] == record["measure_steps"]
    assert record["committed_tokens"] == 2
    assert record["window_seconds"] == sum(record["step_seconds"])
    assert record["metadata"] == {"model": "test"}
    assert not engine.has_work()
    # run_sweep never closes a caller-owned resident engine.
    assert engine.submit([1], 1).cached_tokens == 0


def test_decode_counts_real_steps_and_disables_eos_without_retiring_cohort():
    runner = BenchmarkRunner(Runner())
    engine = Engine(runner, eos_token_id=0, max_running_requests=2, context_length=64)
    record = run_sweep(
        engine,
        [BenchCase(phase="decode", context_length=4, batch_size=2)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=2,
        measure_steps=3,
    )[0]
    assert record["measure_steps"] == 3
    assert record["committed_tokens"] == 6
    assert (
        record["measured_context_start_min"]
        == record["measured_context_start_max"]
        == 6
    )
    assert record["measured_context_end_min"] == record["measured_context_end_max"] == 9
    assert record["input_tokens"] == 0
    assert record["measured_accept_rate"] is None
    assert not engine.has_work()


@pytest.mark.parametrize("accepted", [0, 3, 4])
def test_fixed_acceptance_changes_engine_commit_and_model_rollback_depth(accepted):
    raw = Runner(drafts=4)
    runner = BenchmarkRunner(raw, accepted_drafts=accepted)
    engine = Engine(runner, max_running_requests=2, context_length=128)
    record = run_sweep(
        engine,
        [BenchCase(phase="decode", context_length=4, batch_size=2)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=1,
        measure_steps=2,
    )[0]
    assert record["committed_tokens"] == 2 * 2 * (accepted + 1)
    assert record["measured_accept_rate"] == accepted / 4
    assert record["measured_context_start_min"] == 4 + accepted + 1
    assert record["measured_context_end_max"] == 4 + 3 * (accepted + 1)
    assert raw.processed_counts == [[accepted + 1] * 2] * 3
    decode = [p for p in raw.plans if p.phase == BatchPhase.DECODE]
    assert [list(p.context_lengths) for p in decode] == [
        [4 + step * (accepted + 1)] * 2 for step in range(3)
    ]


def test_balanced_routes_are_distinct_per_token_and_globally_even():
    scores = torch.randn(13, 8)
    indices = benchmark_expert_indices(scores, 3)
    assert indices.shape == (13, 3)
    assert all(len(set(row)) == 3 for row in indices.tolist())
    counts = indices.flatten().bincount(minlength=8)
    assert int(counts.max() - counts.min()) <= 1
    assert benchmark_expert_indices(scores[:0], 3).shape == (0, 3)
    # Replicas with one token must continue the global assignment sequence.
    routes = torch.cat(
        [benchmark_expert_indices(scores[:1], 3, rank) for rank in range(3)]
    )
    counts = routes.flatten().bincount(minlength=8)
    assert int(counts.max() - counts.min()) <= 1


def test_graph_counters_exclude_prefill_and_decode_warmup():
    class GraphRunner(Runner):
        def __init__(self):
            super().__init__()
            self.replays = {"decode": 0, "prefill": 0}

        @property
        def cuda_graph_stats(self):
            return {
                mode: {
                    "replay_count": count,
                    "eager_fallback_count": count,
                    "fallback_counts": {"test": count},
                }
                for mode, count in self.replays.items()
            }

        def forward(self, plan):
            self.replays[
                "prefill" if plan.phase == BatchPhase.PREFILL else "decode"
            ] += 1
            return super().forward(plan)

    runner = BenchmarkRunner(GraphRunner())
    record = run_sweep(
        Engine(runner, context_length=64),
        [BenchCase(phase="decode", context_length=4, batch_size=1)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=2,
        measure_steps=3,
    )[0]
    assert record["graph_counts"] == {
        f"{mode}.{name}": 3 if mode == "decode" else 0
        for mode in ("decode", "prefill")
        for name in ("replay_count", "eager_fallback_count", "fallback.test")
    }


def test_partial_admission_is_canceled_on_failure():
    class EngineAdapter:
        def __init__(self, runner):
            self.local = Engine(runner, max_running_requests=2, context_length=64)
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("admission failed")
            return self.local.submit(*args, **kwargs)

        def cancel(self, lease):
            return self.local.cancel(lease)

    runner = BenchmarkRunner(Runner())
    engine = EngineAdapter(runner)
    with pytest.raises(ValueError, match="admission failed"):
        run_sweep(
            engine,
            [BenchCase(phase="prefill", context_length=4, batch_size=2)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
        )
    assert not engine.local.has_work()


def test_deepseek_fixed_acceptance_reaches_actual_compressor_rollback(monkeypatch):
    from types import SimpleNamespace

    from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

    runner = DeepSeekV4CudaRunner.__new__(DeepSeekV4CudaRunner)
    runner.device = torch.device("cpu")
    runner.speculative_tokens = 4
    runner._speculation = SpeculativeStats(4, benchmark_accepted_drafts=3)
    observed = []
    monkeypatch.setattr(
        runner,
        "_model",
        SimpleNamespace(
            clear_rejected_speculative_state=lambda positions, counts, block, *, cache: (
                observed.append((positions.tolist(), counts.tolist(), block, cache))
            )
        ),
        raising=False,
    )
    result = runner._accept_speculative(
        [[2, 3, 4, 5]], [[9, 8, 7, 6, 5]], torch.tensor([10]), "cache"
    )
    assert result == [[2, 3, 4, 6]]
    assert observed == [([10], [4], 5, "cache")]


@pytest.mark.parametrize(
    "drafts,targets", [([[1, 2]], [[3, 4, 5]]), ([[1, 2, 3]], [[4]])]
)
def test_fixed_acceptance_retains_shape_validation_without_updating_stats(
    drafts, targets
):
    stats = SpeculativeStats(4, benchmark_accepted_drafts=3)
    with pytest.raises(ValueError):
        stats.accept(drafts, targets)
    assert stats.blocks == stats.drafts == stats.accepted == stats.emitted == 0


def test_rejects_unsupported_or_incomplete_measurements():
    assert BenchmarkRunner(Runner(), accepted_drafts=0).accepted_drafts == 0
    with pytest.raises(ValueError, match="balanced experts"):
        BenchmarkRunner(Runner(), balanced_experts=True)
    for kwargs in ({"accepted_drafts": -1}, {"accepted_drafts": 5}):
        with pytest.raises(ValueError, match="accepted_drafts"):
            BenchmarkRunner(Runner(drafts=4), **kwargs)
    runner = BenchmarkRunner(Runner())
    engine = Engine(runner, max_running_requests=1, context_length=16)
    case = BenchCase(phase="decode", context_length=10, batch_size=1)
    with pytest.raises(ValueError, match="full decode window"):
        run_sweep(engine, [case], runner=runner, prompt_tokens=[1])
    assert not engine.has_work()
    with pytest.raises(ValueError, match="reducer"):
        run_sweep(
            engine, [case], runner=runner, prompt_tokens=[1], gpu_count=2, attn_dp=2
        )
    with pytest.raises(RuntimeError, match="BenchmarkRunner"):
        run_sweep(
            Engine(Runner(), context_length=64),
            [case],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
        )


def test_driver_reduction_uses_slowest_time_and_unique_replica_totals():
    from types import SimpleNamespace

    class Distributed:
        @staticmethod
        def all_gather_object(values, local, *, group):
            values[:] = [local, {**local, "seconds": 2 * local["seconds"]}]

    groups = SimpleNamespace(
        drivers=SimpleNamespace(size=2, handle=object()), distributed=Distributed()
    )
    reduced = distributed_reducer(groups)(
        {
            "seconds": 0.1,
            "input_tokens": 4,
            "committed_tokens": 2,
            "drafts": 4,
            "accepted": 3,
            "sequences": 2,
            "context_start_min": 3,
            "context_start_max": 4,
            "context_end_min": 4,
            "context_end_max": 5,
        }
    )
    assert reduced == {
        "seconds": 0.2,
        "input_tokens": 8,
        "committed_tokens": 4,
        "drafts": 8,
        "accepted": 6,
        "sequences": 4,
        "graph_counts": {},
        "context_start_min": 3,
        "context_start_max": 4,
        "context_end_min": 4,
        "context_end_max": 5,
    }
