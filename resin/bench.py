"""Resident-engine benchmarks with explicit synthetic routing/verification modes.

Install BenchmarkRunner before constructing the Engine and capturing graphs.
Each point owns an uncached request cohort; loading, capture, admission, and
metrics reduction are outside the timed Engine.step calls.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from resin import BatchPhase, BatchPlan, SamplingParams
from resin.backend.speculative import SpeculativeStats


@dataclass(frozen=True, kw_only=True)
class BenchCase:
    phase: Literal["prefill", "decode"]
    context_length: int
    batch_size: int

    def __post_init__(self):
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if any(
            type(n) is not int or n < 1 for n in (self.context_length, self.batch_size)
        ):
            raise ValueError("context_length and batch_size must be positive integers")


class BenchmarkRunner:
    """Delegate the runner contract while observing actual scheduled input rows.

    Synthetic options affect the real router or speculative acceptance boundary;
    they must be configured before graph capture and never imply model quality.
    """

    def __init__(
        self, runner, *, accepted_drafts: int | None = None, balanced_experts=False
    ):
        self.runner = runner
        self.draft_tokens = int(getattr(runner, "speculative_tokens", 0))
        if accepted_drafts is not None:
            if (
                type(accepted_drafts) is not int
                or not 0 <= accepted_drafts <= self.draft_tokens
            ):
                raise ValueError(
                    "accepted_drafts must be between zero and draft_tokens"
                )
            stats = getattr(runner, "_speculation", None)
            if self.draft_tokens and not isinstance(stats, SpeculativeStats):
                raise ValueError("fixed acceptance requires a speculative model runner")
            if isinstance(stats, SpeculativeStats):
                stats.benchmark_accepted_drafts = accepted_drafts
        self.accepted_drafts = accepted_drafts
        if type(balanced_experts) is not bool:
            raise ValueError("balanced_experts must be a bool")
        self.balanced_experts = balanced_experts
        if balanced_experts:
            model = getattr(runner, "_model", None)
            routers = (
                []
                if model is None
                else [
                    m
                    for m in model.modules()
                    if hasattr(m, "benchmark_balanced_experts")
                ]
            )
            if not routers:
                raise ValueError(
                    "balanced experts are unsupported by this model runner"
                )
            groups = getattr(runner, "parallel_groups", None)
            for router in routers:
                router.benchmark_balanced_experts = True
                router.benchmark_replica = (
                    0 if groups is None else groups.attn_dp.local_rank
                )
                if groups is not None:
                    router.benchmark_expert_parallel_size = groups.ep.size
        self._plan: BatchPlan | None = None

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _observe(self, plan: BatchPlan | None):
        self._plan = plan

    @property
    def observation(self) -> dict[str, Any]:
        plan = self._plan
        return (
            {}
            if plan is None
            else {
                "phase": "prefill" if plan.phase == BatchPhase.PREFILL else "decode",
                "input_tokens": len(plan.token_ids),
                "sequences": plan.num_sequences,
            }
        )

    def graph_counts(self):
        counts = {}
        for mode, stats in getattr(self.runner, "cuda_graph_stats", {}).items():
            for name in ("replay_count", "eager_fallback_count"):
                counts[f"{mode}.{name}"] = stats[name]
            for reason, value in stats["fallback_counts"].items():
                counts[f"{mode}.fallback.{reason}"] = value
        return counts

    def forward(self, plan):
        self._observe(plan)
        return self.runner.forward(plan)

    def forward_distributed(self, plan, context):
        self._observe(plan)
        return self.runner.forward_distributed(plan, context)

    def speculation(self):
        stats = getattr(self.runner, "_speculation", None)
        return (0, 0) if stats is None else (stats.drafts, stats.accepted)


def distributed_reducer(groups):
    """Reduce unique attention-replica driver counters after each timed step."""

    drivers = groups.drivers
    if drivers is None:
        raise ValueError("only attention-replica drivers run benchmark sweeps")

    def reduce_step(local):
        values = [local]
        if drivers.size > 1:
            values = [None] * drivers.size
            groups.distributed.all_gather_object(values, local, group=drivers.handle)
        return {
            "seconds": max(v["seconds"] for v in values),
            "replica_sequences": [v["sequences"] for v in values],
            **{
                key: sum(v[key] for v in values)
                for key in (
                    "input_tokens",
                    "committed_tokens",
                    "drafts",
                    "accepted",
                    "sequences",
                )
            },
            "graph_counts": {
                key: sum(v.get("graph_counts", {}).get(key, 0) for v in values)
                for key in {key for v in values for key in v.get("graph_counts", {})}
            },
            "context_start_min": min(v["context_start_min"] for v in values),
            "context_start_max": max(v["context_start_max"] for v in values),
            "context_end_min": min(v["context_end_min"] for v in values),
            "context_end_max": max(v["context_end_max"] for v in values),
        }

    return reduce_step


def run_sweep(
    engine,
    cases: Iterable[BenchCase],
    *,
    runner: BenchmarkRunner,
    prompt_tokens: Sequence[int],
    warmup_steps: int = 5,
    measure_steps: int = 20,
    gpu_count: int = 1,
    attn_dp: int = 1,
    synchronize: Callable[[], None] = lambda: None,
    reduce_step: Callable[[dict], dict] | None = None,
    metadata: dict | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Measure prefill cohorts or a steady decode window on an existing Engine.

    batch_size counts requests per attention-DP replica. Prefill finishes whole
    cohorts, so its actual step count can exceed the requested minimum when
    scheduling chunks. Decode records exactly the requested number of steps;
    its context grows naturally during warmup and measurement. All DP drivers
    call together with distributed_reducer; records contain global token counts
    and the slowest driver's step duration. The caller retains engine ownership.
    on_result receives each completed record after request cleanup, outside timing.
    """

    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("measure_steps", measure_steps, 1),
        ("gpu_count", gpu_count, 1),
        ("attn_dp", attn_dp, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if gpu_count % attn_dp:
        raise ValueError("gpu_count must be divisible by attn_dp")
    if attn_dp > 1 and reduce_step is None:
        raise ValueError("attention-DP benchmarks require a step reducer")
    if not len(prompt_tokens):
        raise ValueError("prompt_tokens must not be empty")
    local = getattr(engine, "local", engine)
    if local.has_work():
        raise ValueError("benchmark engine must not have active requests")
    reduce = reduce_step or (lambda value: value)
    records = []

    def step():
        before_drafts, before_accepted = runner.speculation()
        before_graphs = runner.graph_counts()
        runner._plan = None
        synchronize()
        started = time.perf_counter()
        events = engine.step()
        synchronize()
        elapsed = time.perf_counter() - started
        plan = runner._plan
        if plan is None:
            raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        drafts, accepted = runner.speculation()
        graph_counts = runner.graph_counts()
        starts = plan.context_lengths.tolist()
        if plan.phase == BatchPhase.PREFILL:
            offsets = plan.token_indptr.tolist()
            ends = [start + b - a for start, a, b in zip(starts, offsets, offsets[1:])]
        else:
            counts = Counter(event.request_id for event in events)
            ends = [
                start + counts[int(sequence)]
                for start, sequence in zip(starts, plan.sequence_ids)
            ]
        measured = reduce(
            {
                "seconds": elapsed,
                "input_tokens": runner.observation["input_tokens"],
                "sequences": runner.observation["sequences"],
                "replica_sequences": [runner.observation["sequences"]],
                "committed_tokens": len(events),
                "drafts": drafts - before_drafts,
                "accepted": accepted - before_accepted,
                "context_start_min": min(starts),
                "context_start_max": max(starts),
                "context_end_min": min(ends),
                "context_end_max": max(ends),
                "graph_counts": {
                    key: value - before_graphs.get(key, 0)
                    for key, value in graph_counts.items()
                },
            }
        )
        return measured

    for case in cases:
        if case.batch_size > local.max_running_requests:
            raise ValueError("batch_size exceeds the engine request capacity")
        if case.context_length >= local.context_length:
            raise ValueError("context_length leaves no space for generated tokens")
        tokens = (
            list(prompt_tokens)
            * ((case.context_length + len(prompt_tokens) - 1) // len(prompt_tokens))
        )[: case.context_length]
        leases = []
        samples = []
        warmed = 0
        result = None

        def admit(max_new_tokens):
            local.clear_prefix_cache()
            leases.clear()
            for _ in range(case.batch_size):
                leases.append(
                    engine.submit(
                        tokens,
                        max_new_tokens,
                        stop_token_ids=[],
                        sampling_params=SamplingParams(temperature=0.0),
                    )
                )

        try:
            if case.phase == "prefill":
                for target, measured in ((warmup_steps, False), (measure_steps, True)):
                    count = 0
                    while count < target:
                        admit(1)
                        while not all(lease.done for lease in leases):
                            result = step()
                            if runner.observation["phase"] != "prefill":
                                raise RuntimeError("prefill benchmark entered decode")
                            count += 1
                            if measured:
                                samples.append(result)
                        if not measured:
                            warmed = count
            else:
                maximum = 1 + (warmup_steps + measure_steps + 1) * (
                    runner.draft_tokens + 1
                )
                if case.context_length + maximum > local.context_length:
                    raise ValueError("engine context cannot fit the full decode window")
                admit(maximum)
                while local.next_phase() == BatchPhase.PREFILL:
                    step()
                for index in range(warmup_steps + measure_steps):
                    result = step()
                    if runner.observation["phase"] != "decode":
                        raise RuntimeError("decode benchmark did not execute decode")
                    if result["sequences"] != case.batch_size * attn_dp:
                        raise RuntimeError(
                            "decode batch was not fully scheduled: "
                            f"scheduled_global={result['sequences']}, "
                            f"expected_global={case.batch_size * attn_dp}, "
                            f"scheduled_local={runner.observation['sequences']}, "
                            f"expected_per_replica={case.batch_size}, "
                            f"context_length={case.context_length}, phase={case.phase}"
                        )
                    if any(lease.done for lease in leases):
                        raise RuntimeError("decode cohort retired during measurement")
                    if index >= warmup_steps:
                        samples.append(result)
                warmed = warmup_steps
            seconds = [sample["seconds"] for sample in samples]
            inputs = (
                sum(s["input_tokens"] for s in samples)
                if case.phase == "prefill"
                else 0
            )
            drafts = sum(s["drafts"] for s in samples)
            records.append(
                {
                    "schema_version": 1,
                    "phase": case.phase,
                    "context_length": case.context_length,
                    "batch_size": case.batch_size,
                    "global_batch_size": case.batch_size * attn_dp,
                    "measured_global_batch_size_min": min(
                        s["sequences"] for s in samples
                    ),
                    "measured_global_batch_size_max": max(
                        s["sequences"] for s in samples
                    ),
                    "gpu_count": gpu_count,
                    "attn_dp": attn_dp,
                    "prefill_tokens": inputs / len(samples),
                    "input_tokens": inputs,
                    "draft_tokens": runner.draft_tokens,
                    "accepted_drafts": runner.accepted_drafts,
                    "measured_accept_rate": sum(s["accepted"] for s in samples) / drafts
                    if drafts
                    else None,
                    "balanced_experts": runner.balanced_experts,
                    "balanced_experts_scope": "physical_routing_rows"
                    if runner.balanced_experts
                    else None,
                    "warmup_steps": warmed,
                    "measure_steps": len(samples),
                    "step_seconds": seconds,
                    "window_seconds": sum(seconds),
                    "committed_tokens": sum(s["committed_tokens"] for s in samples),
                    "measured_context_start_min": samples[0]["context_start_min"],
                    "measured_context_start_max": samples[0]["context_start_max"],
                    "measured_context_end_min": samples[-1]["context_end_min"],
                    "measured_context_end_max": samples[-1]["context_end_max"],
                    "graph_counts": {
                        key: sum(s["graph_counts"].get(key, 0) for s in samples)
                        for key in {key for s in samples for key in s["graph_counts"]}
                    },
                    "metadata": dict(metadata or {}),
                }
            )
        except Exception as error:
            diagnostic = dict(
                **asdict(case),
                error=f"{type(error).__name__}: {error}",
                attn_dp=attn_dp,
                actual_global=(result or {}).get("sequences"),
                expected_global=case.batch_size * attn_dp,
                observation=runner.observation,
                per_replica=(result or {}).get("replica_sequences"),
                cache={
                    name: getattr(local, name, None)
                    for name in (
                        "ready_request_count",
                        "active_request_count",
                        "new_token_ratio",
                        "cache_parent_capacity",
                        "cache_free_parent_count",
                        "cache_occupied_block_count",
                    )
                },
            )
            print("[bench-error] " + json.dumps(diagnostic), flush=True)
            raise
        finally:
            if not local.poisoned:
                for lease in leases:
                    if not lease.done:
                        engine.cancel(lease)
                local.clear_prefix_cache()
        if on_result is not None:
            on_result(records[-1])
    return records
