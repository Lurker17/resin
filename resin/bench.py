"""Resident-engine benchmarks with explicit synthetic routing/verification modes.

Install BenchmarkRunner before constructing the Engine and capturing graphs.
Each point owns an uncached request cohort; loading, capture, and admission
are untimed. Pipelined decode measures continuous completion intervals, with
metrics reduced only after the window so in-flight GPU work stays timed.
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
    shared_prefix_tokens: int = 0

    def __post_init__(self):
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if any(
            type(n) is not int or n < 1 for n in (self.context_length, self.batch_size)
        ):
            raise ValueError("context_length and batch_size must be positive integers")
        if (
            type(self.shared_prefix_tokens) is not int
            or not 0 <= self.shared_prefix_tokens < self.context_length
        ):
            raise ValueError(
                "shared_prefix_tokens must be an integer in [0, context_length)"
            )


class _ObservedCompletion:
    """Attribute a pipelined sample to its harvested plan, not the next launch."""

    def __init__(self, runner, plan, completion):
        self._runner, self._plan, self._completion = runner, plan, completion

    @property
    def feedback(self):
        return self._completion.feedback

    def result(self):
        output = self._completion.result()
        self._runner._observe(self._plan)
        self._runner._completed_positions = getattr(self._completion, "positions", None)
        return output


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
        self._completed_positions = None

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _observe(self, plan: BatchPlan | None):
        self._plan = plan
        self._completed_positions = None

    @property
    def completed_context_lengths(self):
        if self._plan is None:
            raise RuntimeError("no benchmark result has been harvested")
        return (
            self._completed_positions
            if self._completed_positions is not None
            else self._plan.context_lengths
        )

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
        stats = getattr(self.runner, "cuda_graph_counts", None)
        if stats is None:
            stats = getattr(self.runner, "cuda_graph_stats", {})
        for mode, stats in stats.items():
            for name in ("replay_count", "eager_fallback_count"):
                counts[f"{mode}.{name}"] = stats[name]
            for reason, value in stats["fallback_counts"].items():
                counts[f"{mode}.fallback.{reason}"] = value
        return counts

    def forward(self, plan):
        self._observe(plan)
        return self.runner.forward(plan)

    def forward_distributed(self, plan, context, *, prepared=None):
        self._observe(plan)
        return (
            self.runner.forward_distributed(plan, context)
            if prepared is None
            else self.runner.forward_distributed(plan, context, prepared=prepared)
        )

    def enqueue(self, plan, *, feedback=None):
        enqueue = getattr(self.runner, "enqueue", None)
        if enqueue is None:
            return self.forward(plan)
        output = enqueue(plan, feedback=feedback)
        if callable(getattr(output, "result", None)):
            return _ObservedCompletion(self, plan, output)
        self._observe(plan)
        return output

    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        enqueue = getattr(self.runner, "enqueue_distributed", None)
        if enqueue is None:
            return self.forward_distributed(plan, context, prepared=prepared)
        kwargs = {"feedback": feedback}
        if prepared is not None:
            kwargs["prepared"] = prepared
        output = enqueue(plan, context, **kwargs)
        if callable(getattr(output, "result", None)):
            return _ObservedCompletion(self, plan, output)
        self._observe(plan)
        return output

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
            "overlap_counts": {
                key: sum(v.get("overlap_counts", {}).get(key, 0) for v in values)
                for key in ("enqueues", "continuations")
            },
            "context_start_min": min(v["context_start_min"] for v in values),
            "context_start_max": max(v["context_start_max"] for v in values),
            "context_end_min": min(v["context_end_min"] for v in values),
            "context_end_max": max(v["context_end_max"] for v in values),
            "device_phase_ms": {
                phase: max(v.get("device_phase_ms", {}).get(phase, 0) for v in values)
                for phase in {
                    key for v in values for key in v.get("device_phase_ms", {})
                }
            },
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
    trace_step: Callable[[BenchCase, Callable[[], dict]], None] | None = None,
) -> list[dict]:
    """Measure prefill cohorts or a steady decode window on an existing Engine.

    batch_size counts requests per attention-DP replica. Prefill finishes whole
    cohorts, so its actual step count can exceed the requested minimum when
    scheduling chunks. Decode records exactly the requested number of steps;
    its context grows naturally during warmup and measurement. All DP drivers
    call together with distributed_reducer; records contain global token counts
    and the slowest driver's completion interval. Overlapped decode defers these
    reductions until the continuous measurement window ends. The caller retains
    engine ownership.
    on_result receives each completed record after request cleanup, outside timing.
    trace_step may run one extra decode step (two with overlap) after recording
    measurements. Its outputs are excluded from records; all distributed drivers
    must participate in that step.
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
    overlap = bool(getattr(local, "overlap", False)) and callable(
        getattr(runner.runner, "enqueue", None)
    )
    if local.has_work():
        raise ValueError("benchmark engine must not have active requests")
    reduce = reduce_step or (lambda value: value)
    records = []
    previous_completion = None

    def step(*, defer_reduce=False):
        nonlocal previous_completion
        before_drafts, before_accepted = runner.speculation()
        before_graphs = runner.graph_counts()
        before_overlap = getattr(local, "overlap_stats", (0, 0))
        runner._plan = None
        if not overlap:
            synchronize()
        started = (
            previous_completion
            if defer_reduce and previous_completion is not None
            else time.perf_counter()
        )
        # Filling an empty pipeline produces no host result. Include any such
        # submission in the next completion interval, including boundary bubbles.
        while True:
            events = engine.step()
            if runner._plan is not None:
                break
            if not getattr(local, "overlap_pending", False):
                raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        if not overlap:
            synchronize()
        finished = time.perf_counter()
        elapsed = finished - started
        previous_completion = finished
        plan = runner._plan
        if plan is None:
            raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        drafts, accepted = runner.speculation()
        graph_counts = runner.graph_counts()
        after_overlap = getattr(local, "overlap_stats", (0, 0))
        starts = getattr(
            runner, "completed_context_lengths", plan.context_lengths
        ).tolist()
        if plan.phase == BatchPhase.PREFILL:
            offsets = plan.token_indptr.tolist()
            ends = [start + b - a for start, a, b in zip(starts, offsets, offsets[1:])]
        else:
            counts = Counter(event.request_id for event in events)
            ends = [
                start + counts[int(sequence)]
                for start, sequence in zip(starts, plan.sequence_ids)
            ]
        measured = {
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
            "device_phase_ms": (
                getattr(runner, "last_dspark_phase_ms", None) or {}
                if plan.phase == BatchPhase.DECODE
                else {}
            ),
            "graph_counts": {
                key: value - before_graphs.get(key, 0)
                for key, value in graph_counts.items()
            },
            "overlap_counts": dict(
                zip(
                    ("enqueues", "continuations"),
                    (
                        after - before
                        for before, after in zip(before_overlap, after_overlap)
                    ),
                    strict=True,
                )
            ),
        }
        return measured if defer_reduce else reduce(measured)

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
        nonce = 0
        cached_tokens = []
        shared = case.shared_prefix_tokens

        def admit(max_output_tokens):
            nonlocal nonce
            if not shared:
                local.clear_prefix_cache()
            leases.clear()
            for _ in range(case.batch_size):
                request_tokens = tokens
                if shared:
                    if nonce >= runner.vocab_size:
                        raise ValueError(
                            "shared-prefix sweep exhausted distinct private tokens"
                        )
                    request_tokens = tokens.copy()
                    request_tokens[shared] = nonce
                    nonce += 1
                lease = engine.submit(
                    request_tokens,
                    sampling_params=SamplingParams(
                        max_output_tokens=max_output_tokens,
                        stop_token_ids=[],
                        temperature=0.0,
                    ),
                )
                leases.append(lease)
                cached_tokens.append(lease.cached_tokens)
                if lease.cached_tokens != shared:
                    raise ValueError(
                        f"shared prefix reuse differs from requested checkpoint: "
                        f"requested={shared}, cached={lease.cached_tokens}"
                    )

        try:
            local.clear_prefix_cache()
            if shared:
                if local.prefix_cache_stats is None:
                    raise ValueError(
                        "shared-prefix benchmarks require Engine(prefix_cache=True)"
                    )
                leases.append(
                    engine.submit(
                        tokens[:shared],
                        sampling_params=SamplingParams(
                            max_output_tokens=1, stop_token_ids=[], temperature=0.0
                        ),
                    )
                )
                while not leases[0].done:
                    step()  # Seed the one shared prefix outside warmup/measurement.
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
                # Reserve a complete extra step for optional tracing, including
                # maximum speculative acceptance and the initial prefill token.
                maximum = 1 + (warmup_steps + measure_steps + (3 if overlap else 1)) * (
                    runner.draft_tokens + 1
                )
                if case.context_length + maximum > local.context_length:
                    raise ValueError("engine context cannot fit the full decode window")
                admit(maximum)
                while local.next_phase() == BatchPhase.PREFILL:
                    step()
                for index in range(warmup_steps + measure_steps):
                    measuring = index >= warmup_steps
                    if overlap and measuring and warmup_steps == 0 and index == 0:
                        previous_completion = None
                    result = step(defer_reduce=overlap)
                    if runner.observation["phase"] != "decode":
                        raise RuntimeError("decode benchmark did not execute decode")
                    expected = case.batch_size * (1 if overlap else attn_dp)
                    if result["sequences"] != expected:
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
                    if measuring:
                        samples.append(result)
                if overlap:
                    # A per-step reduction or CUDA synchronization lets future
                    # GPU work run outside the timer. Reduce only after the
                    # continuous completion window has finished.
                    samples = [reduce(sample) for sample in samples]
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
                    "shared_prefix_tokens": shared,
                    "private_prompt_tokens": case.context_length - shared,
                    "cached_tokens_min": min(cached_tokens),
                    "cached_tokens_max": max(cached_tokens),
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
                    **(
                        {
                            "device_phase_ms": [
                                sample["device_phase_ms"] for sample in samples
                            ]
                        }
                        if any(sample.get("device_phase_ms") for sample in samples)
                        else {}
                    ),
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
                    "overlap_counts": {
                        key: sum(
                            s.get("overlap_counts", {}).get(key, 0) for s in samples
                        )
                        for key in ("enqueues", "continuations")
                    },
                    "metadata": dict(metadata or {}),
                }
            )
            if case.phase == "decode" and trace_step is not None:
                synchronize()
                trace_step(case, lambda: step(defer_reduce=overlap))
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
