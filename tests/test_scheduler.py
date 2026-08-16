from __future__ import annotations

import inspect
from typing import Any

import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    Engine,
    ResourceBinding,
    RunnerConfig,
    SchedulerConfig,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        return [
            plan.token_ids[end - 1] + 1
            for _, end in map(plan.token_range, range(plan.num_sequences))
        ]

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class ContinuationRecordingRunner(RecordingRunner):
    runner_config = RunnerConfig(continuation_prefill=True)


class ControlledAsyncRunner:
    def __init__(
        self,
        poll_results: list[list[int] | None] | None = None,
        *,
        dispatch_failures: int = 0,
    ) -> None:
        self.poll_results = list(poll_results or [])
        self.dispatch_failures = dispatch_failures
        self.dispatch_plans: list[BatchPlan] = []
        self.released: list[int] = []
        self._accepted: set[object] = set()

    def dispatch(self, plan: BatchPlan) -> object:
        self.dispatch_plans.append(plan)
        if self.dispatch_failures:
            self.dispatch_failures -= 1
            raise RuntimeError("dispatch rejected")
        handle = object()
        self._accepted.add(handle)
        return handle

    def poll(self, handle: object) -> list[int] | None:
        if handle not in self._accepted:
            raise ValueError("foreign completion")
        if not self.poll_results:
            return None
        result = self.poll_results.pop(0)
        if result is not None:
            self._accepted.remove(handle)
        return result

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class ContinuationControlledAsyncRunner(ControlledAsyncRunner):
    runner_config = RunnerConfig(continuation_prefill=True)


class TestSchedulerConfigContract:
    def test_runner_prefill_limit_is_an_execution_bound(self) -> None:
        class LimitedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=2,
            )

        runner = LimitedRunner()
        engine = Engine(runner)
        request = engine.submit([1, 2, 3, 4, 5], 1)

        engine.step()

        assert runner.plans[0].token_indptr == [0, 2]
        assert not request.done
        engine.close()

    def test_runner_batch_limit_is_consumed_in_request_order(self) -> None:
        class LimitedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=8192,
                max_batch_tokens=4,
            )

        runner = LimitedRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([1, 2, 3, 4], 1)
        second = engine.submit([5, 6, 7, 8], 1)

        engine.step()

        assert runner.plans[0].sequence_ids == [first]
        assert runner.plans[0].token_indptr == [0, 4]
        assert first.done
        assert not second.done
        engine.close()

    def test_window_prefill_spans_multiple_pages_by_default(self) -> None:
        layout = CacheLayout(
            8,
            [CacheGroupLayout(2, retention=CacheRetention.WINDOW, window_tokens=8)],
        )

        class MultiPageRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=8,
                max_batch_tokens=8,
                cache_layout=layout,
            )

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.bound_layout = binding.cache_layout

        runner = MultiPageRunner()
        engine = Engine(runner)
        request = engine.submit([1, 2, 3, 4, 5, 6], 1)

        events = engine.step()

        assert runner.plans[0].token_ids == [1, 2, 3, 4, 5, 6]
        assert [event.request_id for event in events] == [request]
        assert request.done
        engine.close()

    def test_config_is_frozen_keyword_only_and_validated(self) -> None:
        default = SchedulerConfig()
        assert (
            repr(default)
            == "SchedulerConfig(max_batch_tokens=None, prefill_chunk_size=8192, schedule_conservativeness=1.0)"
        )

        with pytest.raises(TypeError):
            SchedulerConfig(17)
        limited = SchedulerConfig(max_batch_tokens=17, prefill_chunk_size=5)
        assert (limited.max_batch_tokens, limited.prefill_chunk_size) == (17, 5)
        assert (
            repr(limited)
            == "SchedulerConfig(max_batch_tokens=17, prefill_chunk_size=5, schedule_conservativeness=1.0)"
        )
        for field in ("max_batch_tokens", "prefill_chunk_size"):
            with (
                pytest.raises(ValueError, match=f"{field} must be positive"),
            ):
                SchedulerConfig(**{field: 0})
        for value in (-0.1, float("inf"), float("nan")):
            with (
                pytest.raises(
                    ValueError,
                    match="schedule_conservativeness must be finite and non-negative",
                ),
            ):
                SchedulerConfig(schedule_conservativeness=value)
        aggressive = SchedulerConfig(schedule_conservativeness=0.0)
        assert aggressive.schedule_conservativeness == 0.0
        default_engine = Engine(RecordingRunner())
        assert default_engine.new_token_ratio == 0.7
        default_engine.close()
        conservative = Engine(
            RecordingRunner(),
            scheduler_config=SchedulerConfig(schedule_conservativeness=2.0),
        )
        assert conservative.new_token_ratio == 1.0
        conservative.close()

    def test_continuation_capability_is_exact_and_only_read_when_enabled(self) -> None:
        class Missing:
            def forward(self, plan):
                return [0] * plan.num_sequences

        class FalseRunner(Missing):
            runner_config = RunnerConfig(continuation_prefill=False)

        class IntegerTrueRunner(Missing):
            runner_config = 1

        class RaisingDescriptor(Missing):
            @property
            def runner_config(self):
                raise RuntimeError("capability descriptor accessed")

        with pytest.raises(RuntimeError, match="capability descriptor accessed"):
            Engine(RaisingDescriptor())
        with pytest.raises(TypeError, match="exactly a RunnerConfig"):
            Engine(IntegerTrueRunner())
        for runner in (Missing(), FalseRunner()):
            engine = Engine(
                runner,
                scheduler_config=SchedulerConfig(prefill_chunk_size=2),
            )
            engine.submit([1, 2, 3], 1)
            with pytest.raises(
                ValueError, match="runner_config.continuation_prefill is false"
            ):
                engine.dispatch()
            engine.close()

        engine = Engine(
            ContinuationRecordingRunner(),
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        assert engine.scheduler_config.prefill_chunk_size == 2
        engine.close()

        layout = CacheLayout(2, [CacheGroupLayout(2)])

        class ContinuationCachedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                cache_layout=layout,
            )

            def __init__(self) -> None:
                super().__init__()
                self.binding = None

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.binding = (binding.cache_pool_id, binding.cache_layout)

        runner = ContinuationCachedRunner()
        cached_engine = Engine(
            runner,
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        assert runner.binding == (cached_engine.cache_pool_id, layout)
        cached_engine.close()

    def test_builtin_runner_continuation_capabilities_are_explicit(self) -> None:
        from resin_models_cuda.gpt_oss import GptOssCudaRunner
        from resin_models_cuda.lfm2 import Lfm2CudaRunner
        from resin_models_torch.lfm2 import Lfm2TorchRunner

        gpt_source = inspect.getsource(GptOssCudaRunner.__init__)
        lfm_source = inspect.getsource(Lfm2CudaRunner.__init__)
        assert (
            "RunnerConfig(" in gpt_source and "continuation_prefill=True" in gpt_source
        )
        assert "max_batch_tokens=_PREFILL_CHUNK_SIZE" in gpt_source
        assert (
            "RunnerConfig(" in lfm_source and "continuation_prefill=True" in lfm_source
        )
        assert Lfm2TorchRunner.runner_config.continuation_prefill is True

    def test_engine_freezes_a_copy_of_the_effective_config(self) -> None:
        runner = RecordingRunner()
        config = SchedulerConfig(max_batch_tokens=17)
        engine = Engine(runner, scheduler_config=config)
        assert engine.scheduler_config.max_batch_tokens == 17
        with pytest.raises(TypeError):
            Engine(runner, scheduler_config=object())

        default = Engine(RecordingRunner())
        assert default.scheduler_config.max_batch_tokens is None

    def test_batch_plan_batch_key_defaults_and_range_validation(self) -> None:
        default = BatchPlan([7], [0], [1], [0, 1])
        keyed = BatchPlan([7], [0], [1], [0, 1], batch_key=9)
        assert default.batch_key is None
        assert keyed.batch_key == 9
        with pytest.raises(OverflowError):
            BatchPlan([7], [0], [1], [0, 1], batch_key=2**64)

        engine = Engine(RecordingRunner())
        with pytest.raises(OverflowError):
            engine.submit([1], 1, priority=2**31)
        with pytest.raises(OverflowError):
            engine.submit([1], 1, batch_key=2**64)

    def test_batch_plan_freezes_phase_and_sampling_rows(self) -> None:
        plan = BatchPlan(
            [7, 8, 9],
            [0, 4, 0],
            [10, 20, 30],
            [0, 1, 2, 3],
            sample_sequence_indices=[1, 2],
        )
        assert plan.phase == BatchPhase.PREFILL
        assert plan.sample_sequence_indices == [1, 2]
        assert plan.num_sequences == 3
        assert len(RecordingRunner().forward(plan)) == 3
        with pytest.raises(AttributeError):
            plan.sample_sequence_indices = [0]

    def test_batch_plan_phase_is_batch_homogeneous(self) -> None:
        prefill = BatchPlan(
            [7, 8],
            [0, 0],
            [1, 2],
            [0, 1, 2],
        )
        assert prefill.phase == BatchPhase.PREFILL
        assert prefill.sample_sequence_indices == [0, 1]

        decode = BatchPlan(
            [7, 8],
            [3, 5],
            [1, 2],
            [0, 1, 2],
            phase=BatchPhase.DECODE,
        )
        assert decode.phase == BatchPhase.DECODE

    def test_batch_plan_rejects_invalid_sampling_rows(self) -> None:
        for value in ([2, 1], [1, 1], [3]):
            with pytest.raises(ValueError):
                BatchPlan(
                    [7, 8, 9],
                    [0, 4, 0],
                    [10, 20, 30],
                    [0, 1, 2, 3],
                    sample_sequence_indices=value,
                )

        with pytest.raises(TypeError, match="prefill_sequence_indices"):
            BatchPlan(
                [7],
                [0],
                [1],
                [0, 1],
                prefill_sequence_indices=[0],
            )


class TestNativePolicyIntegration:
    def test_default_config_keeps_survivors_ahead_of_waiting_requests(self) -> None:
        def trace(config: SchedulerConfig | None) -> list[list[int]]:
            runner = RecordingRunner()
            kwargs: dict[str, Any] = {"max_running_requests": 2}
            if config is not None:
                kwargs["scheduler_config"] = config
            engine = Engine(runner, **kwargs)
            first = engine.submit([1], max_new_tokens=3)
            second = engine.submit([10], max_new_tokens=1)
            third = engine.submit([20], max_new_tokens=1)
            engine.step()
            engine.step()
            engine.close()
            assert [first, second, third] == [1, 2, 3]
            return [plan.sequence_ids for plan in runner.plans]

        expected = [[1, 2], [3]]
        assert trace(None) == expected
        assert trace(SchedulerConfig()) == expected

    def test_priority_is_strict_stable_and_never_backfills(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=4)
        low = engine.submit([10], 1, priority=0, batch_key=7)
        high_keyed_first = engine.submit([20], 1, priority=5, batch_key=7)
        high_none = engine.submit([30], 1, priority=5)
        high_keyed_second = engine.submit([40], 1, priority=5, batch_key=7)
        middle = engine.submit([50], 1, priority=4, batch_key=7)

        engine.step()
        assert runner.plans[-1].sequence_ids == [high_keyed_first, high_keyed_second]
        assert runner.plans[-1].batch_key == 7
        assert runner.plans[-1].num_sequences == 2

        engine.step()
        assert runner.plans[-1].sequence_ids == [high_none]
        assert runner.plans[-1].batch_key is None
        engine.step()
        assert runner.plans[-1].sequence_ids == [middle]
        engine.step()
        assert runner.plans[-1].sequence_ids == [low]

    def test_none_and_zero_are_distinct_homogeneous_keys(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=3)
        none_first = engine.submit([1], 1)
        zero = engine.submit([2], 1, batch_key=0)
        none_second = engine.submit([3], 1)

        engine.step()
        assert runner.plans[-1].sequence_ids == [none_first, none_second]
        assert runner.plans[-1].batch_key is None
        engine.step()
        assert runner.plans[-1].sequence_ids == [zero]
        assert runner.plans[-1].batch_key == 0

    def test_fresh_prefill_runs_before_the_decode_batch_resumes(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        decode = engine.submit([10], 2)
        engine.step()
        fresh = engine.submit([20], 1)
        engine.step()
        assert runner.plans[-1].sequence_ids == [fresh]
        assert runner.plans[-1].phase == BatchPhase.PREFILL
        engine.step()
        assert runner.plans[-1].sequence_ids == [decode]
        assert runner.plans[-1].phase == BatchPhase.DECODE

    def test_survivors_stay_ahead_of_inflight_submissions(self) -> None:
        runner = ControlledAsyncRunner([None, [11, 21], [12, 22]])
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], 3)
        second = engine.submit([20], 3)
        engine.dispatch()
        engine.submit([30], 1)
        engine.submit([40], 1)
        assert engine.poll() == []
        engine.poll()

        engine.dispatch()
        assert runner.dispatch_plans[-1].sequence_ids == [first, second]
        engine.poll()
        engine.close()

    def test_noncontiguous_async_rejection_restores_exact_cohort_and_age(
        self,
    ) -> None:
        runner = ControlledAsyncRunner([[11, 31], [21]], dispatch_failures=1)
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], 1, priority=3, batch_key=7)
        other_key = engine.submit([20], 1, priority=3)
        third = engine.submit([30], 1, priority=3, batch_key=7)
        lower = engine.submit([40], 1, priority=2, batch_key=7)

        with pytest.raises(RuntimeError, match="dispatch rejected"):
            engine.dispatch()
        assert not engine.poisoned
        assert engine.ready_request_count == 4

        assert engine.dispatch() == 2
        failed, retry = runner.dispatch_plans[:2]
        assert (failed.batch_id, retry.batch_id) == (1, 2)
        assert failed.sequence_ids == [first, third]
        assert retry.sequence_ids == [first, third]
        for field in (
            "batch_key",
            "phase",
            "token_ids",
            "token_indptr",
            "context_lengths",
            "sample_sequence_indices",
        ):
            assert getattr(failed, field) == getattr(retry, field), field
        engine.poll()

        engine.dispatch()
        assert runner.dispatch_plans[-1].sequence_ids == [other_key]
        engine.poll()
        engine.close()
        assert lower.done
        assert lower.take_result() == []

    def test_ready_and_inflight_cancellation_preserve_selected_row_order(self) -> None:
        ready_runner = RecordingRunner()
        ready_engine = Engine(ready_runner, max_running_requests=3)
        first = ready_engine.submit([10], 1)
        middle = ready_engine.submit([20], 1)
        third = ready_engine.submit([30], 1)
        assert ready_engine.cancel(middle)
        ready_engine.step()
        assert ready_runner.plans[-1].sequence_ids == [first, third]

        async_runner = ControlledAsyncRunner([[101, 301], [201]])
        async_engine = Engine(async_runner, max_running_requests=2)
        first = async_engine.submit([100], 1, priority=2, batch_key=7)
        other_key = async_engine.submit([200], 1, priority=2)
        third = async_engine.submit([300], 1, priority=2, batch_key=7)
        async_engine.dispatch()
        assert async_runner.dispatch_plans[-1].sequence_ids == [first, third]
        assert async_engine.cancel(third)
        events = async_engine.poll()
        assert [event.request_id for event in events] == [first]
        assert async_runner.released == [first, third]

        async_engine.dispatch()
        assert async_runner.dispatch_plans[-1].sequence_ids == [other_key]
        async_engine.poll()

    def test_decode_and_partial_prefill_are_dispatched_separately(
        self,
    ) -> None:
        runner = ContinuationRecordingRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            eos_token_id=22,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=3,
                prefill_chunk_size=2,
            ),
        )
        decoding = engine.submit([10], max_new_tokens=2)
        assert [event.token_id for event in engine.step()] == [11]
        partial = engine.submit([20, 21, 22, 23], max_new_tokens=1)

        events = engine.step()
        plan = runner.plans[-1]
        assert plan.sequence_ids == [partial]
        assert plan.phase == BatchPhase.PREFILL
        assert plan.token_ids == [20, 21]
        assert plan.context_lengths == [0]
        assert plan.sample_sequence_indices == []
        assert events == []
        assert not partial.done
        assert runner.released == []

        events = engine.step()
        assert runner.plans[-1].phase == BatchPhase.PREFILL
        assert runner.plans[-1].context_lengths == [2]
        assert runner.plans[-1].sample_sequence_indices == [0]
        assert [event.request_id for event in events] == [partial]
        assert partial.done
        assert partial.take_result() == [24]
        assert runner.released == [partial]

        events = engine.step()
        assert runner.plans[-1].sequence_ids == [decoding]
        assert runner.plans[-1].phase == BatchPhase.DECODE
        assert runner.plans[-1].token_ids == [11]
        assert [event.request_id for event in events] == [decoding]
        assert runner.released == [partial, decoding]

    def test_partial_prefill_survivor_finishes_before_waiting_request(self) -> None:
        runner = ContinuationRecordingRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=1,
                prefill_chunk_size=1,
            ),
        )
        first = engine.submit([1, 2], 1)
        second = engine.submit([10, 11], 1)
        while engine.has_work():
            engine.step()
        assert (first.take_result(), second.take_result()) == ([3], [12])
        assert [plan.sequence_ids[0] for plan in runner.plans] == [1, 1, 2, 2]

    def test_async_partial_cancel_discards_dense_output_and_compact_output_poisoned(
        self,
    ) -> None:
        runner = ContinuationControlledAsyncRunner([None, [99]])
        engine = Engine(
            runner,
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        request = engine.submit([1, 2, 3], 1)
        engine.dispatch()
        assert runner.dispatch_plans[-1].sample_sequence_indices == []
        assert engine.poll() == []
        assert engine.cancel(request)
        assert engine.poll() == []
        assert request.done
        assert request.take_result() == []
        assert runner.released == [request]

        compact = ContinuationControlledAsyncRunner([[]])
        poisoned = Engine(
            compact,
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        poisoned.submit([1, 2], 1)
        poisoned.dispatch()
        with pytest.raises(RuntimeError, match="1 sequences"):
            poisoned.poll()
        assert poisoned.poisoned

    def test_cleanup_failure_does_not_publish_a_sampled_token(self) -> None:
        class FailingRelease(ContinuationRecordingRunner):
            def release(self, request_ids: list[int]) -> None:
                raise RuntimeError("cleanup failed")

        engine = Engine(
            FailingRelease(),
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        request = engine.submit([4], 1)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.step()
        assert engine.poisoned
        assert not request.done
