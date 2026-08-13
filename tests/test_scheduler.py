from __future__ import annotations

import unittest
from typing import Any

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
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
    supports_continuation_prefill = True


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
    supports_continuation_prefill = True


class SchedulerConfigContractTests(unittest.TestCase):
    def test_runner_prefill_limit_is_an_execution_bound(self) -> None:
        class LimitedRunner(ContinuationRecordingRunner):
            prefill_chunk_size = 2

        runner = LimitedRunner()
        engine = Engine(runner)
        request = engine.submit([1, 2, 3, 4, 5], 1)

        engine.step()

        self.assertEqual(runner.plans[0].sequence_indptr, [0, 2])
        self.assertFalse(request.done)
        engine.close()

    def test_config_is_frozen_keyword_only_and_validated(self) -> None:
        default = SchedulerConfig()
        self.assertEqual(default.queue_mode, "round_robin")
        self.assertEqual(
            repr(default),
            "SchedulerConfig(queue_mode='round_robin', max_batch_tokens=None, "
            "prefill_chunk_size=8192)",
        )

        fifo = SchedulerConfig(queue_mode="fifo")
        self.assertEqual(fifo.queue_mode, "fifo")
        with self.assertRaises(AttributeError):
            fifo.queue_mode = "round_robin"
        with self.assertRaises(TypeError):
            SchedulerConfig("fifo")
        with self.assertRaisesRegex(
            ValueError, "queue_mode must be 'round_robin' or 'fifo'"
        ):
            SchedulerConfig(queue_mode="priority")
        limited = SchedulerConfig(max_batch_tokens=17, prefill_chunk_size=5)
        self.assertEqual(
            (limited.max_batch_tokens, limited.prefill_chunk_size),
            (17, 5),
        )
        self.assertEqual(
            repr(limited),
            "SchedulerConfig(queue_mode='round_robin', max_batch_tokens=17, "
            "prefill_chunk_size=5)",
        )
        for field in ("max_batch_tokens", "prefill_chunk_size"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"{field} must be positive"),
            ):
                SchedulerConfig(**{field: 0})

    def test_continuation_capability_is_exact_and_only_read_when_enabled(self) -> None:
        class Missing:
            def forward(self, plan):
                return [0] * plan.num_sequences

        class FalseRunner(Missing):
            supports_continuation_prefill = False

        class IntegerTrueRunner(Missing):
            supports_continuation_prefill = 1

        class RaisingDescriptor(Missing):
            @property
            def supports_continuation_prefill(self):
                raise RuntimeError("capability descriptor accessed")

        # Unmanaged construction must not inspect an optional descriptor.
        Engine(RaisingDescriptor()).close()
        for runner in (Missing(), FalseRunner(), IntegerTrueRunner()):
            with self.subTest(runner=type(runner).__name__):
                engine = Engine(
                    runner,
                    scheduler_config=SchedulerConfig(prefill_chunk_size=2),
                )
                engine.submit([1, 2, 3], 1)
                with self.assertRaisesRegex(
                    ValueError, "supports_continuation_prefill is True"
                ):
                    engine.dispatch()
                engine.close()
        raising_engine = Engine(
            RaisingDescriptor(),
            scheduler_config=SchedulerConfig(max_batch_tokens=2),
        )
        raising_engine.submit([1, 2, 3], 1)
        with self.assertRaisesRegex(RuntimeError, "capability descriptor accessed"):
            raising_engine.dispatch()
        raising_engine.close()

        engine = Engine(
            ContinuationRecordingRunner(),
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        self.assertEqual(engine.scheduler_config.prefill_chunk_size, 2)
        engine.close()

        layout = CacheLayout(2, [CacheGroupLayout(2)])

        class ContinuationCachedRunner(ContinuationRecordingRunner):
            cache_layout = layout

            def __init__(self) -> None:
                super().__init__()
                self.binding = None

            def bind_cache_pool(self, pool_id, bound_layout):
                self.binding = (pool_id, bound_layout)
                return pool_id

        runner = ContinuationCachedRunner()
        cached_engine = Engine(
            runner,
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        self.assertEqual(runner.binding, (cached_engine.cache_pool_id, layout))
        cached_engine.close()

    def test_builtin_runner_continuation_capabilities_are_explicit(self) -> None:
        from resin.backends.gpt_oss.cuda import GptOssCudaRunner
        from resin.backends.gpt_oss.reference import GptOssRunner
        from resin.backends.lfm2.cuda import Lfm2CudaRunner
        from resin.backends.lfm2.reference import Lfm2TorchRunner

        self.assertIs(GptOssCudaRunner.supports_continuation_prefill, True)
        self.assertIs(Lfm2CudaRunner.supports_continuation_prefill, True)
        self.assertIs(Lfm2TorchRunner.supports_continuation_prefill, True)
        self.assertIs(GptOssRunner.supports_continuation_prefill, True)

    def test_engine_freezes_a_copy_of_the_effective_config(self) -> None:
        runner = RecordingRunner()
        engine = Engine(
            runner,
            scheduler_config=SchedulerConfig(queue_mode="fifo"),
        )
        self.assertEqual(engine.scheduler_config.queue_mode, "fifo")
        with self.assertRaises(TypeError):
            Engine(runner, scheduler_config=object())

        default = Engine(RecordingRunner())
        self.assertEqual(default.scheduler_config.queue_mode, "round_robin")

    def test_batch_plan_batch_key_defaults_and_range_validation(self) -> None:
        default = BatchPlan([1], [0], [7], [0, 1], [0])
        keyed = BatchPlan([1], [0], [7], [0, 1], [0], batch_key=9)
        self.assertIsNone(default.batch_key)
        self.assertEqual(keyed.batch_key, 9)
        with self.assertRaises(OverflowError):
            BatchPlan([1], [0], [7], [0, 1], [0], batch_key=2**64)

        engine = Engine(RecordingRunner())
        with self.assertRaises(OverflowError):
            engine.submit([1], 1, priority=2**31)
        with self.assertRaises(OverflowError):
            engine.submit([1], 1, batch_key=2**64)

    def test_batch_plan_freezes_phase_and_sampling_rows(self) -> None:
        plan = BatchPlan(
            [10, 20, 30],
            [0, 4, 0],
            [7, 8, 9],
            [0, 1, 2, 3],
            [0, 4, 0],
            sample_sequence_indices=[1, 2],
        )
        self.assertEqual(plan.phase, BatchPhase.PREFILL)
        self.assertEqual(plan.sample_sequence_indices, [1, 2])
        self.assertEqual(plan.num_sequences, 3)
        self.assertEqual(len(RecordingRunner().forward(plan)), 3)
        with self.assertRaises(AttributeError):
            plan.sample_sequence_indices = [0]

    def test_batch_plan_phase_is_batch_homogeneous(self) -> None:
        prefill = BatchPlan(
            [1, 2],
            [0, 0],
            [7, 8],
            [0, 1, 2],
            [0, 0],
        )
        self.assertEqual(prefill.phase, BatchPhase.PREFILL)
        self.assertEqual(prefill.sample_sequence_indices, [0, 1])

        decode = BatchPlan(
            [1, 2],
            [3, 5],
            [7, 8],
            [0, 1, 2],
            [3, 5],
            phase=BatchPhase.DECODE,
        )
        self.assertEqual(decode.phase, BatchPhase.DECODE)

    def test_batch_plan_rejects_invalid_sampling_rows(self) -> None:
        for value in ([2, 1], [1, 1], [3]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BatchPlan(
                    [10, 20, 30],
                    [0, 4, 0],
                    [7, 8, 9],
                    [0, 1, 2, 3],
                    [0, 4, 0],
                    sample_sequence_indices=value,
                )

        with self.assertRaisesRegex(TypeError, "prefill_sequence_indices"):
            BatchPlan(
                [1],
                [0],
                [7],
                [0, 1],
                [0],
                prefill_sequence_indices=[0],
            )


class NativePolicyIntegrationTests(unittest.TestCase):
    def test_default_config_has_exact_legacy_round_robin_trace(self) -> None:
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
            self.assertEqual([first, second, third], [1, 2, 3])
            return [plan.sequence_ids for plan in runner.plans]

        expected = [[1, 2], [3]]
        self.assertEqual(trace(None), expected)
        self.assertEqual(trace(SchedulerConfig()), expected)

    def test_priority_is_strict_stable_and_never_backfills(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=4)
        low = engine.submit([10], 1, priority=0, batch_key=7)
        high_keyed_first = engine.submit([20], 1, priority=5, batch_key=7)
        high_none = engine.submit([30], 1, priority=5)
        high_keyed_second = engine.submit([40], 1, priority=5, batch_key=7)
        middle = engine.submit([50], 1, priority=4, batch_key=7)

        engine.step()
        self.assertEqual(
            runner.plans[-1].sequence_ids,
            [high_keyed_first, high_keyed_second],
        )
        self.assertEqual(runner.plans[-1].batch_key, 7)
        self.assertEqual(runner.plans[-1].num_sequences, 2)

        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [high_none])
        self.assertIsNone(runner.plans[-1].batch_key)
        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [middle])
        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [low])

    def test_none_and_zero_are_distinct_homogeneous_keys(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=3)
        none_first = engine.submit([1], 1)
        zero = engine.submit([2], 1, batch_key=0)
        none_second = engine.submit([3], 1)

        engine.step()
        self.assertEqual(
            runner.plans[-1].sequence_ids,
            [none_first, none_second],
        )
        self.assertIsNone(runner.plans[-1].batch_key)
        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [zero])
        self.assertEqual(runner.plans[-1].batch_key, 0)

    def test_decode_survivor_is_separate_from_fresh_prefill(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        decode = engine.submit([10], 2)
        engine.step()
        fresh = engine.submit([20], 1)
        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [decode])
        self.assertEqual(runner.plans[-1].phase, BatchPhase.DECODE)
        engine.step()
        self.assertEqual(runner.plans[-1].sequence_ids, [fresh])
        self.assertEqual(runner.plans[-1].phase, BatchPhase.PREFILL)

    def test_fifo_and_round_robin_diverge_only_in_survivor_turn_refresh(self) -> None:
        def next_batch(queue_mode: str) -> list[int]:
            runner = ControlledAsyncRunner([None, [11, 21], [31, 41]])
            engine = Engine(
                runner,
                max_running_requests=2,
                scheduler_config=SchedulerConfig(queue_mode=queue_mode),
            )
            first = engine.submit([10], 3)
            second = engine.submit([20], 3)
            third = engine.submit([30], 1)
            engine.dispatch()
            fourth = engine.submit([40], 1)
            self.assertEqual(engine.poll(), [])
            engine.poll()

            engine.dispatch()
            selected = runner.dispatch_plans[-1].sequence_ids
            engine.poll()
            engine.close()
            if queue_mode == "fifo":
                self.assertEqual(selected, [first, second])
            else:
                self.assertEqual(selected, [third, fourth])
            return selected

        self.assertEqual(next_batch("round_robin"), [3, 4])
        self.assertEqual(next_batch("fifo"), [1, 2])

    def test_noncontiguous_async_rejection_restores_exact_cohort_and_age(
        self,
    ) -> None:
        runner = ControlledAsyncRunner([[11, 31], [21]], dispatch_failures=1)
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], 1, priority=3, batch_key=7)
        other_key = engine.submit([20], 1, priority=3)
        third = engine.submit([30], 1, priority=3, batch_key=7)
        lower = engine.submit([40], 1, priority=2, batch_key=7)

        with self.assertRaisesRegex(RuntimeError, "dispatch rejected"):
            engine.dispatch()
        self.assertFalse(engine.poisoned)
        self.assertEqual(engine.ready_request_count, 4)

        self.assertEqual(engine.dispatch(), 2)
        failed, retry = runner.dispatch_plans[:2]
        self.assertEqual((failed.batch_id, retry.batch_id), (1, 2))
        self.assertEqual(failed.sequence_ids, [first, third])
        self.assertEqual(retry.sequence_ids, [first, third])
        for field in (
            "batch_key",
            "phase",
            "token_ids",
            "positions",
            "sequence_indptr",
            "context_lengths",
            "sample_sequence_indices",
        ):
            self.assertEqual(getattr(failed, field), getattr(retry, field), field)
        engine.poll()

        engine.dispatch()
        self.assertEqual(runner.dispatch_plans[-1].sequence_ids, [other_key])
        engine.poll()
        engine.close()
        self.assertTrue(lower.done)
        self.assertEqual(lower.take_result(), [])

    def test_ready_and_inflight_cancellation_preserve_selected_row_order(self) -> None:
        ready_runner = RecordingRunner()
        ready_engine = Engine(ready_runner, max_running_requests=3)
        first = ready_engine.submit([10], 1)
        middle = ready_engine.submit([20], 1)
        third = ready_engine.submit([30], 1)
        self.assertTrue(ready_engine.cancel(middle))
        ready_engine.step()
        self.assertEqual(ready_runner.plans[-1].sequence_ids, [first, third])

        async_runner = ControlledAsyncRunner([[101, 301], [201]])
        async_engine = Engine(async_runner, max_running_requests=2)
        first = async_engine.submit([100], 1, priority=2, batch_key=7)
        other_key = async_engine.submit([200], 1, priority=2)
        third = async_engine.submit([300], 1, priority=2, batch_key=7)
        async_engine.dispatch()
        self.assertEqual(async_runner.dispatch_plans[-1].sequence_ids, [first, third])
        self.assertTrue(async_engine.cancel(third))
        events = async_engine.poll()
        self.assertEqual([event.request_id for event in events], [first])
        self.assertEqual(async_runner.released, [first, third])

        async_engine.dispatch()
        self.assertEqual(async_runner.dispatch_plans[-1].sequence_ids, [other_key])
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
        self.assertEqual([event.token_id for event in engine.step()], [11])
        partial = engine.submit([20, 21, 22, 23], max_new_tokens=1)

        events = engine.step()
        plan = runner.plans[-1]
        self.assertEqual(plan.sequence_ids, [decoding])
        self.assertEqual(plan.phase, BatchPhase.DECODE)
        self.assertEqual(plan.token_ids, [11])
        self.assertEqual(plan.context_lengths, [1])
        self.assertEqual(plan.sample_sequence_indices, [0])
        self.assertEqual([event.request_id for event in events], [decoding])
        self.assertEqual([event.token_id for event in events], [12])
        self.assertFalse(partial.done)
        self.assertEqual(runner.released, [decoding])

        events = engine.step()
        self.assertEqual(events, [])
        self.assertEqual(runner.plans[-1].phase, BatchPhase.PREFILL)
        self.assertEqual(runner.plans[-1].context_lengths, [0])
        self.assertEqual(runner.plans[-1].sample_sequence_indices, [])
        self.assertFalse(partial.done)

        events = engine.step()
        self.assertEqual(runner.plans[-1].context_lengths, [2])
        self.assertEqual(runner.plans[-1].sample_sequence_indices, [0])
        self.assertEqual([event.request_id for event in events], [partial])
        self.assertTrue(partial.done)
        self.assertEqual(partial.take_result(), [24])
        self.assertEqual(runner.released, [decoding, partial])

    def test_partial_prefill_survivors_obey_fifo_and_round_robin(self) -> None:
        def trace(queue_mode: str) -> list[int]:
            runner = ContinuationRecordingRunner()
            engine = Engine(
                runner,
                max_running_requests=2,
                scheduler_config=SchedulerConfig(
                    queue_mode=queue_mode,
                    max_batch_tokens=1,
                    prefill_chunk_size=1,
                ),
            )
            first = engine.submit([1, 2], 1)
            second = engine.submit([10, 11], 1)
            while engine.has_work():
                engine.step()
            self.assertEqual((first.take_result(), second.take_result()), ([3], [12]))
            return [plan.sequence_ids[0] for plan in runner.plans]

        self.assertEqual(trace("fifo"), [1, 1, 2, 2])
        self.assertEqual(trace("round_robin"), [1, 2, 1, 2])

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
        self.assertEqual(runner.dispatch_plans[-1].sample_sequence_indices, [])
        self.assertEqual(engine.poll(), [])
        self.assertTrue(engine.cancel(request))
        self.assertEqual(engine.poll(), [])
        self.assertTrue(request.done)
        self.assertEqual(request.take_result(), [])
        self.assertEqual(runner.released, [request])

        compact = ContinuationControlledAsyncRunner([[]])
        poisoned = Engine(
            compact,
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        poisoned.submit([1, 2], 1)
        poisoned.dispatch()
        with self.assertRaisesRegex(RuntimeError, "1 sequences"):
            poisoned.poll()
        self.assertTrue(poisoned.poisoned)

    def test_cleanup_failure_does_not_publish_a_sampled_token(self) -> None:
        class FailingRelease(ContinuationRecordingRunner):
            def release(self, request_ids: list[int]) -> None:
                raise RuntimeError("cleanup failed")

        engine = Engine(
            FailingRelease(),
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        request = engine.submit([4], 1)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            engine.step()
        self.assertTrue(engine.poisoned)
        self.assertFalse(request.done)


if __name__ == "__main__":
    unittest.main()
