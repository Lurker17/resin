from __future__ import annotations

import gc
import unittest
from collections.abc import Callable

from resin import BatchPlan, CacheGroupLayout, CacheLayout, Engine, StatePool

PollResult = list[int] | tuple[int, ...] | BaseException | None


def history_layout(
    parent_capacity: int,
    block_tokens: int = 2,
    *,
    group_count: int = 1,
) -> CacheLayout:
    return CacheLayout(
        parent_capacity,
        [CacheGroupLayout(block_tokens) for _ in range(group_count)],
    )


class ControlledAsyncRunner:
    def __init__(
        self,
        poll_results: list[PollResult] | None = None,
        *,
        dispatch_failures: int = 0,
        release_error: BaseException | None = None,
        cache_layout: CacheLayout | None = None,
    ) -> None:
        self.poll_results = list(poll_results or [])
        self.dispatch_failures = dispatch_failures
        self.release_error = release_error
        self.dispatch_plans: list[BatchPlan] = []
        self.poll_handles: list[object] = []
        self.released: list[int] = []
        self.cache_binding: tuple[int, CacheLayout] | None = None
        if cache_layout is not None:
            self.cache_layout = cache_layout
        self._accepted: dict[object, BatchPlan] = {}

    def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
        self.cache_binding = (pool_id, layout)
        return pool_id

    def dispatch(self, plan: BatchPlan) -> object:
        self.dispatch_plans.append(plan)
        if self.dispatch_failures:
            self.dispatch_failures -= 1
            raise RuntimeError("dispatch rejected before acceptance")
        handle = object()
        self._accepted[handle] = plan
        return handle

    def poll(self, handle: object):
        if handle not in self._accepted:
            raise ValueError("foreign or stale handle")
        self.poll_handles.append(handle)
        if not self.poll_results:
            return None
        result = self.poll_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result is not None:
            self._accepted.pop(handle)
        return result

    def release(self, request_ids: list[int]) -> None:
        if self.release_error is not None:
            raise self.release_error
        self.released.extend(request_ids)


class AsyncEngineContractTests(unittest.TestCase):
    def test_sync_runner_keeps_step_and_explicit_dispatch_poll_behavior(self) -> None:
        class SyncRunner:
            def __init__(self) -> None:
                self.plans = []
                self.released = []

            def forward(self, plan):
                self.plans.append(plan)
                return [plan.token_ids[end - 1] + 1 for end in plan.sequence_indptr[1:]]

            def release(self, request_ids):
                self.released.extend(request_ids)

        runner = SyncRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], max_new_tokens=2)
        second = engine.submit([10], max_new_tokens=1)

        self.assertFalse(engine.asynchronous)
        self.assertEqual(engine.ready_request_count, 2)
        batch_id = engine.dispatch()
        self.assertEqual(batch_id, 1)
        self.assertEqual(engine.ready_request_count, 1)
        self.assertEqual(engine.in_flight_request_count, 1)
        self.assertEqual(engine.in_flight_batch_id, batch_id)
        first_events = engine.poll()
        self.assertEqual([event.request_id for event in first_events], [first])
        self.assertFalse(first.done)

        # Existing step() remains one dispatch plus one poll. Round-robin keeps
        # the previously unscheduled request at the front.
        second_events = engine.step()
        self.assertEqual([event.request_id for event in second_events], [second])
        self.assertTrue(second.done)
        self.assertEqual(second.take_result(), [11])
        self.assertEqual(runner.released, [second])

    def test_marked_sync_failure_quarantines_accepted_native_resources(self) -> None:
        class PostAcceptError(RuntimeError):
            _resin_post_accept = True

        class SyncRunner:
            cache_layout = history_layout(2)

            def __init__(self) -> None:
                self.calls = 0
                self.released = []

            def forward(self, plan):
                del plan
                self.calls += 1
                raise PostAcceptError("CUDA quiescence failed")

            @staticmethod
            def bind_cache_pool(pool_id, layout):
                del layout
                return pool_id

            def release(self, request_ids):
                self.released.extend(request_ids)

        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = SyncRunner()
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([1, 2], max_new_tokens=1, state_handles=[state])
        pool.release([state])

        with self.assertRaisesRegex(PostAcceptError, "quiescence failed"):
            engine.dispatch()

        self.assertTrue(engine.poisoned)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(runner.released, [])
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (0, 1)
        )
        self.assertEqual(engine.in_flight_batch_id, 1)
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual(pool.access_counts(state), (0, True))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (1, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "resources are quarantined"):
            engine.cancel(request)
        with self.assertRaisesRegex(RuntimeError, "resources are quarantined"):
            engine.close()
        self.assertEqual(runner.calls, 1)

    def test_unmarked_sync_failure_remains_preaccept_and_retryable(self) -> None:
        class SyncRunner:
            cache_layout = history_layout(2)

            def __init__(self) -> None:
                self.calls = 0

            def forward(self, plan):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("retryable forward failure")
                return [plan.token_ids[-1] + 1]

            @staticmethod
            def bind_cache_pool(pool_id, layout):
                del layout
                return pool_id

        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = SyncRunner()
        engine = Engine(runner, state_pool=pool)
        engine.submit([1, 2], max_new_tokens=1, state_handles=[state])
        pool.release([state])

        with self.assertRaisesRegex(RuntimeError, "retryable forward failure"):
            engine.dispatch()
        self.assertFalse(engine.poisoned)
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (1, 0)
        )
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

        self.assertEqual(engine.dispatch(), 2)
        self.assertEqual([event.token_id for event in engine.poll()], [3])
        self.assertEqual(runner.calls, 2)
        self.assertFalse(pool.contains(state))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (2, 0),
        )

    def test_malformed_sync_result_quarantines_after_forward_returns(self) -> None:
        for label, malformed, error_type in (
            ("not a token sequence", object(), TypeError),
            ("wrong token count", [], RuntimeError),
        ):
            with self.subTest(label=label):

                class SyncRunner:
                    cache_layout = history_layout(2)

                    def __init__(self) -> None:
                        self.calls = 0
                        self.released: list[int] = []

                    def forward(self, plan):
                        del plan
                        self.calls += 1
                        return malformed

                    @staticmethod
                    def bind_cache_pool(pool_id, layout):
                        del layout
                        return pool_id

                    def release(self, request_ids):
                        self.released.extend(request_ids)

                pool = StatePool(1)
                state = pool.allocate()[0]
                runner = SyncRunner()
                engine = Engine(runner, state_pool=pool)
                request = engine.submit([1, 2], max_new_tokens=1, state_handles=[state])
                pool.release([state])

                with self.assertRaises(error_type):
                    engine.dispatch()

                self.assertTrue(engine.poisoned)
                self.assertEqual(runner.calls, 1)
                self.assertEqual(runner.released, [])
                self.assertEqual(
                    (engine.ready_request_count, engine.in_flight_request_count),
                    (0, 1),
                )
                self.assertEqual(pool.counts(state), (1, 1))
                self.assertEqual(pool.access_counts(state), (0, True))
                self.assertEqual(
                    (
                        engine.cache_free_parent_count,
                        engine.cache_occupied_block_count,
                    ),
                    (1, 1),
                )
                with self.assertRaisesRegex(RuntimeError, "resources are quarantined"):
                    engine.cancel(request)
                self.assertEqual(runner.calls, 1)

    def test_marked_async_dispatch_failure_quarantines_accepted_resources(
        self,
    ) -> None:
        class PostAcceptError(RuntimeError):
            _resin_post_accept = True

        class AsyncRunner:
            cache_layout = history_layout(2)

            def __init__(self) -> None:
                self.calls = 0
                self.released: list[int] = []

            @staticmethod
            def bind_cache_pool(pool_id, layout):
                del layout
                return pool_id

            def dispatch(self, plan):
                del plan
                self.calls += 1
                raise PostAcceptError("dispatch lost its accepted completion")

            @staticmethod
            def poll(completion):
                raise AssertionError(f"lost completion was polled: {completion!r}")

            def release(self, request_ids):
                self.released.extend(request_ids)

        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = AsyncRunner()
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([1, 2], max_new_tokens=1, state_handles=[state])
        pool.release([state])

        with self.assertRaisesRegex(PostAcceptError, "lost its accepted completion"):
            engine.dispatch()

        self.assertTrue(engine.poisoned)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(runner.released, [])
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (0, 1)
        )
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual(pool.access_counts(state), (0, True))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (1, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "resources are quarantined"):
            engine.cancel(request)
        self.assertEqual(runner.calls, 1)

    def test_pending_completion_keeps_state_pinned_and_cache_blocks_owned(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(1),
        )
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([1, 2], max_new_tokens=1, state_handles=[state])
        pool.release([state])

        self.assertTrue(engine.asynchronous)
        self.assertEqual(engine.dispatch(), 1)
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (0, 1)
        )
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (0, 1),
        )

        self.assertEqual(engine.poll(), [])
        self.assertEqual(pool.counts(state), (1, 1))
        with self.assertRaises(MemoryError):
            engine.submit([9], max_new_tokens=1)

        events = engine.poll()
        self.assertEqual([event.token_id for event in events], [7])
        self.assertTrue(events[0].finished)
        self.assertEqual(runner.released, [request])
        self.assertFalse(pool.contains(state))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (1, 0),
        )
        self.assertEqual(engine.in_flight_request_count, 0)

    def test_in_flight_cancel_waits_for_terminal_poll_and_discards_output(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner(
            [None, [99]],
            cache_layout=history_layout(2),
        )
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([4], max_new_tokens=3, state_handles=[state])
        pool.release([state])
        engine.dispatch()

        self.assertTrue(engine.cancel(request))
        self.assertTrue(engine.cancel(request))
        self.assertEqual(engine.poll(), [])
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual(engine.cache_occupied_block_count, 2)
        with self.assertRaises(MemoryError):
            engine.submit([8], max_new_tokens=1)

        self.assertEqual(engine.poll(), [])
        self.assertTrue(request.done)
        self.assertEqual(request.take_result(), [])
        self.assertEqual(runner.released, [request])
        self.assertFalse(pool.contains(state))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (2, 0),
        )
        self.assertFalse(engine.has_work())

    def test_in_flight_cancel_preserves_result_alignment_for_survivors(self) -> None:
        runner = ControlledAsyncRunner([[11, 21]])
        engine = Engine(runner, max_running_requests=2)
        canceled = engine.submit([10], max_new_tokens=2)
        surviving = engine.submit([20], max_new_tokens=2)
        engine.dispatch()

        self.assertTrue(engine.cancel(canceled))
        events = engine.poll()

        self.assertEqual(
            [(event.request_id, event.token_id) for event in events],
            [(surviving, 21)],
        )
        self.assertTrue(canceled.done)
        self.assertEqual(canceled.take_result(), [])
        self.assertFalse(surviving.done)
        self.assertEqual(runner.released, [canceled])
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (1, 0)
        )

    def test_pending_completion_owns_all_ordered_cache_groups(self) -> None:
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(2, block_tokens=1, group_count=2),
        )
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)
        engine.dispatch()

        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (0, 2),
        )
        self.assertEqual(engine.poll(), [])
        with self.assertRaises(MemoryError):
            engine.submit([2], max_new_tokens=1)

        events = engine.poll()
        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (2, 0),
        )

    def test_ready_cancel_retires_existing_physical_state_immediately(self) -> None:
        runner = ControlledAsyncRunner([[3]], cache_layout=history_layout(2))
        engine = Engine(runner)
        request = engine.submit([1, 2], max_new_tokens=2)

        engine.dispatch()
        events = engine.poll()
        self.assertFalse(events[0].finished)
        self.assertEqual(engine.ready_request_count, 1)
        self.assertEqual(engine.cache_occupied_block_count, 2)

        self.assertTrue(engine.cancel(request))
        self.assertFalse(engine.cancel(request))
        self.assertTrue(request.done)
        self.assertEqual(request.take_result(), [3])
        self.assertEqual(runner.released, [request])
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (2, 0),
        )
        self.assertFalse(engine.has_work())

    def test_dispatch_rejection_restores_ready_order_and_temporary_pins(self) -> None:
        pool = StatePool(3)
        first_state, second_state, tail_state = pool.allocate(3)
        runner = ControlledAsyncRunner([[11, 21]], dispatch_failures=1)
        engine = Engine(runner, max_running_requests=2, state_pool=pool)
        first = engine.submit([10], max_new_tokens=1, state_handles=[first_state])
        second = engine.submit([20], max_new_tokens=1, state_handles=[second_state])
        tail = engine.submit([30], max_new_tokens=1, state_handles=[tail_state])
        pool.release([first_state, second_state, tail_state])

        with self.assertRaisesRegex(RuntimeError, "before acceptance"):
            engine.dispatch()
        self.assertFalse(engine.poisoned)
        self.assertEqual(
            (engine.ready_request_count, engine.in_flight_request_count), (3, 0)
        )
        self.assertEqual(pool.counts(first_state), (1, 0))
        self.assertEqual(pool.counts(second_state), (1, 0))
        self.assertEqual(pool.counts(tail_state), (1, 0))

        self.assertEqual(engine.dispatch(), 2)
        self.assertEqual(
            [plan.sequence_ids for plan in runner.dispatch_plans],
            [[first, second], [first, second]],
        )
        failed_plan, retry_plan = runner.dispatch_plans
        self.assertEqual((failed_plan.batch_id, retry_plan.batch_id), (1, 2))
        for field in (
            "phase",
            "token_ids",
            "positions",
            "sequence_ids",
            "sequence_indptr",
            "context_lengths",
            "state_slots",
            "state_generations",
            "state_indptr",
        ):
            self.assertEqual(
                getattr(failed_plan, field),
                getattr(retry_plan, field),
                field,
            )
        events = engine.poll()
        self.assertEqual([event.request_id for event in events], [first, second])
        self.assertEqual(engine.ready_request_count, 1)

        runner.poll_results.append([31])
        tail_events = engine.step()
        self.assertEqual([event.request_id for event in tail_events], [tail])
        self.assertEqual(runner.dispatch_plans[-1].sequence_ids, [tail])

    def test_canceled_terminal_and_surviving_rows_preserve_tail_rotation(self) -> None:
        runner = ControlledAsyncRunner([[11, 21, 31], [41], [32]])
        engine = Engine(runner, max_running_requests=3)
        canceled = engine.submit([10], max_new_tokens=2)
        terminal = engine.submit([20], max_new_tokens=1)
        surviving = engine.submit([30], max_new_tokens=2)
        untouched = engine.submit([40], max_new_tokens=1)

        engine.dispatch()
        self.assertTrue(engine.cancel(canceled))
        first_events = engine.poll()
        self.assertEqual(
            [event.request_id for event in first_events],
            [terminal, surviving],
        )
        self.assertEqual(runner.released, [canceled, terminal])

        second_events = engine.step()
        self.assertEqual(
            runner.dispatch_plans[-1].sequence_ids,
            [untouched],
        )
        self.assertEqual(
            [event.request_id for event in second_events],
            [untouched],
        )

        third_events = engine.step()
        self.assertEqual(runner.dispatch_plans[-1].sequence_ids, [surviving])
        self.assertEqual(
            [event.request_id for event in third_events],
            [surviving],
        )

    def test_async_step_dispatches_and_polls_exactly_once(self) -> None:
        runner = ControlledAsyncRunner([None, [5]])
        engine = Engine(runner)
        request = engine.submit([4], max_new_tokens=1)

        self.assertEqual(engine.step(), [])
        self.assertEqual(len(runner.dispatch_plans), 1)
        self.assertEqual(len(runner.poll_handles), 1)
        events = engine.step()
        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(len(runner.dispatch_plans), 1)
        self.assertEqual(len(runner.poll_handles), 2)

    def test_poll_errors_and_protocol_violations_poison_and_quarantine(self) -> None:
        cases: tuple[tuple[str, PollResult, type[BaseException], str], ...] = (
            ("exception", RuntimeError("poll exploded"), RuntimeError, "poll exploded"),
            ("not_list", (7,), TypeError, "Python list"),
            ("wrong_length", [], RuntimeError, "1 sequences"),
            ("bad_token", [-1], OverflowError, "out of range"),
        )
        for name, result, error_type, message in cases:
            with self.subTest(name=name):
                pool = StatePool(1)
                state = pool.allocate()[0]
                runner = ControlledAsyncRunner(
                    [result],
                    cache_layout=history_layout(1),
                )
                engine = Engine(runner, state_pool=pool)
                request = engine.submit([1], max_new_tokens=1, state_handles=[state])
                pool.release([state])
                engine.dispatch()

                with self.assertRaisesRegex(error_type, message):
                    engine.poll()
                self.assertTrue(engine.poisoned)
                self.assertIn("poll", engine.poison_error)
                self.assertEqual(engine.in_flight_request_count, 1)
                self.assertFalse(request.done)
                self.assertEqual(pool.counts(state), (1, 1))
                self.assertEqual(
                    (
                        engine.cache_free_parent_count,
                        engine.cache_occupied_block_count,
                    ),
                    (0, 1),
                )
                with self.assertRaisesRegex(RuntimeError, "poisoned"):
                    engine.poll()
                with self.assertRaisesRegex(RuntimeError, "poisoned"):
                    engine.dispatch()

    def test_terminal_release_failure_poison_quarantines_before_native_reuse(
        self,
    ) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner(
            [[7]],
            release_error=RuntimeError("physical cleanup failed"),
            cache_layout=history_layout(1),
        )
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([1], max_new_tokens=1, state_handles=[state])
        pool.release([state])
        engine.dispatch()

        with self.assertRaisesRegex(RuntimeError, "physical cleanup failed"):
            engine.poll()
        self.assertTrue(engine.poisoned)
        self.assertIn("release", engine.poison_error)
        self.assertFalse(request.done)
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (0, 1),
        )

    def test_close_drains_pending_work_and_retires_ready_requests(self) -> None:
        pool = StatePool(2)
        pending_state, ready_state = pool.allocate(2)
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(2),
        )
        engine = Engine(
            runner,
            max_running_requests=1,
            state_pool=pool,
        )
        pending = engine.submit([1], max_new_tokens=2, state_handles=[pending_state])
        ready = engine.submit([2], max_new_tokens=2, state_handles=[ready_state])
        pool.release([pending_state, ready_state])
        engine.dispatch()

        engine.close()

        self.assertTrue(engine.closed)
        self.assertFalse(engine.poisoned)
        self.assertFalse(engine.has_work())
        self.assertTrue(pending.done)
        self.assertEqual(pending.take_result(), [])
        self.assertTrue(ready.done)
        self.assertEqual(ready.take_result(), [])
        self.assertEqual(runner.released, [pending, ready])
        self.assertEqual(len(runner.poll_handles), 2)
        self.assertFalse(pool.contains(pending_state))
        self.assertFalse(pool.contains(ready_state))
        self.assertEqual(
            (engine.cache_free_parent_count, engine.cache_occupied_block_count),
            (2, 0),
        )

        # close is idempotent and does not run physical cleanup twice.
        engine.close()
        self.assertEqual(runner.released, [pending, ready])
        self.assertEqual(engine.poll(), [])

        closed_operations = (
            ("submit", lambda: engine.submit([3], max_new_tokens=1)),
            ("dispatch", engine.dispatch),
            ("step", engine.step),
            ("cancel", lambda: engine.cancel(pending)),
        )
        for name, operation in closed_operations:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RuntimeError, "closed"),
            ):
                operation()

    def test_context_manager_closes_without_suppressing_exceptions(self) -> None:
        runner = ControlledAsyncRunner()
        with self.assertRaisesRegex(ValueError, "body failed"):
            with Engine(runner) as engine:
                request = engine.submit([1], max_new_tokens=1)
                raise ValueError("body failed")

        self.assertTrue(engine.closed)
        self.assertTrue(request.done)
        self.assertEqual(request.take_result(), [])
        self.assertEqual(runner.released, [request])

    def test_destructor_releases_ready_state_pool_ownership(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner()
        engine = Engine(runner, state_pool=pool)
        request = engine.submit([1], max_new_tokens=1, state_handles=[state])
        pool.release([state])
        self.assertEqual(pool.counts(state), (1, 0))

        del engine
        gc.collect()

        self.assertEqual(runner.released, [request])
        self.assertFalse(pool.contains(state))
        self.assertEqual(pool.free_count, 1)

    def test_destructor_drains_healthy_pending_work_before_runner(self) -> None:
        destruction_order: list[str] = []

        class Completion:
            def __del__(self) -> None:
                destruction_order.append("completion")

        class OrderedRunner:
            def __init__(self) -> None:
                self.accepted: set[Completion] = set()
                self.poll_count = 0

            def dispatch(self, plan: BatchPlan) -> Completion:
                completion = Completion()
                self.accepted.add(completion)
                return completion

            def poll(self, completion: Completion) -> list[int] | None:
                self.poll_count += 1
                if self.poll_count == 1:
                    return None
                self.accepted.remove(completion)
                return [7]

            def release(self, request_ids: list[int]) -> None:
                pass

            def __del__(self) -> None:
                destruction_order.append("runner")

        pool = StatePool(1)
        state = pool.allocate()[0]
        engine = Engine(OrderedRunner(), state_pool=pool)
        engine.submit([1], max_new_tokens=1, state_handles=[state])
        pool.release([state])
        engine.dispatch()

        del engine
        gc.collect()

        self.assertEqual(destruction_order, ["completion", "runner"])
        self.assertFalse(pool.contains(state))

    def test_destructor_quarantines_undrainable_shared_state(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner()
        engine = Engine(runner, state_pool=pool)
        engine.submit([1], max_new_tokens=1, state_handles=[state])
        pool.release([state])
        engine.dispatch()

        del engine
        gc.collect()

        self.assertGreater(len(runner.poll_handles), 0)
        self.assertEqual(runner.released, [])
        self.assertEqual(pool.counts(state), (1, 1))
        self.assertEqual((pool.free_count, pool.retiring_count), (0, 0))


class RunnerContractConstructionTests(unittest.TestCase):
    def test_async_contract_wins_over_forward_and_bound_callables_are_cached(
        self,
    ) -> None:
        class DescriptorRunner:
            def __init__(self) -> None:
                self.lookups = {"dispatch": 0, "poll": 0, "release": 0}
                self.forward_calls = 0
                self._handle = object()

            @property
            def dispatch(self) -> Callable[[BatchPlan], object]:
                self.lookups["dispatch"] += 1
                return self._dispatch

            @property
            def poll(self) -> Callable[[object], list[int]]:
                self.lookups["poll"] += 1
                return self._poll

            @property
            def release(self) -> Callable[[list[int]], None]:
                self.lookups["release"] += 1
                return lambda request_ids: None

            def _dispatch(self, plan):
                return self._handle

            def _poll(self, handle):
                self.assert_handle(handle)
                return [2]

            def assert_handle(self, handle):
                assert handle is self._handle

            def forward(self, plan):
                self.forward_calls += 1
                return [99]

        runner = DescriptorRunner()
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)
        events = engine.step()
        self.assertTrue(engine.asynchronous)
        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(runner.forward_calls, 0)
        self.assertEqual(runner.lookups, {"dispatch": 1, "poll": 1, "release": 1})

    def test_constructor_rejects_partial_noncallable_and_raising_descriptors(
        self,
    ) -> None:
        class DispatchOnly:
            def dispatch(self, plan):
                return object()

            def forward(self, plan):
                return [1]

        class PollOnly:
            def poll(self, handle):
                return [1]

            def forward(self, plan):
                return [1]

        class BadDispatch:
            dispatch = 7

            def poll(self, handle):
                return [1]

        class BadForward:
            forward = 7

        class BadRelease:
            release = 7

            def forward(self, plan):
                return [1]

        class RaisingRelease:
            def forward(self, plan):
                return [1]

            @property
            def release(self):
                raise RuntimeError("release descriptor failed")

        for runner in (DispatchOnly(), PollOnly()):
            with (
                self.subTest(runner=type(runner).__name__),
                self.assertRaisesRegex(TypeError, "supplied together"),
            ):
                Engine(runner)
        with self.assertRaisesRegex(TypeError, "dispatch must be callable"):
            Engine(BadDispatch())
        with self.assertRaisesRegex(TypeError, "forward must be callable"):
            Engine(BadForward())
        with self.assertRaisesRegex(TypeError, "release must be callable"):
            Engine(BadRelease())
        with self.assertRaisesRegex(RuntimeError, "release descriptor failed"):
            Engine(RaisingRelease())


if __name__ == "__main__":
    unittest.main()
