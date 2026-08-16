from __future__ import annotations

import gc
from collections.abc import Callable

import pytest

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ResourceBinding,
    RunnerConfig,
)

PollResult = list[int] | list[list[int]] | tuple[int, ...] | BaseException | None


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
            self.runner_config = RunnerConfig(cache_layout=cache_layout)
        self._accepted: dict[object, BatchPlan] = {}

    def bind_resources(self, binding: ResourceBinding) -> None:
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is not None
        self.cache_binding = (binding.cache_pool_id, binding.cache_layout)

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


class TestAsyncEngineContract:
    def test_sync_runner_keeps_step_and_explicit_dispatch_poll_behavior(self) -> None:
        class SyncRunner:
            def __init__(self) -> None:
                self.plans = []
                self.released = []

            def forward(self, plan):
                self.plans.append(plan)
                return [plan.token_ids[end - 1] + 1 for end in plan.token_indptr[1:]]

            def release(self, request_ids):
                self.released.extend(request_ids)

        runner = SyncRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], max_new_tokens=2)
        second = engine.submit([10], max_new_tokens=1)

        assert not engine.asynchronous
        assert engine.ready_request_count == 2
        batch_id = engine.dispatch()
        assert batch_id == 1
        assert engine.ready_request_count == 1
        assert engine.in_flight_request_count == 1
        assert engine.in_flight_batch_id == batch_id
        first_events = engine.poll()
        assert [event.request_id for event in first_events] == [first]
        assert not first.done

        # Existing step() remains one dispatch plus one poll. Sticky admission
        # keeps the surviving request running.
        second_events = engine.step()
        assert [event.request_id for event in second_events] == [first]
        assert first.done
        assert not second.done
        final_events = engine.step()
        assert [event.request_id for event in final_events] == [second]
        assert second.take_result() == [11]
        assert runner.released == [first, second]

    def test_marked_sync_failure_quarantines_accepted_native_resources(self) -> None:
        class PostAcceptError(RuntimeError):
            _resin_post_accept = True

        class SyncRunner:
            runner_config = RunnerConfig(cache_layout=history_layout(2))

            def __init__(self) -> None:
                self.calls = 0
                self.released = []

            def forward(self, plan):
                del plan
                self.calls += 1
                raise PostAcceptError("CUDA quiescence failed")

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> None:
                del binding

            def release(self, request_ids):
                self.released.extend(request_ids)

        runner = SyncRunner()
        engine = Engine(runner)
        request = engine.submit([1, 2], max_new_tokens=1)

        with pytest.raises(PostAcceptError, match="quiescence failed"):
            engine.dispatch()

        assert engine.poisoned
        assert runner.calls == 1
        assert runner.released == []
        assert (engine.ready_request_count, engine.in_flight_request_count) == (0, 1)
        assert engine.in_flight_batch_id == 1
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            1,
            1,
        )
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.cancel(request)
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.close()
        assert runner.calls == 1

    def test_unmarked_sync_failure_remains_preaccept_and_retryable(self) -> None:
        class SyncRunner:
            runner_config = RunnerConfig(cache_layout=history_layout(2))

            def __init__(self) -> None:
                self.calls = 0

            def forward(self, plan):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("retryable forward failure")
                return [plan.token_ids[-1] + 1]

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> None:
                del binding

        runner = SyncRunner()
        engine = Engine(runner)
        engine.submit([1, 2], max_new_tokens=1)

        with pytest.raises(RuntimeError, match="retryable forward failure"):
            engine.dispatch()
        assert not engine.poisoned
        assert (engine.ready_request_count, engine.in_flight_request_count) == (1, 0)

        assert engine.dispatch() == 2
        assert [event.token_id for event in engine.poll()] == [3]
        assert runner.calls == 2
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )

    def test_malformed_sync_result_quarantines_after_forward_returns(self) -> None:
        for label, malformed, error_type in (
            ("not a token sequence", object(), TypeError),
            ("wrong token count", [], RuntimeError),
        ):

            class SyncRunner:
                runner_config = RunnerConfig(cache_layout=history_layout(2))

                def __init__(self) -> None:
                    self.calls = 0
                    self.released: list[int] = []

                def forward(self, plan):
                    del plan
                    self.calls += 1
                    return malformed

                @staticmethod
                def bind_resources(binding: ResourceBinding) -> None:
                    del binding

                def release(self, request_ids):
                    self.released.extend(request_ids)

            runner = SyncRunner()
            engine = Engine(runner)
            request = engine.submit([1, 2], max_new_tokens=1)

            with pytest.raises(error_type):
                engine.dispatch()

            assert engine.poisoned
            assert runner.calls == 1
            assert runner.released == []
            assert (engine.ready_request_count, engine.in_flight_request_count) == (
                0,
                1,
            )
            assert (
                engine.cache_free_parent_count,
                engine.cache_occupied_block_count,
            ) == (1, 1)
            with pytest.raises(RuntimeError, match="resources are quarantined"):
                engine.cancel(request)
            assert runner.calls == 1

    def test_marked_async_dispatch_failure_quarantines_accepted_resources(
        self,
    ) -> None:
        class PostAcceptError(RuntimeError):
            _resin_post_accept = True

        class AsyncRunner:
            runner_config = RunnerConfig(cache_layout=history_layout(2))

            def __init__(self) -> None:
                self.calls = 0
                self.released: list[int] = []

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> None:
                del binding

            def dispatch(self, plan):
                del plan
                self.calls += 1
                raise PostAcceptError("dispatch lost its accepted completion")

            @staticmethod
            def poll(completion):
                raise AssertionError(f"lost completion was polled: {completion!r}")

            def release(self, request_ids):
                self.released.extend(request_ids)

        runner = AsyncRunner()
        engine = Engine(runner)
        request = engine.submit([1, 2], max_new_tokens=1)

        with pytest.raises(PostAcceptError, match="lost its accepted completion"):
            engine.dispatch()

        assert engine.poisoned
        assert runner.calls == 1
        assert runner.released == []
        assert (engine.ready_request_count, engine.in_flight_request_count) == (0, 1)
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            1,
            1,
        )
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.cancel(request)
        assert runner.calls == 1

    def test_pending_completion_keeps_cache_blocks_owned(self) -> None:
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(1),
        )
        engine = Engine(runner)
        request = engine.submit([1, 2], max_new_tokens=1)

        assert engine.asynchronous
        assert engine.dispatch() == 1
        assert (engine.ready_request_count, engine.in_flight_request_count) == (0, 1)
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            0,
            1,
        )

        assert engine.poll() == []
        waiting = engine.submit([9], max_new_tokens=1)
        assert engine.cache_occupied_block_count == 1

        events = engine.poll()
        assert [event.token_id for event in events] == [7]
        assert events[0].finished
        assert runner.released == [request]
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            1,
            0,
        )
        assert engine.in_flight_request_count == 0
        assert engine.cancel(waiting)

    def test_in_flight_cancel_waits_for_terminal_poll_and_discards_output(self) -> None:
        runner = ControlledAsyncRunner(
            [None, [99]],
            cache_layout=history_layout(2),
        )
        engine = Engine(runner)
        request = engine.submit([4], max_new_tokens=3)
        engine.dispatch()

        assert engine.cancel(request)
        assert engine.cancel(request)
        assert engine.poll() == []
        assert engine.cache_occupied_block_count == 1
        waiting = engine.submit([8], max_new_tokens=1)
        assert engine.cache_occupied_block_count == 1

        assert engine.poll() == []
        assert request.done
        assert request.take_result() == []
        assert runner.released == [request]
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )
        assert engine.cancel(waiting)
        assert runner.released == [request, waiting]
        assert not engine.has_work()

    def test_in_flight_cancel_preserves_result_alignment_for_survivors(self) -> None:
        runner = ControlledAsyncRunner([[11, 21]])
        engine = Engine(runner, max_running_requests=2)
        canceled = engine.submit([10], max_new_tokens=2)
        surviving = engine.submit([20], max_new_tokens=2)
        engine.dispatch()

        assert engine.cancel(canceled)
        events = engine.poll()

        assert [(event.request_id, event.token_id) for event in events] == [
            (surviving, 21)
        ]
        assert canceled.done
        assert canceled.take_result() == []
        assert not surviving.done
        assert runner.released == [canceled]
        assert (engine.ready_request_count, engine.in_flight_request_count) == (1, 0)

    def test_pending_completion_owns_all_ordered_cache_groups(self) -> None:
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(2, block_tokens=1, group_count=2),
        )
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)
        engine.dispatch()

        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            0,
            2,
        )
        assert engine.poll() == []
        waiting = engine.submit([2], max_new_tokens=1)
        assert engine.cache_occupied_block_count == 2

        events = engine.poll()
        assert [event.request_id for event in events] == [request]
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )
        assert engine.cancel(waiting)

    def test_ready_cancel_retires_existing_cache_immediately(self) -> None:
        runner = ControlledAsyncRunner([[3]], cache_layout=history_layout(2))
        engine = Engine(runner)
        request = engine.submit([1, 2], max_new_tokens=2)

        engine.dispatch()
        events = engine.poll()
        assert not events[0].finished
        assert engine.ready_request_count == 1
        assert engine.cache_occupied_block_count == 1

        assert engine.cancel(request)
        assert not engine.cancel(request)
        assert request.done
        assert request.take_result() == [3]
        assert runner.released == [request]
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )
        assert not engine.has_work()

    def test_dispatch_rejection_restores_ready_order(self) -> None:
        runner = ControlledAsyncRunner([[11, 21]], dispatch_failures=1)
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], max_new_tokens=1)
        second = engine.submit([20], max_new_tokens=1)
        tail = engine.submit([30], max_new_tokens=1)

        with pytest.raises(RuntimeError, match="before acceptance"):
            engine.dispatch()
        assert not engine.poisoned
        assert (engine.ready_request_count, engine.in_flight_request_count) == (3, 0)

        assert engine.dispatch() == 2
        assert [plan.sequence_ids.tolist() for plan in runner.dispatch_plans] == [
            [first, second],
            [first, second],
        ]
        failed_plan, retry_plan = runner.dispatch_plans
        assert (failed_plan.batch_id, retry_plan.batch_id) == (1, 2)
        for field in (
            "phase",
            "token_ids",
            "sequence_ids",
            "token_indptr",
            "context_lengths",
        ):
            left = getattr(failed_plan, field)
            right = getattr(retry_plan, field)
            assert (
                left.tolist() == right.tolist()
                if hasattr(left, "tolist")
                else left == right
            ), field
        events = engine.poll()
        assert [event.request_id for event in events] == [first, second]
        assert engine.ready_request_count == 1

        runner.poll_results.append([31])
        tail_events = engine.step()
        assert [event.request_id for event in tail_events] == [tail]
        assert runner.dispatch_plans[-1].sequence_ids.tolist() == [tail]

    def test_canceled_terminal_and_surviving_rows_preserve_tail_rotation(self) -> None:
        runner = ControlledAsyncRunner([[11, 21, 31], [41], [32]])
        engine = Engine(runner, max_running_requests=3)
        canceled = engine.submit([10], max_new_tokens=2)
        terminal = engine.submit([20], max_new_tokens=1)
        surviving = engine.submit([30], max_new_tokens=2)
        untouched = engine.submit([40], max_new_tokens=1)

        engine.dispatch()
        assert engine.cancel(canceled)
        first_events = engine.poll()
        assert [event.request_id for event in first_events] == [terminal, surviving]
        assert runner.released == [canceled, terminal]

        second_events = engine.step()
        assert runner.dispatch_plans[-1].sequence_ids.tolist() == [untouched]
        assert [event.request_id for event in second_events] == [untouched]

        third_events = engine.step()
        assert runner.dispatch_plans[-1].sequence_ids.tolist() == [surviving]
        assert [event.request_id for event in third_events] == [surviving]

    def test_async_step_dispatches_and_polls_exactly_once(self) -> None:
        runner = ControlledAsyncRunner([None, [5]])
        engine = Engine(runner)
        request = engine.submit([4], max_new_tokens=1)

        assert engine.step() == []
        assert len(runner.dispatch_plans) == 1
        assert len(runner.poll_handles) == 1
        events = engine.step()
        assert [event.request_id for event in events] == [request]
        assert len(runner.dispatch_plans) == 1
        assert len(runner.poll_handles) == 2

    def test_poll_errors_and_protocol_violations_poison_and_quarantine(self) -> None:
        cases: tuple[tuple[str, PollResult, type[BaseException], str], ...] = (
            ("exception", RuntimeError("poll exploded"), RuntimeError, "poll exploded"),
            ("not_list", (7,), TypeError, "Python list"),
            ("wrong_length", [], RuntimeError, "1 sequences"),
            ("bad_token", [-1], OverflowError, "out of range"),
        )
        for name, result, error_type, message in cases:
            runner = ControlledAsyncRunner(
                [result],
                cache_layout=history_layout(1),
            )
            engine = Engine(runner)
            request = engine.submit([1], max_new_tokens=1)
            engine.dispatch()

            with pytest.raises(error_type, match=message):
                engine.poll()
            assert engine.poisoned
            assert "poll" in engine.poison_error
            assert engine.in_flight_request_count == 1
            assert not request.done
            assert (
                engine.cache_free_parent_count,
                engine.cache_occupied_block_count,
            ) == (0, 1)
            with pytest.raises(RuntimeError, match="poisoned"):
                engine.poll()
            with pytest.raises(RuntimeError, match="poisoned"):
                engine.dispatch()

    def test_terminal_release_failure_poison_quarantines_before_native_reuse(
        self,
    ) -> None:
        runner = ControlledAsyncRunner(
            [[7]],
            release_error=RuntimeError("physical cleanup failed"),
            cache_layout=history_layout(1),
        )
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)
        engine.dispatch()

        with pytest.raises(RuntimeError, match="physical cleanup failed"):
            engine.poll()
        assert engine.poisoned
        assert "release" in engine.poison_error
        assert not request.done
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            0,
            1,
        )

    def test_close_drains_pending_work_and_retires_ready_requests(self) -> None:
        runner = ControlledAsyncRunner(
            [None, [7]],
            cache_layout=history_layout(2),
        )
        engine = Engine(
            runner,
            max_running_requests=1,
        )
        pending = engine.submit([1], max_new_tokens=2)
        ready = engine.submit([2], max_new_tokens=2)
        engine.dispatch()

        engine.close()

        assert engine.closed
        assert not engine.poisoned
        assert not engine.has_work()
        assert pending.done
        assert pending.take_result() == []
        assert ready.done
        assert ready.take_result() == []
        assert runner.released == [pending, ready]
        assert len(runner.poll_handles) == 2
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )

        # close is idempotent and does not run physical cleanup twice.
        engine.close()
        assert runner.released == [pending, ready]
        assert engine.poll() == []

        closed_operations = (
            ("submit", lambda: engine.submit([3], max_new_tokens=1)),
            ("dispatch", engine.dispatch),
            ("step", engine.step),
            ("cancel", lambda: engine.cancel(pending)),
        )
        for name, operation in closed_operations:
            with (
                pytest.raises(RuntimeError, match="closed"),
            ):
                operation()

    def test_context_manager_closes_without_suppressing_exceptions(self) -> None:
        runner = ControlledAsyncRunner()
        with pytest.raises(ValueError, match="body failed"):
            with Engine(runner) as engine:
                request = engine.submit([1], max_new_tokens=1)
                raise ValueError("body failed")

        assert engine.closed
        assert request.done
        assert request.take_result() == []
        assert runner.released == [request]

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

        engine = Engine(OrderedRunner())
        engine.submit([1], max_new_tokens=1)
        engine.dispatch()

        del engine
        gc.collect()

        assert destruction_order == ["completion", "runner"]

    def test_destructor_quarantines_undrainable_completion(self) -> None:
        runner = ControlledAsyncRunner(cache_layout=history_layout(1))
        engine = Engine(runner)
        engine.submit([1], max_new_tokens=1)
        engine.dispatch()

        del engine
        gc.collect()

        assert len(runner.poll_handles) > 0
        assert runner.released == []


class TestRunnerContractConstruction:
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
        assert engine.asynchronous
        assert [event.request_id for event in events] == [request]
        assert runner.forward_calls == 0
        assert runner.lookups == {"dispatch": 1, "poll": 1, "release": 1}

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
                pytest.raises(TypeError, match="supplied together"),
            ):
                Engine(runner)
        with pytest.raises(TypeError, match="dispatch must be callable"):
            Engine(BadDispatch())
        with pytest.raises(TypeError, match="forward must be callable"):
            Engine(BadForward())
        with pytest.raises(TypeError, match="release must be callable"):
            Engine(BadRelease())
        with pytest.raises(RuntimeError, match="release descriptor failed"):
            Engine(RaisingRelease())
