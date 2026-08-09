from __future__ import annotations

import unittest

from resin import BatchPlan, Engine, StatePool


class RecordingSyncRunner:
    def __init__(self, on_forward=None) -> None:
        self.on_forward = on_forward
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        if self.on_forward is not None:
            self.on_forward(plan)
        return [100 + row for row in range(plan.num_sequences)]

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class ControlledAsyncRunner:
    def __init__(
        self,
        poll_results: list[list[int] | BaseException | None],
        *,
        release_error: BaseException | None = None,
    ) -> None:
        self.poll_results = list(poll_results)
        self.release_error = release_error
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []
        self.handle = object()

    def dispatch(self, plan: BatchPlan) -> object:
        self.plans.append(plan)
        return self.handle

    def poll(self, handle: object) -> list[int] | None:
        if handle is not self.handle:
            raise ValueError("foreign completion handle")
        result = self.poll_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def release(self, request_ids: list[int]) -> None:
        if self.release_error is not None:
            raise self.release_error
        self.released.extend(request_ids)


class BatchPlanStateAccessTests(unittest.TestCase):
    @staticmethod
    def plan(**state_metadata) -> BatchPlan:
        if state_metadata.get("state_slots"):
            state_metadata.setdefault("state_pool_id", 1)
        return BatchPlan(
            [11],
            [0],
            [7],
            [0, 1],
            [0],
            prefill_sequence_indices=[0],
            decode_sequence_indices=[],
            **state_metadata,
        )

    def test_omission_defaults_to_write_and_getter_is_frozen_copy(self) -> None:
        default = self.plan(
            state_slots=[3],
            state_generations=[9],
            state_indptr=[0, 1],
        )
        self.assertEqual(default.state_access_modes, ["write"])

        explicit = self.plan(
            state_slots=[3, 4],
            state_generations=[9, 10],
            state_indptr=[0, 2],
            state_access_modes=["read", "write"],
        )
        returned = explicit.state_access_modes
        returned[0] = "write"
        self.assertEqual(explicit.state_access_modes, ["read", "write"])
        with self.assertRaises(AttributeError):
            explicit.state_access_modes = ["write", "write"]

    def test_explicit_modes_validate_exact_length_and_vocabulary(self) -> None:
        metadata = {
            "state_slots": [3],
            "state_generations": [9],
            "state_indptr": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "one item per state handle"):
            self.plan(**metadata, state_access_modes=[])
        with self.assertRaisesRegex(ValueError, "must be 'read' or 'write'"):
            self.plan(**metadata, state_access_modes=["Read"])

    def test_submit_validation_is_atomic_before_state_retain(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        engine = Engine(RecordingSyncRunner(), state_pool=pool)

        with self.assertRaisesRegex(ValueError, "one item per state handle"):
            engine.submit(
                [1],
                max_new_tokens=1,
                state_handles=[state],
                state_access_modes=[],
            )
        with self.assertRaisesRegex(ValueError, "must be 'read' or 'write'"):
            engine.submit(
                [1],
                max_new_tokens=1,
                state_handles=[state],
                state_access_modes=["shared"],
            )

        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))
        self.assertEqual(engine.ready_request_count, 0)


class EngineStateAccessTests(unittest.TestCase):
    def test_submit_omission_defaults_to_exclusive_write(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        observed: list[tuple[int, bool]] = []

        def inspect(plan: BatchPlan) -> None:
            self.assertEqual(plan.state_access_modes, ["write"])
            observed.append(pool.access_counts(state))

        engine = Engine(RecordingSyncRunner(inspect), state_pool=pool)
        engine.submit([1], max_new_tokens=1, state_handles=[state])
        engine.step()

        self.assertEqual(observed, [(0, True)])
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_readers_of_same_state_co_batch_under_one_shared_lease(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        observed: list[tuple[tuple[int, int], tuple[int, bool]]] = []

        def inspect(plan: BatchPlan) -> None:
            self.assertEqual(plan.state_access_modes, ["read", "read"])
            observed.append((pool.counts(state), pool.access_counts(state)))

        runner = RecordingSyncRunner(inspect)
        engine = Engine(runner, max_running_requests=2, state_pool=pool)
        first = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )
        second = engine.submit(
            [2],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )

        events = engine.step()

        self.assertEqual([event.request_id for event in events], [first, second])
        self.assertEqual(runner.plans[0].sequence_ids, [first, second])
        self.assertEqual(runner.plans[0].state_indptr, [0, 1, 2])
        self.assertEqual(observed, [((3, 2), (1, False))])
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_writer_reader_conflict_truncates_then_later_progresses(self) -> None:
        pool = StatePool(2)
        state, disjoint_state = pool.allocate(2)
        observed: list[tuple[int, bool]] = []

        def inspect(plan: BatchPlan) -> None:
            observed.append(pool.access_counts(state))

        runner = RecordingSyncRunner(inspect)
        engine = Engine(runner, max_running_requests=3, state_pool=pool)
        writer = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["write"],
        )
        reader = engine.submit(
            [2],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )
        disjoint = engine.submit(
            [3],
            max_new_tokens=1,
            state_handles=[disjoint_state],
            state_access_modes=["write"],
        )

        first_events = engine.step()
        second_events = engine.step()

        self.assertEqual([event.request_id for event in first_events], [writer])
        self.assertEqual(
            [event.request_id for event in second_events], [reader, disjoint]
        )
        self.assertEqual(
            [plan.sequence_ids for plan in runner.plans],
            [[writer], [reader, disjoint]],
        )
        self.assertEqual(
            [plan.state_access_modes for plan in runner.plans],
            [["write"], ["read", "write"]],
        )
        self.assertEqual(observed, [(0, True), (1, False)])
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_duplicate_occurrences_preserve_modes_but_write_dominates(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        observed: list[tuple[tuple[int, int], tuple[int, bool]]] = []

        def inspect(plan: BatchPlan) -> None:
            self.assertEqual(plan.state_access_modes, ["read", "write"])
            observed.append((pool.counts(state), pool.access_counts(state)))

        runner = RecordingSyncRunner(inspect)
        engine = Engine(runner, state_pool=pool)
        request = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state, state],
            state_access_modes=["read", "write"],
        )

        events = engine.step()

        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(runner.plans[0].state_indptr, [0, 2])
        self.assertEqual(observed, [((3, 2), (0, True))])
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_sync_rejection_releases_access_and_retry_reacquires_it(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]

        class RejectOnceRunner(RecordingSyncRunner):
            def __init__(self) -> None:
                super().__init__()
                self.access_during_forward: list[tuple[int, bool]] = []

            def forward(self, plan: BatchPlan) -> list[int]:
                self.plans.append(plan)
                self.access_during_forward.append(pool.access_counts(state))
                if len(self.plans) == 1:
                    raise RuntimeError("rejected before acceptance")
                return [7]

        runner = RejectOnceRunner()
        engine = Engine(runner, state_pool=pool)
        request = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["write"],
        )

        with self.assertRaisesRegex(RuntimeError, "before acceptance"):
            engine.step()
        self.assertFalse(engine.poisoned)
        self.assertEqual(engine.ready_request_count, 1)
        self.assertEqual(pool.counts(state), (2, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

        events = engine.step()

        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(runner.access_during_forward, [(0, True), (0, True)])
        self.assertEqual(
            [plan.state_access_modes for plan in runner.plans],
            [["write"], ["write"]],
        )
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_async_pending_cancel_holds_then_releases_shared_access(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner([None, [7]])
        engine = Engine(runner, state_pool=pool)
        request = engine.submit(
            [1],
            max_new_tokens=2,
            state_handles=[state],
            state_access_modes=["read"],
        )

        self.assertEqual(engine.dispatch(), 1)
        self.assertEqual(pool.access_counts(state), (1, False))
        self.assertTrue(engine.cancel(request))
        self.assertEqual(engine.poll(), [])
        self.assertEqual(pool.access_counts(state), (1, False))

        self.assertEqual(engine.poll(), [])
        self.assertEqual(runner.released, [request])
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_shared_read_survivor_reacquires_after_peer_retires(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = RecordingSyncRunner()
        engine = Engine(runner, max_running_requests=2, state_pool=pool)
        finished = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )
        survivor = engine.submit(
            [2],
            max_new_tokens=2,
            state_handles=[state],
            state_access_modes=["read"],
        )

        first = engine.step()

        self.assertEqual([event.request_id for event in first], [finished, survivor])
        self.assertEqual(pool.counts(state), (2, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

        second = engine.step()

        self.assertEqual([event.request_id for event in second], [survivor])
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_access_arbitrates_across_engines_sharing_one_pool(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        pending_runner = ControlledAsyncRunner([None, [7]])
        pending = Engine(pending_runner, state_pool=pool)
        pending.submit(
            [1],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )
        pending.dispatch()
        self.assertEqual(pool.access_counts(state), (1, False))

        peer_observed: list[tuple[int, bool]] = []
        peer = Engine(
            RecordingSyncRunner(
                lambda _plan: peer_observed.append(pool.access_counts(state))
            ),
            state_pool=pool,
        )
        peer.submit(
            [2],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["read"],
        )
        peer.step()
        self.assertEqual(peer_observed, [(2, False)])
        self.assertEqual(pool.access_counts(state), (1, False))

        writer_runner = RecordingSyncRunner()
        writer = Engine(writer_runner, state_pool=pool)
        writer_request = writer.submit(
            [3],
            max_new_tokens=1,
            state_handles=[state],
            state_access_modes=["write"],
        )
        with self.assertRaisesRegex(ValueError, "active lease"):
            writer.dispatch()
        self.assertEqual(writer.ready_request_count, 1)
        self.assertEqual(pool.access_counts(state), (1, False))

        self.assertEqual(pending.poll(), [])
        self.assertEqual(len(pending.poll()), 1)
        self.assertEqual(pool.access_counts(state), (0, False))
        self.assertEqual(
            [event.request_id for event in writer.step()], [writer_request]
        )
        self.assertEqual(writer.in_flight_batch_id, None)
        self.assertEqual(writer_runner.plans[0].batch_id, 2)
        self.assertEqual(pool.access_counts(state), (0, False))

    def test_close_drains_pending_write_and_releases_transferred_state(self) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = ControlledAsyncRunner([None, [7]])
        engine = Engine(runner, state_pool=pool)
        engine.submit([1], max_new_tokens=1, state_handles=[state])
        pool.release([state])
        engine.dispatch()
        self.assertEqual(pool.access_counts(state), (0, True))

        engine.close()

        self.assertTrue(engine.closed)
        self.assertEqual(pool.free_count, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            pool.access_counts(state)

    def test_protocol_and_release_failures_quarantine_exclusive_access(self) -> None:
        cases = (
            (
                "protocol",
                ControlledAsyncRunner([[]]),
                RuntimeError,
                "1 sequences",
                "poll",
            ),
            (
                "release",
                ControlledAsyncRunner(
                    [[7]], release_error=RuntimeError("cleanup failed")
                ),
                RuntimeError,
                "cleanup failed",
                "release",
            ),
        )
        for name, runner, error_type, message, poison_stage in cases:
            with self.subTest(name=name):
                pool = StatePool(1)
                state = pool.allocate()[0]
                engine = Engine(runner, state_pool=pool)
                request = engine.submit(
                    [1],
                    max_new_tokens=1,
                    state_handles=[state],
                    state_access_modes=["write"],
                )
                engine.dispatch()

                with self.assertRaisesRegex(error_type, message):
                    engine.poll()

                self.assertTrue(engine.poisoned)
                self.assertIn(poison_stage, engine.poison_error)
                self.assertFalse(request.done)
                self.assertEqual(pool.counts(state), (2, 1))
                self.assertEqual(pool.access_counts(state), (0, True))


if __name__ == "__main__":
    unittest.main()
