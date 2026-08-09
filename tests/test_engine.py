from __future__ import annotations

import unittest

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RequestLease,
    StatePool,
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


class EngineTests(unittest.TestCase):
    def test_engine_context_length_and_request_limit_are_top_level(self) -> None:
        engine = Engine(RecordingRunner(), max_running_requests=3, context_length=4)
        self.assertEqual(engine.max_running_requests, 3)
        self.assertEqual(engine.context_length, 4)
        with self.assertRaisesRegex(ValueError, "exceeds Engine context_length 4"):
            engine.submit([1, 2, 3], max_new_tokens=3)
        with self.assertRaisesRegex(ValueError, "context_length must be positive"):
            Engine(RecordingRunner(), context_length=0)

    def test_request_lease_owns_result_without_engine_retention(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit([1], max_new_tokens=1)

        self.assertIsInstance(request, RequestLease)
        self.assertEqual(request.id, request.request_id)
        self.assertFalse(request.done)
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            request.take_result()

        engine.step()

        self.assertTrue(request.done)
        self.assertFalse(request.cancelled)
        self.assertFalse(hasattr(engine, "output"))
        self.assertEqual(request.take_result(), [2])
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            request.take_result()

    def test_request_lease_is_bound_to_its_engine(self) -> None:
        first = Engine(RecordingRunner())
        second = Engine(RecordingRunner())
        request = first.submit([1], max_new_tokens=1)

        with self.assertRaisesRegex(ValueError, "different Engine"):
            second.cancel(request)
        self.assertEqual(first.ready_request_count, 1)
        self.assertEqual(second.ready_request_count, 0)

    def test_rust_owns_prefill_decode_and_completion_transitions(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([1, 2], max_new_tokens=2)
        second = engine.submit([8], max_new_tokens=1)

        initial = engine.step()
        self.assertEqual([event.token_id for event in initial], [3, 9])
        self.assertEqual(runner.plans[0].mode, "prefill")
        self.assertEqual(runner.plans[0].sequence_indptr, [0, 2, 3])
        self.assertEqual(runner.plans[0].prefill_sequence_indices, [0, 1])
        self.assertEqual(runner.plans[0].decode_sequence_indices, [])
        self.assertEqual(runner.plans[0].sample_sequence_indices, [0, 1])
        self.assertEqual(runner.released, [second])

        final = engine.step()
        self.assertEqual([event.token_id for event in final], [4])
        self.assertEqual(runner.plans[1].mode, "decode")
        self.assertEqual(runner.plans[1].positions, [2])
        self.assertEqual(runner.plans[1].prefill_sequence_indices, [])
        self.assertEqual(runner.plans[1].decode_sequence_indices, [0])
        self.assertEqual(runner.plans[1].sample_sequence_indices, [0])
        self.assertFalse(engine.has_work())
        self.assertTrue(first.done)
        self.assertEqual(first.take_result(), [3, 4])
        self.assertTrue(second.done)
        self.assertEqual(second.take_result(), [9])

    def test_runner_must_return_one_token_per_sequence(self) -> None:
        class BadRunner:
            def forward(self, plan):
                return []

        engine = Engine(BadRunner())
        engine.submit([1], max_new_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "1 sequences"):
            engine.step()

    def test_batch_limit_rotates_requests_fairly(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], max_new_tokens=3)
        second = engine.submit([10], max_new_tokens=1)
        engine.step()
        event = engine.step()[0]
        self.assertEqual(event.request_id, second)
        self.assertFalse(first.done)
        self.assertTrue(second.done)
        self.assertEqual(second.take_result(), [11])

    def test_mixed_terminal_and_surviving_rows_rotate_behind_ready_tail(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        surviving = engine.submit([1], max_new_tokens=3)
        terminal = engine.submit([10], max_new_tokens=1)
        untouched = engine.submit([20], max_new_tokens=1)

        first_events = engine.step()
        self.assertEqual(
            [event.request_id for event in first_events],
            [surviving, terminal],
        )
        self.assertEqual(runner.plans[0].sequence_ids, [surviving, terminal])

        second_events = engine.step()
        self.assertEqual(
            [event.request_id for event in second_events],
            [untouched, surviving],
        )
        self.assertEqual(runner.plans[1].sequence_ids, [untouched, surviving])
        self.assertTrue(untouched.done)
        self.assertEqual(untouched.take_result(), [21])
        self.assertFalse(surviving.done)

    def test_public_batch_plan_validates_indptr(self) -> None:
        with self.assertRaisesRegex(ValueError, "end at the token count"):
            BatchPlan(
                [1, 2],
                [0, 1],
                [7],
                [0, 1],
                [0],
                prefill_sequence_indices=[0],
                decode_sequence_indices=[],
            )

    def test_engine_pins_native_state_through_forward_and_release(self) -> None:
        pool = StatePool(2)
        handle = pool.allocate()[0]

        class StateAwareRunner:
            def __init__(self) -> None:
                self.plan = None

            def forward(self, plan):
                self.plan = plan
                self.assert_pinned()
                return [7]

            def release(self, request_ids):
                self.assert_pinned()

            @staticmethod
            def assert_pinned() -> None:
                assert pool.counts(handle) == (1, 1)

        runner = StateAwareRunner()
        engine = Engine(runner, eos_token_id=7, state_pool=pool)
        request = engine.submit([11], max_new_tokens=2, state_handles=[handle])
        self.assertEqual(pool.counts(handle), (2, 0))
        pool.release([handle])  # Transfer the caller's reference to the engine.

        event = engine.step()[0]
        self.assertTrue(event.finished)
        self.assertTrue(request.done)
        self.assertEqual(request.take_result(), [7])
        self.assertEqual(runner.plan.state_slots, [handle.slot])
        self.assertEqual(runner.plan.state_generations, [handle.generation])
        self.assertEqual(runner.plan.state_indptr, [0, 1])
        self.assertFalse(pool.contains(handle))
        self.assertEqual(pool.free_count, 2)

    def test_rust_owns_flat_cache_allocation_and_sparse_group_metadata(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(2)])

        class CachedRunner(RecordingRunner):
            cache_layout = layout

            def __init__(self) -> None:
                super().__init__()
                self.binding = None

            def bind_cache_pool(self, pool_id, bound_layout):
                self.binding = (pool_id, bound_layout)
                return pool_id

        runner = CachedRunner()
        engine = Engine(runner)
        request = engine.submit([4, 5, 6], max_new_tokens=2)

        self.assertEqual(runner.binding, (engine.cache_pool_id, layout))
        self.assertIs(runner.binding[1], layout)
        self.assertEqual(engine.cache_occupied_block_count, 2)
        self.assertEqual(engine.cache_free_parent_count, 1)

        engine.step()
        prefill = runner.plans[0]
        self.assertEqual(prefill.cache_pool_id, engine.cache_pool_id)
        self.assertEqual(len(prefill.cache_groups), 1)
        group = prefill.cache_groups[0]
        self.assertEqual(group.group_index, 0)
        self.assertEqual(group.block_indptr, [0, 2])
        self.assertEqual(group.first_logical_blocks, [0])
        self.assertEqual(
            group.append_block_ids, [group.block_ids[0]] * 2 + [group.block_ids[1]]
        )
        self.assertEqual(group.append_block_offsets, [0, 1, 0])

        engine.step()
        decode = runner.plans[1]
        self.assertEqual(decode.positions, [3])
        decode_group = decode.cache_groups[0]
        self.assertEqual(decode_group.block_ids, group.block_ids)
        self.assertEqual(decode_group.append_block_ids, [group.block_ids[1]])
        self.assertEqual(decode_group.append_block_offsets, [1])

        self.assertEqual(runner.released, [request])
        self.assertEqual(engine.cache_occupied_block_count, 0)
        self.assertEqual(engine.cache_free_parent_count, 3)


if __name__ == "__main__":
    unittest.main()
