from __future__ import annotations

import unittest

import resin
from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    Engine,
)


class RecordingCachedRunner:
    supports_continuation_prefill = True

    def __init__(
        self,
        layout: CacheLayout,
        *,
        forward_failures: int = 0,
    ) -> None:
        self.cache_layout = layout
        self.forward_failures = forward_failures
        self.bindings: list[tuple[int, CacheLayout]] = []
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
        self.bindings.append((pool_id, layout))
        return pool_id

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        if self.forward_failures:
            self.forward_failures -= 1
            raise RuntimeError("runner rejected the batch")
        return [
            plan.token_ids[end - 1] + 1
            for _, end in map(plan.token_range, range(plan.num_sequences))
        ]

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class FlatCacheApiTests(unittest.TestCase):
    def test_flat_cache_descriptors_are_exported(self) -> None:
        for name in ("CacheGroupLayout", "CacheLayout", "CacheGroupPlan"):
            self.assertIn(name, resin.__all__)
            self.assertTrue(hasattr(resin, name))

    def test_layout_is_frozen_ordered_and_derives_kernel_geometry(self) -> None:
        coarse = CacheGroupLayout(4)
        fine = CacheGroupLayout(2, blocks_per_parent=2)
        source = [coarse, fine]
        layout = CacheLayout(3, source)
        source.reverse()

        self.assertEqual(
            [(group.block_tokens, group.blocks_per_parent) for group in layout.groups],
            [(4, 1), (2, 2)],
        )
        self.assertEqual(layout.group_count, 2)
        self.assertEqual(layout.group_kernel_page_capacity(0), 3)
        self.assertEqual(layout.group_kernel_page_count(0), 4)
        self.assertEqual(layout.group_kernel_page_capacity(1), 6)
        self.assertEqual(layout.group_kernel_page_count(1), 7)
        with self.assertRaises(AttributeError):
            layout.parent_capacity = 4

    def test_cache_retention_is_only_history_or_window(self) -> None:
        history = CacheGroupLayout(4)
        window = CacheGroupLayout(4, retention="window", window_tokens=4)

        self.assertEqual(history.retention, "history")
        self.assertIsNone(history.window_tokens)
        self.assertEqual(window.retention, "window")
        self.assertEqual(window.window_tokens, 4)
        with self.assertRaises(AttributeError):
            _ = history.resident_blocks
        with self.assertRaisesRegex(TypeError, "resident_blocks"):
            CacheGroupLayout(4, resident_blocks=1)
        with self.assertRaisesRegex(ValueError, "'history' or 'window'"):
            CacheGroupLayout(4, retention="state")

    def test_manual_sparse_group_plan_is_the_only_batch_cache_metadata(self) -> None:
        group = CacheGroupPlan(
            0,
            [0, 2],
            [3, 4],
            [0],
            [3, 4],
            [1, 0],
        )
        plan = BatchPlan(
            [8, 9],
            [1, 2],
            [11],
            [0, 2],
            [1],
            cache_pool_id=7,
            cache_groups=[group],
        )

        self.assertEqual(plan.cache_pool_id, 7)
        self.assertEqual(len(plan.cache_groups), 1)
        frozen = plan.cache_groups[0]
        self.assertEqual(frozen.group_index, group.group_index)
        self.assertEqual(frozen.block_indptr, [0, 2])
        self.assertEqual(frozen.block_ids, [3, 4])
        self.assertEqual(frozen.first_logical_blocks, [0])
        self.assertEqual(frozen.append_block_ids, [3, 4])
        self.assertEqual(frozen.append_block_offsets, [1, 0])

    def test_binding_requires_an_exact_layout_callable_and_pool_ack(self) -> None:
        layout = CacheLayout(2, [CacheGroupLayout(2)])

        class MissingBinder:
            cache_layout = layout

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        with self.assertRaisesRegex(TypeError, "bind_cache_pool"):
            Engine(MissingBinder())

        class BadLayout(MissingBinder):
            cache_layout = {"parent_capacity": 2}

            @staticmethod
            def bind_cache_pool(pool_id: int, bound_layout: object) -> int:
                del bound_layout
                return pool_id

        with self.assertRaisesRegex(TypeError, "exactly a CacheLayout"):
            Engine(BadLayout())

        class BadAck(MissingBinder):
            def __init__(self, acknowledgement: object) -> None:
                self.acknowledgement = acknowledgement

            def bind_cache_pool(self, pool_id: int, bound_layout: CacheLayout):
                del pool_id, bound_layout
                return self.acknowledgement

        with self.assertRaisesRegex(TypeError, "exactly an int"):
            Engine(BadAck(True))
        with self.assertRaisesRegex(ValueError, "acknowledged cache pool"):
            Engine(BadAck(999_999))

        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        self.assertGreater(engine.cache_pool_id, 0)
        self.assertEqual(runner.bindings, [(engine.cache_pool_id, layout)])
        self.assertIs(runner.bindings[0][1], layout)

    def test_explicit_none_layout_is_unmanaged_and_ignores_stray_binder(self) -> None:
        class UnmanagedRunner:
            cache_layout = None

            def __init__(self) -> None:
                self.bind_calls = 0

            def bind_cache_pool(self, pool_id: int, layout: object) -> int:
                del pool_id, layout
                self.bind_calls += 1
                raise AssertionError("an unmanaged runner must not bind a cache pool")

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        runner = UnmanagedRunner()
        engine = Engine(runner)
        self.assertIsNone(engine.cache_pool_id)
        self.assertEqual(runner.bind_calls, 0)
        engine.close()


class FlatCacheEngineTests(unittest.TestCase):
    def test_one_history_group_uses_positive_sparse_block_ids(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        request = engine.submit([4, 5, 6], max_new_tokens=2)

        self.assertEqual(engine.cache_free_parent_count, 3)
        self.assertEqual(engine.cache_occupied_block_count, 0)
        first_event = engine.step()[0]
        self.assertEqual(engine.cache_free_parent_count, 1)
        self.assertEqual(engine.cache_occupied_block_count, 2)
        self.assertFalse(first_event.finished)
        prefill = runner.plans[0]
        self.assertEqual(prefill.cache_pool_id, engine.cache_pool_id)
        self.assertEqual(len(prefill.cache_groups), 1)
        group = prefill.cache_groups[0]
        self.assertEqual(group.group_index, 0)
        self.assertEqual(group.block_indptr, [0, 2])
        self.assertEqual(group.first_logical_blocks, [0])
        self.assertEqual(len(group.block_ids), 2)
        self.assertTrue(all(block_id > 0 for block_id in group.block_ids))
        self.assertEqual(
            group.append_block_ids,
            [group.block_ids[0], group.block_ids[0], group.block_ids[1]],
        )
        self.assertEqual(group.append_block_offsets, [0, 1, 0])

        final_event = engine.step()[0]
        self.assertTrue(final_event.finished)
        decode_group = runner.plans[1].cache_groups[0]
        self.assertEqual(decode_group.block_ids, group.block_ids)
        self.assertEqual(decode_group.append_block_ids, [group.block_ids[1]])
        self.assertEqual(decode_group.append_block_offsets, [1])
        self.assertEqual(runner.released, [request])
        self.assertEqual(engine.cache_free_parent_count, 3)
        self.assertEqual(engine.cache_occupied_block_count, 0)

    def test_ordered_groups_share_one_parent_pool_with_group_specific_ids(self) -> None:
        layout = CacheLayout(
            3,
            [
                CacheGroupLayout(4),
                CacheGroupLayout(2, blocks_per_parent=2),
            ],
        )
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        engine.submit([1, 2, 3], max_new_tokens=2)

        self.assertEqual(engine.cache_free_parent_count, 3)
        self.assertEqual(engine.cache_occupied_block_count, 0)
        engine.step()
        plan = runner.plans[0]
        self.assertEqual([group.group_index for group in plan.cache_groups], [0, 1])

        coarse, fine = plan.cache_groups
        self.assertEqual(coarse.block_indptr, [0, 1])
        self.assertEqual(coarse.first_logical_blocks, [0])
        self.assertEqual(coarse.append_block_offsets, [0, 1, 2])
        self.assertEqual(coarse.append_block_ids, [coarse.block_ids[0]] * 3)

        self.assertEqual(fine.block_indptr, [0, 2])
        self.assertEqual(fine.first_logical_blocks, [0])
        self.assertEqual(fine.append_block_offsets, [0, 1, 0])
        self.assertEqual(
            fine.append_block_ids,
            [fine.block_ids[0], fine.block_ids[0], fine.block_ids[1]],
        )

        engine.step()
        self.assertEqual(engine.cache_free_parent_count, 3)
        self.assertEqual(engine.cache_occupied_block_count, 0)

    def test_preaccept_rejection_keeps_pages_owned_and_retry_uses_same_ids(
        self,
    ) -> None:
        layout = CacheLayout(1, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout, forward_failures=1)
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)

        with self.assertRaisesRegex(RuntimeError, "rejected the batch"):
            engine.dispatch()
        self.assertFalse(engine.poisoned)
        self.assertEqual(engine.ready_request_count, 1)
        self.assertEqual(engine.in_flight_request_count, 0)
        self.assertEqual(engine.cache_free_parent_count, 0)
        self.assertEqual(engine.cache_occupied_block_count, 1)

        self.assertEqual(engine.dispatch(), 2)
        events = engine.poll()
        self.assertEqual([event.request_id for event in events], [request])
        self.assertEqual(
            runner.plans[0].cache_groups[0].block_ids,
            runner.plans[1].cache_groups[0].block_ids,
        )
        self.assertEqual(engine.cache_free_parent_count, 1)
        self.assertEqual(engine.cache_occupied_block_count, 0)

    def test_window_group_exposes_a_chronological_suffix_without_copy_metadata(
        self,
    ) -> None:
        layout = CacheLayout(
            2,
            [CacheGroupLayout(4, retention="window", window_tokens=4)],
        )
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        engine.submit(list(range(12)), max_new_tokens=1)

        self.assertEqual(engine.step(), [])
        first = runner.plans[-1].cache_groups[0]
        self.assertEqual(first.first_logical_blocks, [0])
        self.assertEqual(len(first.block_ids), 1)
        self.assertEqual(first.append_block_ids, [first.block_ids[0]] * 4)

        self.assertEqual(engine.step(), [])
        second = runner.plans[-1].cache_groups[0]
        self.assertEqual(second.first_logical_blocks, [0])
        self.assertEqual(len(second.block_ids), 2)
        self.assertEqual(second.block_ids[0], first.block_ids[0])
        self.assertEqual(second.append_block_ids, [second.block_ids[1]] * 4)

        events = engine.step()
        self.assertEqual(len(events), 1)
        third = runner.plans[-1].cache_groups[0]
        self.assertEqual(third.first_logical_blocks, [1])
        self.assertEqual(third.block_ids, [second.block_ids[1], second.block_ids[0]])
        self.assertEqual(third.append_block_ids, [second.block_ids[0]] * 4)
        self.assertEqual(engine.cache_occupied_block_count, 0)
        self.assertEqual(engine.cache_free_parent_count, 2)


if __name__ == "__main__":
    unittest.main()
