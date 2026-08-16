from __future__ import annotations

import pytest

import resin
from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
    Engine,
    ResourceBinding,
    RunnerConfig,
)


class RecordingCachedRunner:
    def __init__(
        self,
        layout: CacheLayout,
        *,
        forward_failures: int = 0,
    ) -> None:
        self.cache_layout = layout
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=layout,
        )
        self.forward_failures = forward_failures
        self.bindings: list[tuple[int, CacheLayout]] = []
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def bind_resources(self, binding: ResourceBinding) -> None:
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is not None
        self.bindings.append((binding.cache_pool_id, binding.cache_layout))

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


class TestFlatCacheApi:
    def test_flat_cache_descriptors_are_exported(self) -> None:
        for name in ("CacheGroupLayout", "CacheLayout", "CacheGroupPlan"):
            assert name in resin.__all__
            assert hasattr(resin, name)

    def test_layout_is_frozen_ordered_and_derives_kernel_geometry(self) -> None:
        coarse = CacheGroupLayout(4)
        fine = CacheGroupLayout(2, blocks_per_parent=2)
        source = [coarse, fine]
        layout = CacheLayout(3, source)
        source.reverse()

        assert [
            (group.block_tokens, group.blocks_per_parent) for group in layout.groups
        ] == [(4, 1), (2, 2)]
        assert layout.group_count == 2
        assert layout.group_kernel_page_capacity(0) == 3
        assert layout.group_kernel_page_count(0) == 4
        assert layout.group_kernel_page_capacity(1) == 6
        assert layout.group_kernel_page_count(1) == 7
        with pytest.raises(AttributeError):
            layout.parent_capacity = 4

    def test_cache_retention_is_only_history_or_window(self) -> None:
        history = CacheGroupLayout(4)
        window = CacheGroupLayout(4, retention=CacheRetention.WINDOW, window_tokens=4)

        assert history.retention is CacheRetention.HISTORY
        assert history.window_tokens is None
        assert window.retention is CacheRetention.WINDOW
        assert window.window_tokens == 4
        with pytest.raises(AttributeError):
            _ = history.resident_blocks
        with pytest.raises(TypeError, match="resident_blocks"):
            CacheGroupLayout(4, resident_blocks=1)
        with pytest.raises(TypeError):
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
            [11],
            [1],
            [8, 9],
            [0, 2],
            cache_groups=[group],
        )

        assert len(plan.cache_groups) == 1
        frozen = plan.cache_groups[0]
        assert frozen.group_index == group.group_index
        assert frozen.block_indptr == [0, 2]
        assert frozen.block_ids == [3, 4]
        assert frozen.first_logical_blocks == [0]
        assert frozen.append_block_ids == [3, 4]
        assert frozen.append_block_offsets == [1, 0]

    def test_binding_requires_an_exact_layout_callable_and_pool_ack(self) -> None:
        layout = CacheLayout(2, [CacheGroupLayout(2)])

        class MissingBinder:
            runner_config = RunnerConfig(cache_layout=layout)

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        with pytest.raises(TypeError, match="bind_resources"):
            Engine(MissingBinder())

        class BadLayout(MissingBinder):
            runner_config = {"cache_layout": {"parent_capacity": 2}}

            @staticmethod
            def bind_resources(binding: object) -> None:
                del binding

        with pytest.raises(TypeError, match="exactly a RunnerConfig"):
            Engine(BadLayout())

        class BadAck(MissingBinder):
            def __init__(self, acknowledgement: object) -> None:
                self.acknowledgement = acknowledgement

            def bind_resources(self, binding: ResourceBinding):
                del binding
                return self.acknowledgement

        with pytest.raises(TypeError, match="must return None"):
            Engine(BadAck(True))
        with pytest.raises(TypeError, match="must return None"):
            Engine(BadAck(999_999))

        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        assert engine.cache_pool_id > 0
        assert runner.bindings == [(engine.cache_pool_id, layout)]
        assert runner.bindings[0][1] is layout

    def test_explicit_none_layout_is_unmanaged_and_ignores_stray_binder(self) -> None:
        class UnmanagedRunner:
            def __init__(self) -> None:
                self.bind_calls = 0

            def bind_resources(self, binding: object) -> None:
                del binding
                self.bind_calls += 1
                raise AssertionError("an unmanaged runner must not bind a cache pool")

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        runner = UnmanagedRunner()
        engine = Engine(runner)
        assert engine.cache_pool_id is None
        assert runner.bind_calls == 0
        engine.close()


class TestFlatCacheEngine:
    def test_opt_in_lru_prefix_cache_skips_the_shared_prefill(self) -> None:
        layout = CacheLayout(4, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner, prefix_cache=True)

        first = engine.submit([1, 2, 3, 4], max_new_tokens=1)
        while engine.has_work():
            engine.step()
        assert first.done
        assert engine.prefix_cache_stats == (2, 0, 1, 0)
        assert engine.cache_occupied_block_count == 2

        second = engine.submit([1, 2, 3, 4, 5], max_new_tokens=1)
        assert engine.prefix_cache_stats == (2, 1, 1, 0)
        engine.step()
        reused = runner.plans[-1]
        assert reused.token_ids == [5]
        assert reused.context_lengths == [4]
        assert second.done

        engine.clear_prefix_cache()
        assert engine.prefix_cache_stats == (0, 1, 1, 2)
        assert engine.cache_occupied_block_count == 0

    def test_prefix_cache_is_disabled_by_default(self) -> None:
        runner = RecordingCachedRunner(CacheLayout(2, [CacheGroupLayout(2)]))
        engine = Engine(runner)
        engine.submit([1, 2], max_new_tokens=1)
        engine.step()
        assert engine.prefix_cache_stats is None
        assert engine.cache_occupied_block_count == 0

    def test_one_history_group_uses_positive_sparse_block_ids(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        request = engine.submit([4, 5, 6], max_new_tokens=2)

        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0
        first_event = engine.step()[0]
        assert engine.cache_free_parent_count == 1
        assert engine.cache_occupied_block_count == 2
        assert not first_event.finished
        prefill = runner.plans[0]
        assert len(prefill.cache_groups) == 1
        group = prefill.cache_groups[0]
        assert group.group_index == 0
        assert group.block_indptr == [0, 2]
        assert group.first_logical_blocks == [0]
        assert len(group.block_ids) == 2
        assert all(block_id > 0 for block_id in group.block_ids)
        assert group.append_block_ids == [
            group.block_ids[0],
            group.block_ids[0],
            group.block_ids[1],
        ]
        assert group.append_block_offsets == [0, 1, 0]

        final_event = engine.step()[0]
        assert final_event.finished
        decode_group = runner.plans[1].cache_groups[0]
        assert decode_group.block_ids == group.block_ids
        assert decode_group.append_block_ids == [group.block_ids[1]]
        assert decode_group.append_block_offsets == [1]
        assert runner.released == [request]
        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0

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

        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0
        engine.step()
        plan = runner.plans[0]
        assert [group.group_index for group in plan.cache_groups] == [0, 1]

        coarse, fine = plan.cache_groups
        assert coarse.block_indptr == [0, 1]
        assert coarse.first_logical_blocks == [0]
        assert coarse.append_block_offsets == [0, 1, 2]
        assert coarse.append_block_ids == [coarse.block_ids[0]] * 3

        assert fine.block_indptr == [0, 2]
        assert fine.first_logical_blocks == [0]
        assert fine.append_block_offsets == [0, 1, 0]
        assert fine.append_block_ids == [
            fine.block_ids[0],
            fine.block_ids[0],
            fine.block_ids[1],
        ]

        engine.step()
        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0

    def test_preaccept_rejection_keeps_pages_owned_and_retry_uses_same_ids(
        self,
    ) -> None:
        layout = CacheLayout(1, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout, forward_failures=1)
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=1)

        with pytest.raises(RuntimeError, match="rejected the batch"):
            engine.dispatch()
        assert not engine.poisoned
        assert engine.ready_request_count == 1
        assert engine.in_flight_request_count == 0
        assert engine.cache_free_parent_count == 0
        assert engine.cache_occupied_block_count == 1

        assert engine.dispatch() == 2
        events = engine.poll()
        assert [event.request_id for event in events] == [request]
        assert (
            runner.plans[0].cache_groups[0].block_ids
            == runner.plans[1].cache_groups[0].block_ids
        )
        assert engine.cache_free_parent_count == 1
        assert engine.cache_occupied_block_count == 0

    def test_window_group_exposes_a_chronological_suffix_without_copy_metadata(
        self,
    ) -> None:
        layout = CacheLayout(
            4,
            [CacheGroupLayout(4, retention=CacheRetention.WINDOW, window_tokens=12)],
        )
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        engine.submit(list(range(12)), max_new_tokens=1)

        events = engine.step()
        assert len(events) == 1
        prefill = runner.plans[-1].cache_groups[0]
        assert prefill.first_logical_blocks == [0]
        assert len(prefill.block_ids) == 3
        assert prefill.append_block_ids == [
            page for page in prefill.block_ids for _ in range(4)
        ]
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 4
