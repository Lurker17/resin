from __future__ import annotations

import numpy as np
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


def _block_addresses(group: CacheGroupPlan) -> list[tuple[int, int]]:
    return list(zip(group.block_slab_ids, group.block_slotnums, strict=True))


def _append_addresses(group: CacheGroupPlan) -> list[tuple[int, int]]:
    return list(zip(group.append_slab_ids, group.append_slotnums, strict=True))


class TestFlatCacheApi:
    def test_flat_cache_descriptors_are_exported(self) -> None:
        for name in ("CacheGroupLayout", "CacheLayout", "CacheGroupPlan"):
            assert name in resin.__all__
            assert hasattr(resin, name)

    def test_layout_is_frozen_ordered_and_derives_kernel_geometry(self) -> None:
        coarse = CacheGroupLayout(4)
        fine = CacheGroupLayout(2, blocks_per_slab=2)
        source = [coarse, fine]
        layout = CacheLayout(3, source)
        source.reverse()

        assert [
            (group.block_tokens, group.blocks_per_slab) for group in layout.groups
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
            [0, 0],
            [0],
            [3, 4],
            [0, 0],
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
        assert frozen.block_indptr.tolist() == [0, 2]
        assert frozen.block_slab_ids.tolist() == [3, 4]
        assert frozen.block_slotnums.tolist() == [0, 0]
        assert frozen.block_table_start_indices.tolist() == [0]
        assert frozen.append_slab_ids.tolist() == [3, 4]
        assert frozen.append_slotnums.tolist() == [0, 0]
        assert frozen.append_block_offsets.tolist() == [1, 0]

    def test_cache_group_numeric_metadata_is_read_only_zero_copy_numpy(self) -> None:
        group = CacheGroupPlan(
            0,
            np.asarray([0, 2], dtype=np.uint32),
            np.asarray([3, 4], dtype=np.uint32),
            np.asarray([0, 0], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            np.asarray([3, 4], dtype=np.uint32),
            np.asarray([0, 0], dtype=np.uint32),
            np.asarray([1, 0], dtype=np.uint32),
        )
        for name in (
            "block_indptr",
            "block_slab_ids",
            "block_slotnums",
            "block_table_start_indices",
            "append_slab_ids",
            "append_slotnums",
            "append_block_offsets",
        ):
            array = getattr(group, name)
            assert array.dtype == np.uint32
            assert array.base is group
            assert not array.flags.writeable
            assert np.shares_memory(array, getattr(group, name))
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

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
        assert reused.token_ids.tolist() == [5]
        assert reused.context_lengths.tolist() == [4]
        assert second.done

        engine.clear_prefix_cache()
        assert engine.prefix_cache_stats == (0, 1, 1, 2)
        assert engine.cache_occupied_block_count == 0

    def test_media_digest_identity_controls_prefix_reuse(self) -> None:
        class Media:
            token_ids = [1, 2, 3, 4]

            def __init__(self, identity: list[int]) -> None:
                self.cache_key = identity

        layout = CacheLayout(4, [CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner, prefix_cache=True)

        first = Media([1, 101, 102, 4])
        engine.submit(media=first, max_new_tokens=1)
        while engine.has_work():
            engine.step()

        same_media = Media([1, 101, 102, 4])
        engine.submit(media=same_media, max_new_tokens=1)
        while engine.has_work():
            engine.step()
        assert runner.plans[-1].context_lengths.tolist() == [2]
        assert runner.plans[-1].media == (same_media,)

        changed_media = Media([1, 201, 202, 4])
        engine.submit(media=changed_media, max_new_tokens=1)
        engine.step()
        assert runner.plans[-1].context_lengths.tolist() == [0]

    def test_media_atomic_prefill_span_is_not_cut_at_cache_alignment(self) -> None:
        class Media:
            token_ids = [1, 2, 3, 4, 5, 6]
            cache_key = (1, 20, 30, 40, 50, 6)
            token_spans = ((1, 4),)

        runner = RecordingCachedRunner(CacheLayout(4, [CacheGroupLayout(4)]))
        engine = Engine(runner, prefix_cache=True)

        engine.submit(media=Media(), max_new_tokens=1)
        engine.step()

        assert runner.plans[0].token_ids.tolist() == [1, 2, 3, 4, 5]
        assert runner.plans[0].context_lengths.tolist() == [0]

    def test_prefix_cache_is_disabled_by_default(self) -> None:
        runner = RecordingCachedRunner(CacheLayout(2, [CacheGroupLayout(2)]))
        engine = Engine(runner)
        engine.submit([1, 2], max_new_tokens=1)
        engine.step()
        assert engine.prefix_cache_stats is None
        assert engine.cache_occupied_block_count == 0

    def test_one_history_group_uses_positive_sparse_physical_addresses(self) -> None:
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
        assert group.block_indptr.tolist() == [0, 2]
        assert group.block_table_start_indices.tolist() == [0]
        assert len(group.block_slab_ids) == 2
        assert all(parent > 0 for parent in group.block_slab_ids)
        assert group.block_slotnums.tolist() == [0, 0]
        assert _append_addresses(group) == [
            _block_addresses(group)[0],
            _block_addresses(group)[0],
            _block_addresses(group)[1],
        ]
        assert group.append_block_offsets.tolist() == [0, 1, 0]

        final_event = engine.step()[0]
        assert final_event.finished
        decode_group = runner.plans[1].cache_groups[0]
        assert _block_addresses(decode_group) == _block_addresses(group)
        assert _append_addresses(decode_group) == [_block_addresses(group)[1]]
        assert decode_group.append_block_offsets.tolist() == [1]
        assert runner.released == [request]
        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0

    def test_ordered_groups_share_one_parent_pool_with_group_specific_ids(self) -> None:
        layout = CacheLayout(
            3,
            [
                CacheGroupLayout(4),
                CacheGroupLayout(2, blocks_per_slab=2),
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
        assert coarse.block_indptr.tolist() == [0, 1]
        assert coarse.block_table_start_indices.tolist() == [0]
        assert coarse.append_block_offsets.tolist() == [0, 1, 2]
        assert _append_addresses(coarse) == [_block_addresses(coarse)[0]] * 3

        assert fine.block_indptr.tolist() == [0, 2]
        assert fine.block_table_start_indices.tolist() == [0]
        assert fine.append_block_offsets.tolist() == [0, 1, 0]
        assert _append_addresses(fine) == [
            _block_addresses(fine)[0],
            _block_addresses(fine)[0],
            _block_addresses(fine)[1],
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
            _block_addresses(runner.plans[0].cache_groups[0])
            == _block_addresses(runner.plans[1].cache_groups[0])
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
        assert prefill.block_table_start_indices.tolist() == [0]
        assert len(prefill.block_slab_ids) == 3
        assert _append_addresses(prefill) == [
            address for address in _block_addresses(prefill) for _ in range(4)
        ]
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 4
