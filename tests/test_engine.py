from __future__ import annotations

import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RequestLease,
    ResourceBinding,
    RunnerConfig,
    SchedulerConfig,
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


class TestEngine:
    def test_engine_context_length_and_request_limit_are_top_level(self) -> None:
        engine = Engine(RecordingRunner(), max_running_requests=3, context_length=4)
        assert engine.max_running_requests == 3
        assert engine.context_length == 4
        with pytest.raises(ValueError, match="exceeds Engine context_length 4"):
            engine.submit([1, 2, 3], max_new_tokens=3)
        with pytest.raises(ValueError, match="context_length must be positive"):
            Engine(RecordingRunner(), context_length=0)

    def test_missing_generation_limit_uses_remaining_model_context(self) -> None:
        engine = Engine(RecordingRunner(), context_length=3)
        request = engine.submit([1, 2])
        while engine.has_work():
            engine.step()
        assert request.take_result() == [3, 4]

    def test_request_lease_owns_result_without_engine_retention(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit([1], max_new_tokens=1)

        assert isinstance(request, RequestLease)
        assert request.id == request.request_id
        assert not request.done
        with pytest.raises(RuntimeError, match="not completed"):
            request.take_result()

        engine.step()

        assert request.done
        assert not request.cancelled
        assert not hasattr(engine, "output")
        assert request.take_result() == [2]
        with pytest.raises(RuntimeError, match="already been consumed"):
            request.take_result()

    def test_request_lease_is_bound_to_its_engine(self) -> None:
        first = Engine(RecordingRunner())
        second = Engine(RecordingRunner())
        request = first.submit([1], max_new_tokens=1)

        with pytest.raises(ValueError, match="different Engine"):
            second.cancel(request)
        assert first.ready_request_count == 1
        assert second.ready_request_count == 0

    def test_rust_owns_prefill_decode_and_completion_transitions(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([1, 2], max_new_tokens=2)
        second = engine.submit([8], max_new_tokens=1)

        initial = engine.step()
        assert [event.token_id for event in initial] == [3, 9]
        assert runner.plans[0].phase == BatchPhase.PREFILL
        assert runner.plans[0].sequence_indptr == [0, 2, 3]
        assert runner.plans[0].sample_sequence_indices == [0, 1]
        assert runner.released == [second]

        final = engine.step()
        assert [event.token_id for event in final] == [4]
        assert runner.plans[1].phase == BatchPhase.DECODE
        assert runner.plans[1].positions == [2]
        assert runner.plans[1].sample_sequence_indices == [0]
        assert not engine.has_work()
        assert first.done
        assert first.take_result() == [3, 4]
        assert second.done
        assert second.take_result() == [9]

    def test_runner_must_return_one_token_per_sequence(self) -> None:
        class BadRunner:
            def forward(self, plan):
                return []

        engine = Engine(BadRunner())
        engine.submit([1], max_new_tokens=1)
        with pytest.raises(RuntimeError, match="1 sequences"):
            engine.step()

    def test_decode_runner_can_commit_a_verified_token_block(self) -> None:
        class BlockRunner:
            def __init__(self) -> None:
                self.plans: list[BatchPlan] = []

            def forward(self, plan: BatchPlan):
                self.plans.append(plan)
                if plan.phase == BatchPhase.PREFILL:
                    return [2]
                if len(self.plans) == 2:
                    return [[3, 4, 5]]
                return [[6, 7]]

        runner = BlockRunner()
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=6)

        assert [event.token_id for event in engine.step()] == [2]
        block = engine.step()
        assert [event.token_id for event in block] == [3, 4, 5]
        assert [event.index for event in block] == [1, 2, 3]
        assert not any(event.finished for event in block)

        final = engine.step()
        assert [event.token_id for event in final] == [6, 7]
        assert [event.index for event in final] == [4, 5]
        assert [event.finished for event in final] == [False, True]
        assert runner.plans[2].token_ids == [5]
        assert runner.plans[2].positions == [4]
        assert request.take_result() == [2, 3, 4, 5, 6, 7]

    def test_managed_cache_leases_declared_speculative_write_lookahead(self) -> None:
        layout = CacheLayout(2, [CacheGroupLayout(2)])

        class BlockRunner:
            runner_config = RunnerConfig(
                cache_layout=layout,
                cache_write_lookahead_tokens=2,
            )

            def __init__(self) -> None:
                self.plans: list[BatchPlan] = []

            @staticmethod
            def bind_resources(_binding: ResourceBinding) -> None:
                return None

            def forward(self, plan: BatchPlan):
                self.plans.append(plan)
                return [2] if plan.phase == BatchPhase.PREFILL else [[3, 4, 5]]

        runner = BlockRunner()
        assert runner.runner_config.cache_write_lookahead_tokens == 2
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=4)

        engine.step()
        events = engine.step()

        assert [event.token_id for event in events] == [3, 4, 5]
        decode = runner.plans[1]
        assert len(decode.cache_groups[0].block_ids) == 2
        assert len(decode.cache_groups[0].append_block_ids) == 1
        assert request.take_result() == [2, 3, 4, 5]

    def test_verified_block_stops_at_eos_and_discards_its_suffix(self) -> None:
        class BlockRunner:
            @staticmethod
            def forward(plan: BatchPlan):
                if plan.phase == BatchPhase.PREFILL:
                    return [2]
                return [[3, 7, 8]]

        engine = Engine(BlockRunner(), eos_token_id=7)
        request = engine.submit([1], max_new_tokens=8)
        engine.step()

        events = engine.step()

        assert [event.token_id for event in events] == [3, 7]
        assert [event.finished for event in events] == [False, True]
        assert [event.finish_reason for event in events] == [None, "stop"]
        assert request.take_result() == [2, 3, 7]

    def test_request_stop_tokens_override_engine_eos(self) -> None:
        engine = Engine(RecordingRunner(), eos_token_id=7)
        request = engine.submit([1], max_new_tokens=5, stop_token_ids=[3])

        first = engine.step()[0]
        second = engine.step()[0]

        assert not first.finished
        assert first.finish_reason is None
        assert second.finished
        assert second.finish_reason == "stop"
        assert request.take_result() == [2, 3]

    def test_length_finish_reason_is_explicit(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit([1], max_new_tokens=1)

        event = engine.step()[0]

        assert event.finished
        assert event.finish_reason == "length"
        assert request.take_result() == [2]

    def test_speculative_rows_are_rejected_during_prefill(self) -> None:
        class BadRunner:
            @staticmethod
            def forward(plan: BatchPlan):
                return [[2, 3] for _ in range(plan.num_sequences)]

        engine = Engine(BadRunner())
        engine.submit([1], max_new_tokens=2)

        with pytest.raises(RuntimeError, match="only supported during decode"):
            engine.step()

    def test_batch_limit_keeps_running_request_ahead_of_waiter(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], max_new_tokens=3)
        second = engine.submit([10], max_new_tokens=1)
        engine.step()
        event = engine.step()[0]
        assert event.request_id == first
        assert not first.done
        assert not second.done
        engine.step()
        event = engine.step()[0]
        assert event.request_id == second
        assert second.done
        assert second.take_result() == [11]

    def test_ready_prefill_fills_a_vacancy_before_decode_resumes(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        surviving = engine.submit([1], max_new_tokens=3)
        terminal = engine.submit([10], max_new_tokens=1)
        untouched = engine.submit([20], max_new_tokens=1)

        first_events = engine.step()
        assert [event.request_id for event in first_events] == [surviving, terminal]
        assert runner.plans[0].sequence_ids == [surviving, terminal]

        second_events = engine.step()
        assert [event.request_id for event in second_events] == [untouched]
        assert runner.plans[1].sequence_ids == [untouched]
        assert untouched.done
        assert not surviving.done

        third_events = engine.step()
        assert [event.request_id for event in third_events] == [surviving]
        assert runner.plans[2].sequence_ids == [surviving]

        fourth_events = engine.step()
        assert [event.request_id for event in fourth_events] == [surviving]
        assert runner.plans[3].sequence_ids == [surviving]
        assert untouched.take_result() == [21]

    def test_public_batch_plan_validates_indptr(self) -> None:
        with pytest.raises(ValueError, match="end at the token count"):
            BatchPlan(
                [1, 2],
                [0, 1],
                [7],
                [0, 1],
                [0],
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
        assert pool.counts(handle) == (2, 0)
        pool.release([handle])  # Transfer the caller's reference to the engine.

        event = engine.step()[0]
        assert event.finished
        assert request.done
        assert request.take_result() == [7]
        assert runner.plan.state_slots == [handle.slot]
        assert runner.plan.state_generations == [handle.generation]
        assert runner.plan.state_indptr == [0, 1]
        assert not pool.contains(handle)
        assert pool.free_count == 2

    def test_rust_owns_flat_cache_allocation_and_sparse_group_metadata(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(2)])

        class CachedRunner(RecordingRunner):
            runner_config = RunnerConfig(cache_layout=layout)

            def __init__(self) -> None:
                super().__init__()
                self.binding = None

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.binding = (binding.cache_pool_id, binding.cache_layout)

        runner = CachedRunner()
        engine = Engine(runner)
        request = engine.submit([4, 5, 6], max_new_tokens=2)

        assert runner.binding == (engine.cache_pool_id, layout)
        assert runner.binding[1] is layout
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 3

        engine.step()
        assert engine.cache_occupied_block_count == 2
        assert engine.cache_free_parent_count == 1
        prefill = runner.plans[0]
        assert prefill.cache_pool_id == engine.cache_pool_id
        assert len(prefill.cache_groups) == 1
        group = prefill.cache_groups[0]
        assert group.group_index == 0
        assert group.block_indptr == [0, 2]
        assert group.first_logical_blocks == [0]
        assert group.append_block_ids == [group.block_ids[0]] * 2 + [group.block_ids[1]]
        assert group.append_block_offsets == [0, 1, 0]

        engine.step()
        decode = runner.plans[1]
        assert decode.positions == [3]
        decode_group = decode.cache_groups[0]
        assert decode_group.block_ids == group.block_ids
        assert decode_group.append_block_ids == [group.block_ids[1]]
        assert decode_group.append_block_offsets == [1]

        assert runner.released == [request]
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 3

    def test_kv_grows_on_demand_and_retracts_for_replay(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(1)])

        class TinyCachedRunner(RecordingRunner):
            runner_config = RunnerConfig(
                cache_layout=layout,
                continuation_prefill=True,
            )

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.binding = (binding.cache_pool_id, binding.cache_layout)

        runner = TinyCachedRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            context_length=3,
            scheduler_config=SchedulerConfig(schedule_conservativeness=0.0),
        )
        first = engine.submit([1], max_new_tokens=3)
        second = engine.submit([10], max_new_tokens=3)

        engine.step()
        assert engine.cache_occupied_block_count == 2
        assert runner.plans[-1].sequence_ids == [first, second]

        while engine.has_work():
            engine.step()

        assert second.take_result() == [11, 12, 13]
        assert first.take_result() == [2, 3, 4]
        assert runner.released == [first, second, first]
        first_replay = next(
            plan
            for plan in runner.plans[1:]
            if plan.sequence_ids == [first] and plan.phase == BatchPhase.PREFILL
        )
        assert first_replay.sample_sequence_indices == []
        assert engine.cache_occupied_block_count == 0

    def test_conservativeness_limits_initial_kv_admission(self) -> None:
        layout = CacheLayout(3, [CacheGroupLayout(1)])

        class TinyCachedRunner(RecordingRunner):
            runner_config = RunnerConfig(cache_layout=layout)

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> None:
                del binding

        runner = TinyCachedRunner()
        engine = Engine(runner, max_running_requests=2, context_length=3)
        first = engine.submit([1], max_new_tokens=3)
        second = engine.submit([10], max_new_tokens=3)
        engine.step()
        assert runner.plans[-1].sequence_ids == [first]
        assert not second.done
        engine.close()
