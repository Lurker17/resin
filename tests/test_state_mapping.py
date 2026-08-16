from __future__ import annotations

import pytest

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ResourceBinding,
    RunnerConfig,
    StateMappingConfig,
    StatePool,
    mint_state_mapping_arena_id,
)

PROTOCOL = "synthetic.recurrent.v1"
MAPPING = StateMappingConfig(PROTOCOL, ["write"])
CHECKPOINT_MAPPING = StateMappingConfig(
    PROTOCOL,
    ["write"],
    checkpoint_granularity=2,
)


class MappedRunner:
    runner_config = RunnerConfig(state_mapping=MAPPING)

    def __init__(self, arena_id: int | None = None) -> None:
        self.arena_id = arena_id or mint_state_mapping_arena_id()
        self.bindings: list[ResourceBinding] = []
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def bind_resources(self, binding: ResourceBinding) -> int:
        assert binding.state_mapping.protocol == PROTOCOL
        assert binding.state_mapping.access_modes == ["write"]
        assert binding.state_pool_id is not None
        assert binding.state_capacity is not None
        if binding.required_state_arena_id not in (None, self.arena_id):
            raise ValueError("runner cannot bind the required recurrent arena")
        self.bindings.append(binding)
        return self.arena_id

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        return [plan.token_ids[end - 1] + 1 for end in plan.sequence_indptr[1:]]

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class TestStateMappingDescriptor:
    def test_descriptor_is_frozen_exact_and_validated(self) -> None:
        assert MAPPING.protocol == PROTOCOL
        assert MAPPING.access_modes == ["write"]
        assert MAPPING.handles_per_sequence == 1
        assert MAPPING.checkpoint_granularity is None
        assert CHECKPOINT_MAPPING.checkpoint_granularity == 2
        with pytest.raises(ValueError, match="nonempty"):
            StateMappingConfig("", ["write"])
        with pytest.raises(ValueError, match="at least one"):
            StateMappingConfig(PROTOCOL, [])
        with pytest.raises(ValueError, match="'read' or 'write'"):
            StateMappingConfig(PROTOCOL, ["append"])
        with pytest.raises(ValueError, match="positive"):
            StateMappingConfig(PROTOCOL, ["write"], checkpoint_granularity=0)
        with pytest.raises(ValueError, match="exactly one write"):
            StateMappingConfig(
                PROTOCOL,
                ["read", "write"],
                checkpoint_granularity=2,
            )
        with pytest.raises(AttributeError):
            MAPPING.protocol = "drift"

    def test_resourceful_runner_requires_one_exact_config_and_binder(self) -> None:
        class MissingBinder:
            runner_config = RunnerConfig(state_mapping=MAPPING)

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        with pytest.raises(TypeError, match="bind_resources"):
            Engine(MissingBinder())

        class BadConfig(MissingBinder):
            runner_config = {"state_mapping": MAPPING}

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> int:
                del binding
                return 1

        with pytest.raises(TypeError, match="exactly a RunnerConfig"):
            Engine(BadConfig())

        class BadAck(MappedRunner):
            def bind_resources(self, binding: ResourceBinding):
                del binding
                return True

        with pytest.raises(TypeError, match="positive state arena id"):
            Engine(BadAck())


class TestEngineOwnedState:
    def test_engine_creates_pool_allocates_at_admission_and_retires_slots(self) -> None:
        runner = MappedRunner()
        engine = Engine(runner, max_running_requests=2)

        assert engine.state_pool_id == runner.bindings[0].state_pool_id
        assert runner.bindings[0].state_capacity == 2
        assert engine.state_free_slot_count == 2
        request = engine.submit([4, 5], max_new_tokens=1)
        assert engine.state_free_slot_count == 2

        events = engine.step()

        assert [event.request_id for event in events] == [request]
        plan = runner.plans[0]
        assert len(plan.state_slots) == 1
        assert plan.state_indptr == [0, 1]
        assert plan.state_access_modes == ["write"]
        assert plan.state_pool_id == engine.state_pool_id
        assert engine.state_free_slot_count == 2

    def test_external_pool_limits_admission_and_exact_arena_is_permanent(self) -> None:
        pool = StatePool(1)
        first_runner = MappedRunner()
        first = Engine(first_runner, max_running_requests=4, state_pool=pool)
        assert pool.physical_mapping == (PROTOCOL, first_runner.arena_id)
        assert first_runner.bindings[0].state_capacity == 1

        same = Engine(MappedRunner(first_runner.arena_id), state_pool=pool)
        assert same.state_pool_id == first.state_pool_id
        with pytest.raises(ValueError, match="required recurrent arena"):
            Engine(MappedRunner(), state_pool=pool)

        first.close()
        same.close()

    def test_explicit_handles_preserve_sharing_and_mode_validation(self) -> None:
        pool = StatePool(1)
        runner = MappedRunner()
        engine = Engine(runner, state_pool=pool)
        handle = pool.allocate()[0]

        request = engine.submit([1], 1, state_handles=[handle])
        assert pool.counts(handle) == (2, 0)
        pool.release([handle])
        engine.step()
        assert request.done
        assert not pool.contains(handle)

        handle = pool.allocate()[0]
        with pytest.raises(ValueError, match="access modes"):
            engine.submit(
                [1],
                1,
                state_handles=[handle],
                state_access_modes=["read"],
            )
        pool.release([handle])

    def test_shared_write_handle_serializes_conflicting_rows(self) -> None:
        pool = StatePool(1)
        runner = MappedRunner()
        engine = Engine(runner, max_running_requests=2, state_pool=pool)
        handle = pool.allocate()[0]
        first = engine.submit([1], 1, state_handles=[handle])
        second = engine.submit([2], 1, state_handles=[handle])
        pool.release([handle])

        engine.step()
        assert runner.plans[0].sequence_ids == [first]
        engine.step()
        assert runner.plans[1].sequence_ids == [second]

    def test_auto_pool_capacity_rejects_impossible_mapping(self) -> None:
        mapping = StateMappingConfig(PROTOCOL, ["read", "write"])

        class TwoStateRunner(MappedRunner):
            runner_config = RunnerConfig(state_mapping=mapping)

        with pytest.raises(ValueError, match="at least 2"):
            Engine(TwoStateRunner(), state_pool=StatePool(1))


class TestCombinedResources:
    def test_cache_and_state_are_bound_once_with_one_resource_identity(self) -> None:
        layout = CacheLayout(4, [CacheGroupLayout(2)])

        class CombinedRunner(MappedRunner):
            runner_config = RunnerConfig(cache_layout=layout, state_mapping=MAPPING)

        runner = CombinedRunner()
        engine = Engine(runner)
        binding = runner.bindings[0]
        assert binding.cache_pool_id == engine.cache_pool_id
        assert binding.cache_layout is layout
        assert binding.state_pool_id == engine.state_pool_id

    def test_prefix_cache_rejects_opaque_recurrent_state(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_granularity"):
            Engine(MappedRunner(), prefix_cache=True)

    def test_checkpointed_prefix_restores_read_only_state_then_copies_on_write(
        self,
    ) -> None:
        layout = CacheLayout(8, [CacheGroupLayout(2)])

        class CheckpointRunner(MappedRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                cache_layout=layout,
                state_mapping=CHECKPOINT_MAPPING,
            )

        runner = CheckpointRunner()
        engine = Engine(runner, max_running_requests=1, prefix_cache=True)
        assert runner.bindings[0].state_capacity == 2

        first = engine.submit([1, 2, 3, 4], max_new_tokens=1)
        while engine.has_work():
            engine.step()
        assert first.done
        assert engine.prefix_cache_stats == (2, 0, 1, 0)
        assert engine.state_free_slot_count == 0
        checkpoint_slot = runner.plans[-1].state_slots[-1]

        second = engine.submit([1, 2, 3, 4, 5], max_new_tokens=1)
        while engine.has_work():
            engine.step()
        assert second.done
        restored = runner.plans[-1]
        assert restored.token_ids == [5]
        assert restored.positions == [4]
        assert restored.state_indptr == [0, 2]
        assert restored.state_access_modes == ["read", "write"]
        assert restored.state_slots[0] == checkpoint_slot
        assert restored.state_slots[1] != checkpoint_slot
        assert engine.prefix_cache_stats == (1, 1, 1, 1)
        assert engine.state_free_slot_count == 1
        engine.close()
        assert engine.state_free_slot_count == 2

    def test_state_only_checkpoint_cache_uses_the_same_lru_cow_contract(self) -> None:
        class StateOnlyRunner(MappedRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                state_mapping=CHECKPOINT_MAPPING,
            )

        runner = StateOnlyRunner()
        engine = Engine(runner, max_running_requests=1, prefix_cache=True)
        first = engine.submit([8, 9, 10, 11], max_new_tokens=1)
        while engine.has_work():
            engine.step()
        assert first.done
        assert engine.prefix_cache_stats == (2, 0, 1, 0)

        second = engine.submit([8, 9, 10, 11, 12], max_new_tokens=1)
        while engine.has_work():
            engine.step()
        restored = runner.plans[-1]
        assert second.done
        assert restored.token_ids == [12]
        assert restored.state_access_modes == ["read", "write"]
        assert engine.prefix_cache_stats == (1, 1, 1, 1)
