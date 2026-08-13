from __future__ import annotations

import unittest

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    StateMappingConfig,
    StatePool,
    mint_state_mapping_arena_id,
)

PROTOCOL = "lfm2.conv_state.v1"
SYNTHETIC_SSM_PROTOCOL = "synthetic.ssm_state.v7"
TEST_CACHE_LAYOUT = CacheLayout(8, [CacheGroupLayout(4)])


class MappedRunner:
    supports_recurrent_state_mapping = True
    recurrent_state_mapping = StateMappingConfig(PROTOCOL, ["write"])

    def __init__(self, arena_id: int | None = None) -> None:
        self.arena_id = arena_id or mint_state_mapping_arena_id()
        self.bindings: list[tuple[int, int, str]] = []
        self.plans: list[BatchPlan] = []

    def bind_recurrent_state_mapping(
        self, pool_id: int, capacity: int, protocol: str
    ) -> tuple[int, int, str, int]:
        self.bindings.append((pool_id, capacity, protocol))
        return pool_id, capacity, protocol, self.arena_id

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        return [100 + row for row in range(plan.num_sequences)]


class PlainRunner:
    def __init__(self) -> None:
        self.plans: list[BatchPlan] = []

    def forward(self, plan: BatchPlan) -> list[int]:
        self.plans.append(plan)
        return [100 + row for row in range(plan.num_sequences)]


class SyntheticSsmRunner(MappedRunner):
    recurrent_state_mapping = StateMappingConfig(
        SYNTHETIC_SSM_PROTOCOL,
        ["read", "write"],
    )


class StateMappingConfigTests(unittest.TestCase):
    def test_descriptor_is_frozen_owned_and_derives_handle_count(self) -> None:
        source_modes = ["read", "write"]
        config = StateMappingConfig(SYNTHETIC_SSM_PROTOCOL, source_modes)
        source_modes[:] = ["write"]

        self.assertEqual(config.protocol, SYNTHETIC_SSM_PROTOCOL)
        self.assertEqual(config.access_modes, ["read", "write"])
        self.assertEqual(config.handles_per_sequence, 2)
        self.assertIn("synthetic.ssm_state.v7", repr(config))
        with self.assertRaises(AttributeError):
            config.protocol = "changed"

    def test_descriptor_rejects_empty_or_malformed_fields(self) -> None:
        for protocol in ("", "   "):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, "protocol must be nonempty"):
                    StateMappingConfig(protocol, ["write"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            StateMappingConfig("synthetic.empty.v1", [])
        with self.assertRaisesRegex(ValueError, "'read' or 'write'"):
            StateMappingConfig("synthetic.bad_mode.v1", ["borrow"])


class BatchPlanStateMappingTests(unittest.TestCase):
    def test_manual_plan_identity_contract_is_explicit(self) -> None:
        stateless = BatchPlan([1], [0], [7], [0, 1], [0])
        self.assertIsNone(stateless.engine_id)
        self.assertIsNone(stateless.state_pool_id)

        stateful = BatchPlan(
            [1],
            [0],
            [7],
            [0, 1],
            [0],
            state_pool_id=9,
            state_slots=[3],
            state_generations=[4],
            state_access_modes=["write"],
            state_indptr=[0, 1],
        )
        self.assertIsNone(stateful.engine_id)
        self.assertEqual(stateful.state_pool_id, 9)
        self.assertEqual(stateful.state_slots, [3])
        self.assertEqual(stateful.state_generations, [4])
        self.assertEqual(stateful.state_access_modes, ["write"])
        self.assertEqual(stateful.state_indptr, [0, 1])

        with self.assertRaisesRegex(ValueError, "state_pool_id is required"):
            BatchPlan(
                [1],
                [0],
                [7],
                [0, 1],
                [0],
                state_slots=[3],
                state_generations=[4],
                state_indptr=[0, 1],
            )
        with self.assertRaisesRegex(ValueError, "must be None"):
            BatchPlan([1], [0], [7], [0, 1], [0], state_pool_id=9)
        with self.assertRaisesRegex(ValueError, "engine_id must be positive"):
            BatchPlan([1], [0], [7], [0, 1], [0], engine_id=0)


class EngineStateMappingTests(unittest.TestCase):
    def test_native_arena_mint_is_positive_and_process_unique(self) -> None:
        ids = [mint_state_mapping_arena_id() for _ in range(3)]
        self.assertTrue(all(arena_id > 0 for arena_id in ids))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, sorted(ids))

    def test_exact_binding_is_idempotent_and_pool_operations_are_unchanged(
        self,
    ) -> None:
        pool = StatePool(2)
        runner = MappedRunner()
        first = Engine(runner, state_pool=pool)
        second = Engine(runner, state_pool=pool)

        expected = (PROTOCOL, runner.arena_id)
        self.assertEqual(pool.physical_mapping, expected)
        self.assertEqual(
            runner.bindings,
            [(pool.pool_id, 2, PROTOCOL), (pool.pool_id, 2, PROTOCOL)],
        )

        state = pool.allocate()[0]
        pool.retain([state])
        self.assertEqual(pool.counts(state), (2, 0))
        pool.release([state])
        self.assertEqual(pool.counts(state), (1, 0))
        self.assertEqual(pool.physical_mapping, expected)

        first.close()
        second.close()
        self.assertEqual(pool.physical_mapping, expected)

    def test_divergent_arena_binding_is_rejected_without_changing_claim(self) -> None:
        pool = StatePool(1)
        first_runner = MappedRunner()
        first = Engine(first_runner, state_pool=pool)
        original = pool.physical_mapping

        class CachedMappedRunner(MappedRunner):
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__()
                self.combined_bindings: list[
                    tuple[int, CacheLayout, int, int, str]
                ] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                raise AssertionError("dual runners must not use the cache-only binder")

            def bind_recurrent_state_mapping(
                self, pool_id: int, capacity: int, protocol: str
            ) -> tuple[int, int, str, int]:
                raise AssertionError("dual runners must not use the state-only binder")

            def bind_cache_and_recurrent_state(
                self,
                cache_pool_id: int,
                layout: CacheLayout,
                state_pool_id: int,
                capacity: int,
                protocol: str,
                required_arena_id: int | None,
            ) -> tuple[int, int, int, str, int]:
                if required_arena_id != first_runner.arena_id:
                    raise AssertionError(
                        "combined bind received the wrong required arena"
                    )
                self.combined_bindings.append(
                    (cache_pool_id, layout, state_pool_id, capacity, protocol)
                )
                return (
                    cache_pool_id,
                    state_pool_id,
                    capacity,
                    protocol,
                    self.arena_id,
                )

        second_runner = CachedMappedRunner()
        self.assertNotEqual(first_runner.arena_id, second_runner.arena_id)
        with self.assertRaisesRegex(ValueError, "acknowledged combined resources"):
            Engine(second_runner, state_pool=pool)

        self.assertEqual(len(second_runner.combined_bindings), 1)
        self.assertIs(second_runner.combined_bindings[0][1], TEST_CACHE_LAYOUT)
        self.assertEqual(second_runner.bindings, [])
        self.assertEqual(pool.physical_mapping, original)
        first.close()

    def test_binding_capability_requires_pool_descriptor_callable_and_exact_ack(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "requires an Engine state_pool"):
            Engine(MappedRunner())

        class CachedMappedRunner(MappedRunner):
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__()
                self.cache_bindings: list[tuple[int, CacheLayout]] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                self.cache_bindings.append((pool_id, layout))
                return pool_id

        cached_runner = CachedMappedRunner()
        with self.assertRaisesRegex(ValueError, "requires an Engine state_pool"):
            Engine(cached_runner)
        self.assertEqual(cached_runner.cache_bindings, [])

        pool = StatePool(1)

        class MissingDescriptor(PlainRunner):
            supports_recurrent_state_mapping = True

        with self.assertRaisesRegex(TypeError, "recurrent_state_mapping is missing"):
            Engine(MissingDescriptor(), state_pool=pool)

        class MalformedDescriptor(PlainRunner):
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = {
                "protocol": PROTOCOL,
                "access_modes": ["write"],
            }

        with self.assertRaisesRegex(TypeError, "exactly a StateMappingConfig"):
            Engine(MalformedDescriptor(), state_pool=pool)
        self.assertIsNone(pool.physical_mapping)

        class MissingBinder:
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = StateMappingConfig(PROTOCOL, ["write"])

            @staticmethod
            def forward(plan: BatchPlan) -> list[int]:
                return [0] * plan.num_sequences

        with self.assertRaisesRegex(TypeError, "bind_recurrent_state_mapping"):
            Engine(MissingBinder(), state_pool=pool)
        self.assertIsNone(pool.physical_mapping)

        for drift in ("pool", "capacity", "protocol", "arena"):
            with self.subTest(drift=drift):
                candidate = StatePool(1)
                values: list[int | str] = [candidate.pool_id, 1, PROTOCOL, 41]
                if drift == "pool":
                    values[0] = candidate.pool_id + 1
                elif drift == "capacity":
                    values[1] = 2
                elif drift == "protocol":
                    values[2] = "lfm2.conv_state.v2"
                else:
                    values[3] = 0
                acknowledgement = tuple(values)

                class BadAck(MappedRunner):
                    def bind_recurrent_state_mapping(
                        self, pool_id: int, capacity: int, protocol: str
                    ) -> tuple[int, int, str, int]:
                        return acknowledgement

                with self.assertRaisesRegex(ValueError, "acknowledged"):
                    Engine(BadAck(), state_pool=candidate)
                self.assertIsNone(candidate.physical_mapping)

        class BooleanArenaAck(MappedRunner):
            def bind_recurrent_state_mapping(
                self, pool_id: int, capacity: int, protocol: str
            ) -> tuple[int, int, str, bool]:
                return pool_id, capacity, protocol, True

        candidate = StatePool(1)
        with self.assertRaisesRegex(TypeError, "exactly"):
            Engine(BooleanArenaAck(), state_pool=candidate)
        self.assertIsNone(candidate.physical_mapping)

    def test_dual_resources_use_one_exact_combined_binding(self) -> None:
        class DualRunner(PlainRunner):
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = StateMappingConfig(PROTOCOL, ["write"])
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__()
                self.arena_id = mint_state_mapping_arena_id()
                self.combined_calls: list[tuple[int, CacheLayout, int, int, str]] = []
                self.individual_calls: list[str] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                del pool_id, layout
                self.individual_calls.append("cache")
                raise AssertionError("cache-only binder must not be called")

            def bind_recurrent_state_mapping(
                self, pool_id: int, capacity: int, protocol: str
            ) -> tuple[int, int, str, int]:
                del pool_id, capacity, protocol
                self.individual_calls.append("state")
                raise AssertionError("state-only binder must not be called")

            def bind_cache_and_recurrent_state(
                self,
                cache_pool_id: int,
                layout: CacheLayout,
                state_pool_id: int,
                capacity: int,
                protocol: str,
                required_arena_id: int | None,
            ) -> tuple[int, int, int, str, int]:
                if required_arena_id is not None:
                    raise AssertionError(
                        "fresh StatePool unexpectedly required an arena"
                    )
                self.combined_calls.append(
                    (cache_pool_id, layout, state_pool_id, capacity, protocol)
                )
                return (
                    cache_pool_id,
                    state_pool_id,
                    capacity,
                    protocol,
                    self.arena_id,
                )

        pool = StatePool(3)
        runner = DualRunner()
        engine = Engine(runner, state_pool=pool)

        self.assertEqual(runner.individual_calls, [])
        self.assertEqual(len(runner.combined_calls), 1)
        cache_pool_id, layout, state_pool_id, capacity, protocol = (
            runner.combined_calls[0]
        )
        self.assertEqual(cache_pool_id, engine.cache_pool_id)
        self.assertIs(layout, TEST_CACHE_LAYOUT)
        self.assertEqual(
            (state_pool_id, capacity, protocol), (pool.pool_id, 3, PROTOCOL)
        )
        self.assertEqual(pool.physical_mapping, (PROTOCOL, runner.arena_id))
        engine.close()

    def test_dual_binding_failure_never_calls_individual_binders_or_mutates_pool(
        self,
    ) -> None:
        class MissingCombined(PlainRunner):
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = StateMappingConfig(PROTOCOL, ["write"])
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__()
                self.individual_calls: list[str] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                del pool_id, layout
                self.individual_calls.append("cache")
                return 1

            def bind_recurrent_state_mapping(
                self, pool_id: int, capacity: int, protocol: str
            ) -> tuple[int, int, str, int]:
                del pool_id, capacity, protocol
                self.individual_calls.append("state")
                return 1, 1, PROTOCOL, 1

        pool = StatePool(2)
        missing = MissingCombined()
        with self.assertRaisesRegex(TypeError, "bind_cache_and_recurrent_state"):
            Engine(missing, state_pool=pool)
        self.assertEqual(missing.individual_calls, [])
        self.assertIsNone(pool.physical_mapping)

        class RaisingCombined(MissingCombined):
            def __init__(self) -> None:
                super().__init__()
                self.combined_calls = 0

            def bind_cache_and_recurrent_state(self, *args: object) -> object:
                del args
                self.combined_calls += 1
                raise RuntimeError("synthetic combined bind failure")

        raising = RaisingCombined()
        with self.assertRaisesRegex(RuntimeError, "synthetic combined bind failure"):
            Engine(raising, state_pool=pool)
        self.assertEqual(raising.combined_calls, 1)
        self.assertEqual(raising.individual_calls, [])
        self.assertIsNone(pool.physical_mapping)

    def test_dual_binding_rejects_malformed_or_drifted_ack_before_pool_commit(
        self,
    ) -> None:
        class BadCombinedAck(PlainRunner):
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = StateMappingConfig(PROTOCOL, ["write"])
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self, mode: str) -> None:
                super().__init__()
                self.mode = mode

            def bind_cache_and_recurrent_state(
                self,
                cache_pool_id: int,
                layout: CacheLayout,
                state_pool_id: int,
                capacity: int,
                protocol: str,
                required_arena_id: int | None,
            ) -> object:
                if required_arena_id is not None:
                    raise AssertionError(
                        "fresh StatePool unexpectedly required an arena"
                    )
                self.assert_layout(layout)
                values: list[object] = [
                    cache_pool_id,
                    state_pool_id,
                    capacity,
                    protocol,
                    91,
                ]
                if self.mode == "not_tuple":
                    return values
                if self.mode == "boolean_arena":
                    values[4] = True
                elif self.mode == "cache_drift":
                    values[0] = cache_pool_id + 1
                return tuple(values)

            @staticmethod
            def assert_layout(layout: CacheLayout) -> None:
                if layout is not TEST_CACHE_LAYOUT:
                    raise AssertionError("combined bind received the wrong layout")

        for mode, error_type, message in (
            ("not_tuple", TypeError, "five-item tuple"),
            ("boolean_arena", TypeError, "exactly"),
            ("cache_drift", ValueError, "acknowledged combined resources"),
        ):
            with self.subTest(mode=mode):
                candidate = StatePool(2)
                with self.assertRaisesRegex(error_type, message):
                    Engine(BadCombinedAck(mode), state_pool=candidate)
                self.assertIsNone(candidate.physical_mapping)

    def test_descriptor_validation_fails_before_cache_binding_callback(self) -> None:
        class MalformedCachedRunner(PlainRunner):
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = "synthetic.ssm_state.v7"
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__()
                self.cache_bindings: list[tuple[int, CacheLayout]] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                self.cache_bindings.append((pool_id, layout))
                return pool_id

        runner = MalformedCachedRunner()
        with self.assertRaisesRegex(TypeError, "exactly a StateMappingConfig"):
            Engine(runner, state_pool=StatePool(1))
        self.assertEqual(runner.cache_bindings, [])

    def test_native_core_uses_descriptor_and_ignores_legacy_protocol_alias(
        self,
    ) -> None:
        class AliasDriftRunner(MappedRunner):
            recurrent_state_mapping_protocol = object()

        pool = StatePool(1)
        runner = AliasDriftRunner()
        engine = Engine(runner, state_pool=pool)
        self.assertEqual(pool.physical_mapping, (PROTOCOL, runner.arena_id))
        engine.close()

    def test_permanent_pool_binding_rejects_protocol_drift(self) -> None:
        pool = StatePool(2)
        first_runner = MappedRunner()
        first = Engine(first_runner, state_pool=pool)
        original = pool.physical_mapping

        class CachedSyntheticSsmRunner(SyntheticSsmRunner):
            cache_layout = TEST_CACHE_LAYOUT

            def __init__(self) -> None:
                super().__init__(arena_id=first_runner.arena_id)
                self.cache_bindings: list[tuple[int, CacheLayout]] = []

            def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
                self.cache_bindings.append((pool_id, layout))
                return pool_id

        second_runner = CachedSyntheticSsmRunner()
        with self.assertRaisesRegex(ValueError, "already bound"):
            Engine(second_runner, state_pool=pool)
        self.assertEqual(second_runner.cache_bindings, [])
        self.assertEqual(second_runner.bindings, [])
        self.assertEqual(pool.physical_mapping, original)
        first.close()

    def test_non_opted_in_runners_do_not_bind_a_pool(self) -> None:
        pool = StatePool(1)
        Engine(PlainRunner(), state_pool=pool).close()
        self.assertIsNone(pool.physical_mapping)

        class TruthyButNotTrue(PlainRunner):
            supports_recurrent_state_mapping = 1

            @staticmethod
            def bind_recurrent_state_mapping(*args):
                raise AssertionError("must not be called")

        Engine(TruthyButNotTrue(), state_pool=pool).close()
        self.assertIsNone(pool.physical_mapping)

    def test_single_write_descriptor_preserves_lfm_admission_defaults(self) -> None:
        pool = StatePool(2)
        first_state, second_state = pool.allocate(2)
        engine = Engine(MappedRunner(), state_pool=pool)

        invalid = [
            {},
            {"state_handles": [first_state], "state_access_modes": ["read"]},
            {
                "state_handles": [first_state, second_state],
                "state_access_modes": ["write", "write"],
            },
        ]
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "exactly one state handle"):
                    engine.submit([1], max_new_tokens=1, **metadata)

        self.assertEqual(engine.ready_request_count, 0)
        self.assertEqual(pool.counts(first_state), (1, 0))
        self.assertEqual(pool.counts(second_state), (1, 0))

        request = engine.submit([1], max_new_tokens=1, state_handles=[first_state])
        event = engine.step()[0]
        self.assertEqual(event.request_id, request)
        self.assertEqual(pool.counts(first_state), (1, 0))

    def test_arbitrary_ssm_descriptor_enforces_two_handle_mixed_mode_pattern(
        self,
    ) -> None:
        pool = StatePool(3)
        read_state, write_state, extra_state = pool.allocate(3)
        runner = SyntheticSsmRunner()
        engine = Engine(runner, state_pool=pool)

        invalid = [
            {},
            {"state_handles": [read_state]},
            {"state_handles": [read_state, write_state]},
            {
                "state_handles": [read_state, write_state],
                "state_access_modes": ["write", "read"],
            },
            {
                "state_handles": [read_state, write_state, extra_state],
                "state_access_modes": ["read", "write", "write"],
            },
        ]
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "exactly 2 state handles"):
                    engine.submit([1], max_new_tokens=1, **metadata)

        self.assertEqual(engine.ready_request_count, 0)
        self.assertEqual(
            pool.physical_mapping, (SYNTHETIC_SSM_PROTOCOL, runner.arena_id)
        )
        self.assertEqual(pool.counts(read_state), (1, 0))
        self.assertEqual(pool.counts(write_state), (1, 0))

        request = engine.submit(
            [1],
            max_new_tokens=1,
            state_handles=[read_state, write_state],
            state_access_modes=["read", "write"],
        )
        event = engine.step()[0]
        self.assertEqual(event.request_id, request)
        plan = runner.plans[-1]
        self.assertEqual(plan.state_indptr, [0, 2])
        self.assertEqual(plan.state_access_modes, ["read", "write"])
        self.assertEqual(
            plan.state_slots,
            [read_state.slot, write_state.slot],
        )
        self.assertEqual(pool.counts(read_state), (1, 0))
        self.assertEqual(pool.counts(write_state), (1, 0))

    def test_engine_owns_descriptor_snapshot_after_construction(self) -> None:
        pool = StatePool(2)
        read_state, write_state = pool.allocate(2)
        runner = SyntheticSsmRunner()
        engine = Engine(runner, state_pool=pool)
        runner.recurrent_state_mapping = StateMappingConfig(
            SYNTHETIC_SSM_PROTOCOL,
            ["write"],
        )

        with self.assertRaisesRegex(ValueError, "exactly 2 state handles"):
            engine.submit([1], 1, state_handles=[write_state])
        request = engine.submit(
            [1],
            1,
            state_handles=[read_state, write_state],
            state_access_modes=["read", "write"],
        )
        self.assertEqual(engine.step()[0].request_id, request)

    def test_cross_engine_local_request_collision_has_distinct_plan_identity(
        self,
    ) -> None:
        pool = StatePool(1)
        state = pool.allocate()[0]
        runner = MappedRunner()
        first = Engine(runner, state_pool=pool)
        second = Engine(runner, state_pool=pool)

        first_request = first.submit([1], 1, state_handles=[state])
        first.step()
        second_request = second.submit([2], 1, state_handles=[state])
        second.step()

        self.assertEqual((first_request, second_request), (1, 1))
        first_plan, second_plan = runner.plans
        self.assertEqual(first_plan.sequence_ids, [1])
        self.assertEqual(second_plan.sequence_ids, [1])
        self.assertIsNotNone(first_plan.engine_id)
        self.assertIsNotNone(second_plan.engine_id)
        self.assertNotEqual(first_plan.engine_id, second_plan.engine_id)
        self.assertEqual(first_plan.state_pool_id, pool.pool_id)
        self.assertEqual(second_plan.state_pool_id, pool.pool_id)
        self.assertEqual(first_plan.state_slots, [state.slot])
        self.assertEqual(second_plan.state_slots, [state.slot])
        self.assertEqual(first_plan.state_generations, [state.generation])
        self.assertEqual(second_plan.state_generations, [state.generation])
        self.assertEqual(first_plan.state_access_modes, ["write"])
        self.assertEqual(second_plan.state_access_modes, ["write"])


if __name__ == "__main__":
    unittest.main()
