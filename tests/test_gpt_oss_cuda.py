from __future__ import annotations

import importlib.util
import inspect
import math
import subprocess
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

from resin.backends.gpt_oss_cuda import (
    _CUDA_GRAPH_WARMUP_STEPS,
    GptOssCudaRunner,
    _CudaGraphDecodeRegistryEntry,
    _fixed_width_full_page_table,
    _invoke_fa4_b22,
    _normalize_cuda_graph_decode_shapes,
)


def _group_plan(group_index, rows, first_logical_blocks, positions, indptr):
    block_indptr = [0]
    block_ids = []
    append_block_ids = []
    append_block_offsets = []
    for sequence_index, row in enumerate(rows):
        block_ids.extend(row)
        block_indptr.append(len(block_ids))
        first = first_logical_blocks[sequence_index]
        token_start, token_end = indptr[sequence_index : sequence_index + 2]
        for position in positions[token_start:token_end]:
            append_block_ids.append(row[position // 128 - first])
            append_block_offsets.append(position % 128)
    return SimpleNamespace(
        group_index=group_index,
        block_indptr=block_indptr,
        block_ids=block_ids,
        first_logical_blocks=list(first_logical_blocks),
        append_block_ids=append_block_ids,
        append_block_offsets=append_block_offsets,
    )


def _flat_plan(
    context_lengths,
    query_lengths,
    *,
    prefill_sequence_indices=None,
    decode_sequence_indices=None,
    cache_pool_id=41,
):
    sequence_count = len(context_lengths)
    assert len(query_lengths) == sequence_count
    sequence_indptr = [0]
    positions = []
    history_rows = []
    window_rows = []
    window_first = []
    for sequence_index, (context, query) in enumerate(
        zip(context_lengths, query_lengths, strict=True)
    ):
        assert context >= 0 and query > 0
        positions.extend(range(context, context + query))
        sequence_indptr.append(len(positions))
        post_length = context + query
        last_block = (post_length - 1) // 128
        history_rows.append(
            tuple(
                1 + sequence_index * 16 + logical for logical in range(last_block + 1)
            )
        )
        first = max(0, last_block - 1)
        window_first.append(first)
        window_rows.append(
            tuple(
                129 + sequence_index * 16 + logical % 2
                for logical in range(first, last_block + 1)
            )
        )

    if prefill_sequence_indices is None and decode_sequence_indices is None:
        prefill_sequence_indices = list(range(sequence_count))
        decode_sequence_indices = []
    assert prefill_sequence_indices is not None
    assert decode_sequence_indices is not None
    history = _group_plan(
        0,
        history_rows,
        [0] * sequence_count,
        positions,
        sequence_indptr,
    )
    window = _group_plan(
        1,
        window_rows,
        window_first,
        positions,
        sequence_indptr,
    )
    return SimpleNamespace(
        cache_pool_id=cache_pool_id,
        cache_groups=(history, window),
        token_ids=list(range(1, len(positions) + 1)),
        positions=positions,
        sequence_ids=list(range(100, 100 + sequence_count)),
        num_sequences=sequence_count,
        sequence_indptr=sequence_indptr,
        context_lengths=list(context_lengths),
        prefill_sequence_indices=list(prefill_sequence_indices),
        decode_sequence_indices=list(decode_sequence_indices),
        sample_sequence_indices=list(range(sequence_count)),
    )


def _prefill_plan():
    return _flat_plan([0, 130], [128, 5])


def _decode_plan():
    return _flat_plan(
        [130, 260],
        [1, 1],
        prefill_sequence_indices=[],
        decode_sequence_indices=[0, 1],
    )


def _mixed_plan():
    return _flat_plan(
        [130, 260],
        [1, 1],
        prefill_sequence_indices=[0],
        decode_sequence_indices=[1],
    )


def _graph_decode_inputs(batch_size, logical_width):
    plan = SimpleNamespace(
        num_sequences=batch_size,
        sequence_ids=list(range(batch_size)),
        token_ids=list(range(batch_size)),
        positions=list(range(batch_size)),
    )
    batch = SimpleNamespace(
        token_indices=tuple(range(batch_size)),
        sequence_indices=tuple(range(batch_size)),
        q_lengths=(1,) * batch_size,
        full_page_table_width=logical_width,
        sliding_page_table_width=2,
    )
    return plan, batch


def _cuda_graph_registry_runner(shapes):
    shapes = _normalize_cuda_graph_decode_shapes(shapes)
    runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
    runner._cuda_graph_decode_shapes = shapes
    runner._cuda_graph_decode_registry = {
        shape: _CudaGraphDecodeRegistryEntry(shape) for shape in shapes
    }
    runner._cuda_graph_capture_count = 0
    runner._cuda_graph_replay_count = 0
    runner._cuda_graph_hit_count = 0
    runner._cuda_graph_miss_count = 0
    runner._cuda_graph_eager_fallback_count = 0
    runner._cuda_graph_prefill_fallback_count = 0
    runner._cuda_graph_shape_fallback_count = 0
    runner._cuda_graph_warmup_count = 0
    runner._cuda_graph_poisoned_error = None
    runner._diagnose_nonfinite = False
    runner.device = "cuda:0"
    runner._torch = SimpleNamespace(
        cuda=SimpleNamespace(
            memory_allocated=lambda device: 0,
            synchronize=lambda device: None,
            device=lambda device: nullcontext(),
        )
    )

    def new_state(shape):
        return SimpleNamespace(
            shape=shape,
            graph=None,
            output_ids=SimpleNamespace(tolist=lambda: [shape[0]]),
        )

    def capture(state, *, allocation_before, shape):
        del allocation_before
        state.graph = SimpleNamespace(replay=lambda: None)
        runner._cuda_graph_capture_count += 1
        entry = runner._cuda_graph_decode_registry[shape]
        entry.capture_count += 1
        entry.capture_ms = float(shape[0])
        entry.capture_memory_bytes = shape[1]

    runner._new_cuda_graph_decode_state = new_state
    runner._stage_cuda_graph_decode = lambda *args: None
    runner._capture_cuda_graph_decode = capture
    return runner


class GptOssCudaImportTests(unittest.TestCase):
    def test_import_is_lazy(self) -> None:
        script = """
import sys
import resin.backends.gpt_oss_cuda
for name in ('torch', 'flash_attn', 'flashinfer', 'transformers', 'tokenizers'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)


class GptOssCudaNativePlanTests(unittest.TestCase):
    def test_runner_exposes_only_the_flat_grouped_cache_path(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        self.assertIs(GptOssCudaRunner.supports_continuation_prefill, True)
        self.assertEqual(GptOssCudaRunner.execution_mode, "flat_cache_fa4_b22")
        self.assertIn("bind_cache_pool", source)
        self.assertIn("ordered_history_window_groups", source)
        self.assertIn("plan.cache_groups", source)

    def test_cache_pool_binding_is_exact_idempotent_and_permanent(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner.cache_layout = object()
        runner._cache_pool_id = None

        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        fresh = GptOssCudaRunner.__new__(GptOssCudaRunner)
        fresh.cache_layout = object()
        fresh._cache_pool_id = None
        with self.assertRaisesRegex(ValueError, "frozen cache_layout object"):
            fresh.bind_cache_pool(41, object())
        with self.assertRaisesRegex(ValueError, "different cache pool"):
            runner.bind_cache_pool(42, runner.cache_layout)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fresh = GptOssCudaRunner.__new__(GptOssCudaRunner)
                fresh.cache_layout = object()
                fresh._cache_pool_id = None
                fresh.bind_cache_pool(invalid, fresh.cache_layout)

    def test_plan_binding_requires_the_one_bound_pool(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._cache_pool_id = None
        with self.assertRaisesRegex(ValueError, "not bound"):
            runner._validate_plan_binding(_prefill_plan())
        runner._cache_pool_id = 41
        runner._validate_plan_binding(_prefill_plan())
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner._validate_plan_binding(_flat_plan([0], [1], cache_pool_id=42))

    def test_prefill_tables_are_derived_from_history_and_window_csr(self) -> None:
        batches, decode = GptOssCudaRunner._attention_batches(_prefill_plan())

        self.assertIsNone(decode)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual((batch.phase, batch.round_index), ("prefill", 0))
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.q_lengths, (128, 5))
        self.assertEqual(batch.token_indices[:3], (0, 1, 2))
        self.assertEqual(batch.token_indices[-3:], (130, 131, 132))
        self.assertEqual(batch.full_page_table_width, 2)
        self.assertEqual(batch.full_page_table, (1, 1, 17, 18))
        self.assertEqual(batch.full_seqused_k, (128, 135))
        self.assertEqual(batch.sliding_page_table_width, 2)
        self.assertEqual(batch.sliding_page_table, (129, 129, 145, 146))
        self.assertEqual(batch.sliding_seqused_k, (128, 135))
        self.assertEqual(batch.full_append_slots[0], 128)
        self.assertEqual(batch.full_append_slots[-1], 18 * 128 + 6)
        self.assertEqual(batch.sliding_append_slots[-1], 146 * 128 + 6)

    def test_one_token_continuation_remains_explicit_prefill(self) -> None:
        plan = _flat_plan([260], [1])
        prefill, decode = GptOssCudaRunner._attention_batches(plan)

        self.assertIsNone(decode)
        self.assertEqual(len(prefill), 1)
        batch = prefill[0]
        self.assertEqual(batch.q_lengths, (1,))
        self.assertEqual(batch.token_indices, (0,))
        self.assertEqual(batch.full_page_table, (1, 2, 3))
        self.assertEqual(batch.full_seqused_k, (261,))
        self.assertEqual(batch.sliding_page_table, (130, 129))
        self.assertEqual(batch.sliding_seqused_k, (133,))

    def test_mixed_rows_keep_explicit_attention_phases(self) -> None:
        prefill, decode = GptOssCudaRunner._attention_batches(_mixed_plan())

        self.assertEqual(len(prefill), 1)
        self.assertEqual(prefill[0].sequence_indices, (0,))
        self.assertEqual(prefill[0].token_indices, (0,))
        self.assertIsNotNone(decode)
        assert decode is not None
        self.assertEqual(decode.sequence_indices, (1,))
        self.assertEqual(decode.token_indices, (1,))

    def test_decode_tables_and_append_slots_come_from_sparse_groups(self) -> None:
        batch = GptOssCudaRunner._decode_batch(_decode_plan())

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.q_lengths, (1, 1))
        self.assertEqual(batch.full_page_table_width, 3)
        self.assertEqual(batch.full_page_table, (1, 2, 1, 17, 18, 19))
        self.assertEqual(batch.full_seqused_k, (131, 261))
        self.assertEqual(batch.sliding_page_table, (129, 130, 146, 145))
        self.assertEqual(batch.sliding_seqused_k, (131, 133))
        self.assertEqual(batch.full_append_slots, (2 * 128 + 2, 19 * 128 + 4))
        self.assertEqual(
            batch.sliding_append_slots,
            (130 * 128 + 2, 145 * 128 + 4),
        )

    def test_frozen_append_destinations_are_a_validation_oracle(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[1].append_block_ids[0] += 1
        with self.assertRaisesRegex(ValueError, "append destinations"):
            GptOssCudaRunner._attention_batches(plan)

    def test_group_order_suffix_and_parent_ownership_fail_closed(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[0].group_index = 1
        with self.assertRaisesRegex(ValueError, "ordered"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([260], [1])
        plan.cache_groups[1].first_logical_blocks[0] = 0
        with self.assertRaisesRegex(ValueError, "active suffix"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([0], [1])
        plan.cache_groups[1].block_ids[0] = plan.cache_groups[0].block_ids[0]
        plan.cache_groups[1].append_block_ids[0] = plan.cache_groups[0].block_ids[0]
        with self.assertRaisesRegex(ValueError, "cannot be interpreted"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([0], [1])
        with self.assertRaisesRegex(ValueError, "outside"):
            GptOssCudaRunner._attention_batches(plan, parent_capacity=64)

    def test_window_prefill_boundary_is_validated_before_device_work(self) -> None:
        plan = _flat_plan([127], [2])
        with self.assertRaisesRegex(ValueError, "within one logical"):
            GptOssCudaRunner._attention_batches(plan)

    def test_cuda_graph_padding_repeats_only_owned_history_pages(self) -> None:
        batch = GptOssCudaRunner._decode_batch(_decode_plan())
        assert batch is not None
        self.assertEqual(
            _fixed_width_full_page_table(batch, 5),
            (1, 2, 1, 1, 1, 17, 18, 19, 17, 17),
        )
        self.assertEqual(_fixed_width_full_page_table(batch, 3), batch.full_page_table)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _fixed_width_full_page_table(batch, 2)

    def test_failed_eager_work_must_quiesce_or_be_post_accept(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        synchronized = []
        runner.device = "cuda:0"
        runner._torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda device: synchronized.append(device))
        )
        runner._synchronize_failed_eager_forward()
        self.assertEqual(synchronized, ["cuda:0"])

        runner._torch.cuda.synchronize = lambda device: (_ for _ in ()).throw(
            RuntimeError(f"{device} synchronize failed")
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined") as caught:
            runner._synchronize_failed_eager_forward()
        self.assertIs(getattr(caught.exception, "_resin_post_accept", None), True)

    def test_cuda_graph_uses_three_exact_decode_warmups(self) -> None:
        self.assertEqual(_CUDA_GRAPH_WARMUP_STEPS, 3)
        source = inspect.getsource(GptOssCudaRunner._capture_cuda_graph_decode)
        self.assertIn("range(_CUDA_GRAPH_WARMUP_STEPS)", source)
        self.assertIn(
            "self._cuda_graph_warmup_count += _CUDA_GRAPH_WARMUP_STEPS",
            source,
        )
        warmup_snapshot = "warmup_output_ids = state.output_ids.clone()"
        validation_replay = "graph.replay()"
        parity_check = "self._validate_cuda_graph_decode_output_ids("
        self.assertIn(warmup_snapshot, source)
        self.assertLess(source.index(warmup_snapshot), source.index(validation_replay))
        self.assertLess(source.index(validation_replay), source.index(parity_check))

    def test_cuda_graph_capture_and_replay_pin_the_runner_device(self) -> None:
        events = []

        class Context:
            def __init__(self, name, value=None):
                self.name = name
                self.value = value

            def __enter__(self):
                events.append((f"{self.name}:enter", self.value))

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                events.append((f"{self.name}:exit", self.value))

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_stream(self, stream):
                events.append(("wait_stream", self.name, stream.name))

        class Graph:
            @staticmethod
            def replay():
                events.append(("replay",))

        class Cuda:
            @staticmethod
            def current_stream(device):
                events.append(("current_stream", device))
                return Stream("current")

            @staticmethod
            def Stream(*, device):
                events.append(("new_stream", device))
                return Stream("warmup")

            @staticmethod
            def stream(stream):
                return Context("stream", stream.name)

            @staticmethod
            def device(device):
                return Context("device", device)

            @staticmethod
            def CUDAGraph():
                events.append(("new_graph",))
                return Graph()

            @staticmethod
            def graph(graph):
                del graph
                return Context("graph")

            @staticmethod
            def synchronize(device):
                events.append(("synchronize", device))

            @staticmethod
            def memory_allocated(device):
                events.append(("memory_allocated", device))
                return 100

        class Torch:
            cuda = Cuda

            @staticmethod
            def inference_mode():
                return Context("inference")

        class OutputIds:
            @staticmethod
            def clone():
                events.append(("clone_output",))
                return "warmup output"

            @staticmethod
            def tolist():
                return [1]

        runner = _cuda_graph_registry_runner(((1, 4),))
        runner._torch = Torch
        runner.device = "cuda:1"
        runner._cuda_graph_decode_body = lambda state: events.append(("body",))
        runner._validate_first_forward_logits = lambda logits: None
        runner._validate_cuda_graph_decode_output_ids = lambda *args: None
        state = SimpleNamespace(graph=None, output_ids=OutputIds())
        runner._cuda_graph_decode_registry[(1, 4)].state = state

        GptOssCudaRunner._capture_cuda_graph_decode(
            runner, state, allocation_before=25, shape=(1, 4)
        )
        device_enter = events.index(("device:enter", "cuda:1"))
        graph_created = events.index(("new_graph",))
        graph_capture = events.index(("graph:enter", None))
        validation_replay = events.index(("replay",))
        device_exit = events.index(("device:exit", "cuda:1"))
        self.assertLess(device_enter, graph_created)
        self.assertLess(graph_created, graph_capture)
        self.assertLess(graph_capture, validation_replay)
        self.assertLess(validation_replay, device_exit)

        events.clear()
        runner._stage_cuda_graph_decode = lambda *args: None
        self.assertEqual(
            runner._forward_cuda_graph_decode(object(), object(), (1, 4)), [1]
        )
        self.assertEqual(
            events,
            [
                ("device:enter", "cuda:1"),
                ("replay",),
                ("device:exit", "cuda:1"),
            ],
        )

    def test_cuda_graph_validation_replay_mismatch_is_fail_stop(self) -> None:
        class Cuda:
            @staticmethod
            def memory_allocated(device):
                self.assertEqual(device, "cuda:0")
                return 123

            @staticmethod
            def synchronize(device):
                self.assertEqual(device, "cuda:0")

        class Torch:
            cuda = Cuda

            @staticmethod
            def equal(left, right):
                return left == right

        runner = _cuda_graph_registry_runner(((2, 4),))
        runner._torch = Torch
        runner.device = "cuda:0"
        runner._cuda_graph_replay_count = 7
        runner._cuda_graph_hit_count = 7
        state = SimpleNamespace(graph=None)
        runner._new_cuda_graph_decode_state = lambda shape: state
        runner._stage_cuda_graph_decode = lambda *args: None

        def reject_mismatched_replay(*args, **kwargs):
            runner._validate_cuda_graph_decode_output_ids((11, 12), (11, 13))

        runner._capture_cuda_graph_decode = reject_mismatched_replay
        with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
            runner._forward_cuda_graph_decode(object(), object(), (2, 4))

        self.assertIs(runner._cuda_graph_decode_registry[(2, 4)].state, state)
        self.assertIsNone(state.graph)
        self.assertEqual(runner._cuda_graph_capture_count, 0)
        self.assertEqual(runner._cuda_graph_replay_count, 7)
        self.assertEqual(runner._cuda_graph_hit_count, 7)
        self.assertEqual(runner._cuda_graph_miss_count, 1)
        self.assertIn("decode graph capture", runner._cuda_graph_poisoned_error)
        self.assertIn("do not exactly match", runner._cuda_graph_poisoned_error)

    def test_cuda_graph_shape_registry_is_immutable_sorted_and_validated(
        self,
    ) -> None:
        constructor_parameters = inspect.signature(GptOssCudaRunner).parameters
        self.assertIn("cuda_graph_decode_shapes", constructor_parameters)
        self.assertNotIn("cuda_graph_decode_shape", constructor_parameters)
        self.assertEqual(_normalize_cuda_graph_decode_shapes(None), ())
        self.assertEqual(
            _normalize_cuda_graph_decode_shapes(((2, 72), (1, 72), (2, 65))),
            ((1, 72), (2, 65), (2, 72)),
        )
        self.assertEqual(
            _normalize_cuda_graph_decode_shapes({2: 72, 1: 65}),
            ((1, 65), (2, 72)),
        )
        self.assertEqual(_normalize_cuda_graph_decode_shapes(((32, 72),)), ((32, 72),))
        self.assertEqual(
            _normalize_cuda_graph_decode_shapes(
                (shape for shape in ((2, 72), (1, 65)))
            ),
            ((1, 65), (2, 72)),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _normalize_cuda_graph_decode_shapes(((1, 65), (1, 65)))
        for invalid in (
            "1x65",
            ((0, 65),),
            ((1, False),),
            ([1, 65],),
            {1: 0},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                _normalize_cuda_graph_decode_shapes(invalid)

    def test_cuda_graph_constructor_rejects_oversize_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            GptOssCudaRunner(
                cache_parent_capacity=8,
                cuda_graph_decode_shapes=((1, 8), (2, 9)),
            )

    def test_cuda_graph_registry_selects_smallest_fitting_exact_shape(self) -> None:
        runner = _cuda_graph_registry_runner(((3, 8), (2, 8), (2, 4)))

        plan, width_three = _graph_decode_inputs(2, 3)
        self.assertEqual(
            runner._select_cuda_graph_decode_shape(plan, width_three), (2, 4)
        )
        self.assertIsNone(runner._cuda_graph_fallback_kind(plan, (), width_three))

        plan, width_five = _graph_decode_inputs(2, 5)
        self.assertEqual(
            runner._select_cuda_graph_decode_shape(plan, width_five), (2, 8)
        )
        self.assertIsNone(runner._cuda_graph_fallback_kind(plan, (), width_five))

        plan, wrong_batch = _graph_decode_inputs(1, 3)
        self.assertIsNone(runner._select_cuda_graph_decode_shape(plan, wrong_batch))
        self.assertEqual(
            runner._cuda_graph_fallback_kind(plan, (), wrong_batch), "shape"
        )

    def test_cuda_graph_registry_owns_isolated_states_and_stats(self) -> None:
        runner = _cuda_graph_registry_runner(((2, 4), (1, 4)))
        plan_one, batch_one = _graph_decode_inputs(1, 3)
        plan_two, batch_two = _graph_decode_inputs(2, 3)

        self.assertEqual(
            runner._forward_cuda_graph_decode(plan_one, batch_one, (1, 4)), [1]
        )
        self.assertEqual(
            runner._forward_cuda_graph_decode(plan_two, batch_two, (2, 4)), [2]
        )
        self.assertEqual(
            runner._forward_cuda_graph_decode(plan_one, batch_one, (1, 4)), [1]
        )

        first = runner._cuda_graph_decode_registry[(1, 4)].state
        second = runner._cuda_graph_decode_registry[(2, 4)].state
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertIsNot(first.graph, second.graph)

        stats = runner.cuda_graph_stats
        self.assertEqual(stats["configured_shapes"], ((1, 4), (2, 4)))
        self.assertEqual(stats["captured_shapes"], ((1, 4), (2, 4)))
        self.assertEqual(stats["capture_count"], 2)
        self.assertEqual(stats["replay_count"], 1)
        self.assertEqual(stats["graph_hit_count"], 3)
        self.assertEqual(stats["graph_miss_count"], 0)
        by_shape = {row["shape"]: row for row in stats["shape_stats"]}
        self.assertEqual(by_shape[(1, 4)]["capture_count"], 1)
        self.assertEqual(by_shape[(1, 4)]["replay_count"], 1)
        self.assertEqual(by_shape[(1, 4)]["graph_hit_count"], 2)
        self.assertEqual(by_shape[(2, 4)]["capture_count"], 1)
        self.assertEqual(by_shape[(2, 4)]["replay_count"], 0)
        self.assertEqual(by_shape[(2, 4)]["graph_hit_count"], 1)
        by_shape[(1, 4)]["capture_count"] = 99
        self.assertEqual(runner.cuda_graph_stats["shape_stats"][0]["capture_count"], 1)

    def test_cuda_graph_registry_allocates_exact_independent_device_buffers(
        self,
    ) -> None:
        allocations = []

        class Tensor:
            def __init__(self, shape, dtype, device):
                self.shape = shape
                self.dtype = dtype
                self.device = device

        class Torch:
            int32 = "int32"
            long = "long"
            bfloat16 = "bfloat16"

            @staticmethod
            def empty(*shape, dtype, device):
                if len(shape) == 1 and isinstance(shape[0], tuple):
                    shape = shape[0]
                tensor = Tensor(shape, dtype, device)
                allocations.append(tensor)
                return tensor

            @staticmethod
            def arange(stop, *, dtype, device):
                tensor = Tensor((stop,), dtype, device)
                allocations.append(tensor)
                return tensor

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = Torch
        runner.device = "cuda:1"
        runner.hidden_size = 16
        one = runner._new_cuda_graph_decode_state((1, 4))
        two = runner._new_cuda_graph_decode_state((2, 8))

        def persistent_tensors(state):
            batch = state.attention_batch
            return (
                state.token_ids,
                state.positions,
                batch.token_indices,
                batch.sequence_indices,
                batch.last_token_indices,
                batch.cu_seqlens_q,
                batch.full_page_table,
                batch.full_seqused_k,
                batch.sliding_page_table,
                batch.sliding_seqused_k,
                batch.full_append_slots,
                batch.sliding_append_slots,
                state.final_hidden,
                state.output_ids,
            )

        one_tensors = persistent_tensors(one)
        two_tensors = persistent_tensors(two)
        self.assertEqual(len({id(tensor) for tensor in one_tensors}), 14)
        self.assertEqual(len({id(tensor) for tensor in two_tensors}), 14)
        self.assertTrue(set(map(id, one_tensors)).isdisjoint(map(id, two_tensors)))
        self.assertEqual(one.token_ids.shape, (1,))
        self.assertEqual(one.attention_batch.cu_seqlens_q.shape, (2,))
        self.assertEqual(one.attention_batch.full_page_table.shape, (1, 4))
        self.assertEqual(one.attention_batch.sliding_page_table.shape, (1, 2))
        self.assertEqual(one.final_hidden.shape, (1, 16))
        self.assertEqual(one.output_ids.shape, (1,))
        self.assertEqual(two.token_ids.shape, (2,))
        self.assertEqual(two.attention_batch.cu_seqlens_q.shape, (3,))
        self.assertEqual(two.attention_batch.full_page_table.shape, (2, 8))
        self.assertEqual(two.attention_batch.sliding_page_table.shape, (2, 2))
        self.assertEqual(two.final_hidden.shape, (2, 16))
        self.assertEqual(two.output_ids.shape, (2,))
        self.assertTrue(allocations)
        self.assertTrue(all(tensor.device == "cuda:1" for tensor in allocations))

    def test_single_shape_registry_uses_plural_per_shape_stats(self) -> None:
        runner = _cuda_graph_registry_runner(((2, 4),))
        plan, batch = _graph_decode_inputs(2, 3)
        runner._forward_cuda_graph_decode(plan, batch, (2, 4))
        runner._forward_cuda_graph_decode(plan, batch, (2, 4))

        state = runner._cuda_graph_decode_registry[(2, 4)].state
        self.assertIsNotNone(state)
        stats = runner.cuda_graph_stats
        self.assertEqual(stats["configured_shapes"], ((2, 4),))
        self.assertTrue(stats["captured"])
        self.assertEqual(stats["capture_count"], 1)
        self.assertEqual(stats["replay_count"], 1)
        self.assertEqual(stats["graph_hit_count"], 2)
        self.assertNotIn("configured_shape", stats)
        self.assertNotIn("capture_ms", stats)
        self.assertNotIn("capture_memory_bytes", stats)
        shape_stats = stats["shape_stats"]
        self.assertEqual(len(shape_stats), 1)
        self.assertEqual(shape_stats[0]["capture_ms"], 2.0)
        self.assertEqual(shape_stats[0]["capture_memory_bytes"], 4)

    def test_unconfigured_cuda_graph_shape_falls_back_without_graph_miss(
        self,
    ) -> None:
        runner = _cuda_graph_registry_runner(((1, 4), (2, 4)))
        plan, batch = _graph_decode_inputs(3, 3)
        fallback = runner._cuda_graph_fallback_kind(plan, (), batch)
        self.assertEqual(fallback, "shape")
        runner._record_cuda_graph_fallback(fallback)
        self.assertEqual(runner._cuda_graph_eager_fallback_count, 1)
        self.assertEqual(runner._cuda_graph_shape_fallback_count, 1)
        self.assertEqual(runner._cuda_graph_miss_count, 0)
        self.assertEqual(runner.cuda_graph_stats["captured_shapes"], ())

    def test_registry_prelaunch_failure_cleans_only_selected_shape(self) -> None:
        runner = _cuda_graph_registry_runner(((1, 4), (2, 4)))
        plan_one, batch_one = _graph_decode_inputs(1, 3)
        runner._forward_cuda_graph_decode(plan_one, batch_one, (1, 4))
        first_state = runner._cuda_graph_decode_registry[(1, 4)].state

        def reject_second(state, *args):
            del args
            if state.shape == (2, 4):
                raise ValueError("bad second shape metadata")

        runner._stage_cuda_graph_decode = reject_second
        plan_two, batch_two = _graph_decode_inputs(2, 3)
        with self.assertRaisesRegex(ValueError, "bad second shape metadata"):
            runner._forward_cuda_graph_decode(plan_two, batch_two, (2, 4))

        self.assertIs(runner._cuda_graph_decode_registry[(1, 4)].state, first_state)
        self.assertIsNone(runner._cuda_graph_decode_registry[(2, 4)].state)
        self.assertIsNone(runner._cuda_graph_poisoned_error)
        self.assertEqual(runner._cuda_graph_miss_count, 0)

    def test_registry_unsafe_failure_retains_states_and_globally_fail_stops(
        self,
    ) -> None:
        runner = _cuda_graph_registry_runner(((1, 4), (2, 4)))
        plan_one, batch_one = _graph_decode_inputs(1, 3)
        runner._forward_cuda_graph_decode(plan_one, batch_one, (1, 4))
        first_state = runner._cuda_graph_decode_registry[(1, 4)].state

        def fail_capture(state, *, allocation_before, shape):
            del state, allocation_before, shape
            raise RuntimeError("second capture failed")

        runner._capture_cuda_graph_decode = fail_capture
        plan_two, batch_two = _graph_decode_inputs(2, 3)
        with self.assertRaisesRegex(RuntimeError, "second capture failed"):
            runner._forward_cuda_graph_decode(plan_two, batch_two, (2, 4))

        second_state = runner._cuda_graph_decode_registry[(2, 4)].state
        self.assertIs(runner._cuda_graph_decode_registry[(1, 4)].state, first_state)
        self.assertIsNotNone(second_state)
        self.assertEqual(runner._cuda_graph_miss_count, 1)
        self.assertEqual(runner._cuda_graph_decode_registry[(2, 4)].miss_count, 1)
        self.assertIn("decode graph capture", runner._cuda_graph_poisoned_error)
        with self.assertRaisesRegex(RuntimeError, "construct a fresh runner"):
            runner._raise_if_cuda_graph_poisoned()

    def test_cuda_graph_body_preserves_rust_sequence_permutation(self) -> None:
        calls = []

        class FinalHidden:
            def index_copy_(self, dimension, indices, values):
                calls.append(("index_copy", dimension, indices, values))

        class Model:
            @staticmethod
            def lm_head(hidden):
                calls.append(("lm_head", hidden))
                return "logits"

        class Torch:
            @staticmethod
            def argmax(logits, *, dim, out):
                calls.append(("argmax", logits, dim, out))

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = Torch
        runner._model = Model
        runner._forward_batch = lambda token_ids, positions, batch: "last_hidden"
        state = SimpleNamespace(
            token_ids="tokens",
            positions="positions",
            attention_batch=SimpleNamespace(sequence_indices="rust:[1,0]"),
            final_hidden=FinalHidden(),
            output_ids="output_ids",
        )

        self.assertEqual(runner._cuda_graph_decode_body(state), "logits")
        self.assertEqual(
            calls,
            [
                ("index_copy", 0, "rust:[1,0]", "last_hidden"),
                ("lm_head", state.final_hidden),
                ("argmax", "logits", -1, "output_ids"),
            ],
        )

    def test_failed_cuda_graph_capture_is_fail_stop_before_any_next_work(
        self,
    ) -> None:
        class Cuda:
            @staticmethod
            def memory_allocated(device):
                self.assertEqual(device, "cuda:0")
                return 123

            @staticmethod
            def synchronize(device):
                self.assertEqual(device, "cuda:0")

        runner = _cuda_graph_registry_runner(((2, 4),))
        runner._torch = SimpleNamespace(cuda=Cuda)
        runner.device = "cuda:0"
        state = SimpleNamespace(graph=None)
        runner._new_cuda_graph_decode_state = lambda shape: state
        runner._stage_cuda_graph_decode = lambda *args: None

        def fail_capture(*args, **kwargs):
            raise RuntimeError("capture failed")

        runner._capture_cuda_graph_decode = fail_capture
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            runner._forward_cuda_graph_decode(object(), object(), (2, 4))
        self.assertIs(runner._cuda_graph_decode_registry[(2, 4)].state, state)
        self.assertEqual(runner._cuda_graph_miss_count, 1)
        self.assertTrue(runner.cuda_graph_stats["poisoned"])
        self.assertIn("capture failed", runner.cuda_graph_stats["poison_error"])

        calls = []
        runner._validate_plan_binding = lambda plan: calls.append(plan)
        with self.assertRaisesRegex(RuntimeError, "construct a fresh runner"):
            runner.forward(object())
        self.assertEqual(calls, [])

    def test_prelaunch_staging_validation_failure_remains_retryable(self) -> None:
        class Cuda:
            @staticmethod
            def memory_allocated(device):
                del device
                return 0

        runner = _cuda_graph_registry_runner(((2, 4),))
        runner._torch = SimpleNamespace(cuda=Cuda)
        runner.device = "cuda:0"
        state = SimpleNamespace(graph=None)
        runner._new_cuda_graph_decode_state = lambda shape: state

        def reject_staging(*args):
            raise ValueError("bad host metadata")

        runner._stage_cuda_graph_decode = reject_staging
        with self.assertRaisesRegex(ValueError, "bad host metadata"):
            runner._forward_cuda_graph_decode(object(), object(), (2, 4))
        self.assertIsNone(runner._cuda_graph_decode_registry[(2, 4)].state)
        self.assertIsNone(runner._cuda_graph_poisoned_error)
        self.assertEqual(runner._cuda_graph_miss_count, 0)

    def test_device_staging_failure_is_conservatively_fail_stop(self) -> None:
        class Cuda:
            @staticmethod
            def memory_allocated(device):
                del device
                return 0

        runner = _cuda_graph_registry_runner(((2, 4),))
        runner._torch = SimpleNamespace(cuda=Cuda)
        runner.device = "cuda:0"
        state = SimpleNamespace(graph=None)
        runner._new_cuda_graph_decode_state = lambda shape: state

        def fail_device_copy(*args):
            raise RuntimeError("CUDA copy failed")

        runner._stage_cuda_graph_decode = fail_device_copy
        with self.assertRaisesRegex(RuntimeError, "CUDA copy failed"):
            runner._forward_cuda_graph_decode(object(), object(), (2, 4))
        self.assertIs(runner._cuda_graph_decode_registry[(2, 4)].state, state)
        self.assertIn("decode graph staging", runner._cuda_graph_poisoned_error)
        self.assertEqual(runner._cuda_graph_miss_count, 1)

    def test_failed_cuda_graph_replay_poisons_without_counting_a_hit(self) -> None:
        class Graph:
            @staticmethod
            def replay():
                raise RuntimeError("illegal replay")

        runner = _cuda_graph_registry_runner(((2, 4),))
        runner._torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda device: None,
                device=lambda device: nullcontext(),
            )
        )
        runner.device = "cuda:0"
        runner._cuda_graph_decode_registry[(2, 4)].state = SimpleNamespace(
            graph=Graph()
        )
        runner._cuda_graph_replay_count = 7
        runner._cuda_graph_hit_count = 7
        runner._stage_cuda_graph_decode = lambda *args: None

        with self.assertRaisesRegex(RuntimeError, "illegal replay"):
            runner._forward_cuda_graph_decode(object(), object(), (2, 4))
        self.assertIn("decode graph replay", runner._cuda_graph_poisoned_error)
        self.assertEqual(runner._cuda_graph_miss_count, 1)
        self.assertEqual(runner._cuda_graph_replay_count, 7)
        self.assertEqual(runner._cuda_graph_hit_count, 7)

    def test_cuda_attention_has_no_fp32_promotion_or_kv_head_expansion(self) -> None:
        source = inspect.getsource(GptOssCudaRunner._attention)
        self.assertNotIn(".float(", source)
        self.assertNotIn("repeat_interleave", source)
        self.assertNotIn("repeat(", source)
        self.assertIn("index_copy_", source)

    def test_runner_has_no_python_request_cache_or_page_allocator(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        self.assertNotIn("_request_pages", source)
        self.assertNotIn("_request_lengths", source)
        self.assertNotIn("_free_pages", source)
        self.assertNotIn("_logical_pages", source)
        self.assertNotIn("_sequence_cache", source)

    def test_one_zeroed_parent_allocation_backs_both_group_views(self) -> None:
        source = inspect.getsource(GptOssCudaRunner.__init__)
        self.assertEqual(source.count("torch.zeros("), 1)
        self.assertIn("self.cache_parent_capacity + 1", source)
        self.assertIn("self._cache_parent_storage = torch.zeros(", source)
        self.assertIn(
            "self._window_k_cache = self._history_k_cache",
            source,
        )
        self.assertIn(
            "self._window_v_cache = self._history_v_cache",
            source,
        )

    def test_nonfinite_checks_are_opt_in_and_cover_attention_and_moe(self) -> None:
        class TorchMustNotRun:
            @staticmethod
            def isfinite(tensor):
                del tensor
                raise AssertionError("normal inference must not check finiteness")

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._diagnose_nonfinite = False
        runner._torch = TorchMustNotRun
        runner._check_finite(object(), "hidden")

        attention_source = inspect.getsource(GptOssCudaRunner._attention)
        forward_source = inspect.getsource(GptOssCudaRunner._forward_batch)
        for boundary in (
            "query",
            "key",
            "value",
            "fa4_output",
            "attention_out",
        ):
            self.assertIn(f'"{boundary}"', attention_source)
        for boundary in (
            "round_input_hidden",
            "normalized_attention_input",
            "post_attention_residual",
            "normalized_mlp_input",
            "moe_output",
            "post_mlp_hidden",
            "round_output_hidden",
        ):
            self.assertIn(f'"{boundary}"', forward_source)

    def test_nonfinite_error_contains_exact_frozen_batch_context(self) -> None:
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FirstIndex:
            @staticmethod
            def tolist():
                return [0, 3]

        class Nonzero:
            @staticmethod
            def __getitem__(index):
                del index
                return FirstIndex()

        class NonfiniteMask:
            @staticmethod
            def sum():
                return Scalar(1)

            @staticmethod
            def nonzero(*, as_tuple):
                assert not as_tuple
                return Nonzero()

        class FiniteMask:
            @staticmethod
            def all():
                return Scalar(False)

            @staticmethod
            def __invert__():
                return NonfiniteMask()

        class FiniteValues:
            @staticmethod
            def numel():
                return 3

            @staticmethod
            def min():
                return Scalar(-2.0)

            @staticmethod
            def max():
                return Scalar(4.0)

        class Tensor:
            shape = (2, 4)
            dtype = "bfloat16"

            def detach(self):
                return self

            @staticmethod
            def masked_select(mask):
                del mask
                return FiniteValues()

        class FakeTorch:
            @staticmethod
            def isfinite(tensor):
                del tensor
                return FiniteMask()

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._diagnose_nonfinite = True
        runner._torch = FakeTorch
        batch = SimpleNamespace(
            phase="prefill",
            round_index=2,
            diagnostic_context=(
                "full_page_table=(0, 1, 2); full_seqused_k=(260,); "
                "full_seqused_minmax=(260, 260); "
                "sliding_page_table=(11, 10); sliding_seqused_k=(132,); "
                "sliding_seqused_minmax=(132, 132)"
            ),
        )

        with self.assertRaises(FloatingPointError) as caught:
            runner._check_finite(
                Tensor(),
                "fa4_output",
                batch=batch,
                layer_index=7,
                layer_type="sliding_attention",
            )
        message = str(caught.exception)
        for expected in (
            "tensor=fa4_output",
            "phase=prefill",
            "round=2",
            "layer=7",
            "layer_type=sliding_attention",
            "first_nonfinite_index=(0, 3)",
            "finite_minmax=(-2.0, 4.0)",
            "full_page_table=(0, 1, 2)",
            "full_seqused_k=(260,)",
            "sliding_page_table=(11, 10)",
            "sliding_seqused_k=(132,)",
        ):
            self.assertIn(expected, message)


class GptOssCudaFa4CallTests(unittest.TestCase):
    def test_exact_b22_surface_for_full_and_sliding_attention(self) -> None:
        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return ("output", None)

        common = {
            "cu_seqlens_q": object(),
            "max_seqlen_q": 1,
            "page_table": object(),
            "seqused_k": object(),
            "min_seqlen_k": 129,
            "max_seqlen_k": 8193,
            "softmax_scale": 1 / math.sqrt(64),
            "learnable_sink": object(),
        }
        self.assertEqual(
            _invoke_fa4_b22(kernel, "q", "k", "v", sliding=False, **common),
            "output",
        )
        self.assertEqual(calls[-1][0], ("q", "k", "v"))
        full = calls[-1][1]
        self.assertTrue(full["causal"])
        self.assertEqual(full["window_size"], (None, None))
        self.assertTrue(full["pack_gqa"])
        self.assertEqual(full["num_splits"], 1)
        self.assertIs(full["learnable_sink"], common["learnable_sink"])
        self.assertEqual(full["softmax_scale"], 0.125)
        self.assertIsNone(full["cu_seqlens_k"])

        _invoke_fa4_b22(kernel, "q", "k", "v", sliding=True, **common)
        sliding = calls[-1][1]
        self.assertEqual(sliding["window_size"], (127, 0))
        self.assertTrue(sliding["causal"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class GptOssCudaTinyOracleTests(unittest.TestCase):
    def test_rotary_path_matches_reference_formula_without_dtype_promotion(
        self,
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = torch
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -3.0]],
                [[-2.0, 1.0, 0.25, 0.75], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=torch.bfloat16,
        )
        cos = torch.tensor([[[1.0, 0.5], [0.25, 1.0]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[0.0, 0.5], [1.0, 0.0]]], dtype=torch.bfloat16)
        first, second = hidden.chunk(2, dim=-1)
        expected = torch.cat(
            (
                first * cos.squeeze(0).unsqueeze(1)
                - second * sin.squeeze(0).unsqueeze(1),
                second * cos.squeeze(0).unsqueeze(1)
                + first * sin.squeeze(0).unsqueeze(1),
            ),
            dim=-1,
        )

        actual = runner._apply_yarn_rotary(hidden, cos, sin)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
