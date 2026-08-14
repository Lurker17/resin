from __future__ import annotations

import gc
import inspect
import math
import unittest
import weakref
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from resin import BatchPhase
from resin.backend.cuda.execution import CudaExecutor
from resin.backend.cuda.mapped_state import MappedStateArena
from resin.backend.cuda.paged_kv import PagedKvArena
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import TorchSampler


class _FakeTensor:
    def __init__(
        self, values, *, dtype, device, pin_memory=False, log=None, tolist_error=None
    ) -> None:
        self.values = tuple(values)
        self._shape = (len(self.values),)
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory
        self.log = log
        self.source = None
        self.tolist_error = tolist_error

    @property
    def shape(self):
        return self._shape

    def to(self, *, device, non_blocking):
        if self.log is not None:
            self.log.append(("h2d", device, non_blocking))
        result = _FakeTensor(self.values, dtype=self.dtype, device=device, log=self.log)
        result.source = self
        return result

    def copy_(self, source, *, non_blocking):
        if self.log is not None:
            self.log.append(("d2h", source.device, non_blocking))
        self.values = source.values
        self.source = source
        return self

    def view(self, *shape):
        result = _FakeTensor(
            self.values,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            log=self.log,
        )
        result.source = self
        result._shape = tuple(shape)
        return result

    def __getitem__(self, index):
        values = self.values[index]
        if not isinstance(values, tuple):
            values = (values,)
        return _FakeTensor(values, dtype=self.dtype, device=self.device, log=self.log)

    def unsqueeze(self, dimension):
        shape = list(self._shape)
        shape.insert(dimension, 1)
        result = self.view(*shape)
        return result

    def squeeze(self, dimension=None):
        shape = list(self._shape)
        if dimension is None:
            shape = [value for value in shape if value != 1]
        elif shape[dimension] == 1:
            del shape[dimension]
        result = self.view(*shape)
        return result

    def tolist(self):
        if self.log is not None:
            self.log.append(("tolist",))
        if self.tolist_error:
            error = self.tolist_error
            self.tolist_error = None
            raise error
        return list(self.values)


class _LifetimeResource:
    def __init__(self, name, log) -> None:
        self.name = name
        self.log = log

    def __del__(self) -> None:
        self.log.append(("resource.del", self.name))


class _FakeSlabView:
    def __init__(self, slab, key, shape) -> None:
        self.slab = slab
        self.key = key
        self._shape = tuple(shape)
        self.dtype = slab.dtype
        self.device = slab.device
        self.copied_from = None

    @property
    def shape(self):
        return self._shape

    def zero_(self):
        self.slab.log.append(("slab.zero", self.key))
        self.slab.zeroed.append(self.key)
        return self

    def unsqueeze(self, dimension):
        shape = list(self._shape)
        shape.insert(dimension, 1)
        return _FakeSlabView(self.slab, self.key, shape)

    def copy_(self, source):
        self.copied_from = source
        self.slab.log.append(("slab.copy", self.key, source))
        self.slab.copies.append((self.key, source))
        return self


class _FakeSlab:
    def __init__(self, shape, *, dtype, device, log) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.log = log
        self.zeroed = []
        self.copies = []

    def __getitem__(self, key):
        first, second = key
        if isinstance(first, slice):
            shape = (self.shape[0], *self.shape[2:])
        else:
            shape = self.shape[2:]
        return _FakeSlabView(self, (first, second), shape)


class _FakeStream:
    def __init__(self, log) -> None:
        self.log = log
        self.ready = False
        self.query_error = None

    def query(self) -> bool:
        self.log.append(("stream.query", self.ready))
        if self.query_error is not None:
            error = self.query_error
            self.query_error = None
            raise error
        return self.ready


class _FakeEvent:
    def __init__(self, log, record_error=None) -> None:
        self.log = log
        self.ready = False
        self.recorded_stream = None
        self.record_error = record_error
        self.synchronize_error = None
        self.query_error = None

    def record(self, stream) -> None:
        if self.record_error is not None:
            error = self.record_error
            self.record_error = None
            raise error
        self.recorded_stream = stream
        self.log.append(("event.record", stream))

    def query(self) -> bool:
        self.log.append(("event.query", self.ready))
        if self.query_error is not None:
            error = self.query_error
            self.query_error = None
            raise error
        return self.ready

    def synchronize(self) -> None:
        self.log.append(("event.synchronize",))
        if self.synchronize_error is not None:
            error = self.synchronize_error
            self.synchronize_error = None
            raise error
        self.ready = True


class _FakeStreamContext:
    def __init__(self, cuda, stream) -> None:
        self.cuda = cuda
        self.stream = stream

    def __enter__(self):
        if self.cuda.active_stream is not None:
            raise AssertionError("nested fake CUDA stream context")
        self.cuda.active_stream = self.stream
        self.cuda.log.append(("stream.enter", self.stream))

    def __exit__(self, exc_type, exc, traceback):
        self.cuda.log.append(("stream.exit", self.stream))
        self.cuda.active_stream = None


class _FakeCuda:
    def __init__(self, log) -> None:
        self.log = log
        self.active_stream = None
        self.events = []
        self.event_record_errors = []

    def Event(self):
        record_error = (
            self.event_record_errors.pop(0) if self.event_record_errors else None
        )
        event = _FakeEvent(self.log, record_error)
        self.events.append(event)
        return event

    def stream(self, stream):
        return _FakeStreamContext(self, stream)


class _FakeTorch:
    int32 = "int32"
    long = "long"
    bfloat16 = "bfloat16"

    def __init__(self) -> None:
        self.log = []
        self.cuda = _FakeCuda(self.log)

    def tensor(self, values, *, dtype, device, pin_memory=False):
        self.log.append(("tensor", tuple(values), dtype, device, pin_memory))
        return _FakeTensor(
            values,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            log=self.log,
        )

    def empty(self, shape, *, dtype, device, pin_memory=False):
        self.log.append(("empty", shape, dtype, device, pin_memory))
        return _FakeTensor(
            [0] * shape[0],
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            log=self.log,
        )

    def zeros(self, shape, *, dtype, device):
        self.log.append(("zeros", tuple(shape), dtype, device))
        return _FakeSlab(shape, dtype=dtype, device=device, log=self.log)

    def cat(self, tensors, dim=0):
        self.log.append(("cat", tuple(tensors), dim))
        values = tuple(value for tensor in tensors for value in tensor.values)
        return _FakeTensor(values, dtype=tensors[0].dtype, device=tensors[0].device)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _RecordingDecode:
    def __init__(self, cuda=None) -> None:
        self.calls = []
        self.cuda = cuda

    def plan(self, *args, **kwargs) -> None:
        if self.cuda is not None and self.cuda.active_stream is None:
            raise AssertionError("FlashInfer planning escaped the private stream")
        if self.cuda is not None:
            self.cuda.log.append(("decode.plan", self.cuda.active_stream))
        self.calls.append((args, kwargs))


class Lfm2CudaImportTests(unittest.TestCase):
    def test_physical_cache_slab_reserves_zero_as_the_null_page(self) -> None:
        source = inspect.getsource(PagedKvArena.__init__)

        self.assertIn("self.capacity + 1", source)
        self.assertIn("self.keys = self.storage[0]", source)
        self.assertIn("self.values = self.storage[1]", source)

    def test_physical_cache_slab_zeroes_null_page_and_unwritten_tails(self) -> None:
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        constructor = inspect.getsource(Lfm2CudaRunner.__init__)
        self.assertIn("self._kv = PagedKvArena(", constructor)
        self.assertIn("self.cache_layout = CacheLayout(", constructor)
        self.assertIs(Lfm2CudaRunner.supports_continuation_prefill, True)
        self.assertIs(Lfm2CudaRunner.supports_recurrent_state_mapping, True)
        self.assertEqual(
            Lfm2CudaRunner.recurrent_state_mapping.protocol,
            "lfm2.conv_state.v1",
        )
        self.assertEqual(
            Lfm2CudaRunner.recurrent_state_mapping.access_modes,
            ["write"],
        )
        self.assertEqual(
            Lfm2CudaRunner.recurrent_state_mapping.handles_per_sequence,
            1,
        )
        self.assertTrue(callable(Lfm2CudaRunner.bind_cache_and_recurrent_state))
        self.assertTrue(callable(Lfm2CudaRunner.dispatch))
        self.assertTrue(callable(Lfm2CudaRunner.poll))
        self.assertFalse(hasattr(Lfm2CudaRunner, "recurrent_state_mapping_protocol"))
        self.assertFalse(hasattr(Lfm2CudaRunner, "bind_cache_pool"))
        self.assertFalse(hasattr(Lfm2CudaRunner, "bind_recurrent_state_mapping"))
        self.assertFalse(hasattr(Lfm2CudaRunner, "forward"))


class Lfm2CudaCachePoolBoundaryTests(unittest.TestCase):
    @staticmethod
    def _runner():
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        runner = Lfm2CudaRunner.__new__(Lfm2CudaRunner)
        runner._torch = _FakeTorch()
        runner._flashinfer = _LifetimeResource("flashinfer", runner._torch.log)
        runner._fa4 = _LifetimeResource("fa4", runner._torch.log)
        runner._model = _LifetimeResource("model", runner._torch.log)
        runner._decode_workspace = _LifetimeResource(
            "decode_workspace", runner._torch.log
        )
        runner.device = "cuda:fake"
        runner.page_size = 16
        runner.max_pages = 8
        runner.cache_layout = object()
        runner._kv = PagedKvArena.__new__(PagedKvArena)
        runner._kv.capacity = runner.max_pages
        runner._kv.page_size = runner.page_size
        runner._kv.binding = CachePoolBinding(runner.cache_layout, resource="LFM2 KV")
        runner._kv.storage = _LifetimeResource("kv_storage", runner._torch.log)
        runner._kv.keys = _LifetimeResource("k_cache", runner._torch.log)
        runner._kv.values = _LifetimeResource("v_cache", runner._torch.log)
        runner._conv_layers = [0, 2, 4, 5]
        runner._conv_slot = {
            layer: slot for slot, layer in enumerate(runner._conv_layers)
        }
        runner.hidden_size = 8
        runner.conv_cache_length = 3
        runner._decode = _RecordingDecode()
        runner.num_query_heads = 4
        runner.num_kv_heads = 2
        runner.head_dim = 64
        runner.scale = 0.125
        runner._stream = _FakeStream(runner._torch.log)
        runner._state = MappedStateArena(
            runner._torch,
            runner.device,
            runner._stream,
            protocol="lfm2.conv_state.v1",
            layer_indices=runner._conv_layers,
            row_shape=(runner.hidden_size, runner.conv_cache_length - 1),
            dtype=runner._torch.bfloat16,
            label="LFM2 recurrent state",
        )
        runner._sampler = TorchSampler(runner._torch)
        runner._executor = CudaExecutor(
            runner._torch,
            runner.device,
            runner._stream,
            resources=runner._completion_resources,
        )
        runner.vocab_size = 128
        return runner

    @staticmethod
    def _plan(**overrides):
        group_fields = {
            "group_index": 0,
            "block_indptr": [0, 1, 3],
            "block_ids": [5, 3, 7],
            "first_logical_blocks": [0, 0],
            "append_block_ids": [5, 5, 7],
            "append_block_offsets": [0, 1, 0],
        }
        for name in tuple(group_fields):
            if name in overrides:
                group_fields[name] = overrides.pop(name)
        fields = {
            "cache_pool_id": 41,
            "cache_groups": (SimpleNamespace(**group_fields),),
            "positions": [0, 1, 16],
            "sequence_ids": [10, 11],
            "engine_id": 71,
            "sequence_indptr": [0, 2, 3],
            "context_lengths": [0, 16],
            "phase": BatchPhase.PREFILL,
            "sample_sequence_indices": [0, 1],
            "token_ids": [1, 2, 3],
            "state_pool_id": 53,
            "state_slots": [0, 1],
            "state_generations": [1, 1],
            "state_access_modes": ["write", "write"],
            "state_indptr": [0, 1, 2],
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    @staticmethod
    def _bind(
        runner,
        *,
        cache_pool_id=41,
        state_pool_id=53,
        capacity=4,
    ):
        return runner.bind_cache_and_recurrent_state(
            cache_pool_id,
            runner.cache_layout,
            state_pool_id,
            capacity,
            "lfm2.conv_state.v1",
            None,
        )

    def test_combined_binding_requires_the_exact_layout(self) -> None:
        runner = self._runner()
        with self.assertRaisesRegex(ValueError, "frozen layout object"):
            runner.bind_cache_and_recurrent_state(
                41,
                object(),
                53,
                4,
                "lfm2.conv_state.v1",
                None,
            )
        self.assertIsNone(runner._kv.binding.pool_id)
        self.assertIsNone(runner._state.binding)

    def test_combined_binding_is_atomic_exact_and_idempotent(self) -> None:
        runner = self._runner()
        first = self._bind(runner, capacity=7)
        second = self._bind(runner, capacity=7)

        self.assertEqual(first[:4], (41, 53, 7, "lfm2.conv_state.v1"))
        self.assertGreater(first[-1], 0)
        self.assertEqual(second, first)
        self.assertEqual(runner._kv.binding.pool_id, 41)
        self.assertEqual(runner._state.binding, (53, 7, "lfm2.conv_state.v1"))
        self.assertEqual(runner._state.slab.shape, (4, 7, 8, 2))

        with self.assertRaisesRegex(ValueError, "different cache pool"):
            runner.bind_cache_and_recurrent_state(
                42,
                runner.cache_layout,
                53,
                7,
                "lfm2.conv_state.v1",
                None,
            )
        with self.assertRaisesRegex(ValueError, "different mapped-state"):
            runner.bind_cache_and_recurrent_state(
                41,
                runner.cache_layout,
                54,
                7,
                "lfm2.conv_state.v1",
                None,
            )

    def test_combined_binding_reuses_the_exact_registered_arena_object(self) -> None:
        from resin.backend.cuda.mapped_state import _MAPPED_STATE_ARENAS

        _MAPPED_STATE_ARENAS.clear()
        try:
            first = self._runner()
            with patch(
                "resin.backend.cuda.mapped_state.mint_state_mapping_arena_id",
                return_value=1234,
                create=True,
            ):
                first_ack = first.bind_cache_and_recurrent_state(
                    41,
                    first.cache_layout,
                    53,
                    7,
                    "lfm2.conv_state.v1",
                    None,
                )

            second = self._runner()
            with patch(
                "resin.backend.cuda.mapped_state.mint_state_mapping_arena_id",
                side_effect=AssertionError("a required arena must not be reminted"),
                create=True,
            ) as mint:
                second_ack = second.bind_cache_and_recurrent_state(
                    42,
                    second.cache_layout,
                    53,
                    7,
                    "lfm2.conv_state.v1",
                    1234,
                )

            mint.assert_not_called()
            self.assertEqual(first_ack[-1], 1234)
            self.assertEqual(second_ack[-1], 1234)
            self.assertIs(second._state.slab, first._state.slab)
            self.assertIs(
                second._state.generations,
                first._state.generations,
            )
            self.assertIs(second._state.owners, first._state.owners)
            self.assertFalse(
                any(entry[0] == "zeros" for entry in second._torch.log),
                "reusing an arena must not allocate replacement bytes",
            )

            incompatible = self._runner()
            incompatible.hidden_size += 1
            incompatible._state.row_shape = (
                incompatible.hidden_size,
                incompatible.conv_cache_length - 1,
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                incompatible.bind_cache_and_recurrent_state(
                    43,
                    incompatible.cache_layout,
                    53,
                    7,
                    "lfm2.conv_state.v1",
                    1234,
                )
            self.assertIsNone(incompatible._kv.binding.pool_id)
            self.assertIsNone(incompatible._state.binding)
        finally:
            _MAPPED_STATE_ARENAS.clear()

    def test_combined_binding_allocation_failure_publishes_neither_identity(
        self,
    ) -> None:
        runner = self._runner()

        def fail_allocation(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("synthetic recurrent slab allocation failure")

        runner._torch.zeros = fail_allocation
        with patch(
            "resin.backend.cuda.mapped_state.mint_state_mapping_arena_id",
            return_value=1234,
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic recurrent slab"):
                runner.bind_cache_and_recurrent_state(
                    41,
                    runner.cache_layout,
                    53,
                    7,
                    "lfm2.conv_state.v1",
                    None,
                )

        self.assertIsNone(runner._kv.binding.pool_id)
        self.assertIsNone(runner._state.binding)
        self.assertIsNone(runner._state.arena_id)
        self.assertIsNone(runner._state.slab)
        self.assertIsNone(runner._state.generations)
        self.assertIsNone(runner._state.owners)

    def test_engine_combined_binding_rejects_unregistered_prebound_native_arena(
        self,
    ) -> None:
        from resin import (
            CacheGroupLayout,
            CacheLayout,
            Engine,
            StateMappingConfig,
            StatePool,
        )

        arena_id = 8765
        pool = StatePool(4)

        class SeedRunner:
            supports_recurrent_state_mapping = True
            recurrent_state_mapping = StateMappingConfig(
                "lfm2.conv_state.v1", ["write"]
            )

            @staticmethod
            def bind_recurrent_state_mapping(pool_id, capacity, protocol):
                return pool_id, capacity, protocol, arena_id

            @staticmethod
            def forward(plan):
                return [0] * plan.num_sequences

        Engine(SeedRunner(), state_pool=pool).close()
        self.assertEqual(pool.physical_mapping, ("lfm2.conv_state.v1", arena_id))

        runner = self._runner()
        runner.cache_layout = CacheLayout(8, [CacheGroupLayout(16)])
        runner._kv.binding.layout = runner.cache_layout
        with patch(
            "resin.backend.cuda.mapped_state.mint_state_mapping_arena_id",
            side_effect=AssertionError("prebound native arena must not be reminted"),
            create=True,
        ) as mint:
            with self.assertRaisesRegex(ValueError, "not registered in this process"):
                Engine(runner, state_pool=pool)

        mint.assert_not_called()
        self.assertIsNone(runner._kv.binding.pool_id)
        self.assertIsNone(runner._state.arena_id)
        self.assertEqual(pool.physical_mapping, ("lfm2.conv_state.v1", arena_id))

    def test_runner_only_tensorizes_frozen_rust_kernel_metadata(self) -> None:
        runner = self._runner()
        self._bind(runner)
        plan = self._plan()

        metadata = runner._metadata(plan)

        expected_cuda = {
            "block_indptr": ([0, 1, 3], "int32"),
            "block_ids": ([5, 3, 7], "int32"),
            "last_block_lengths": ([2, 1], "int32"),
            "append_batch_indices": ([0, 0, 1], "int32"),
            "append_positions": ([0, 1, 16], "int32"),
            "prefill_token_indices": ([0, 1], "long"),
            "prefill_cu_seqlens": ([0, 2], "int32"),
            "continuation_prefill_token_indices": ([2], "long"),
            "continuation_prefill_cu_seqlens": ([0, 1], "int32"),
            "decode_token_indices": ([], "long"),
        }
        expected_host = {
            "decode_block_indptr": ([0], "int32"),
            "decode_block_ids": ([], "int32"),
            "decode_last_block_lengths": ([], "int32"),
        }
        self.assertEqual(metadata["prefill_max_q_len"], 2)
        self.assertEqual(metadata["continuation_prefill_max_q_len"], 1)
        for name, (values, dtype) in expected_cuda.items():
            self.assertEqual(metadata[name].values, tuple(values))
            self.assertEqual(metadata[name].dtype, dtype)
            self.assertEqual(metadata[name].device, "cuda:fake")
            self.assertEqual(metadata["_pinned"][name].device, "cpu")
            self.assertTrue(metadata["_pinned"][name].pin_memory)
        for name, (values, dtype) in expected_host.items():
            self.assertEqual(metadata[name].values, tuple(values))
            self.assertEqual(metadata[name].dtype, dtype)
            self.assertEqual(metadata[name].device, "cpu")
            self.assertTrue(metadata[name].pin_memory)
            self.assertIs(metadata[name], metadata["_pinned"][name])

        self.assertFalse(runner._plan_decode(plan, metadata))
        self.assertEqual(runner._decode.calls, [])

    def test_frozen_append_destinations_are_validated_before_tensor_staging(
        self,
    ) -> None:
        runner = self._runner()
        self._bind(runner)
        plan = self._plan(append_block_ids=[5, 3, 7])

        with self.assertRaisesRegex(ValueError, "append destinations"):
            runner.dispatch(plan)

        self.assertFalse(runner._torch.cuda.events)
        self.assertFalse(any(entry[0] == "tensor" for entry in runner._torch.log))

    def test_decode_metadata_needs_no_python_request_history(self) -> None:
        runner = self._runner()
        self._bind(runner)

        metadata = runner._metadata(
            self._plan(
                block_indptr=[0, 2],
                block_ids=[6, 2],
                first_logical_blocks=[0],
                append_block_ids=[2],
                append_block_offsets=[2],
                positions=[18],
                sequence_ids=[10],
                sequence_indptr=[0, 1],
                context_lengths=[18],
                phase=BatchPhase.DECODE,
                sample_sequence_indices=[0],
                token_ids=[3],
            )
        )
        self.assertEqual(metadata["block_ids"].values, (6, 2))
        self.assertEqual(metadata["append_positions"].values, (18,))

    def test_explicit_continuation_rows_use_only_rust_post_append_pages(self) -> None:
        from resin.models.lfm2.plan import build_cache_metadata

        plan = self._plan(
            block_indptr=[0, 1, 3, 6],
            block_ids=[5, 2, 7, 4, 6, 1],
            first_logical_blocks=[0, 0, 0],
            append_block_ids=[5, 5, 7, 7, 1],
            append_block_offsets=[0, 1, 1, 2, 0],
            positions=[0, 1, 5, 6, 8],
            sequence_ids=[10, 11, 12],
            sequence_indptr=[0, 2, 4, 5],
            context_lengths=[0, 5, 8],
            phase=BatchPhase.PREFILL,
            sample_sequence_indices=[0, 1, 2],
            token_ids=[1, 2, 3, 4, 5],
        )

        metadata = build_cache_metadata(plan, 4).prefill

        self.assertEqual(metadata.dense_token_indices, (0, 1))
        self.assertEqual(metadata.dense_cu_seqlens, (0, 2))
        self.assertEqual(metadata.continuation_token_indices, (2, 3, 4))
        self.assertEqual(metadata.continuation_cu_seqlens, (0, 2, 3))
        # Row 2 has one query token but remains explicit prefill. It therefore
        # shares the continuation page table rather than entering decode.
        self.assertEqual(metadata.continuation_seqused_k, (7, 9))
        self.assertEqual(metadata.continuation_page_table_width, 3)
        self.assertEqual(
            metadata.continuation_page_table,
            (2, 7, 2, 4, 6, 1),
        )
        self.assertEqual(metadata.continuation_min_seqlen_k, 7)
        self.assertEqual(metadata.continuation_max_seqlen_k, 9)

    def test_continuation_kernel_views_are_pinned_nonblocking_tensorizations(
        self,
    ) -> None:
        runner = self._runner()
        runner.page_size = 4
        self._bind(runner)
        plan = self._plan(
            block_indptr=[0, 1, 3, 6],
            block_ids=[5, 2, 7, 4, 6, 1],
            first_logical_blocks=[0, 0, 0],
            append_block_ids=[5, 5, 7, 7, 1],
            append_block_offsets=[0, 1, 1, 2, 0],
            positions=[0, 1, 5, 6, 8],
            sequence_ids=[10, 11, 12],
            sequence_indptr=[0, 2, 4, 5],
            context_lengths=[0, 5, 8],
            phase=BatchPhase.PREFILL,
            sample_sequence_indices=[0, 1, 2],
            token_ids=[1, 2, 3, 4, 5],
        )

        metadata = runner._metadata(plan)

        self.assertEqual(metadata["prefill_token_indices"].values, (0, 1))
        self.assertEqual(
            metadata["continuation_prefill_token_indices"].values,
            (2, 3, 4),
        )
        self.assertEqual(
            metadata["continuation_prefill_cu_seqlens"].values,
            (0, 2, 3),
        )
        self.assertEqual(
            metadata["continuation_prefill_page_table"].values,
            (2, 7, 2, 4, 6, 1),
        )
        self.assertEqual(
            metadata["continuation_prefill_page_table"].shape,
            (2, 3),
        )
        self.assertEqual(
            metadata["continuation_prefill_seqused_k"].values,
            (7, 9),
        )
        for name in (
            "continuation_prefill_token_indices",
            "continuation_prefill_cu_seqlens",
            "continuation_prefill_page_table",
            "continuation_prefill_seqused_k",
        ):
            self.assertTrue(metadata["_pinned"][name].pin_memory)
            self.assertEqual(metadata["_pinned"][name].device, "cpu")

    def test_continuation_rejects_stale_preappend_block_table(self) -> None:
        from resin.models.lfm2.plan import build_cache_metadata

        plan = self._plan(
            block_indptr=[0, 1],
            block_ids=[5],
            first_logical_blocks=[0],
            append_block_ids=[5, 5],
            append_block_offsets=[15, 0],
            sequence_ids=[10],
            sequence_indptr=[0, 2],
            context_lengths=[15],
            phase=BatchPhase.PREFILL,
            sample_sequence_indices=[0],
            token_ids=[1, 2],
            positions=[15, 16],
        )
        with self.assertRaisesRegex(ValueError, "post-append"):
            build_cache_metadata(plan, 16)

    def test_one_token_prefill_is_not_inferred_to_be_decode(self) -> None:
        from resin.models.lfm2.plan import build_cache_metadata

        plan = self._plan(
            block_indptr=[0, 1, 2],
            block_ids=[5, 7],
            first_logical_blocks=[0, 0],
            append_block_ids=[5, 7],
            append_block_offsets=[8, 8],
            sequence_ids=[10, 11],
            sequence_indptr=[0, 1, 2],
            context_lengths=[8, 8],
            phase=BatchPhase.PREFILL,
            sample_sequence_indices=[0, 1],
            token_ids=[1, 2],
            positions=[8, 8],
        )

        metadata = build_cache_metadata(plan, 16).prefill

        self.assertEqual(metadata.continuation_token_indices, (0, 1))
        self.assertEqual(metadata.continuation_cu_seqlens, (0, 1, 2))
        self.assertEqual(metadata.continuation_seqused_k, (9, 9))

    def test_runner_has_no_logical_page_or_length_mirror(self) -> None:
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        source = inspect.getsource(Lfm2CudaRunner)
        self.assertNotIn("_request_pages", source)
        self.assertNotIn("_request_lengths", source)
        self.assertNotIn("_free_pages", source)
        metadata_source = inspect.getsource(Lfm2CudaRunner._metadata)
        self.assertIn("_preflight_cache", metadata_source)
        from resin.models.lfm2.plan import build_cache_metadata

        adapter_source = inspect.getsource(build_cache_metadata)
        self.assertIn("plan.phase", adapter_source)
        self.assertIn("group.append_block_ids", adapter_source)
        self.assertIn("group.append_block_offsets", adapter_source)
        self.assertIn(
            "_invoke_paged_fa4_b22(",
            inspect.getsource(Lfm2CudaRunner._attention),
        )


class Lfm2CudaRecurrentStateTests(unittest.TestCase):
    @staticmethod
    def _bound_runner():
        runner = Lfm2CudaCachePoolBoundaryTests._runner()
        Lfm2CudaCachePoolBoundaryTests._bind(runner)
        return runner

    def test_binding_allocates_exact_bf16_geometry_once_on_private_stream(self) -> None:
        runner = Lfm2CudaCachePoolBoundaryTests._runner()
        first = Lfm2CudaCachePoolBoundaryTests._bind(runner, capacity=7)
        second = Lfm2CudaCachePoolBoundaryTests._bind(runner, capacity=7)

        self.assertEqual(first[:4], (41, 53, 7, "lfm2.conv_state.v1"))
        self.assertGreater(first[-1], 0)
        self.assertEqual(second, first)
        self.assertEqual(runner._state.slab.shape, (4, 7, 8, 2))
        self.assertEqual(runner._state.slab.dtype, "bfloat16")
        self.assertEqual(runner._state.generations, [None] * 7)
        self.assertEqual(runner._state.owners, [None] * 7)
        enter = runner._torch.log.index(("stream.enter", runner._stream))
        allocation = runner._torch.log.index(
            ("zeros", (4, 7, 8, 2), "bfloat16", "cuda:fake")
        )
        exit_index = runner._torch.log.index(("stream.exit", runner._stream))
        self.assertLess(enter, allocation)
        self.assertLess(allocation, exit_index)

        with self.assertRaisesRegex(ValueError, "different mapped-state"):
            runner.bind_cache_and_recurrent_state(
                41,
                runner.cache_layout,
                54,
                7,
                "lfm2.conv_state.v1",
                None,
            )
        with self.assertRaisesRegex(ValueError, "different mapped-state"):
            runner.bind_cache_and_recurrent_state(
                41,
                runner.cache_layout,
                53,
                8,
                "lfm2.conv_state.v1",
                None,
            )

    def test_binding_rejects_invalid_identity_capacity_and_protocol_before_mint(
        self,
    ) -> None:
        cases = (
            ((0, 4, "lfm2.conv_state.v1"), "pool id"),
            ((True, 4, "lfm2.conv_state.v1"), "pool id"),
            ((53, 0, "lfm2.conv_state.v1"), "capacity"),
            ((53, True, "lfm2.conv_state.v1"), "capacity"),
            ((53, 4, "lfm2.conv_state.v2"), "protocol"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                runner = Lfm2CudaCachePoolBoundaryTests._runner()
                with patch(
                    "resin.backend.cuda.mapped_state.mint_state_mapping_arena_id",
                    return_value=1234,
                    create=True,
                ) as mint:
                    with self.assertRaisesRegex(ValueError, message):
                        runner.bind_cache_and_recurrent_state(
                            41,
                            runner.cache_layout,
                            *arguments,
                            None,
                        )
                mint.assert_not_called()
                self.assertIsNone(runner._state.slab)

    def test_preflight_guards_reject_before_start_event_acceptance(self) -> None:
        cases = (
            ({"state_pool_id": 54}, "state pool id"),
            ({"engine_id": 0}, "engine_id"),
            ({"state_indptr": [0, 0, 2]}, "exactly one handle per row"),
            ({"state_slots": [0]}, "exactly one handle per row"),
            ({"state_access_modes": ["write", "read"]}, "write access"),
            ({"state_slots": [0, 4]}, "outside capacity"),
            (
                {"state_slots": [0, 0], "state_generations": [1, 1]},
                "duplicate mapped-state",
            ),
            (
                {"state_slots": [0, 0], "state_generations": [1, 2]},
                "cannot serve two",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                runner = self._bound_runner()
                plan = Lfm2CudaCachePoolBoundaryTests._plan(**overrides)
                with self.assertRaisesRegex(ValueError, message):
                    runner.dispatch(plan)
                self.assertIsNone(runner._executor.inflight)
                self.assertEqual(runner._executor.dispatch_count, 0)
                self.assertFalse(runner._torch.cuda.events)
                self.assertFalse(
                    any(entry[0] == "slab.zero" for entry in runner._torch.log)
                )

    def test_generation_zeroing_reuse_and_exact_engine_request_tombstone(self) -> None:
        runner = self._bound_runner()
        initial = Lfm2CudaCachePoolBoundaryTests._plan()

        rows = runner._state.rows(initial)
        with runner._torch.cuda.stream(runner._stream):
            runner._state.prepare_rows(rows)
        self.assertEqual(
            runner._state.slab.zeroed, [(slice(None), 0), (slice(None), 1)]
        )
        self.assertEqual(runner._state.generations[:2], [1, 1])
        self.assertEqual(runner._state.owners[:2], [(71, 10), (71, 11)])

        same_owner_continuation = Lfm2CudaCachePoolBoundaryTests._plan(
            context_lengths=[2, 17],
            positions=[2, 3, 17],
        )
        same_rows = runner._state.rows(same_owner_continuation)
        with runner._torch.cuda.stream(runner._stream):
            runner._state.prepare_rows(same_rows)
        self.assertEqual(len(runner._state.slab.zeroed), 2)

        next_generation = Lfm2CudaCachePoolBoundaryTests._plan(state_generations=[2, 1])
        with runner._torch.cuda.stream(runner._stream):
            runner._state.prepare_rows(runner._state.rows(next_generation))
        self.assertEqual(len(runner._state.slab.zeroed), 3)
        self.assertEqual(runner._state.generations[0], 2)
        self.assertEqual(runner._state.owners[0], (71, 10))

        stale = Lfm2CudaCachePoolBoundaryTests._plan(state_generations=[1, 1])
        with self.assertRaisesRegex(ValueError, "stale"):
            runner.dispatch(stale)
        self.assertFalse(runner._torch.cuda.events)
        self.assertEqual(len(runner._state.slab.zeroed), 3)

        substitutions = (
            ({"sequence_ids": [99, 11], "state_generations": [2, 1]}, "request"),
            ({"engine_id": 72, "state_generations": [2, 1]}, "Engine"),
        )
        for overrides, message in substitutions:
            with self.subTest(overrides=overrides):
                plan = Lfm2CudaCachePoolBoundaryTests._plan(**overrides)
                with self.assertRaisesRegex(ValueError, message):
                    runner.dispatch(plan)
                self.assertFalse(runner._torch.cuda.events)
                self.assertEqual(len(runner._state.slab.zeroed), 3)

    def test_short_conv_assignment_updates_fixed_layer_slot_view_in_place(self) -> None:
        runner = self._bound_runner()
        rows = (SimpleNamespace(slot=3, generation=1, engine_id=71, request_id=10),)
        plan = SimpleNamespace(sequence_indptr=[0, 2])
        hidden = _FakeTensor(
            [4, 5], dtype="bfloat16", device="cuda:fake", log=runner._torch.log
        )
        replacement = object()
        observations = []

        def layer(value, cache):
            original = cache.value
            cache.value = replacement
            observations.append((original, cache.value))
            return value

        ephemeral = []
        output = runner._convolution(
            2,
            layer,
            hidden,
            plan,
            rows,
            ephemeral,
        )

        self.assertEqual(output.values, (4, 5))
        self.assertIs(observations[0][0], observations[0][1])
        self.assertEqual(
            runner._state.slab.copies,
            [((1, 3), replacement)],
        )
        self.assertIn(observations[0][0], ephemeral)
        self.assertNotIn(replacement, ephemeral)

    def test_source_has_no_request_keyed_convolution_cache(self) -> None:
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        source = inspect.getsource(Lfm2CudaRunner)
        self.assertNotIn("_conv_cache", source)
        self.assertIn("_InPlaceRecurrentState", source)


class Lfm2CudaFa4ContinuationTests(unittest.TestCase):
    def test_exact_b22_paged_surface(self) -> None:
        from resin.models.lfm2.cuda import _invoke_paged_fa4_b22

        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return ("output", "lse")

        values = {
            "cu_seqlens_q": object(),
            "page_table": object(),
            "seqused_k": object(),
        }
        output = _invoke_paged_fa4_b22(
            kernel,
            "q",
            "k_cache",
            "v_cache",
            cu_seqlens_q=values["cu_seqlens_q"],
            max_seqlen_q=7,
            page_table=values["page_table"],
            seqused_k=values["seqused_k"],
            min_seqlen_k=33,
            max_seqlen_k=65,
            softmax_scale=0.125,
        )

        self.assertEqual(output, "output")
        self.assertEqual(calls[0][0], ("q", "k_cache", "v_cache"))
        kwargs = calls[0][1]
        self.assertIs(kwargs["cu_seqlens_q"], values["cu_seqlens_q"])
        self.assertIsNone(kwargs["cu_seqlens_k"])
        self.assertIsNone(kwargs["seqused_q"])
        self.assertIs(kwargs["seqused_k"], values["seqused_k"])
        self.assertIs(kwargs["page_table"], values["page_table"])
        self.assertEqual(kwargs["max_seqlen_q"], 7)
        self.assertEqual(kwargs["min_seqlen_k"], 33)
        self.assertEqual(kwargs["max_seqlen_k"], 65)
        self.assertTrue(kwargs["causal"])
        self.assertEqual(kwargs["window_size"], (None, None))
        self.assertEqual(kwargs["num_splits"], 1)
        self.assertTrue(kwargs["pack_gqa"])
        self.assertFalse(kwargs["deterministic"])
        self.assertFalse(kwargs["return_lse"])

    def test_bottom_right_causal_paged_call_matches_dense_suffix_oracle(self) -> None:
        from resin.models.lfm2.cuda import _invoke_paged_fa4_b22

        query = [[0.5, -0.25], [0.1, 0.8], [-0.4, 0.3]]
        key_cache = [
            [[0.2, 0.1], [0.3, -0.2]],
            [[-0.1, 0.7], [0.6, 0.4]],
            [[0.9, -0.3], [0.2, 0.5]],
            [[-0.5, 0.2], [0.8, -0.4]],
            [[0.4, 0.6], [-0.2, -0.1]],
        ]
        value_cache = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, -0.5], [1.5, 0.5]],
            [[-1.0, 0.2], [0.3, 0.7]],
            [[0.8, -0.1], [-0.4, 1.2]],
            [[0.6, 0.9], [-0.7, 0.4]],
        ]
        cu_seqlens_q = [0, 2, 3]
        page_table = [[2, 0, 3], [1, 4, 1]]
        seqused_k = [5, 3]
        scale = 0.5

        def logical_row(cache, row, length):
            return [token for page in page_table[row] for token in cache[page]][:length]

        def suffix_attention(q_row, k_row, v_row):
            rows = []
            for q_index, q_vector in enumerate(q_row):
                last_key = q_index + len(k_row) - len(q_row)
                scores = [
                    scale * sum(q * k for q, k in zip(q_vector, k_vector))
                    for k_vector in k_row[: last_key + 1]
                ]
                maximum = max(scores)
                unnormalized = [math.exp(score - maximum) for score in scores]
                denominator = sum(unnormalized)
                weights = [weight / denominator for weight in unnormalized]
                rows.append(
                    [
                        sum(
                            weight * value_vector[dimension]
                            for weight, value_vector in zip(weights, v_row)
                        )
                        for dimension in range(len(v_row[0]))
                    ]
                )
            return rows

        expected_rows = []
        for row in range(2):
            q_start, q_end = cu_seqlens_q[row : row + 2]
            length = seqused_k[row]
            expected_rows.append(
                suffix_attention(
                    query[q_start:q_end],
                    logical_row(key_cache, row, length),
                    logical_row(value_cache, row, length),
                )
            )
        expected = [token for row in expected_rows for token in row]

        def paged_oracle(q, k_cache, v_cache, **kwargs):
            rows = []
            cu = kwargs["cu_seqlens_q"]
            lengths = kwargs["seqused_k"]
            table = kwargs["page_table"]
            self.assertTrue(kwargs["causal"])
            for row in range(len(lengths)):
                q_start, q_end = cu[row : row + 2]
                length = lengths[row]
                k_row = [token for page in table[row] for token in k_cache[page]][
                    :length
                ]
                v_row = [token for page in table[row] for token in v_cache[page]][
                    :length
                ]
                rows.extend(suffix_attention(q[q_start:q_end], k_row, v_row))
            return rows

        actual = _invoke_paged_fa4_b22(
            paged_oracle,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=2,
            page_table=page_table,
            seqused_k=seqused_k,
            min_seqlen_k=3,
            max_seqlen_k=5,
            softmax_scale=scale,
        )
        for actual_row, expected_row in zip(actual, expected, strict=True):
            for actual_value, expected_value in zip(
                actual_row, expected_row, strict=True
            ):
                self.assertAlmostEqual(actual_value, expected_value, places=12)


class Lfm2CudaAsyncLifecycleTests(unittest.TestCase):
    @staticmethod
    def _runner_and_plan():
        runner = Lfm2CudaCachePoolBoundaryTests._runner()
        Lfm2CudaCachePoolBoundaryTests._bind(runner)
        runner._decode = _RecordingDecode(runner._torch.cuda)
        plan = Lfm2CudaCachePoolBoundaryTests._plan()

        def run_model(
            self,
            model_plan,
            tokens,
            positions,
            final_indices,
            metadata,
            decode,
            recurrent_rows,
            ephemeral,
        ):
            if self._torch.cuda.active_stream is not self._stream:
                raise AssertionError("model launch escaped the private stream")
            self._torch.log.append(("model", model_plan, decode))
            self._torch.log.append(("recurrent.rows", recurrent_rows))
            return _FakeTensor(
                [17, 23],
                dtype=self._torch.long,
                device=self.device,
                log=self._torch.log,
            )

        runner._run_model = MethodType(run_model, runner)
        return runner, plan

    def test_dispatch_poll_retains_then_retires_exactly_once(self) -> None:
        runner, plan = self._runner_and_plan()

        completion = runner.dispatch(plan)

        self.assertIs(runner._executor.inflight, completion)
        self.assertIs(completion._plan, plan)
        self.assertIsNotNone(completion.staging)
        self.assertTrue(completion.ephemeral)
        for resource in (
            runner._state.slab,
            runner._state.generations,
            runner._state.owners,
        ):
            self.assertTrue(
                any(candidate is resource for candidate in completion._resources)
            )
        self.assertIs(completion._start_event.recorded_stream, runner._stream)
        self.assertIs(completion._done_event.recorded_stream, runner._stream)
        self.assertIsNone(runner.poll(completion))
        self.assertIs(runner._executor.inflight, completion)
        self.assertNotIn(("tolist",), runner._torch.log)

        completion._done_event.ready = True
        self.assertEqual(runner.poll(completion), [17, 23])
        self.assertIsNone(runner._executor.inflight)
        self.assertTrue(completion._retired)
        self.assertIsNone(completion._plan)
        self.assertIsNone(completion._resources)
        self.assertIsNone(completion.staging)
        self.assertIsNone(completion.ephemeral)
        self.assertNotIn(("event.synchronize",), runner._torch.log)
        self.assertNotIn(("stream.synchronize",), runner._torch.log)
        with self.assertRaisesRegex(ValueError, "stale"):
            runner.poll(completion)

        event_syncs = runner._torch.log.count(("event.synchronize",))
        del completion
        gc.collect()
        self.assertEqual(runner._torch.log.count(("event.synchronize",)), event_syncs)

    def test_exact_one_inflight_and_foreign_handle_guards(self) -> None:
        runner, plan = self._runner_and_plan()
        completion = runner.dispatch(plan)
        tensor_calls = sum(entry[0] == "tensor" for entry in runner._torch.log)

        with self.assertRaisesRegex(RuntimeError, "in-flight"):
            runner.dispatch(plan)
        self.assertEqual(
            sum(entry[0] == "tensor" for entry in runner._torch.log), tensor_calls
        )
        with self.assertRaisesRegex(TypeError, "dispatch"):
            runner.poll(object())

        other, _ = self._runner_and_plan()
        with self.assertRaisesRegex(ValueError, "different"):
            other.poll(completion)
        self.assertIs(runner._executor.inflight, completion)

    def test_poll_extraction_failure_preserves_all_lifetimes_for_retry(self) -> None:
        runner, plan = self._runner_and_plan()
        completion = runner.dispatch(plan)
        staging = completion.staging
        ephemeral = completion.ephemeral
        completion._done_event.ready = True
        completion._output_host.tolist_error = RuntimeError("host extraction failed")

        with self.assertRaisesRegex(RuntimeError, "host extraction failed"):
            runner.poll(completion)
        self.assertIs(runner._executor.inflight, completion)
        self.assertIs(completion.staging, staging)
        self.assertIs(completion.ephemeral, ephemeral)
        self.assertFalse(completion._terminal)

        self.assertEqual(runner.poll(completion), [17, 23])
        self.assertIsNone(runner._executor.inflight)

    def test_start_event_failure_is_preacceptance_but_done_failure_is_not(self) -> None:
        runner, plan = self._runner_and_plan()
        runner._torch.cuda.event_record_errors = [RuntimeError("start rejected")]

        with self.assertRaisesRegex(RuntimeError, "start rejected"):
            runner.dispatch(plan)
        self.assertIsNone(runner._executor.inflight)
        self.assertEqual(runner._executor.dispatch_count, 0)

        runner._torch.cuda.event_record_errors = [
            None,
            RuntimeError("done record failed after acceptance"),
        ]
        completion = runner.dispatch(plan)
        self.assertFalse(completion._done_recorded)
        with self.assertRaisesRegex(RuntimeError, "after acceptance"):
            runner.poll(completion)
        self.assertIs(runner._executor.inflight, completion)
        self.assertIsNotNone(completion.staging)

    def test_post_acceptance_launch_error_is_deferred_to_poll_and_quarantined(
        self,
    ) -> None:
        runner, plan = self._runner_and_plan()

        def fail_after_acceptance(*args, **kwargs):
            raise RuntimeError("kernel launch failed after acceptance")

        runner._run_model = fail_after_acceptance
        completion = runner.dispatch(plan)
        staging = completion.staging
        ephemeral = completion.ephemeral
        self.assertIs(runner._executor.inflight, completion)
        self.assertTrue(completion._done_recorded)
        self.assertIsNone(runner.poll(completion))

        completion._done_event.ready = True
        with self.assertRaisesRegex(RuntimeError, "after acceptance"):
            runner.poll(completion)
        self.assertIs(runner._executor.inflight, completion)
        self.assertIs(completion.staging, staging)
        self.assertIs(completion.ephemeral, ephemeral)
        self.assertFalse(completion._terminal)

    def test_all_list_staging_and_copies_are_pinned_and_nonblocking(self) -> None:
        runner, plan = self._runner_and_plan()

        runner.dispatch(plan)

        tensor_entries = [entry for entry in runner._torch.log if entry[0] == "tensor"]
        self.assertTrue(tensor_entries)
        self.assertTrue(all(entry[3:] == ("cpu", True) for entry in tensor_entries))
        h2d = [entry for entry in runner._torch.log if entry[0] == "h2d"]
        self.assertEqual(len(h2d), 15)
        self.assertTrue(all(entry[2] is True for entry in h2d))
        self.assertIn(("d2h", "cuda:fake", True), runner._torch.log)
        self.assertEqual(runner._decode.calls, [])
        model_index = next(
            index
            for index, entry in enumerate(runner._torch.log)
            if entry[0] == "model"
        )
        enter = max(
            index
            for index, entry in enumerate(runner._torch.log[:model_index])
            if entry == ("stream.enter", runner._stream)
        )
        exit = next(
            index
            for index, entry in enumerate(runner._torch.log[model_index:], model_index)
            if entry == ("stream.exit", runner._stream)
        )
        zero_indices = [
            index
            for index, entry in enumerate(runner._torch.log)
            if entry[0] == "slab.zero"
        ]
        self.assertTrue(zero_indices)
        self.assertTrue(all(enter < index < model_index for index in zero_indices))
        for index, entry in enumerate(runner._torch.log):
            if entry[0] in {"h2d", "d2h", "decode.plan", "model", "event.record"}:
                self.assertLess(enter, index)
                self.assertLess(index, exit)

    def test_preflight_validation_does_not_consume_inflight_slot(self) -> None:
        runner, plan = self._runner_and_plan()
        plan.cache_pool_id = 99

        with self.assertRaisesRegex(ValueError, "does not match"):
            runner.dispatch(plan)
        self.assertIsNone(runner._executor.inflight)
        self.assertEqual(runner._executor.dispatch_count, 0)

    def test_completion_cannot_be_retired_while_pending(self) -> None:
        runner, plan = self._runner_and_plan()
        completion = runner.dispatch(plan)

        with self.assertRaisesRegex(RuntimeError, "pending"):
            completion._retire()
        self.assertIs(runner._executor.inflight, completion)

    def test_abandoned_ready_completion_outlives_runner_and_queries_before_release(
        self,
    ) -> None:
        runner, plan = self._runner_and_plan()
        tracked = {
            "flashinfer": weakref.ref(runner._flashinfer),
            "fa4": weakref.ref(runner._fa4),
            "model": weakref.ref(runner._model),
            "k_cache": weakref.ref(runner._kv.keys),
            "v_cache": weakref.ref(runner._kv.values),
            "decode_workspace": weakref.ref(runner._decode_workspace),
            "decode": weakref.ref(runner._decode),
            "state_slab": weakref.ref(runner._state.slab),
        }
        log = runner._torch.log
        runner_ref = weakref.ref(runner)

        completion = runner.dispatch(plan)
        completion._done_event.ready = True
        captured = completion._resources
        self.assertTrue(all(reference() in captured for reference in tracked.values()))
        self.assertIsNotNone(tracked["state_slab"]())

        del captured
        del runner
        gc.collect()
        self.assertIsNone(runner_ref())
        self.assertTrue(all(reference() is not None for reference in tracked.values()))
        self.assertNotIn(("event.query", True), log)

        del completion
        gc.collect()
        query_index = log.index(("event.query", True))
        deletion_indices = [
            index for index, entry in enumerate(log) if entry[0] == "resource.del"
        ]
        self.assertTrue(deletion_indices)
        self.assertTrue(all(query_index < index for index in deletion_indices))
        self.assertTrue(
            all(
                reference() is None
                for name, reference in tracked.items()
                if name != "state_slab"
            )
        )
        # The native arena registry intentionally outlives runner/completion
        # objects so its physical identity and owner tombstones remain reusable.
        self.assertIsNotNone(tracked["state_slab"]())
        self.assertNotIn(("event.synchronize",), log)

    def test_abandonment_queries_ready_stream_when_terminal_event_was_not_recorded(
        self,
    ) -> None:
        runner, plan = self._runner_and_plan()
        runner._torch.cuda.event_record_errors = [
            None,
            RuntimeError("done event rejected"),
        ]
        log = runner._torch.log
        runner._stream.ready = True

        completion = runner.dispatch(plan)
        self.assertFalse(completion._done_recorded)
        del runner
        gc.collect()
        del completion
        gc.collect()

        self.assertIn(("stream.query", True), log)
        self.assertNotIn(("event.synchronize",), log)

    def test_abandonment_timeout_quarantines_every_lifetime(self) -> None:
        from resin.backend.cuda.execution import CUDA_COMPLETION_QUARANTINE

        baseline = len(CUDA_COMPLETION_QUARANTINE)
        runner, plan = self._runner_and_plan()
        resource = runner._model
        resource_ref = weakref.ref(resource)
        del resource
        log = runner._torch.log

        completion = runner.dispatch(plan)
        done_event = completion._done_event
        del runner
        gc.collect()
        del completion
        gc.collect()

        self.assertGreaterEqual(log.count(("event.query", False)), 2)
        self.assertNotIn(("event.synchronize",), log)
        self.assertEqual(len(CUDA_COMPLETION_QUARANTINE), baseline + 1)
        self.assertIsNotNone(resource_ref())
        self.assertNotIn(("resource.del", "model"), log)
        payload = CUDA_COMPLETION_QUARANTINE[-1]
        self.assertIs(payload["_done_event"], done_event)
        self.assertIn(resource_ref(), payload["_resources"])
        self.assertIsInstance(payload["wait_error"], TimeoutError)

        # Test-only cleanup after the fake event can prove completion.
        done_event.ready = True
        self.assertTrue(done_event.query())
        del CUDA_COMPLETION_QUARANTINE[baseline:]
        del payload
        del done_event
        gc.collect()
        self.assertIsNone(resource_ref())

    def test_abandonment_query_error_quarantines_without_retrying(self) -> None:
        from resin.backend.cuda.execution import CUDA_COMPLETION_QUARANTINE

        baseline = len(CUDA_COMPLETION_QUARANTINE)
        runner, plan = self._runner_and_plan()
        resource_ref = weakref.ref(runner._kv.keys)
        log = runner._torch.log

        completion = runner.dispatch(plan)
        done_event = completion._done_event
        done_event.query_error = RuntimeError("CUDA context lost")
        del runner
        gc.collect()
        del completion
        gc.collect()

        self.assertEqual(log.count(("event.query", False)), 1)
        self.assertNotIn(("event.synchronize",), log)
        self.assertEqual(len(CUDA_COMPLETION_QUARANTINE), baseline + 1)
        payload = CUDA_COMPLETION_QUARANTINE[-1]
        self.assertIsInstance(payload["wait_error"], RuntimeError)
        self.assertIsNotNone(resource_ref())

        done_event.ready = True
        self.assertTrue(done_event.query())
        del CUDA_COMPLETION_QUARANTINE[baseline:]
        del payload
        del done_event
        gc.collect()
        self.assertIsNone(resource_ref())


class Lfm2CudaAsyncStaticBoundaryTests(unittest.TestCase):
    def test_abandonment_is_bounded_and_query_only(self) -> None:
        from resin.backend.cuda.execution import CudaCompletion

        source = inspect.getsource(CudaCompletion._abandon)
        self.assertIn("fence.query()", source)
        self.assertIn("deadline", source)
        self.assertIn("_quarantine_lifetimes", source)
        self.assertNotIn(".synchronize(", source)

    def test_poll_queries_without_synchronizing(self) -> None:
        from resin.backend.cuda.execution import CudaExecutor

        source = inspect.getsource(CudaExecutor.poll)
        self.assertIn(".query()", source)
        self.assertNotIn("synchronize", source)

    def test_dispatch_uses_private_stream_and_nonblocking_copies(self) -> None:
        from resin.backend.cuda.execution import CudaExecutor
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        source = inspect.getsource(CudaExecutor.dispatch)
        self.assertIn("torch.cuda.stream(self.stream)", source)
        self.assertIn("completion._done_event.record(self.stream)", source)
        self.assertIn(
            "self._executor.dispatch", inspect.getsource(Lfm2CudaRunner.dispatch)
        )
        self.assertNotIn("torch.cuda.synchronize", inspect.getsource(Lfm2CudaRunner))
        self.assertNotIn("current_stream", inspect.getsource(Lfm2CudaRunner))

    def test_metadata_stages_decode_plan_on_pinned_cpu(self) -> None:
        from resin.backend.cuda.execution import CudaExecutor
        from resin.backend.cuda.staging import TensorStager
        from resin.models.lfm2.cuda import Lfm2CudaRunner

        metadata_source = inspect.getsource(Lfm2CudaRunner._metadata)
        staging_source = inspect.getsource(TensorStager)
        poll_source = inspect.getsource(CudaExecutor.poll)
        dispatch_source = inspect.getsource(Lfm2CudaRunner.dispatch)
        self.assertIn("TensorStager", metadata_source)
        self.assertIn('device="cpu"', staging_source)
        self.assertIn("pin_memory=True", staging_source)
        self.assertIn("non_blocking=True", staging_source)
        self.assertNotIn(".tolist()", dispatch_source)
        self.assertIn(".tolist()", poll_source)


if __name__ == "__main__":
    unittest.main()
