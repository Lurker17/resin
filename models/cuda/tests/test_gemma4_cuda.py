from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import BatchPhase
from resin.backend.cuda.failures import CudaFailureGuard, PostAcceptCudaError
from resin.backend.cuda.graphs import default_decode_graph_shapes
from resin.backend.cuda.runner import CudaRunner
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin.models.gemma4.plan import PAGE_SIZE, SLIDING_WINDOW
from resin_models_cuda.gemma4 import cuda as gemma4_runtime
from resin_models_cuda.gemma4.cuda import (
    Gemma4CudaRunner,
    _Gemma4KvArena,
    _select_nvfp4_backend,
    plan_gemma4_kv_capacity,
)


def test_capacity_planner_accounts_for_fp8_shared_parent_bytes() -> None:
    total = 96 * 1024**3
    free = 63 * 1024**3
    plan = plan_gemma4_kv_capacity(total, free, 0.8)

    assert plan.bytes_per_parent == 25 * 1024**2
    assert plan.protected_free_bytes == math.ceil(total * 0.2)
    assert (
        plan.parent_capacity
        == (free - plan.protected_free_bytes - plan.overhead_bytes)
        // plan.bytes_per_parent
        - 1
    )
    assert plan.overhead_bytes == 7 * 256 * 1024**2
    assert plan.allocation_bytes == (plan.parent_capacity + 1) * plan.bytes_per_parent


def test_fp4_backend_falls_back_when_cudnn_is_too_old() -> None:
    assert _select_nvfp4_backend("auto", 92000) == "cute-dsl"
    assert _select_nvfp4_backend("auto", 92301) == "cudnn"


def test_local_and_global_cache_views_cover_raw_parent_child_rows() -> None:
    config = SimpleNamespace(
        local_layer_count=2,
        global_layer_count=1,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        history_blocks_per_parent=2,
    )
    arena = _Gemma4KvArena(
        torch.device("cpu"),
        config,
        parent_capacity=3,
        dtype=torch.float8_e4m3fn,
    )

    expected = 4 * 2 * 2 * PAGE_SIZE * 2 * 4
    assert arena.storage.numel() == expected
    assert arena.local_keys.untyped_storage().data_ptr() == arena.storage.data_ptr()
    assert arena.global_keys.untyped_storage().data_ptr() == arena.storage.data_ptr()
    assert arena.local_keys.shape == (2, 4, PAGE_SIZE, 2, 4)
    assert arena.global_keys.shape == (1, 8, PAGE_SIZE, 1, 8)

    # Packing is two in this fixture. Raw global page 2 is physical parent 1,
    # child 0; raw page 4 is physical parent 2, child 0. Parent zero retains
    # both null-page children, so there is no compact-ID shift to remember.
    local_parent_bytes = arena.storage[0].numel() * arena.storage.element_size()
    assert arena.local_keys[:, 1].data_ptr() - arena.storage.data_ptr() < (
        2 * local_parent_bytes
    )
    assert arena.global_keys[:, 2].data_ptr() - arena.storage.data_ptr() == (
        local_parent_bytes
    )
    assert arena.global_keys[:, 4].data_ptr() - arena.storage.data_ptr() == (
        2 * local_parent_bytes
    )


def test_bfloat16_cache_capacity_accounts_for_two_byte_elements() -> None:
    fp8 = plan_gemma4_kv_capacity(96 * 1024**3, 63 * 1024**3, 0.8)
    bf16 = plan_gemma4_kv_capacity(
        96 * 1024**3,
        63 * 1024**3,
        0.8,
        cache_element_bytes=2,
    )

    assert bf16.bytes_per_parent == 2 * fp8.bytes_per_parent
    assert bf16.parent_capacity < fp8.parent_capacity


def test_cuda_graphs_use_shared_static_step_lifecycle() -> None:
    parameters = inspect.signature(Gemma4CudaRunner.__init__).parameters
    source = inspect.getsource(Gemma4CudaRunner)
    assert "cuda_graph_decode_shapes" in parameters
    assert "speculative_tokens" in parameters
    assert issubclass(Gemma4CudaRunner, CudaRunner)
    assert "self._configure_cuda_graphs(" in source
    assert "def capture_cuda_graphs" not in source
    assert "CUDAGraph" not in source
    assert ".replay()" not in source


@pytest.mark.parametrize("capacity,expected_width", [(2, 8), (32, 11)])
@pytest.mark.parametrize("speculative_tokens", [0, 3])
def test_default_graph_history_uses_packed_cache_and_context_bounds(
    monkeypatch, capacity, expected_width, speculative_tokens
) -> None:
    class ConfiguredGraphs(Exception):
        pass

    config = SimpleNamespace(
        max_position_embeddings=641,
        history_blocks_per_parent=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        hidden_size=8,
        vocab_size=16,
    )
    model = SimpleNamespace(prepare_attention=lambda: None)
    assistant_config = SimpleNamespace(
        backbone_hidden_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
    )
    received = []

    def initialize(runner, **options):
        runner.device = torch.device("cpu")
        runner.cuda_graph_decode = options["cuda_graph_decode"]
        runner.kv_cache_torch_dtype = torch.float8_e4m3fn

    def configure(runner, modes):
        received.append(modes["decode"].shapes)
        raise ConfiguredGraphs

    monkeypatch.setattr(CudaRunner, "__init__", initialize)
    monkeypatch.setattr(Gemma4CudaRunner, "_configure_cuda_graphs", configure)
    monkeypatch.setattr(gemma4_runtime, "resolve_model_path", lambda *a: "unused")
    monkeypatch.setattr(gemma4_runtime, "resolve_assistant_path", lambda *a: "unused")
    monkeypatch.setattr(
        gemma4_runtime, "load_model", lambda *a, **k: (model, config, {})
    )
    monkeypatch.setattr(
        gemma4_runtime, "load_assistant", lambda *a, **k: (object(), assistant_config)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100 * 1024**3,) * 2)
    with pytest.raises(ConfiguredGraphs):
        Gemma4CudaRunner(
            device="cuda",
            cache_parent_capacity=capacity,
            speculative_tokens=speculative_tokens,
        )
    assert received == [
        default_decode_graph_shapes(
            expected_width, speculative=bool(speculative_tokens), max_batch_size=128
        )
    ]


def test_ordinary_graph_capture_rows_stay_inside_null_page() -> None:
    runner = object.__new__(Gemma4CudaRunner)
    runner.speculative_tokens = 0

    inputs = runner._graph_capture_inputs((128, 4096), vocab_size=256)

    assert inputs["local_append_offsets"] == tuple(
        row % PAGE_SIZE for row in range(128)
    )
    assert inputs["global_append_offsets"] == inputs["local_append_offsets"]


def test_ordinary_graph_padding_rows_stay_inside_null_page() -> None:
    plan = SimpleNamespace(
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        page_indices=torch.tensor([7], dtype=torch.int32).numpy(),
        last_page_lengths=torch.tensor([1], dtype=torch.int32).numpy(),
        append_slots=torch.tensor([7 * PAGE_SIZE], dtype=torch.int64).numpy(),
    )

    inputs = Gemma4CudaRunner._graph_page_inputs(
        # Minimal page-table double covers only fields used by this adapter.
        "local",
        plan,  # ty: ignore[invalid-argument-type]
        active_rows=1,
        graph_rows=128,
        width=81,
    )

    offsets = inputs["local_append_offsets"]
    assert isinstance(offsets, np.ndarray)
    assert max(offsets) < PAGE_SIZE


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_prefill_regions_activate_after_media_and_attention_planning(phase) -> None:
    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cpu")
    runner.cuda_graph_prefill = "breakable"
    runner._prefill_graph_active = False
    setattr(
        runner,
        "_model",
        SimpleNamespace(embed_tokens=lambda ids: ids[:, None].bfloat16()),
    )
    events = []

    def record(name, result):
        def call(*_args):
            events.append((name, runner._prefill_graph_active))
            return result

        return call

    runner._merge_media_embeddings = record("media", {})
    runner._local_custom_mask = record("mask", None)
    runner._device_plan = record("cache", object())
    runner._plan_attention = record("attention", (object(), object()))
    expected = torch.ones(3, 1, dtype=torch.bfloat16)
    runner._run_transformer = record("transformer", expected)
    batch = SimpleNamespace(
        token_ids=(1, 2, 3),
        positions=(130, 131, 132),
        local=object(),
        global_=object(),
        phase=phase,
    )
    # This fixture isolates activation order from native plan construction.
    assert runner._eager_transformer(object(), batch) is expected  # ty: ignore[invalid-argument-type]
    assert events[:-1] == [
        (name, False) for name in ("media", "mask", "cache", "cache", "attention")
    ]
    assert events[-1] == ("transformer", phase == BatchPhase.PREFILL)
    assert not runner._prefill_graph_active


def test_prefill_feed_forward_can_reuse_output_storage_across_layers() -> None:
    runner = object.__new__(Gemma4CudaRunner)
    rows, width = 3, 4
    layers = [
        SimpleNamespace(
            input_layernorm=lambda hidden: hidden * 0.5,
            self_attn=SimpleNamespace(
                layer_type=SLIDING_ATTENTION,
                project_qkv=lambda hidden, *_: (hidden[:, None],) * 3,
                o_proj=lambda hidden: hidden * 0.25,
            ),
            post_attention_layernorm=lambda hidden: hidden + 0.125,
            pre_feedforward_layernorm=lambda hidden: hidden * 0.5,
            mlp=lambda hidden: hidden.square() + 0.25,
            post_feedforward_layernorm=lambda hidden: hidden * 0.25,
            layer_scalar=scale,
        )
        for scale in (0.5, 0.75)
    ]
    setattr(runner, "config", SimpleNamespace(layer_types=(SLIDING_ATTENTION,) * 2))
    setattr(
        runner,
        "_model",
        SimpleNamespace(
            language_model=SimpleNamespace(
                layers=layers,
                rotary_emb=lambda *_: (None, None),
                norm=lambda hidden: hidden * 0.5,
            )
        ),
    )
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            dtype=torch.bfloat16,
            local_keys=torch.zeros(2, 1, PAGE_SIZE, 1, width, dtype=torch.bfloat16),
            local_values=torch.zeros(2, 1, PAGE_SIZE, 1, width, dtype=torch.bfloat16),
        ),
    )
    plan = SimpleNamespace(
        append_pages=torch.zeros(rows, dtype=torch.int64),
        append_offsets=torch.arange(rows),
    )
    wrapper = SimpleNamespace(run=lambda query, _: query)
    hidden = torch.arange(rows * width, dtype=torch.bfloat16).reshape(rows, width) / 16

    def run():
        return runner._run_transformer(
            # Minimal device-plan doubles supply only append destinations.
            hidden,
            torch.arange(rows),
            plan,  # ty: ignore[invalid-argument-type]
            plan,  # ty: ignore[invalid-argument-type]
            wrapper,
            wrapper,
        )

    operations = tuple(
        lambda hidden, layer=layer: runner._feed_forward(layer, hidden)
        for layer in layers
    )
    runner._prefill_feed_forward = operations
    expected = run()
    shared = torch.empty_like(hidden)

    def replay(operation, hidden):
        shared.copy_(operation(hidden))
        return shared

    runner._prefill_feed_forward = tuple(
        lambda hidden, operation=operation: replay(operation, hidden)
        for operation in operations
    )
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("speculative_tokens", [0, 3])
def test_graph_attention_contract_matches_capture_and_replay(
    monkeypatch, speculative_tokens
) -> None:
    import flashinfer

    wrappers = []

    class Wrapper:
        def __init__(self, _workspace, _layout, **options):
            self.options = options
            self.calls = []
            wrappers.append(self)

        def plan(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    for name in (
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
    ):
        monkeypatch.setattr(flashinfer, name, Wrapper, raising=False)
    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cpu")
    setattr(
        runner,
        "config",
        SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=4,
            global_head_dim=8,
            hidden_size=8,
            vocab_size=16,
        ),
    )
    runner.speculative_tokens = speculative_tokens
    runner.attention_config = runner.config
    setattr(runner, "_assistant", object())
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            dtype=torch.float8_e4m3fn,
            local_keys=[None],
            local_values=[None],
            global_keys=[None],
            global_values=[None],
        ),
    )
    setattr(
        runner,
        "_sampler",
        SimpleNamespace(empty=lambda *_: None, graph_inputs=lambda _: {}),
    )
    setattr(runner, "_graph_attention_workspaces", (None, None))
    runner._graph_attention = {}
    runner._graph_spec_attention = {}
    runner._graph_verify_hidden = {}
    shape = (2, 3)
    step = runner._new_decode_graph_step(shape)
    inputs = runner._graph_capture_inputs(shape, vocab_size=16)
    for name, values in inputs.items():
        if name.endswith("_page_indices"):
            assert isinstance(values, tuple)
            inputs[name] = (17 if "local" in name else 23,) * len(values)
    runner._prepare_graph_attention(shape, inputs)

    assert step.inputs.keys() <= inputs.keys()
    assert len(wrappers) == (4 if speculative_tokens else 2)
    for wrapper in wrappers:
        (initial, initial_options), (current, current_options) = wrapper.calls
        assert initial_options == current_options
        assert initial[-4:] == current[-4:]
        local = current[-2] == 4
        assert current[-4:] == (4, 2 if local else 1, 4 if local else 8, PAGE_SIZE)
        assert current_options["window_left"] == (SLIDING_WINDOW - 1 if local else -1)
        assert current_options["disable_split_kv"] is True
        verify = "qo_indptr_buf" in wrapper.options
        assert current_options.get("causal", False) is verify
        assert current[-6].tolist() == [17 if local else 23] * shape[0]
        if verify:
            assert current[0].tolist() == [0, 4, 8]
            assert current[-5].tolist() == [5, 5]
        assert wrapper.options["backend"] == ("auto" if speculative_tokens else "fa2")
        assert wrapper.options.get("use_tensor_cores", False) is (
            not speculative_tokens
        )


@pytest.mark.parametrize("graph", [False, True])
def test_speculative_failure_after_state_update_is_not_retryable(
    monkeypatch, graph
) -> None:
    state = [0]
    synchronized = []
    runner = object.__new__(Gemma4CudaRunner)
    runner.speculative_tokens = 1
    setattr(runner, "cache_layout", object())
    setattr(runner, "_check_cuda_graphs", lambda _phase: None)
    runner._cuda_failure = CudaFailureGuard(
        lambda: synchronized.append(True), label="Gemma speculative decode"
    )
    native = SimpleNamespace(phase=BatchPhase.DECODE)
    monkeypatch.setattr(
        gemma4_runtime, "build_gemma4_batch_plan", lambda *_args, **_kwargs: native
    )

    def fail_after_update(_plan, _batch):
        state[0] += 1
        raise ValueError("sampling failed after state update")

    setattr(
        runner,
        "_speculative_graph_forward",
        fail_after_update if graph else lambda _plan, _batch: None,
    )
    setattr(runner, "_speculative_decode", fail_after_update)

    with pytest.raises(PostAcceptCudaError, match="sampling failed"):
        # The plan double tests failure tagging after a simulated cache write.
        runner.forward(native)  # ty: ignore[invalid-argument-type]

    assert state == [1]
    assert synchronized == [True]
