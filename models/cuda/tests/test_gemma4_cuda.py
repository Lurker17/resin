from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

from resin.models.gemma4.plan import PAGE_SIZE
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
        == (free - plan.protected_free_bytes - plan.workspace_bytes)
        // plan.bytes_per_parent
        - 1
    )
    assert plan.workspace_bytes == 7 * 256 * 1024**2
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
    assert "CudaGraphRunner(" in source
    assert "CUDAGraph" not in source
    assert ".replay()" not in source


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
        "local", plan, active_rows=1, graph_rows=128, width=81
    )

    assert max(inputs["local_append_offsets"]) < PAGE_SIZE


def test_ordinary_graph_decode_uses_graph_stable_attention_plan() -> None:
    source = inspect.getsource(Gemma4CudaRunner._new_decode_graph_step)

    assert "use_tensor_cores=True" in source
    assert 'backend="fa2"' in source
    assert "disable_split_kv=True" in source
