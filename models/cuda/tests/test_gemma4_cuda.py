from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

from resin_models_cuda.gemma4.cuda import (
    Gemma4CudaRunner,
    _Gemma4KvArena,
    _select_nvfp4_backend,
    plan_gemma4_kv_capacity,
)
from resin_models_cuda.gemma4.plan import PAGE_SIZE


def test_capacity_planner_accounts_for_fp8_shared_parent_bytes() -> None:
    total = 96 * 1024**3
    free = 63 * 1024**3
    plan = plan_gemma4_kv_capacity(total, free, 0.8)

    assert plan.bytes_per_parent == 25 * 1024**2
    assert plan.protected_free_bytes == math.ceil(total * 0.2)
    assert (
        plan.parent_capacity
        == (free - plan.protected_free_bytes - 256 * 1024**2) // plan.bytes_per_parent
        - 1
    )
    assert plan.allocation_bytes == (plan.parent_capacity + 1) * plan.bytes_per_parent


def test_fp4_backend_falls_back_when_cudnn_is_too_old() -> None:
    assert _select_nvfp4_backend("auto", 92000) == "cute-dsl"
    assert _select_nvfp4_backend("auto", 92301) == "cudnn"


def test_local_and_global_cache_views_alias_one_physical_slab() -> None:
    config = SimpleNamespace(
        local_layer_count=2,
        global_layer_count=1,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        history_blocks_per_parent=2,
    )
    arena = _Gemma4KvArena(torch, torch.device("cpu"), config, parent_capacity=3)

    expected = 4 * 2 * 2 * PAGE_SIZE * 2 * 4
    assert arena.storage.numel() == expected
    assert arena.local_keys.untyped_storage().data_ptr() == (
        arena.global_keys.untyped_storage().data_ptr()
    )
    assert arena.local_keys.shape == (2, 4, PAGE_SIZE, 2, 4)
    assert arena.global_keys.shape == (1, 8, PAGE_SIZE, 1, 8)


def test_cuda_graphs_use_shared_static_step_lifecycle() -> None:
    parameters = inspect.signature(Gemma4CudaRunner.__init__).parameters
    source = inspect.getsource(Gemma4CudaRunner)
    assert "cuda_graph_decode_shapes" in parameters
    assert "CudaGraphCache(" in source
    assert "CUDAGraph" not in source
    assert ".replay()" not in source
