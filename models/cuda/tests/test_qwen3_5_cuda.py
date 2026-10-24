from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from test_qwen3_5_media import _media

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.runner import CudaRunner
from resin_models_cuda.qwen3_5.checkpoint import DEFAULT_MODEL
from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner


def test_graph_rope_deltas_stay_on_host_and_match_eager_positions() -> None:
    runner = object.__new__(Qwen3_5CudaRunner)
    # No CUDA operations are needed to prepare the graph's host metadata.
    runner.device = torch.device("cpu")
    plan = BatchPlan(
        [1, 2],
        [6, 9],
        [1, 2],
        [0, 1, 2],
        phase=BatchPhase.DECODE,
        media=[_media(), None],
    )
    deltas = runner._host_rope_deltas(plan)
    assert isinstance(deltas, np.ndarray)
    np.testing.assert_array_equal(deltas, [-2, 0])
    adjusted = runner._decode_rotary_positions(plan, torch.tensor([6, 9]))
    torch.testing.assert_close(adjusted, torch.tensor([[[4], [9]]] * 3))


def test_qwen38_is_runner_default_and_public_name_is_consistent() -> None:
    parameters = inspect.signature(Qwen3_5CudaRunner.__init__).parameters

    assert parameters["model"].default == DEFAULT_MODEL == "RadixArk/Qwen3.8-27B-NVFP4"
    assert parameters["speculative_tokens"].default == 0
    assert parameters["kv_cache_dtype"].default == "fp8"
    assert parameters["nvfp4_backend"].default == "auto"
    assert parameters["cuda_graph_decode_shapes"].default is None
    assert Qwen3_5CudaRunner.__name__ == "Qwen3_5CudaRunner"


def test_qwen_runner_uses_optional_bundled_mtp_and_engine_cache() -> None:
    source = inspect.getsource(Qwen3_5CudaRunner)

    assert "include_mtp=speculative_tokens > 0" in source
    assert "CacheGeometry.from_config" in source
    assert "cache_write_lookahead_tokens=lookahead" in source
    assert "self._model.mtp" in source
    assert 'torch.device("cuda", torch.cuda.current_device())' in inspect.getsource(
        CudaRunner
    )


def test_qwen_runner_captures_both_ordinary_and_mtp_decode() -> None:
    source = inspect.getsource(Qwen3_5CudaRunner)

    assert issubclass(Qwen3_5CudaRunner, CudaRunner)
    assert "self._configure_cuda_graphs(" in source
    assert "GraphPagedAttentionRuntime" in source
    assert "def capture_cuda_graphs" not in source
    assert "def _new_decode_graph_step" in source
    assert "self._mtp_forward(" in source
    assert "self._run_cuda_graph(" in source


def test_speculative_decode_capacity_includes_mtp_verification_lookahead() -> None:
    runner = object.__new__(Qwen3_5CudaRunner)
    runner.max_batch_size = 4
    runner.max_seq_len = 32
    runner.speculative_tokens = 3
    valid = BatchPlan(
        [1],
        [26],
        [7],
        [0, 1],
        phase=BatchPhase.DECODE,
    )
    runner._validate_plan(valid)

    too_long = BatchPlan(
        [1],
        [28],
        [7],
        [0, 1],
        phase=BatchPhase.DECODE,
    )
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        runner._validate_plan(too_long)
