from __future__ import annotations

import inspect

import pytest

from resin import BatchPhase, BatchPlan
from resin_models_cuda.qwen3_5 import DEFAULT_MODEL, Qwen3_5CudaRunner


def test_qwen38_is_runner_default_and_public_name_is_consistent() -> None:
    parameters = inspect.signature(Qwen3_5CudaRunner.__init__).parameters

    assert parameters["model"].default == DEFAULT_MODEL == "Qwen/Qwen3.8-27B"
    assert parameters["speculative_tokens"].default == 0
    assert parameters["kv_cache_dtype"].default == "fp8"
    assert Qwen3_5CudaRunner.__name__ == "Qwen3_5CudaRunner"


def test_qwen_runner_uses_optional_bundled_mtp_and_engine_cache() -> None:
    source = inspect.getsource(Qwen3_5CudaRunner)

    assert "include_mtp=speculative_tokens > 0" in source
    assert "CacheGeometry.from_config" in source
    assert "cache_write_lookahead_tokens=lookahead" in source
    assert "self._model.mtp" in source
    assert 'torch.device("cuda", torch.cuda.current_device())' in source


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
