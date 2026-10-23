"""Candidate allocation uses runtime context widths with a fixed reduction tile."""

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy
import torch

ROOT = Path(__file__).parents[1] / "resin_models_cuda/deepseek_v41/upstream"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate_kernel(monkeypatch):
    import sys

    # Evaluate the real factory's shape equations without a CUDA compiler. The
    # prim_func body is deliberately not executed; the GPU probe checks it.
    language = SimpleNamespace(
        dynamic=lambda name: sympy.Symbol(name, integer=True, positive=True),
        ceildiv=lambda value, divisor: sympy.ceiling(value / divisor),
        prim_func=lambda function: function,
        StridedTensor=lambda shape, strides, dtype: SimpleNamespace(
            shape=shape, strides=strides, dtype=dtype
        ),
        Tensor=lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype),
        bfloat16="bfloat16",
        int32="int32",
    )
    tilelang = SimpleNamespace(
        language=language,
        jit=lambda **_kwargs: lru_cache(None),
        PassConfigKey=SimpleNamespace(
            TL_DISABLE_DATA_RACE_CHECK=1,
            TL_ENABLE_LOWER_LDGSTG=2,
            TL_DISABLE_TMA_LOWER=3,
            TL_DISABLE_WARP_SPECIALIZED=4,
        ),
    )
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(sys.modules, "tilelang.language", language)
    monkeypatch.setitem(
        sys.modules, "deep_gemm", SimpleNamespace(get_num_sms=lambda: 148)
    )
    return _load(ROOT / "cuda/candidate_blocks.py", "candidate_kernel_contract")


@pytest.mark.parametrize("block_size", [8, 16])
def test_candidate_factory_keeps_width_and_task_capacity_dynamic(
    candidate_kernel, block_size
):
    kernel = candidate_kernel.candidate_block_max_tl(block_size)
    shapes = kernel.__annotations__
    rows, width = shapes["logits"].shape
    assert str(rows) == "num_rows" and str(width) == "max_model_len"
    for capacity in (17, 16384, 16385, 65539, 131073):
        values = {rows: 257, width: capacity}
        assert (
            shapes["block_scores"].shape[1].subs(values)
            == (capacity + block_size - 1) // block_size
        )
        assert shapes["task_ids"].shape[0].subs(values) == 129 * (
            (capacity + 16383) // 16384
        )
        assert candidate_kernel.candidate_block_max_tl(block_size) is kernel
    assert candidate_kernel.candidate_block_max_tl.cache_info().misses == 1


@pytest.mark.parametrize("block_size", [8, 16])
def test_candidate_wrapper_reuses_factory_while_output_capacity_changes(
    monkeypatch, block_size
):
    import resin_models_cuda.deepseek_v41.upstream.cuda as cuda

    calls = []
    factory_keys = []

    def dispatch(key):
        factory_keys.append(key)
        assert key == block_size

        def call(logits, lengths, scores, counts):
            width = logits.shape[1]
            calls.append((width, scores.shape, scores.stride(0), counts.numel()))
            # Stand in for the compiled operation; this test exercises the
            # production wrapper's factory arguments and output allocation.
            counts[: len(lengths)] = (lengths + block_size - 1) // block_size

        return call

    monkeypatch.setattr(
        cuda,
        "candidate_blocks",
        SimpleNamespace(candidate_block_max_tl=dispatch),
        raising=False,
    )
    wrapper = _load(ROOT / "candidate_blocks.py", "candidate_wrapper_contract")
    for width in (17, 16384, 16385, 65539):
        storage = torch.zeros(5, (width + 15) // 16 * 16, dtype=torch.bfloat16)
        logits = storage[:, :width]
        lengths = torch.tensor(
            [0, 1, block_size + 1, width - 1, width], dtype=torch.int32
        )
        scores, counts = wrapper.candidate_block_max(logits, lengths, block_size)
        blocks = (width + block_size - 1) // block_size
        stride = (blocks + 511) // 512 * 512
        assert scores.shape == (5, stride)
        assert scores.stride(0) * scores.element_size() % 1024 == 0
        assert counts.shape == (5,)
        assert counts.tolist() == [
            (int(value) + block_size - 1) // block_size for value in lengths
        ]
        assert calls[-1] == (width, (5, blocks), stride, 8)
    assert factory_keys == [block_size] * 4


@pytest.mark.parametrize("block_size", [8, 16])
def test_candidate_probe_oracle_covers_empty_partial_and_16k_task_boundaries(
    block_size,
):
    from deploy.v41_candidate_probe import _reference

    width = 16384 + block_size + 1
    lengths = torch.tensor(
        [0, 1, block_size, block_size + 1, 16384, 16385, width], dtype=torch.int32
    )
    logits = -torch.arange(width).expand(len(lengths), -1).bfloat16().clone()
    # Scores outside a causal length should never affect its completed blocks.
    logits.masked_fill_(torch.arange(width)[None] >= lengths[:, None], 1000)
    scores, counts = _reference(logits, lengths, block_size)
    assert counts.tolist() == [
        (int(length) + block_size - 1) // block_size for length in lengths
    ]
    assert (scores[0] == -torch.inf).all()
    for row, count in enumerate(counts.tolist()):
        if count:
            assert scores[row, count - 1] == torch.inf
            for block in range(count - 1):
                # Monotonic nonpositive input makes each block maximum known.
                assert scores[row, block] == logits[row, block * block_size]
    assert not (scores == 1000).any()
