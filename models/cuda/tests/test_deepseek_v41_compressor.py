"""Paged compressor addressing and FP32 accumulation across chunk boundaries."""

import sys
from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v41_contracts import _cache
from test_deepseek_v41_projections import deepgemm as deepgemm
from torch import nn
from torch.nn import functional as F

from resin_models_cuda.deepseek_v41.compressor import (
    PreparedCompressor,
    compress_projected,
)
from resin_models_cuda.deepseek_v41.linear import Linear


@pytest.fixture
def kernel(monkeypatch):
    def factory(*, DIM, COMPRESS_RATIO):
        assert (DIM, COMPRESS_RATIO) == (512, 2)

        def call(scores, values, state, pages, lengths, output):
            positions = lengths.long() - 1
            state[pages.long(), positions % 128, 0] = values
            state[pages.long(), positions % 128, 1] = scores
            output.zero_()
            for index, (page, position) in enumerate(zip(pages, positions)):
                if position % 2:
                    rows = state[page, position % 128 - 1 : position % 128 + 1]
                    output[index] = (rows[:, 0] * rows[:, 1].softmax(0)).sum(0)

        return call

    monkeypatch.setitem(
        sys.modules,
        "resin_models_cuda.deepseek_v41.upstream.cuda.compress",
        SimpleNamespace(compress_and_update_state_decode_tl=factory),
    )


def test_paged_state_view_aliases_value_score_rows_with_layer_offsets():
    cache = _cache()
    positions = torch.tensor([[0, 1, 127, 128, 129], [2, 3, 126, 128, 129]])
    pages = cache.compressor_pages(positions).long()
    for layer in (1, 3):
        values, scores = cache.compressor_rows(layer, positions)
        cache.compressor_values[values] = layer + 10
        cache.compressor_values[scores] = layer + 20
        state = cache.compressor_state(layer)
        assert (state[pages, positions % 128, 0] == layer + 10).all()
        assert (state[pages, positions % 128, 1] == layer + 20).all()


@pytest.mark.parametrize("chunk", [1, 3, 128, 130])
def test_paged_kernel_preserves_partial_windows_and_emits_completed_only(kernel, chunk):
    cache = _cache()
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(2, 130, 512, generator=generator)
    scores = torch.randn(2, 130, 512, generator=generator)
    expected = (
        (values.unflatten(1, (-1, 2)) * scores.unflatten(1, (-1, 2)).softmax(2))
        .sum(2)
        .bfloat16()
    )
    for start in range(0, 130, chunk):
        end = min(start + chunk, 130)
        positions = torch.arange(start, end).expand(2, -1)
        actual = compress_projected(
            scores[:, start:end], values[:, start:end], positions, cache, 1, 2
        )
        complete = (positions[0] + 1) % 2 == 0
        torch.testing.assert_close(
            actual[:, complete],
            expected[:, positions[0, complete] // 2],
            rtol=0,
            atol=0,
        )
        assert (actual[:, ~complete] == 0).all()


def test_prepared_compressor_merges_bf16_projections_with_fp32_output(deepgemm, kernel):
    source = SimpleNamespace(
        layer=1,
        ratio=2,
        norm=nn.Identity(),
        wkv=Linear(32, 512, dtype=torch.float32),
        wgate=Linear(32, 512, dtype=torch.float32),
    )
    with torch.no_grad():
        source.wkv.weight.copy_(torch.randn(512, 32).bfloat16())
        source.wgate.weight.copy_(torch.randn(512, 32).bfloat16())
    prepared = PreparedCompressor(source)
    assert prepared.projection.output_dtype == torch.float32
    hidden = torch.randn(2, 130, 32).bfloat16()
    values = F.linear(hidden.float(), source.wkv.weight).unflatten(1, (-1, 2))
    scores = F.linear(hidden.float(), source.wgate.weight).unflatten(1, (-1, 2))
    expected = (values * scores.softmax(2)).sum(2).bfloat16()
    actual = prepared(hidden, torch.arange(130).expand(2, -1), _cache())
    torch.testing.assert_close(actual[:, 1::2], expected, rtol=0, atol=0)
