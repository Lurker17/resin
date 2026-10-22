"""Prepared FlashMLA projection ABI, independent of CUDA availability."""

import sys
from types import SimpleNamespace

import torch
from torch import nn

from resin_models_cuda.deepseek_v41.attention import OutputProjection


def test_output_projection_rearranges_packed_scale_words_without_numeric_cast(
    monkeypatch,
):
    layer = OutputProjection.__new__(OutputProjection)
    nn.Module.__init__(layer)
    layer.register_buffer(
        "weight", torch.empty(8, 128, 4096, dtype=torch.float8_e4m3fn)
    )
    layer.register_buffer("scale", torch.empty(8, 128, 32, dtype=torch.int32))
    scales = torch.arange(4 * 8 * 32, dtype=torch.int32).reshape(4, 8, 32)
    scales[0, 0, :2] = torch.tensor([0x7F800000, 0x7FC12345], dtype=torch.int32)
    calls = []

    def layout(words):
        assert words.dtype == torch.float32
        return words.transpose(-1, -2).contiguous().transpose(-1, -2)

    def einsum(expression, left, right, output, *, recipe):
        assert expression == "bhr,hdr->bhd"
        assert recipe == (1, 1, 32)
        assert left[1].dtype == torch.int32
        assert torch.equal(left[1], scales)
        assert left[1].stride(0) == 1
        calls.append(output.shape)
        output.zero_()

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(get_mn_major_tma_aligned_tensor=layout, fp8_einsum=einsum),
    )
    output = layer(torch.empty(4, 8, 4096, dtype=torch.float8_e4m3fn), scales)
    assert output.shape == (4, 8, 128)
    assert calls == [output.shape]
