"""Prepared FlashMLA projection ABI, independent of CUDA availability."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from resin_models_cuda.deepseek_v41.attention import OutputProjection, finish_output


@pytest.mark.parametrize("fp8_output", [False, True])
def test_output_projection_rearranges_packed_scale_words_without_numeric_cast(
    monkeypatch, fp8_output
):
    layer = OutputProjection.__new__(OutputProjection)
    nn.Module.__init__(layer)
    layer.fp8_output = fp8_output
    layer.register_buffer(
        "weight", torch.empty(8, 128, 4096, dtype=torch.float8_e4m3fn)
    )
    layer.register_buffer("scale", torch.empty(8, 128, 32, dtype=torch.int32))
    scales = torch.arange(5 * 8 * 32, dtype=torch.int32).reshape(5, 8, 32)
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
        if fp8_output:
            value, sf = output
            assert value.dtype == torch.float8_e4m3fn
            assert sf.dtype == torch.int32
            assert sf.shape == (5, 8)
            assert sf.stride() == (1, 8)
            sf.fill_(0x7F7F7F7F)
        else:
            value = output
            assert value.dtype == torch.bfloat16
        calls.append(value.shape)
        value.zero_()

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_mn_major_tma_aligned_tensor=layout,
            get_tma_aligned_size=lambda rows, size: (rows + 3) // 4 * 4,
            fp8_einsum=einsum,
        ),
    )
    output = layer(torch.empty(5, 8, 4096, dtype=torch.float8_e4m3fn), scales)
    values = output[0] if fp8_output else output
    assert values.shape == (5, 8, 128)
    assert calls == [values.shape]


@pytest.mark.parametrize("fp8_output", [False, True])
def test_finish_output_flattens_groups_without_reshaping_packed_scales(fp8_output):
    values = torch.arange(6 * 2 * 128).reshape(6, 2, 128).bfloat16()
    scales = torch.empty_strided((6, 2), (1, 8), dtype=torch.int32).fill_(0x7F7F7F7F)
    grouped = (values.to(torch.float8_e4m3fn), scales) if fp8_output else values
    context = (object(), object())

    def project_a(*args):
        assert args == context
        return grouped

    def project_b(value):
        if fp8_output:
            data, sf = value
            assert sf is scales
            assert data.shape == (6, 256)
            assert data.data_ptr() == grouped[0].data_ptr()
        else:
            assert value.shape == (6, 256)
        return torch.ones(6, 32)

    attention = SimpleNamespace(wo_a=project_a, wo_b=project_b)
    assert finish_output(attention, context).shape == (6, 32)
