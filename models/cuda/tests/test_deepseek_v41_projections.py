"""Prepared projection layouts and fusion against independent dense arithmetic."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from resin_models_cuda.deepseek_v41.linear import (
    Linear,
    PreparedLinear,
    prepare_linear,
)


@pytest.fixture
def deepgemm(monkeypatch):
    calls = []

    def transform(scales, rows, columns, **kwargs):
        assert scales.shape == (rows, columns // 32)
        assert kwargs == {"recipe": (1, 1, 32), "is_sfa": False}
        return scales

    def quantize(value, **kwargs):
        assert kwargs == {
            "fmt": "e4m3",
            "num_per_channels": 32,
            "round_sf": True,
            "use_packed_ue8m0": True,
            "use_tma_aligned_col_major_sf": True,
        }
        blocks = value.float().unflatten(-1, (-1, 32))
        scales = 2 ** (blocks.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil()
        return (blocks / scales[..., None]).flatten(-2).to(torch.float8_e4m3fn), scales

    def gemm(left, right, output, **kwargs):
        assert output.stride(0) * output.element_size() % 32 == 0
        assert kwargs == {"recipe": (1, 1, 32)}
        calls.append((left, right))
        a = left[0].float() * left[1].repeat_interleave(32, -1)
        b = right[0].float() * right[1].repeat_interleave(32, -1)
        output.copy_(F.linear(a, b))

    def bf16_gemm(left, right, output):
        assert output.stride(0) * output.element_size() % 32 == 0
        output.copy_(F.linear(left.float(), right.float()))

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            transform_sf_into_required_layout=transform,
            fp8_fp4_gemm_nt=gemm,
            bf16_gemm_nt=bf16_gemm,
        ),
    )
    monkeypatch.setattr(
        "resin_models_cuda.deepseek_v4.tilelang_kernels.per_token_cast", quantize
    )
    return calls, quantize


def projection(rows, *, dtype=torch.float8_e4m3fn, start=0):
    layer = Linear(32, rows, dtype=dtype)
    with torch.no_grad():
        values = torch.arange(rows * 32).reshape(rows, 32) % 11 - 5
        layer.weight.copy_(values.to(dtype))
        if layer.scale is not None:
            scales = 2.0 ** torch.arange(start, start + (rows + 31) // 32)
            layer.scale.copy_(scales[:, None].to(torch.float8_e8m0fnu))
    return layer


def test_merged_partial_scale_blocks_preserve_each_checkpoint_weight(deepgemm):
    first, second = projection(33), projection(17, start=3)
    merged = PreparedLinear(first, second)
    assert merged.widths == (33, 17)
    expected_scales = torch.cat(
        (torch.ones(32, 1), torch.full((1, 1), 2), torch.full((17, 1), 8))
    )
    torch.testing.assert_close(merged.scale, expected_scales)
    torch.testing.assert_close(
        merged.weight.float() * merged.scale.repeat_interleave(32, -1),
        torch.cat(
            [
                layer.weight.float()
                * layer.scale.float().repeat_interleave(32, 0)[: layer.out_features]
                for layer in (first, second)
            ]
        ),
    )


def test_merged_projection_casts_once_and_handles_non_aligned_output(deepgemm):
    calls, quantize = deepgemm
    first, second = projection(33), projection(17, start=3)
    merged = PreparedLinear(first, second)
    hidden = torch.linspace(-2, 2, 2 * 3 * 32).reshape(2, 3, 32).bfloat16()
    actual = merged(hidden)
    values, scales = quantize(
        hidden.reshape(-1, 32),
        fmt="e4m3",
        num_per_channels=32,
        round_sf=True,
        use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=True,
    )
    reference = F.linear(
        values.float() * scales.repeat_interleave(32, -1),
        merged.weight.float() * merged.scale.repeat_interleave(32, -1),
    ).bfloat16()
    assert len(calls) == 1
    assert actual.shape == (2, 3, 50)
    assert actual.stride() == (192, 64, 1)
    torch.testing.assert_close(actual.flatten(0, 1), reference, rtol=0, atol=0)


def test_bf16_projection_preserves_reduction_and_fp32_stays_unmodified(deepgemm):
    layer = projection(3, dtype=torch.bfloat16)
    layer.axis, layer.allreduce = 1, lambda value: value * 2
    prepared = prepare_linear(layer)
    hidden = torch.ones(2, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(prepared(hidden), F.linear(hidden, layer.weight) * 2)
    assert prepare_linear(prepared) is prepared
    fp32 = projection(3, dtype=torch.float32)
    assert prepare_linear(fp32) is fp32


@pytest.mark.parametrize("merged", [False, True])
def test_query_permutation_preserves_source_and_index_rows(
    deepgemm, monkeypatch, merged
):
    from resin_models_cuda.deepseek_v41.attention import QueryProjection

    def permute(pair, heads, dim):
        assert (heads, dim) == (1, 32)
        weight, scales = pair
        return weight.float().flip(0).to(weight.dtype), scales.flip(0)

    monkeypatch.setitem(
        sys.modules,
        "flash_mla",
        SimpleNamespace(
            fused_norm_rope_attn_rope_cast=SimpleNamespace(permute_q_b_proj=permute)
        ),
    )
    attention, index = projection(32), projection(17, start=3)
    original = attention.weight.float().clone()
    query = QueryProjection(attention, 1, 32, index if merged else None)
    torch.testing.assert_close(query.weight[:32].float(), original.flip(0))
    torch.testing.assert_close(attention.weight.float(), original, rtol=0, atol=0)
    assert query.weight.data_ptr() != attention.weight.data_ptr()
    if merged:
        torch.testing.assert_close(query.weight[32:].float(), index.weight.float())
    hidden = torch.ones(2, 3, 32, dtype=torch.bfloat16)
    main, index_query = query.project(hidden)
    assert main.shape == (2, 3, 32)
    if merged:
        assert index_query.shape == (2, 3, 17)
    else:
        assert index_query is None
    assert len(deepgemm[0]) == 1


def test_parent_fusion_accepts_already_prepared_child_scales(deepgemm):
    first, second = projection(32), projection(32, start=3)

    def prepared(source):
        rows = source.scale.float().repeat_interleave(32, 0)
        packed = (
            rows.expand(-1, 4)
            .to(torch.float8_e8m0fnu)
            .view(torch.uint8)
            .contiguous()
            .view(torch.int32)
        )
        return SimpleNamespace(weight=source.weight, scale=packed)

    merged = PreparedLinear(prepared(first), prepared(second))
    torch.testing.assert_close(
        merged.scale, torch.cat((torch.ones(32, 1), torch.full((32, 1), 8)))
    )
