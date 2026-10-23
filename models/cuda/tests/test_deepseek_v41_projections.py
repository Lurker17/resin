"""Prepared projection layouts and fusion against independent dense arithmetic."""

import sys
from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v41_model import implementation as implementation
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext
from resin_models_cuda.deepseek_v41.linear import (
    Linear,
    PreparedLinear,
    PreparedVocabHead,
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


def test_prequantized_projection_preserves_scales_and_restores_leading_shape(
    deepgemm, monkeypatch
):
    calls, quantize = deepgemm
    prepared = PreparedLinear(projection(17))
    hidden = torch.randn(2, 3, 32).bfloat16()
    values, scales = quantize(
        hidden.flatten(0, 1),
        fmt="e4m3",
        num_per_channels=32,
        round_sf=True,
        use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=True,
    )

    def no_cast(_value):
        raise AssertionError("prequantized projection must not cast a second time")

    monkeypatch.setattr("resin_models_cuda.deepseek_v41.linear.quantize_fp8", no_cast)
    result = prepared((values.reshape(2, 3, 32), scales))
    assert result.shape == (2, 3, 17)
    assert calls[0][0][0].shape == (6, 32)
    assert calls[0][0][1] is scales
    expected = F.linear(
        values.float() * scales.repeat_interleave(32, -1),
        prepared.weight.float() * prepared.scale.repeat_interleave(32, -1),
    ).bfloat16()
    torch.testing.assert_close(result.flatten(0, 1), expected, rtol=0, atol=0)
    bf16 = PreparedLinear(projection(17, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="prequantized inputs require FP8"):
        bf16((values, scales))
    with pytest.raises(ValueError, match="input width"):
        prepared((values[:, :16], scales))


def test_attention_finish_restores_batch_sequence_shape(implementation):
    from resin_models_cuda.deepseek_v41.attention import finish_output

    values = torch.ones(6, 2, 128, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((6, 2), (1, 8), dtype=torch.int32)
    attention = SimpleNamespace(
        wo_a=lambda *_: (values, scales), wo_b=lambda value: value[0].bfloat16()[:, :32]
    )
    context = (values, scales)
    actual = implementation.Attention._finish(attention, context, torch.empty(2, 3))
    assert actual.shape == (2, 3, 32)
    torch.testing.assert_close(actual.flatten(0, 1), finish_output(attention, context))


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


def test_prepared_vocab_head_preserves_last_row_full_logits_and_fp32_output(deepgemm):
    weight = torch.randn(19, 32).bfloat16().float()
    source = SimpleNamespace(weight=weight, tensor_parallel=SimpleNamespace(size=1))
    head = PreparedVocabHead(source)
    hidden = torch.randn(2, 3, 32).bfloat16()
    expected = F.linear(hidden.float(), weight)
    assert head.weight.dtype == torch.bfloat16
    assert head(hidden).dtype == torch.float32
    torch.testing.assert_close(head.weight.float(), weight, rtol=0, atol=0)
    torch.testing.assert_close(head(hidden), expected[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(head(hidden, full_logits=True), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        head(hidden[:, 0], full_logits=True), expected[:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(source.weight, weight, rtol=0, atol=0)


def test_prepared_vocab_head_rejects_attention_tp_before_preparation():
    source = SimpleNamespace(tensor_parallel=SimpleNamespace(size=2))
    with pytest.raises(ValueError, match="attention TP1"):
        PreparedVocabHead(source)


def test_weight_preparation_lowers_main_and_markov_vocabulary_heads(
    deepgemm, implementation
):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.head = implementation.ParallelHead(
        32, 32, tensor_parallel=TensorParallelContext()
    )
    model.markov = nn.Module()
    model.markov.head = implementation.ParallelHead(
        32, 16, tensor_parallel=TensorParallelContext()
    )
    model.prepare_weights(lambda value: value)
    assert isinstance(model.head, PreparedVocabHead)
    assert isinstance(model.markov.head, PreparedVocabHead)
