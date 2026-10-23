"""Native residual operator layouts, checked without a GPU dependency."""

import sys
from types import SimpleNamespace

import torch

from resin.backend.torch import use_default_dtype
from resin_models_cuda.deepseek_v41 import engram, hc


def test_prenorm_uses_bf16_inputs_fixed_splits_and_fp32_workspaces(monkeypatch):
    hidden = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    fn = torch.randn(24, 32)
    eps = 1e-6

    def gemm(x, weight, out, squares, splits):
        assert x.shape == (6, 32) and x.dtype == torch.bfloat16
        assert weight is fn and splits == 16
        assert out.shape == (16, 6, 24) and out.dtype == torch.float32
        assert squares.shape == (16, 6) and squares.dtype == torch.float32
        for split in range(splits):
            part = slice(2 * split, 2 * split + 2)
            out[split] = x[:, part].float() @ fn[:, part].T
            squares[split] = x[:, part].float().square().sum(-1)

    def kernel(mixes, groups, width, epsilon, splits):
        assert (mixes, groups, width, epsilon, splits) == (24, 1, 32, eps, 16)

        def reduce(partials, squares, sums, sum_squares, output):
            assert partials.shape == (16, 6, 1, 24)
            assert squares.shape == (16, 6, 1)
            sums.copy_(partials.sum(0))
            sum_squares.copy_(squares.sum(0))
            output.copy_(sums[:, 0] * torch.rsqrt(sum_squares / width + eps))

        return reduce

    monkeypatch.setitem(
        sys.modules, "deep_gemm", SimpleNamespace(tf32_hc_prenorm_gemm=gemm)
    )
    monkeypatch.setattr(hc, "_prenorm_kernel", lambda: kernel)
    expected = hc.prenorm(hidden, fn, eps)
    with use_default_dtype(torch.bfloat16):
        actual = hc._prenorm_native(hidden, fn, eps)
    assert actual.shape == (2, 3, 24) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_native_post_keeps_source_destination_order_and_lazy_state(monkeypatch):
    output = torch.tensor([[[2.0, -3.0]]], dtype=torch.bfloat16)
    residual = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]], dtype=torch.bfloat16)
    state = hc.HCLazyPost(
        residual,
        torch.tensor([[[7.0, 11.0]]]),
        torch.tensor([[[0.5, 2.0]]]),
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
    )
    snapshots = [
        value.clone() for value in (output, residual, state.pre, state.post, state.comb)
    ]

    def native(x, trunk, post, comb):
        assert x.shape == (1, 1, 2)
        assert trunk.shape == (1, 1, 2, 2)
        assert post.shape == (1, 1, 2, 1)
        assert torch.equal(comb, state.comb)
        mixed = torch.einsum("...ij,...id->...jd", comb, trunk.float())
        return (mixed + post * x.unsqueeze(-2)).bfloat16()

    monkeypatch.setattr(hc, "_post_kernel", lambda: native)
    expected = torch.tensor([[[[101.0, 138.5], [144.0, 194.0]]]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        hc._materialize_native(output, state), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(hc.materialize(output, state), expected, rtol=0, atol=0)
    for actual, before in zip(
        (output, residual, state.pre, state.post, state.comb), snapshots, strict=True
    ):
        assert torch.equal(actual, before)


def test_native_engram_passes_strided_kv_and_inference_gate_contract(monkeypatch):
    hidden = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    projection = torch.randn(2, 3, 40, dtype=torch.bfloat16)
    weight = torch.randn(4, 8)

    def native(x, key, value, fused_weight, eps, clamp, *, save_for_backward):
        assert x.shape == (6, 4, 8)
        assert key.shape == (6, 4, 8) and key.stride() == (40, 8, 1)
        assert value.shape == (6, 8) and value.stride() == (40, 1)
        assert (
            key.untyped_storage().data_ptr() == projection.untyped_storage().data_ptr()
        )
        assert fused_weight is weight and eps == 1e-20 and clamp == 1e-6
        assert not save_for_backward
        return x + value[:, None], None, None, None, None

    monkeypatch.setattr(
        engram, "_gate_kernels", lambda: SimpleNamespace(engram_gate_fwd=native)
    )
    actual = engram._gate_native(hidden, projection, weight, 1e-20)
    expected = hidden + projection.reshape(2, 3, 5, 8)[..., -1:, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_engram_preparation_retains_float_weight_products():
    module = engram.Engram.__new__(engram.Engram)
    torch.nn.Module.__init__(module)
    module.q_weight = torch.nn.Parameter(
        torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)
    )
    module.k_weight = torch.nn.Parameter(
        torch.tensor([[0.75, 1.125]], dtype=torch.bfloat16)
    )
    module.register_buffer("_gate_weight", None, persistent=False)
    module.prepare_gate()
    assert module._gate_weight.dtype == torch.float32
    assert torch.equal(
        module._gate_weight, module.q_weight.float() * module.k_weight.float()
    )
    assert "_gate_weight" not in module.state_dict()
