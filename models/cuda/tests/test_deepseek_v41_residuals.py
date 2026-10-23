"""Native residual operator layouts, checked without a GPU dependency."""

from types import SimpleNamespace

import torch

from resin_models_cuda.deepseek_v41 import engram, hc


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
