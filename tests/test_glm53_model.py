"""CPU reference checks for GLM's routing, mHC and checkpoint preparation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash import model as glm
from resin_models_cuda.glm5_3_flash.checkpoint import expert_partition
from resin_models_cuda.glm5_3_flash.weights import nvfp4_expert_scales


def tiny_config(**kwargs):
    return replace(Glm53Config(), hidden_size=16, moe_intermediate_size=16, **kwargs)


def test_router_correction_changes_selection_but_not_combination_scores():
    config = tiny_config(n_routed_experts=4, num_experts_per_tok=2)
    projection = torch.zeros(4, 16)
    projection[:, 0] = torch.tensor([2.0, 0.8, -0.5, -2.0])
    bias = torch.tensor([0.0, 0.0, 0.6, 0.8])
    hidden = torch.ones(2, 16)
    ids, weights = glm.Router(config, projection, bias)(hidden)
    # Raw scores prefer expert 0, but selection correction chooses experts 2/3.
    assert set(ids[0].tolist()) == {2, 3}
    original = 1 / (1 + np.exp(-np.array([-0.5, -2.0])))
    expected = original / original.sum() * 2.5
    for row in range(2):
        lookup = dict(zip(ids[row].tolist(), weights[row].tolist(), strict=True))
        np.testing.assert_allclose([lookup[2], lookup[3]], expected, rtol=1e-6)


def test_swiglu_has_asymmetric_gate_and_symmetric_up_clamps():
    gate = torch.tensor([-20.0, -1.0, 4.0, 20.0])
    up = torch.tensor([-20.0, 20.0, -2.0, 20.0])
    expected = np.array([-20.0, -1.0, 4.0, 10.0])
    expected = expected / (1 + np.exp(-expected)) * [-10.0, 10.0, -2.0, 10.0]
    np.testing.assert_allclose(glm.swiglu(gate, up, 10).numpy(), expected, rtol=1e-6)


def test_swiglu_rounds_once_after_fp32_activation_and_product():
    generator = torch.Generator().manual_seed(390)
    gate = (torch.randn(2048, generator=generator) * 6).bfloat16()
    up = (torch.randn(2048, generator=generator) * 6).bfloat16()
    g = np.minimum(gate.float().numpy().astype(np.float64), 10)
    u = np.clip(up.float().numpy().astype(np.float64), -10, 10)
    expected = torch.from_numpy(g / (1 + np.exp(-g)) * u).bfloat16()
    legacy = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    assert torch.count_nonzero(legacy != expected) > 300
    torch.testing.assert_close(glm.swiglu(gate, up, 10), expected, atol=0, rtol=0)


def test_mhc_matches_independent_numpy_reference_and_transposed_residual_mix():
    config = tiny_config()
    generator = torch.Generator().manual_seed(424)
    hidden = torch.randn(3, 4, 16, generator=generator)
    fn = torch.randn(24, 64, generator=generator) / 5
    base = torch.randn(24, generator=generator) / 4
    scale = torch.tensor([0.2, 0.3, 0.4])
    connection = glm.HyperConnection(
        config, {"hc_attn_fn": fn, "hc_attn_base": base, "hc_attn_scale": scale}, "attn"
    )
    value, post, comb = connection(hidden)
    flat = hidden.numpy().reshape(3, 64).astype(np.float64)
    coefficients = (
        flat / np.sqrt((flat * flat).mean(1, keepdims=True) + 1e-5) @ fn.numpy().T
    )
    pre = 1 / (1 + np.exp(-(coefficients[:, :4] * 0.2 + base.numpy()[:4]))) + 1e-6
    expected_post = 2 / (1 + np.exp(-(coefficients[:, 4:8] * 0.3 + base.numpy()[4:8])))
    matrix = (coefficients[:, 8:] * 0.4 + base.numpy()[8:]).reshape(3, 4, 4)
    matrix = np.exp(matrix - matrix.max(-1, keepdims=True))
    matrix = matrix / matrix.sum(-1, keepdims=True) + 1e-6
    matrix /= matrix.sum(-2, keepdims=True) + 1e-6
    for _ in range(19):
        matrix /= matrix.sum(-1, keepdims=True) + 1e-6
        matrix /= matrix.sum(-2, keepdims=True) + 1e-6
    expected = np.einsum("ts,tsh->th", pre, hidden.numpy())
    np.testing.assert_allclose(value.numpy(), expected, atol=4e-7, rtol=1e-5)
    np.testing.assert_allclose(post.numpy(), expected_post, atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(comb.numpy(), matrix, atol=1e-7, rtol=1e-6)
    output = torch.randn(3, 16, generator=generator)
    combined = glm.HyperConnection.post(output, hidden, post, comb)
    expected = expected_post[..., None] * output.numpy()[:, None] + np.einsum(
        "tij,tih->tjh", matrix, hidden.numpy()
    )
    np.testing.assert_allclose(combined.numpy(), expected, atol=4e-7, rtol=1e-5)


def test_nvfp4_scales_preserve_distinct_gate_up_and_clamp_after_dequantization():
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        scales = nvfp4_expert_scales(
            torch.tensor([2.0, 3.0]),
            torch.tensor([4.0, 1.0]),
            torch.tensor([5.0, 7.0]),
            0.5,
            0.25,
            10.0,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    assert all(value.dtype == torch.float32 for value in scales.values())
    torch.testing.assert_close(scales["g1_alphas"], torch.tensor([1.0, 1.5]))
    torch.testing.assert_close(scales["g1_scale_c"], torch.tensor([8.0, 2.0]))
    torch.testing.assert_close(scales["g2_alphas"], torch.tensor([1.25, 1.75]))
    torch.testing.assert_close(
        scales["clamp_limits"] * scales["g1_alphas"], torch.full((2,), 10.0)
    )
    assert scales["input_scale_quant"].item() == 2.0


def groups(ep=4, adp=4, tp=1, expert_tp=1):
    return SimpleNamespace(
        ep=SimpleNamespace(size=ep, local_rank=min(2, ep - 1)),
        attn_dp=SimpleNamespace(size=adp),
        attn_tp=SimpleNamespace(size=tp),
        expert_tp=SimpleNamespace(size=expert_tp, local_rank=0),
        cp=SimpleNamespace(size=1),
        device_domain=SimpleNamespace(size=4),
    )


def test_ep_adp_partition_requires_matching_world_and_no_unsupported_tp():
    assert expert_partition(Glm53Config(), groups()) == (4, 2)
    assert expert_partition(Glm53Config(), groups(ep=1, expert_tp=4)) == (1, 0)
    assert expert_partition(Glm53Config(), groups(ep=2, expert_tp=2)) == (2, 1)
    assert expert_partition(Glm53Config(), None) == (1, 0)
    with pytest.raises(ValueError, match="not attention TP/CP"):
        expert_partition(Glm53Config(), groups(tp=4))
    with pytest.raises(ValueError, match="world size"):
        expert_partition(Glm53Config(), groups(adp=2))
    with pytest.raises(ValueError, match="world size"):
        expert_partition(Glm53Config(), groups(ep=1))
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(replace(Glm53Config(), n_routed_experts=7), groups())


def test_mtp_returns_normalized_recursive_hidden_and_accepts_image_embeddings():
    config = tiny_config(vocab_size=8)
    embedding = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 100

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_norm = torch.linspace(0.5, 1.5, 16)

        def forward(self, embedded, previous, positions, cache):
            return embedded + previous

    draft = Draft()
    model = glm.Glm53Model(config, embedding, torch.ones(16), embedding, [], draft)
    ids = torch.tensor([2, 3])
    previous = torch.full((2, 16), 0.5)
    positions = torch.tensor([1, 2])
    for replacement in (None, torch.full((2, 16), 3.0)):
        logits, hidden = model.mtp_forward(
            ids, previous, positions, None, input_embeddings=replacement
        )
        raw = (embedding[ids] if replacement is None else replacement) + previous
        expected = glm.rms_norm(raw, draft.head_norm, config.rms_norm_eps)
        torch.testing.assert_close(hidden, expected)
        torch.testing.assert_close(logits, expected @ embedding.T)
        assert not torch.equal(hidden, raw)
