"""CPU reference checks for GLM's routing, mHC and checkpoint preparation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash import model as glm
from resin_models_cuda.glm5_3_flash.checkpoint import expert_partition
from resin_models_cuda.glm5_3_flash.weights import WeightReader, nvfp4_expert_scales


def tiny_config(**kwargs):
    return replace(Glm53Config(), hidden_size=16, moe_intermediate_size=16, **kwargs)


def test_native_config_geometry():
    config = Glm53Config.from_dict({"model_type": "glm5_next", "text_config": {}})
    assert len(config.kda_layers) == 34
    assert config.dsa_layers == tuple(range(3, 45, 4))
    assert config.linear_dim == 8192
    assert config.n_routed_experts // 4 == 72
    text = Glm53Config.from_dict({"model_type": "glm5_next_text", "eos_token_id": 7})
    assert text.eos_token_id == (7,)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"mhc": False}, "mHC"),
        ({"mla_use_nope": False}, "NoPE"),
        ({"scoring_func": "softmax"}, "sigmoid"),
        ({"hidden_act": "gelu"}, "SwiGLU"),
        ({"attention_bias": True}, "projection bias"),
        ({"mlp_layer_types": ["sparse"] * 45}, "dense layer count"),
        ({"indexer_types": ["sliding"] * 45}, "full GLM"),
        ({"layer_types": ["linear_attention"]}, "every KDA/DSA"),
        ({"qk_rope_head_dim": 64}, "NoPE"),
        ({"n_group": 8}, "routing group"),
        ({"num_experts_per_tok": 289}, "expert count"),
        ({"hidden_size": 31}, "align"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"linear_lower_bound": 0}, "finite and negative"),
    ],
)
def test_rejects_incompatible_checkpoint(changes, match):
    with pytest.raises(ValueError, match=match):
        Glm53Config.from_dict({"model_type": "glm5_next", "text_config": changes})


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


def test_idle_dense_ffn_skips_zero_row_kernel_launches():
    weights = {
        name + ".weight": torch.eye(16)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    mlp = glm.DenseMLP(weights, 10)

    def fail_if_called(*args):
        raise AssertionError("idle rank launched a dense GEMM")

    mlp.gate.register_forward_pre_hook(fail_if_called)
    hidden = torch.empty(0, 16, dtype=torch.bfloat16)
    result = mlp(hidden)
    assert result.shape == hidden.shape and result.dtype == hidden.dtype


def test_shared_ffn_preserves_original_linear_at_each_verification_column(monkeypatch):
    config = tiny_config(n_routed_experts=4, num_experts_per_tok=2)
    weights = {
        "shared_experts." + name + ".weight": torch.eye(16)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    weights.update(
        {
            "gate.weight": torch.randn(4, 16),
            "gate.e_score_correction_bias": torch.zeros(4),
        }
    )
    model = glm.SparseMLP(
        config, weights, lambda hidden, ids, weights: torch.zeros_like(hidden)
    )
    calls = []
    original = glm.Linear.forward

    def project(self, hidden):
        calls.append(hidden.shape[0])
        return original(self, hidden)

    monkeypatch.setattr(glm.Linear, "forward", project)
    hidden = torch.randn(4, 16)
    model(hidden)
    assert calls == [4, 4, 4]
    calls.clear()
    expected_padded = model(torch.nn.functional.pad(hidden, (0, 0, 0, 60)))[:4]
    calls.clear()
    assert torch.equal(model(hidden, projection_rows=64), expected_padded)
    assert calls == [64] * 3


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("projection_rows", [0, 64])
def test_decoder_projection_metadata_does_not_enter_prefill_callback(
    rows, projection_rows
):
    config = tiny_config()
    weights = {}
    for name in ("attn", "ffn"):
        weights.update(
            {
                f"hc_{name}_fn": torch.zeros(24, 64),
                f"hc_{name}_base": torch.zeros(24),
                f"hc_{name}_scale": torch.ones(3),
            }
        )
    weights.update(
        {
            "input_layernorm.weight": torch.ones(16),
            "post_attention_layernorm.weight": torch.ones(16),
        }
    )
    calls = []

    class Attention(nn.Module):
        def forward(self, hidden, positions, cache, *, projection_rows=0):
            calls.append(("attention", projection_rows))
            return hidden

    class FeedForward(nn.Module):
        def forward(self, hidden, *, projection_rows=0):
            calls.append(("ffn", projection_rows))
            return hidden

    layer = glm.DecoderLayer(
        config, 0, weights, FeedForward(), lambda *args: Attention()
    )

    def prefill(hidden):
        calls.append(("prefill", 0))
        return layer.feed_forward(hidden)

    layer.prefill_ffn = prefill
    hidden = torch.randn(rows, 4, 16)
    for decode in (False, True):
        calls.clear()
        active_rows = projection_rows if decode else 0
        cache = SimpleNamespace(
            is_decode=decode, projection_rows=active_rows, layer=lambda index: None
        )
        layer(hidden, torch.arange(rows), cache)
        assert calls == (
            [("attention", projection_rows), ("ffn", projection_rows)]
            if decode
            else [("attention", 0), ("prefill", 0), ("ffn", 0)]
        )


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


class ExpertReader:
    device = torch.device("cpu")

    def __init__(self, invalid_scale=None):
        self.reads = []
        self.invalid_scale = invalid_scale

    def read(self, name, shape=None, dtype=None, *, cpu=False):
        self.reads.append(name)
        expert, projection, suffix = name.split(".")
        expert = int(expert)
        if suffix == "input_scale":
            # Expert 0 is unowned by EP rank 1, but owns the global calibration max.
            value = 8.0 if expert == 0 else 0.5
            if projection == "down_proj":
                value *= 2
            return torch.tensor(
                self.invalid_scale if self.invalid_scale is not None else value
            )
        if suffix == "weight_scale_2":
            return torch.tensor(
                {"gate_proj": 2.0, "up_proj": 4.0, "down_proj": 3.0}[projection]
            )
        assert shape is not None and dtype is not None
        if suffix == "weight_scale":
            return torch.ones(shape).to(dtype)
        return torch.full(
            shape,
            expert * 10 + {"gate_proj": 1, "up_proj": 2, "down_proj": 3}[projection],
            dtype=dtype,
        )


def test_nvfp4_expert_loading_is_rank_local_with_global_calibration(monkeypatch):
    reader = ExpertReader()
    prepared = []
    monkeypatch.setattr(
        glm, "prepare_nvfp4_moe_weights", lambda *args: prepared.append(args)
    )
    config = tiny_config(n_routed_experts=4, num_experts_per_tok=2)
    experts = glm.Nvfp4Experts(config, reader, "", ep_size=2, ep_rank=1, max_tokens=8)
    assert experts.start == 2 and experts.count == 2
    assert len(prepared) == 1
    assert all(
        int(name.split(".")[0]) >= 2
        for name in reader.reads
        if name.endswith(".weight")
    )
    assert experts.input_scale_quant.item() == 1 / 8
    torch.testing.assert_close(
        experts.w13[0, :16], torch.full((16, 8), 22, dtype=torch.uint8)
    )
    torch.testing.assert_close(
        experts.w13[0, 16:], torch.full((16, 8), 21, dtype=torch.uint8)
    )
    torch.testing.assert_close(experts.w13_scales[0, :16].float(), torch.ones(16, 1))
    torch.testing.assert_close(experts.w13_scales[0, 16:].float(), torch.ones(16, 1))
    torch.testing.assert_close(experts.g1_alphas, torch.full((2,), 16.0))
    torch.testing.assert_close(experts.g2_alphas, torch.full((2,), 48.0))
    assert experts(torch.empty(0, 16), torch.empty(0, 2), torch.empty(0, 2)).shape == (
        0,
        16,
    )
    with pytest.raises(ValueError, match="prepared token bound"):
        experts(torch.empty(9, 16), torch.empty(9, 2), torch.empty(9, 2))


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_nvfp4_calibration_rejected(monkeypatch, invalid):
    monkeypatch.setattr(glm, "prepare_nvfp4_moe_weights", lambda *args: None)
    with pytest.raises(ValueError, match="finite and positive"):
        glm.Nvfp4Experts(
            tiny_config(n_routed_experts=4), ExpertReader(invalid), "", ep_size=2
        )


def test_weight_reader_validates_before_transfer_and_filters_experts(tmp_path):
    tensors = {
        "layer.norm": torch.ones(16, dtype=torch.bfloat16),
        "layer.mlp.experts.0.weight": torch.zeros(16, 16),
    }
    save_file(tensors, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "part.safetensors")})
    )
    reader = WeightReader(tmp_path, "cpu")
    try:
        with pytest.raises(ValueError, match="missing"):
            reader.read("absent")
        with pytest.raises(ValueError, match="expected"):
            reader.read("layer.norm", shape=(32,))
        with pytest.raises(ValueError, match="expected"):
            reader.read("layer.norm", dtype=torch.float32)
        assert reader.loaded_bytes == 0
        selected = reader.prefix("layer.", exclude=(".experts.",))
        assert set(selected) == {"norm"}
        assert reader.loaded_bytes == 32
    finally:
        reader.close()


def groups(ep=4, adp=4, tp=1):
    return SimpleNamespace(
        ep=SimpleNamespace(size=ep, local_rank=2),
        attn_dp=SimpleNamespace(size=adp),
        attn_tp=SimpleNamespace(size=tp),
        expert_tp=SimpleNamespace(size=1),
        cp=SimpleNamespace(size=1),
        device_domain=SimpleNamespace(size=4),
    )


def test_ep_adp_partition_requires_matching_world_and_no_unsupported_tp():
    assert expert_partition(Glm53Config(), groups()) == (4, 2)
    assert expert_partition(Glm53Config(), None) == (1, 0)
    with pytest.raises(ValueError, match="not TP/CP"):
        expert_partition(Glm53Config(), groups(tp=4))
    with pytest.raises(ValueError, match="world size"):
        expert_partition(Glm53Config(), groups(adp=2))
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(replace(Glm53Config(), n_routed_experts=7), groups())


def test_model_collapses_unweighted_streams_and_selects_hidden_before_logits():
    config = tiny_config(vocab_size=8)
    embedding = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 100

    class Streams(nn.Module):
        def forward(self, hidden, positions, cache):
            return hidden + torch.arange(4)[None, :, None]

    model = glm.Glm53Model(config, embedding, torch.ones(16), embedding, [Streams()])
    ids, positions = torch.tensor([1, 4, 7]), torch.tensor([0, 1, 2])
    hidden = model.forward_hidden(ids, positions, None)
    torch.testing.assert_close(hidden, embedding[ids] + 1.5)
    full, state = model(ids, positions, None, return_hidden=True)
    torch.testing.assert_close(state, hidden)
    torch.testing.assert_close(model.logits(hidden[-1:]), full[-1:])
    replacement = torch.ones(3, 16)
    torch.testing.assert_close(
        model.forward_hidden(ids, positions, None, hidden=replacement),
        replacement + 1.5,
    )
    with pytest.raises(RuntimeError, match="MTP weights"):
        model.mtp_forward(ids, hidden, positions, None)


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


def test_mtp_prefill_callback_and_idle_rank_share_the_same_ffn_path():
    config = tiny_config(vocab_size=8)
    weights = {
        key: torch.ones(16)
        for key in (
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "shared_head.norm.weight",
        )
    }
    weights["eh_proj.weight"] = torch.cat((torch.eye(16), torch.eye(16)), -1)

    class NoAttention(nn.Module):
        def forward(self, hidden, positions, cache):
            return torch.zeros_like(hidden)

    mtp = glm.MtpLayer(config, weights, nn.Identity(), lambda *args: NoAttention())
    calls = []

    def callback(hidden):
        calls.append(hidden.shape[0])
        return mtp.feed_forward(hidden)

    mtp.prefill_ffn = callback
    mtp(
        torch.ones(3, 16),
        torch.ones(3, 16),
        torch.arange(3),
        SimpleNamespace(layer=lambda index: None),
    )
    model = glm.Glm53Model(
        config, torch.zeros(8, 16), torch.ones(16), torch.zeros(8, 16), [], mtp
    )
    assert model.forward_mtp_experts_empty().shape == (0, 16)
    assert calls == [3, 0]
