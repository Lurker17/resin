from __future__ import annotations

import torch

from resin_models_cuda.gemma4.config import Gemma4Nvfp4Config
from resin_models_cuda.gemma4.model import (
    Gemma4Attention,
    RotaryEmbedding,
    ScaledEmbedding,
    apply_rotary,
)


def _small_config() -> Gemma4Nvfp4Config:
    return Gemma4Nvfp4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        sliding_window=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        layer_types=("sliding_attention", "full_attention"),
        image_token_id=30,
        video_token_id=31,
        pad_token_id=0,
        eos_token_ids=(1,),
        final_logit_softcapping=30.0,
        local_rope_theta=10_000.0,
        global_rope_theta=1_000_000.0,
        global_partial_rotary_factor=0.25,
        group_size=16,
    )


def test_global_proportional_rope_leaves_nonrotary_channels_unchanged() -> None:
    rotary = RotaryEmbedding(_small_config())
    rotary.materialize(torch.device("cpu"))
    positions = torch.tensor([1, 2])
    cos, sin = rotary(positions, "full_attention", torch.float32)

    assert torch.equal(cos[:, 1:4], torch.ones(2, 3))
    assert torch.equal(cos[:, 5:], torch.ones(2, 3))
    assert torch.equal(sin[:, 1:4], torch.zeros(2, 3))
    assert torch.equal(sin[:, 5:], torch.zeros(2, 3))
    hidden = torch.randn(2, 2, 8)
    rotated = apply_rotary(hidden, cos, sin)
    assert torch.equal(rotated[..., 1:4], hidden[..., 1:4])
    assert torch.equal(rotated[..., 5:], hidden[..., 5:])


def test_attention_packs_checkpoint_projections_after_loading() -> None:
    attention = Gemma4Attention(_small_config(), layer_index=0)
    attention.pack_projections()
    rotary = RotaryEmbedding(_small_config())
    rotary.materialize(torch.device("cpu"))
    cos, sin = rotary(torch.arange(3), "sliding_attention", torch.bfloat16)

    query, key, value = attention.project_qkv(
        torch.randn(3, 16, dtype=torch.bfloat16), cos, sin
    )

    assert query.shape == (3, 2, 8)
    assert key.shape == value.shape == (3, 1, 8)
    assert not hasattr(attention, "q_proj")


def test_embedding_scale_is_materialized_outside_forward() -> None:
    embedding = ScaledEmbedding(_small_config())
    embedding.materialize_scale()

    output = embedding(torch.tensor([1, 2]))

    assert embedding.scale_tensor is not None
    assert output.shape == (2, 16)
    assert "scale_tensor" not in embedding.state_dict()
