"""GPT-OSS names for Resin's reusable sink-attention MoE transformer."""

from resin.backend.transformer import (
    MLP,
    DenseExperts,
    MoEDecoderLayer,
    MoEForCausalLM,
    MoETransformerModel,
    RMSNorm,
    SinkAttention,
    TopKRouter,
    YarnRotaryEmbedding,
    apply_rotary,
)

Attention = SinkAttention
DecoderLayer = MoEDecoderLayer
GptOssForCausalLM = MoEForCausalLM
GptOssModel = MoETransformerModel
RotaryEmbedding = YarnRotaryEmbedding

__all__ = [
    "Attention",
    "DecoderLayer",
    "DenseExperts",
    "GptOssForCausalLM",
    "GptOssModel",
    "MLP",
    "RMSNorm",
    "RotaryEmbedding",
    "TopKRouter",
    "apply_rotary",
]
