"""Backend-neutral GPT-OSS checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GptOssConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    head_dim: int
    sliding_window: int
    rms_norm_eps: float
    rope_theta: float
    initial_context_length: int
    rope_scaling_factor: float
    rope_beta_fast: float
    rope_beta_slow: float
    max_position_embeddings: int
    swiglu_limit: float
    attention_bias: bool
    layer_types: tuple[str, ...]
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_local_experts": self.num_local_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "head_dim": self.head_dim,
            "sliding_window": self.sliding_window,
            "initial_context_length": self.initial_context_length,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"GPT-OSS dimensions must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"unsupported GPT-OSS layer types: {sorted(unsupported)}")
        if self.rope_scaling_factor < 1.0:
            raise ValueError("rope scaling factor must be at least one")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.attention_dropout != 0.0:
            raise ValueError("the GPT-OSS inference runner requires zero dropout")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GptOssConfig:
        if config.get("model_type") != "gpt_oss":
            raise ValueError(
                f"expected a GPT-OSS checkpoint, got {config.get('model_type')!r}"
            )

        num_experts = _same_value(
            config,
            "num_local_experts",
            "num_experts",
            missing="GPT-OSS config is missing num_local_experts",
            conflict="conflicting GPT-OSS expert counts in config",
        )
        top_k = _same_value(
            config,
            "num_experts_per_tok",
            "experts_per_token",
            missing="GPT-OSS config is missing num_experts_per_tok",
            conflict="conflicting GPT-OSS top-k values in config",
        )
        rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
        layer_types = config.get("layer_types")
        if layer_types is None:
            layer_types = [
                "sliding_attention" if index % 2 == 0 else "full_attention"
                for index in range(config["num_hidden_layers"])
            ]
        if config.get("tie_word_embeddings", False):
            raise ValueError("GPT-OSS requires untied input and output embeddings")

        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            head_dim=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            sliding_window=config["sliding_window"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", config.get("rope_theta", 150_000.0)),
            initial_context_length=rope.get(
                "original_max_position_embeddings",
                config.get("initial_context_length", 4096),
            ),
            rope_scaling_factor=rope.get(
                "factor", config.get("rope_scaling_factor", 1.0)
            ),
            rope_beta_fast=rope.get("beta_fast", config.get("rope_ntk_beta", 32.0)),
            rope_beta_slow=rope.get("beta_slow", config.get("rope_ntk_alpha", 1.0)),
            max_position_embeddings=config.get("max_position_embeddings", 131_072),
            swiglu_limit=config.get("swiglu_limit", 7.0),
            attention_bias=config.get("attention_bias", True),
            layer_types=tuple(layer_types),
            attention_dropout=config.get("attention_dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
            pad_token_id=config.get("pad_token_id"),
            eos_token_id=config.get("eos_token_id"),
        )


def _same_value(
    config: dict[str, Any],
    primary: str,
    alias: str,
    *,
    missing: str,
    conflict: str,
):
    first = config.get(primary)
    second = config.get(alias)
    if first is not None and second is not None and first != second:
        raise ValueError(conflict)
    value = first if first is not None else second
    if value is None:
        raise ValueError(missing)
    return value
