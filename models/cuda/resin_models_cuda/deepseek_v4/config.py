"""Validated configuration for native DeepSeek-V4 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class DeepSeekV4Config:
    """The checkpoint geometry Resin relies on for target-model inference."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    rope_head_dim: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    index_topk: int
    num_routed_experts: int
    num_experts_per_token: int
    moe_intermediate_size: int
    hc_mult: int
    expert_dtype: str
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 0

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        """Parse the public DeepSeek-V4 ``config.json`` naming."""

        config = cls(
            vocab_size=int(values["vocab_size"]),
            hidden_size=int(values["hidden_size"]),
            num_hidden_layers=int(values["num_hidden_layers"]),
            num_attention_heads=int(values["num_attention_heads"]),
            num_key_value_heads=int(values["num_key_value_heads"]),
            head_dim=int(values["head_dim"]),
            q_lora_rank=int(values["q_lora_rank"]),
            o_lora_rank=int(values["o_lora_rank"]),
            o_groups=int(values["o_groups"]),
            rope_head_dim=int(values["qk_rope_head_dim"]),
            sliding_window=int(values["sliding_window"]),
            compress_ratios=tuple(int(ratio) for ratio in values["compress_ratios"]),
            index_topk=int(values["index_topk"]),
            num_routed_experts=int(values["n_routed_experts"]),
            num_experts_per_token=int(values["num_experts_per_tok"]),
            moe_intermediate_size=int(values["moe_intermediate_size"]),
            hc_mult=int(values["hc_mult"]),
            expert_dtype=str(values["expert_dtype"]),
            dspark_block_size=int(values.get("dspark_block_size", 0)),
            dspark_noise_token_id=int(values.get("dspark_noise_token_id", 0)),
            dspark_target_layer_ids=tuple(
                int(layer_id) for layer_id in values.get("dspark_target_layer_ids", ())
            ),
            dspark_markov_rank=int(values.get("dspark_markov_rank", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject geometry unsupported by Resin's target kernels."""

        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "rope_head_dim": self.rope_head_dim,
            "sliding_window": self.sliding_window,
            "index_topk": self.index_topk,
            "num_routed_experts": self.num_routed_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "moe_intermediate_size": self.moe_intermediate_size,
            "hc_mult": self.hc_mult,
        }
        if invalid := [name for name, value in positive.items() if value <= 0]:
            raise ValueError(f"DeepSeek-V4 fields must be positive: {invalid}")
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError("compress_ratios must contain one entry per target layer")
        if any(ratio < 0 for ratio in self.compress_ratios):
            raise ValueError("compress_ratios cannot be negative")
        if self.head_dim <= self.rope_head_dim:
            raise ValueError("head_dim must exceed rope_head_dim")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("attention heads must be divisible by output groups")
        if self.expert_dtype != "fp4":
            raise ValueError("the DeepSeek-V4 runner requires FP4 experts")
        if self.dspark_block_size < 0 or self.dspark_markov_rank < 0:
            raise ValueError("DSpark sizes cannot be negative")
        if self.dspark_block_size:
            if not 0 <= self.dspark_noise_token_id < self.vocab_size:
                raise ValueError("DSpark noise token is outside the vocabulary")
            if self.dspark_markov_rank <= 0:
                raise ValueError("DSpark Markov rank must be positive")
            if not self.dspark_target_layer_ids:
                raise ValueError("DSpark target layers cannot be empty")
            if any(
                layer_id < 0 or layer_id >= self.num_hidden_layers
                for layer_id in self.dspark_target_layer_ids
            ):
                raise ValueError("DSpark target layer is outside the target model")

    def validate_tensor_parallel(self, size: int) -> None:
        """Validate every checkpoint axis sharded by Resin's TP layout."""

        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("tensor parallel size must be a positive integer")
        sharded = {
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "o_groups": self.o_groups,
            "num_routed_experts": self.num_routed_experts,
        }
        if invalid := [name for name, value in sharded.items() if value % size]:
            raise ValueError(
                f"DeepSeek-V4 fields are not divisible by TP={size}: {invalid}"
            )


__all__ = ["DeepSeekV4Config"]
