"""Backend-neutral configuration for Gemma 4 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


@dataclass(frozen=True, slots=True)
class Gemma4Nvfp4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    sliding_window: int
    max_position_embeddings: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    image_token_id: int
    video_token_id: int
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    final_logit_softcapping: float
    local_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float
    group_size: int

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_global_key_value_heads": self.num_global_key_value_heads,
            "head_dim": self.head_dim,
            "global_head_dim": self.global_head_dim,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "group_size": self.group_size,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"Gemma 4 dimensions must be positive: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}
        if unsupported:
            raise ValueError(f"unsupported Gemma 4 layer types: {sorted(unsupported)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("local query heads must be divisible by KV heads")
        if self.num_attention_heads % self.num_global_key_value_heads:
            raise ValueError("global query heads must be divisible by KV heads")
        if self.head_dim % 2 or self.global_head_dim % 2:
            raise ValueError("attention head dimensions must be even")
        if not 0.0 < self.global_partial_rotary_factor <= 1.0:
            raise ValueError("global_partial_rotary_factor must be in (0, 1]")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not self.eos_token_ids:
            raise ValueError("eos_token_ids must not be empty")

    @property
    def local_layer_count(self) -> int:
        return self.layer_types.count(SLIDING_ATTENTION)

    @property
    def global_layer_count(self) -> int:
        return self.layer_types.count(FULL_ATTENTION)

    @property
    def history_blocks_per_parent(self) -> int:
        """Pack global pages into a parent the same size as one local page."""

        local = self.local_layer_count * self.num_key_value_heads * self.head_dim
        global_ = (
            self.global_layer_count
            * self.num_global_key_value_heads
            * self.global_head_dim
        )
        if not global_ or local % global_:
            raise ValueError(
                "local/global KV geometry does not share an integral parent"
            )
        return local // global_

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Gemma4Nvfp4Config:
        if config.get("model_type") != "gemma4":
            raise ValueError(
                f"expected a Gemma 4 checkpoint, got {config.get('model_type')!r}"
            )
        if config.get("audio_config") is not None:
            raise ValueError("the CUDA runner does not implement Gemma 4 audio inputs")
        text = config.get("text_config") or {}
        if text.get("model_type") != "gemma4_text":
            raise ValueError("checkpoint is missing a Gemma 4 text configuration")
        required = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": True,
            "use_bidirectional_attention": "vision",
        }
        mismatched = {
            key: (text.get(key), expected)
            for key, expected in required.items()
            if text.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"unsupported Gemma 4 text options: {mismatched}")
        if not config.get("tie_word_embeddings", False):
            raise ValueError("the NVFP4 runner requires tied input/output embeddings")

        quantization = config.get("quantization_config") or {}
        groups = quantization.get("config_groups") or {}
        group = groups.get("group_0") or {}
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        if (
            weights.get("num_bits") != 4
            or weights.get("type") != "float"
            or weights.get("dynamic") is not False
            or activations.get("num_bits") != 4
            or activations.get("type") != "float"
            or activations.get("dynamic") is not False
            or weights.get("group_size") != activations.get("group_size")
        ):
            raise ValueError(
                "checkpoint is not static ModelOpt NVFP4 with matching groups"
            )

        rope = text.get("rope_parameters") or {}
        local_rope = rope.get(SLIDING_ATTENTION) or {}
        global_rope = rope.get(FULL_ATTENTION) or {}
        if local_rope.get("rope_type") != "default":
            raise ValueError("local attention requires default RoPE")
        if global_rope.get("rope_type") != "proportional":
            raise ValueError("global attention requires proportional RoPE")

        eos = config.get("eos_token_id", text.get("eos_token_id"))
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            num_global_key_value_heads=text["num_global_key_value_heads"],
            head_dim=text["head_dim"],
            global_head_dim=text["global_head_dim"],
            sliding_window=text["sliding_window"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            layer_types=tuple(text["layer_types"]),
            image_token_id=config["image_token_id"],
            video_token_id=config["video_token_id"],
            pad_token_id=text["pad_token_id"],
            eos_token_ids=eos_ids,
            final_logit_softcapping=text["final_logit_softcapping"],
            local_rope_theta=local_rope.get("rope_theta", 10_000.0),
            global_rope_theta=global_rope["rope_theta"],
            global_partial_rotary_factor=global_rope.get("partial_rotary_factor", 1.0),
            group_size=weights["group_size"],
        )


__all__ = [
    "FULL_ATTENTION",
    "Gemma4Nvfp4Config",
    "SLIDING_ATTENTION",
]
