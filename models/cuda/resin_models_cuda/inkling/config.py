"""Validated geometry for Inkling-Small and its external DSpark drafter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self


def _dict(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Inkling config field {key!r} must contain an object")
    return value


@dataclass(frozen=True, slots=True)
class InklingConfig:
    """Text and image geometry of Thinking Machines' Inkling-Small checkpoint."""

    vocab_size: int
    unpadded_vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    sconv_kernel_size: int
    dense_layer_count: int
    dense_intermediate_size: int
    moe_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    experts_per_token: int
    route_scale: float
    rms_norm_eps: float
    max_position_embeddings: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    logits_mup_width_multiplier: float
    eos_token_id: int
    image_token_id: int
    vision_patch_size: int
    vision_temporal_patch_size: int
    vision_num_channels: int
    vision_num_hidden_layers: int

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("model_type") != "inkling_mm_model":
            raise ValueError("the Inkling runner requires an inkling_mm_model")
        if values.get("architectures") != ["InklingForConditionalGeneration"]:
            raise ValueError("unsupported Inkling target architecture")
        text = _dict(values, "text_config")
        vision = _dict(values, "vision_config")
        local = tuple(int(index) for index in text["local_layer_ids"])
        result = cls(
            vocab_size=int(text["vocab_size"]),
            unpadded_vocab_size=int(text["unpadded_vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            d_rel=int(text["d_rel"]),
            rel_extent=int(text["rel_extent"]),
            local_layer_ids=local,
            sliding_window_size=int(text["sliding_window_size"]),
            sconv_kernel_size=int(text["sconv_kernel_size"]),
            dense_layer_count=int(text["dense_mlp_idx"]),
            dense_intermediate_size=int(text["dense_intermediate_size"]),
            moe_intermediate_size=int(text["intermediate_size"]),
            num_routed_experts=int(text["n_routed_experts"]),
            num_shared_experts=int(text["n_shared_experts"]),
            experts_per_token=int(text["num_experts_per_tok"]),
            route_scale=float(text["route_scale"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            max_position_embeddings=int(text["model_max_length"]),
            log_scaling_n_floor=int(text["log_scaling_n_floor"]),
            log_scaling_alpha=float(text["log_scaling_alpha"]),
            logits_mup_width_multiplier=float(text["logits_mup_width_multiplier"]),
            eos_token_id=int(values["eos_token_id"]),
            image_token_id=int(values.get("image_token_id", 200054)),
            vision_patch_size=int(vision["patch_size"]),
            vision_temporal_patch_size=int(vision["temporal_patch_size"]),
            vision_num_channels=int(vision["n_channels"]),
            vision_num_hidden_layers=int(vision["n_layers"]),
        )
        result.validate(text, vision)
        return result

    def validate(
        self,
        raw_text: dict[str, Any] | None = None,
        raw_vision: dict[str, Any] | None = None,
    ) -> None:
        positive = {
            field: getattr(self, field)
            for field in (
                "vocab_size",
                "unpadded_vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "d_rel",
                "rel_extent",
                "sliding_window_size",
                "sconv_kernel_size",
                "dense_layer_count",
                "dense_intermediate_size",
                "moe_intermediate_size",
                "num_routed_experts",
                "num_shared_experts",
                "experts_per_token",
                "route_scale",
                "rms_norm_eps",
                "max_position_embeddings",
                "log_scaling_n_floor",
                "logits_mup_width_multiplier",
                "vision_patch_size",
                "vision_temporal_patch_size",
                "vision_num_channels",
                "vision_num_hidden_layers",
            )
        }
        if invalid := [name for name, value in positive.items() if value <= 0]:
            raise ValueError(f"Inkling fields must be positive: {invalid}")
        if self.unpadded_vocab_size > self.vocab_size:
            raise ValueError("Inkling unpadded vocabulary exceeds its embedding table")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Inkling query heads must be divisible by KV heads")
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids):
            raise ValueError("Inkling local layer IDs must be unique")
        if any(
            layer < 0 or layer >= self.num_hidden_layers
            for layer in self.local_layer_ids
        ):
            raise ValueError("Inkling local layer ID is out of range")
        if self.dense_layer_count >= self.num_hidden_layers:
            raise ValueError("Inkling must contain routed MoE layers")
        if self.experts_per_token > self.num_routed_experts:
            raise ValueError("Inkling routes more experts than it owns")
        if not 0 <= self.eos_token_id < self.unpadded_vocab_size:
            raise ValueError("Inkling EOS token lies outside the output vocabulary")
        if not 0 <= self.image_token_id < self.vocab_size:
            raise ValueError("Inkling image token lies outside the embedding table")
        if raw_text is not None:
            required = {
                "torch_dtype": "bfloat16",
                "q_bias": False,
                "o_bias": False,
                "use_embed_norm": True,
                "use_sconv": True,
                "shared_expert_sink": True,
                "use_gate_bias": True,
                "gate_activation": "sigmoid",
                "norm_after_topk": True,
                "use_global_scale": True,
            }
            mismatched = {
                key: (raw_text.get(key), expected)
                for key, expected in required.items()
                if raw_text.get(key) != expected
            }
            if mismatched:
                raise ValueError(f"unsupported Inkling options: {mismatched}")
        if raw_vision is not None:
            required_vision = {
                "vision_encoder_type": "hmlp",
                "decoder_dmodel": self.hidden_size,
                "patch_size": 40,
                "temporal_patch_size": 2,
                "n_channels": 3,
                "n_layers": 4,
                "use_vision_norm": True,
            }
            mismatched = {
                key: (raw_vision.get(key), expected)
                for key, expected in required_vision.items()
                if raw_vision.get(key) != expected
            }
            if mismatched:
                raise ValueError(f"unsupported Inkling vision options: {mismatched}")

    @property
    def local_layer_count(self) -> int:
        return len(self.local_layer_ids)

    @property
    def global_layer_ids(self) -> tuple[int, ...]:
        local = set(self.local_layer_ids)
        return tuple(
            index for index in range(self.num_hidden_layers) if index not in local
        )

    @property
    def global_layer_count(self) -> int:
        return self.num_hidden_layers - self.local_layer_count

    @property
    def vision_scales(self) -> tuple[tuple[int, int, int, int], ...]:
        """Pinned HMLP progression as `(time, height, width, channels)`."""

        return (
            (1, 1, 1, 3),
            (1, 5, 5, 128),
            (1, 10, 10, 320),
            (1, 40, 40, 4800),
            (2, 40, 40, 9600),
        )


@dataclass(frozen=True, slots=True)
class YarnConfig:
    theta: float
    factor: float
    attention_factor: float
    original_context_length: int
    beta_fast: float
    beta_slow: float


@dataclass(frozen=True, slots=True)
class InklingDSparkConfig:
    """Six-layer Qwen3 DSpark geometry trained for Inkling-Small."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    checkpoint_gamma: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    confidence_with_markov: bool
    yarn: YarnConfig

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("architectures") != ["DSparkDraftModel"]:
            raise ValueError("unsupported Inkling speculative architecture")
        if values.get("model_type") != "qwen3":
            raise ValueError("Inkling DSpark must use the Qwen3 backbone")
        dflash = _dict(values, "dflash_config")
        rope = _dict(values, "rope_scaling")
        result = cls(
            vocab_size=int(values["vocab_size"]),
            hidden_size=int(values["hidden_size"]),
            intermediate_size=int(values["intermediate_size"]),
            num_hidden_layers=int(values["num_hidden_layers"]),
            num_attention_heads=int(values["num_attention_heads"]),
            num_key_value_heads=int(values["num_key_value_heads"]),
            head_dim=int(values["head_dim"]),
            rms_norm_eps=float(values["rms_norm_eps"]),
            max_position_embeddings=int(values["max_position_embeddings"]),
            checkpoint_gamma=int(values["block_size"]),
            target_layer_ids=tuple(int(index) for index in dflash["target_layer_ids"]),
            mask_token_id=int(dflash["mask_token_id"]),
            markov_rank=int(values["markov_rank"]),
            markov_head_type=str(values["markov_head_type"]),
            confidence_with_markov=bool(values["confidence_head_with_markov"]),
            yarn=YarnConfig(
                theta=float(values["rope_theta"]),
                factor=float(rope["factor"]),
                attention_factor=float(
                    rope.get(
                        "attention_factor",
                        1.0 + 0.1 * math.log(float(rope["factor"])),
                    )
                ),
                original_context_length=int(rope["original_max_position_embeddings"]),
                beta_fast=float(rope["beta_fast"]),
                beta_slow=float(rope["beta_slow"]),
            ),
        )
        result.validate(values)
        return result

    def validate(self, raw: dict[str, Any] | None = None) -> None:
        if self.num_hidden_layers != 6:
            raise ValueError("Inkling DSpark requires six draft layers")
        if self.markov_rank <= 0 or self.markov_head_type != "vanilla":
            raise ValueError("Inkling DSpark requires its vanilla Markov head")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DSpark query heads must be divisible by KV heads")
        if len(self.target_layer_ids) == 0 or len(set(self.target_layer_ids)) != len(
            self.target_layer_ids
        ):
            raise ValueError("DSpark target taps must be nonempty and unique")
        if raw is not None:
            required = {
                "attention_bias": False,
                "hidden_act": "silu",
                "use_sliding_window": False,
                "confidence_head_with_markov": True,
            }
            mismatched = {
                key: (raw.get(key), expected)
                for key, expected in required.items()
                if raw.get(key) != expected
            }
            if mismatched:
                raise ValueError(f"unsupported Inkling DSpark options: {mismatched}")
            if tuple(raw.get("layer_types") or ()) != ("full_attention",) * 6:
                raise ValueError(
                    "Inkling DSpark requires full attention in every layer"
                )

    def validate_target(self, target: InklingConfig) -> None:
        if self.vocab_size != target.vocab_size:
            raise ValueError("Inkling target and DSpark vocabulary sizes differ")
        if self.hidden_size != target.hidden_size:
            raise ValueError("Inkling target and DSpark hidden sizes differ")
        if any(index >= target.num_hidden_layers for index in self.target_layer_ids):
            raise ValueError("DSpark target tap lies outside the Inkling target")
        if not 0 <= self.mask_token_id < target.vocab_size:
            raise ValueError("DSpark mask token lies outside the target embedding")


__all__ = ["InklingConfig", "InklingDSparkConfig", "YarnConfig"]
