"""Validated text and EAGLE3 configurations for the Kimi K2.5 family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self


def _required_dict(values: dict[str, Any], key: str) -> dict[str, Any]:
    nested = values.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Kimi config field {key!r} must contain an object")
    return nested


class EagleVersion(StrEnum):
    """LightSeek draft checkpoint format."""

    EAGLE3 = "eagle3"
    EAGLE3_1 = "eagle3.1"


@dataclass(frozen=True, slots=True)
class RopeConfig:
    """YaRN parameters shared by target and draft MLA."""

    theta: float
    factor: float
    original_context_length: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        raw_scaling = values.get("rope_scaling")
        if raw_scaling is None:
            return cls(
                theta=float(values["rope_theta"]),
                factor=1.0,
                original_context_length=int(values["max_position_embeddings"]),
                beta_fast=1.0,
                beta_slow=1.0,
                mscale=1.0,
                mscale_all_dim=1.0,
            )
        scaling = _required_dict(values, "rope_scaling")
        if scaling.get("type") != "yarn":
            raise ValueError("Kimi requires YaRN rotary scaling")
        result = cls(
            theta=float(values["rope_theta"]),
            factor=float(scaling["factor"]),
            original_context_length=int(scaling["original_max_position_embeddings"]),
            beta_fast=float(scaling["beta_fast"]),
            beta_slow=float(scaling["beta_slow"]),
            mscale=float(scaling.get("mscale", 1.0)),
            mscale_all_dim=float(scaling.get("mscale_all_dim", 1.0)),
        )
        if (
            min(
                result.theta,
                result.factor,
                result.original_context_length,
                result.beta_fast,
                result.beta_slow,
                result.mscale,
                result.mscale_all_dim,
            )
            <= 0
        ):
            raise ValueError("Kimi YaRN values must be positive")
        return result

    @property
    def attention_scale_multiplier(self) -> float:
        """Return DeepSeek YaRN's additional softmax multiplier."""

        if self.factor <= 1:
            return 1.0
        scale = 0.1 * __import__("math").log(self.factor) + 1.0
        return scale ** (2 * self.mscale_all_dim)


@dataclass(frozen=True, slots=True)
class KimiK25Config:
    """The common Kimi K2.5/K2.6/K2.7 text-model geometry."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    intermediate_size: int
    moe_intermediate_size: int
    first_dense_layers: int
    num_routed_experts: int
    num_shared_experts: int
    experts_per_token: int
    route_scale: float
    normalize_topk: bool
    rms_norm_eps: float
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    rope: RopeConfig

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        """Parse an NVIDIA ModelOpt Kimi conditional-generation config."""

        if values.get("model_type") != "kimi_k25":
            raise ValueError("the Kimi runner requires a Kimi-K2.5-family config")
        architectures = values.get("architectures")
        if architectures != ["KimiK25ForConditionalGeneration"]:
            raise ValueError("unsupported Kimi target architecture")
        quantization = _required_dict(values, "quantization_config")
        if (
            quantization.get("quant_method") != "modelopt"
            or quantization.get("quant_algo") != "NVFP4"
        ):
            raise ValueError(
                "the Kimi runner supports NVIDIA ModelOpt NVFP4 checkpoints only; "
                "native INT4 checkpoints are not supported"
            )
        groups = _required_dict(quantization, "config_groups")
        group = _required_dict(groups, "group_0")
        weight_quant = _required_dict(group, "weights")
        activation_quant = _required_dict(group, "input_activations")
        for name, config in (
            ("weight", weight_quant),
            ("activation", activation_quant),
        ):
            if (
                config.get("type") != "float"
                or int(config.get("num_bits", 0)) != 4
                or int(config.get("group_size", 0)) != 16
                or bool(config.get("dynamic", True))
            ):
                raise ValueError(f"unsupported Kimi NVFP4 {name} format")
        kv_scheme = _required_dict(quantization, "kv_cache_scheme")
        if kv_scheme.get("type") != "float" or int(kv_scheme.get("num_bits", 0)) != 8:
            raise ValueError("the Kimi target requires an FP8 KV-cache scheme")

        text = _required_dict(values, "text_config")
        if text.get("architectures") != ["DeepseekV3ForCausalLM"]:
            raise ValueError("the Kimi text model must use DeepSeek-V3 MLA")
        if text.get("hidden_act") != "silu":
            raise ValueError("the Kimi runner requires SwiGLU feed-forwards")
        if text.get("scoring_func") != "sigmoid":
            raise ValueError("the Kimi runner requires sigmoid expert routing")
        if text.get("topk_method") != "noaux_tc":
            raise ValueError("the Kimi runner requires noaux_tc expert routing")
        result = cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            q_lora_rank=int(text["q_lora_rank"]),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            intermediate_size=int(text["intermediate_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            first_dense_layers=int(text["first_k_dense_replace"]),
            num_routed_experts=int(text["n_routed_experts"]),
            num_shared_experts=int(text["n_shared_experts"]),
            experts_per_token=int(text["num_experts_per_tok"]),
            route_scale=float(text["routed_scaling_factor"]),
            normalize_topk=bool(text["norm_topk_prob"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            bos_token_id=int(values["bos_token_id"]),
            eos_token_id=int(values["eos_token_id"]),
            pad_token_id=int(values["pad_token_id"]),
            rope=RopeConfig.from_hugging_face(text),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            name: getattr(self, name)
            for name in (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "intermediate_size",
                "moe_intermediate_size",
                "first_dense_layers",
                "num_routed_experts",
                "num_shared_experts",
                "experts_per_token",
                "route_scale",
                "rms_norm_eps",
                "max_position_embeddings",
            )
        }
        if invalid := [name for name, value in positive.items() if value <= 0]:
            raise ValueError(f"Kimi fields must be positive: {invalid}")
        if self.first_dense_layers >= self.num_hidden_layers:
            raise ValueError("Kimi must contain routed MoE layers")
        if self.experts_per_token > self.num_routed_experts:
            raise ValueError("Kimi routes more experts than it owns")
        if any(
            not 0 <= token_id < self.vocab_size
            for token_id in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        ):
            raise ValueError("Kimi special token is outside the vocabulary")

    def validate_tensor_parallel(self, size: int) -> None:
        """Validate every axis partitioned by the native TP=4 layout."""

        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("tensor parallel size must be a positive integer")
        axes = {
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_routed_experts": self.num_routed_experts,
        }
        if invalid := [name for name, value in axes.items() if value % size]:
            raise ValueError(f"Kimi fields are not divisible by TP={size}: {invalid}")


@dataclass(frozen=True, slots=True)
class KimiK25DraftConfig:
    """One-layer BF16 EAGLE3/3.1 MLA draft geometry."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    aux_layer_ids: tuple[int, ...]
    fc_norm: bool
    norm_output: bool
    rope: RopeConfig

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("architectures") != ["Eagle3DeepseekV2ForCausalLM"]:
            raise ValueError("unsupported Kimi speculative architecture")
        if int(values.get("num_hidden_layers", 0)) != 1:
            raise ValueError("the Kimi EAGLE3 draft must contain one layer")
        eagle = _required_dict(values, "eagle_config")
        layer_ids = tuple(
            int(value) for value in eagle["eagle_aux_hidden_state_layer_ids"]
        )
        result = cls(
            hidden_size=int(values["hidden_size"]),
            intermediate_size=int(values["intermediate_size"]),
            num_attention_heads=int(values["num_attention_heads"]),
            q_lora_rank=int(values["q_lora_rank"]),
            kv_lora_rank=int(values["kv_lora_rank"]),
            qk_nope_head_dim=int(values["qk_nope_head_dim"]),
            qk_rope_head_dim=int(values["qk_rope_head_dim"]),
            v_head_dim=int(values["v_head_dim"]),
            vocab_size=int(values["vocab_size"]),
            rms_norm_eps=float(values["rms_norm_eps"]),
            max_position_embeddings=int(values["max_position_embeddings"]),
            aux_layer_ids=layer_ids,
            fc_norm=bool(values.get("fc_norm", False)),
            norm_output=bool(values.get("norm_output", False)),
            rope=RopeConfig.from_hugging_face(values),
        )
        if len(set(layer_ids)) != len(layer_ids) or not layer_ids:
            raise ValueError("Kimi EAGLE auxiliary layers must be unique")
        if result.fc_norm != result.norm_output:
            raise ValueError("Kimi EAGLE3.1 requires both fc_norm and norm_output")
        return result

    @property
    def eagle_version(self) -> EagleVersion:
        """Return the validated draft checkpoint generation."""

        return EagleVersion.EAGLE3_1 if self.fc_norm else EagleVersion.EAGLE3

    def validate_target(self, target: KimiK25Config) -> None:
        """Reject a draft that does not match its target model."""

        shared = (
            "hidden_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "vocab_size",
            "max_position_embeddings",
        )
        if mismatch := [
            name for name in shared if getattr(self, name) != getattr(target, name)
        ]:
            raise ValueError(f"Kimi draft geometry does not match target: {mismatch}")
        if any(
            layer_id < 0 or layer_id >= target.num_hidden_layers
            for layer_id in self.aux_layer_ids
        ):
            raise ValueError("Kimi EAGLE auxiliary layer is outside the target")

    @property
    def target_capture_layer_ids(self) -> tuple[int, ...]:
        """Return zero-based target layers whose outputs feed the draft."""

        # LightSeek's K2.5/K2.6 checkpoints name the *next* layer boundary
        # ([2, 30, 58]), while K2.7 names the produced layer outputs directly
        # ([1, 29, 57]). This is the same compatibility rule used by SGLang's
        # set_eagle3_layers_to_capture implementation.
        if self.aux_layer_ids[0] == 1:
            return self.aux_layer_ids
        return tuple(layer_id - 1 for layer_id in self.aux_layer_ids)


__all__ = ["EagleVersion", "KimiK25Config", "KimiK25DraftConfig", "RopeConfig"]
