from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from resin.models.inkling.config import InklingConfig, InklingDSparkConfig


def _target() -> dict[str, Any]:
    local = [index for index in range(42) if index not in (5, 11, 17, 23, 29, 35, 41)]
    return {
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model",
        "eos_token_id": 200006,
        "text_config": {
            "model_max_length": 1_048_576,
            "torch_dtype": "bfloat16",
            "hidden_size": 4096,
            "num_hidden_layers": 42,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "d_rel": 16,
            "rel_extent": 1024,
            "q_bias": False,
            "o_bias": False,
            "log_scaling_n_floor": 128000,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "use_embed_norm": True,
            "local_layer_ids": local,
            "dense_mlp_idx": 2,
            "use_sconv": True,
            "sconv_kernel_size": 4,
            "logits_mup_width_multiplier": 16.0,
            "sliding_window_size": 512,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "shared_expert_sink": True,
            "dense_intermediate_size": 16384,
            "intermediate_size": 2048,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "gate_activation": "sigmoid",
            "norm_after_topk": True,
            "use_global_scale": True,
        },
        "vision_config": {
            "vision_encoder_type": "hmlp",
            "decoder_dmodel": 4096,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4,
            "use_vision_norm": True,
        },
    }


def _draft() -> dict[str, Any]:
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "hidden_act": "silu",
        "use_sliding_window": False,
        "confidence_head_with_markov": True,
        "block_size": 7,
        "dflash_config": {
            "mask_token_id": 200064,
            "target_layer_ids": [1, 6, 12, 17, 23, 28, 34, 39],
        },
        "vocab_size": 201024,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 6,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1_048_576,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "layer_types": ["full_attention"] * 6,
        "rope_theta": 8_000_000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 128.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    }


def test_pinned_inkling_and_dspark_geometry() -> None:
    target = InklingConfig.from_hugging_face(_target())
    draft = InklingDSparkConfig.from_hugging_face(_draft())
    draft.validate_target(target)
    assert target.local_layer_count == 35
    assert target.global_layer_ids == (5, 11, 17, 23, 29, 35, 41)
    assert target.vision_scales[-1] == (2, 40, 40, 9600)
    assert draft.checkpoint_gamma == 7
    assert draft.target_layer_ids == (1, 6, 12, 17, 23, 28, 34, 39)
    assert draft.mask_token_id < target.vocab_size
    assert draft.yarn.attention_factor == pytest.approx(1.0 + 0.1 * math.log(128.0))


def test_draft_mask_token_is_checked_against_padded_embedding_rows() -> None:
    values = copy.deepcopy(_draft())
    values["dflash_config"]["mask_token_id"] = 201024  # type: ignore[index]
    draft = InklingDSparkConfig.from_hugging_face(values)
    with pytest.raises(ValueError, match="mask token"):
        draft.validate_target(InklingConfig.from_hugging_face(_target()))


def test_unsupported_target_gate_semantics_are_rejected() -> None:
    values = copy.deepcopy(_target())
    values["text_config"]["norm_after_topk"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="unsupported Inkling options"):
        InklingConfig.from_hugging_face(values)
