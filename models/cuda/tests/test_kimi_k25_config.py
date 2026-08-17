from __future__ import annotations

import copy

import pytest

from resin_models_cuda.kimi_k25.checkpoint import MODEL_SPECS, model_spec
from resin_models_cuda.kimi_k25.config import (
    EagleVersion,
    KimiK25Config,
    KimiK25DraftConfig,
)


def target_values() -> dict[str, object]:
    rope = {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    return {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "type": "float",
                        "num_bits": 4,
                        "group_size": 16,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 4,
                        "group_size": 16,
                        "dynamic": False,
                    },
                }
            },
            "kv_cache_scheme": {"type": "float", "num_bits": 8},
        },
        "text_config": {
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "vocab_size": 163840,
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "first_k_dense_replace": 1,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "routed_scaling_factor": 2.827,
            "norm_topk_prob": True,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 262144,
            "rope_theta": 50000.0,
            "rope_scaling": rope,
        },
    }


def draft_values(*, eagle31: bool = True) -> dict[str, object]:
    result = {
        "architectures": ["Eagle3DeepseekV2ForCausalLM"],
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 1,
        "num_attention_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 262144,
        "rope_theta": 50000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [2, 30, 58],
        },
    }
    if eagle31:
        result.update(fc_norm=True, norm_output=True)
    return result


def test_nvidia_nvfp4_target_geometry_and_tp() -> None:
    config = KimiK25Config.from_hugging_face(target_values())

    config.validate_tensor_parallel(4)
    assert config.num_routed_experts == 384
    assert config.kv_lora_rank + config.qk_rope_head_dim == 576
    assert config.rope.attention_scale_multiplier > 1.0


def test_native_int4_is_rejected_before_model_allocation() -> None:
    values = copy.deepcopy(target_values())
    values["quantization_config"]["quant_method"] = "compressed-tensors"  # type: ignore[index]
    values["quantization_config"]["quant_algo"] = "INT4"  # type: ignore[index]

    with pytest.raises(ValueError, match="native INT4"):
        KimiK25Config.from_hugging_face(values)


def test_k27_native_long_context_rope() -> None:
    values = target_values()
    values["text_config"]["rope_scaling"] = None  # type: ignore[index]

    config = KimiK25Config.from_hugging_face(values)

    assert config.rope.factor == 1.0
    assert config.rope.original_context_length == 262144
    assert config.rope.attention_scale_multiplier == 1.0


@pytest.mark.parametrize("eagle31", [False, True])
def test_eagle3_and_eagle31_match_the_shared_target(eagle31: bool) -> None:
    target = KimiK25Config.from_hugging_face(target_values())
    draft = KimiK25DraftConfig.from_hugging_face(draft_values(eagle31=eagle31))

    draft.validate_target(target)
    assert draft.fc_norm is eagle31
    assert draft.norm_output is eagle31
    assert draft.eagle_version is (
        EagleVersion.EAGLE3_1 if eagle31 else EagleVersion.EAGLE3
    )
    assert draft.target_capture_layer_ids == (1, 29, 57)


def test_partial_eagle31_normalization_is_rejected() -> None:
    values = draft_values()
    values["norm_output"] = False

    with pytest.raises(ValueError, match="both fc_norm and norm_output"):
        KimiK25DraftConfig.from_hugging_face(values)


def test_k27_direct_aux_layer_numbering_is_preserved() -> None:
    target = KimiK25Config.from_hugging_face(target_values())
    values = draft_values()
    values["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = [1, 29, 57]  # type: ignore[index]
    draft = KimiK25DraftConfig.from_hugging_face(values)

    draft.validate_target(target)
    assert draft.target_capture_layer_ids == (1, 29, 57)


def test_only_current_kimi_releases_are_pinned() -> None:
    assert tuple(MODEL_SPECS) == ("k2.5", "k2.6", "k2.7-code")
    assert model_spec().key == "k2.6"
    assert model_spec("k2.5").eagle_version is EagleVersion.EAGLE3
    assert model_spec("k2.6").eagle_version is EagleVersion.EAGLE3_1
    assert model_spec("k2.7-code").eagle_version is EagleVersion.EAGLE3_1
    with pytest.raises(ValueError, match="must be one of"):
        model_spec("k2")
