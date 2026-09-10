from __future__ import annotations

import copy

import pytest

from resin.models.gemma4.config import Gemma4Nvfp4Config


def gemma4_config_dict() -> dict:
    layer_types = [
        kind
        for _ in range(10)
        for kind in ["sliding_attention"] * 5 + ["full_attention"]
    ]
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "audio_config": None,
        "image_token_id": 258880,
        "video_token_id": 258884,
        "eos_token_id": [1, 106],
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "vocab_size": 262144,
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "num_global_key_value_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "layer_types": layer_types,
            "pad_token_id": 0,
            "final_logit_softcapping": 30.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": True,
            "use_bidirectional_attention": "vision",
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                }
            }
        },
    }


def test_nvidia_gemma4_geometry_has_integral_shared_cache_parents() -> None:
    config = Gemma4Nvfp4Config.from_dict(gemma4_config_dict())

    assert config.local_layer_count == 50
    assert config.global_layer_count == 10
    assert config.history_blocks_per_parent == 10
    assert config.eos_token_ids == (1, 106)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("text_config", "attention_k_eq_v"), False, "unsupported"),
        (("text_config", "hidden_activation"), "silu", "unsupported"),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "group_size",
            ),
            32,
            "matching groups",
        ),
    ],
)
def test_config_rejects_checkpoint_options_the_runner_cannot_execute(
    path: tuple[str, ...], value: object, message: str
) -> None:
    raw = copy.deepcopy(gemma4_config_dict())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        Gemma4Nvfp4Config.from_dict(raw)
