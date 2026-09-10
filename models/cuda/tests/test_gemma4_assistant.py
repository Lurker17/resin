from __future__ import annotations

from resin.models.gemma4.assistant import Gemma4AssistantConfig
from resin_models_cuda.gemma4.assistant_checkpoint import build_meta_assistant


def _raw_config() -> dict:
    return {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 5376,
        "use_ordered_embeddings": False,
        "text_config": {
            "model_type": "gemma4_text",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "num_kv_shared_layers": 4,
            "tie_word_embeddings": True,
            "use_double_wide_mlp": False,
            "vocab_size": 262144,
            "hidden_size": 1024,
            "intermediate_size": 8192,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "num_global_key_value_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    }


def test_assistant_config_pins_the_google_31b_geometry() -> None:
    config = Gemma4AssistantConfig.from_dict(_raw_config())

    assert config.backbone_hidden_size == 5376
    assert config.hidden_size == 1024
    assert config.intermediate_size == 8192
    assert config.layer_types[-1] == "full_attention"


def test_assistant_meta_model_matches_checkpoint_tensor_shapes() -> None:
    model, _ = build_meta_assistant(_raw_config())
    state = model.state_dict()

    assert state["model.embed_tokens.weight"].shape == (262144, 1024)
    assert state["pre_projection.weight"].shape == (1024, 10752)
    assert state["post_projection.weight"].shape == (5376, 1024)
    assert state["model.layers.0.self_attn.q_proj.weight"].shape == (8192, 1024)
    assert state["model.layers.3.self_attn.q_proj.weight"].shape == (16384, 1024)
    assert not any("k_proj" in key or "v_proj" in key for key in state)
