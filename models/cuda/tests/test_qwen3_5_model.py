from __future__ import annotations

import torch
from test_qwen3_5_config import _qwen38

from resin_models_cuda.qwen3_5.checkpoint import _select_tensor_files
from resin_models_cuda.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5.model import (
    Qwen35ForConditionalGeneration,
    Qwen35GatedDeltaNet,
    Qwen35RMSNorm,
)


def _tiny_config() -> Qwen35Config:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    return Qwen35Config.from_hugging_face(values)


def test_qwen_offset_rms_norm_uses_one_plus_checkpoint_weight() -> None:
    norm = Qwen35RMSNorm(4, 1e-6)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        norm.weight.fill_(0.5)
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(hidden), expected * 1.5)


def test_delta_net_chunk_matches_recurrent_steps() -> None:
    torch.manual_seed(8)
    module = Qwen35GatedDeltaNet(_tiny_config())
    hidden = torch.randn(2, 6, 32)

    expected, expected_recurrent, expected_convolution = module(hidden, None, None)
    recurrent = None
    convolution = None
    pieces = []
    for step in range(hidden.shape[1]):
        output, recurrent, convolution = module(
            hidden[:, step : step + 1], recurrent, convolution
        )
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recurrent, expected_recurrent, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(convolution, expected_convolution)


def test_target_and_optional_mtp_names_match_checkpoint() -> None:
    model = Qwen35ForConditionalGeneration(_tiny_config(), include_mtp=True)
    names = set(model.state_dict())

    assert "model.language_model.embed_tokens.weight" in names
    assert "model.language_model.layers.0.linear_attn.A_log" in names
    assert "model.language_model.layers.3.self_attn.q_proj.weight" in names
    assert "lm_head.weight" in names
    assert "mtp.fc.weight" in names
    assert "mtp.layers.0.self_attn.q_proj.weight" in names
    assert not any("mtp.embed_tokens" in name for name in names)


def test_native_text_model_matches_transformers_reference() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel as ReferenceModel,
    )

    values = _qwen38()["text_config"]  # type: ignore[index]
    values.update(  # type: ignore[union-attr]
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    config = Qwen35Config.from_hugging_face(values)  # type: ignore[arg-type]
    native = Qwen35ForConditionalGeneration(config, include_mtp=False).language_model
    reference = ReferenceModel(Qwen3_5TextConfig.from_dict(values))  # type: ignore[arg-type]
    reference.load_state_dict(native.state_dict())
    token_ids = torch.tensor([[1, 7, 12, 4]])
    positions = torch.arange(4)[None]

    with torch.no_grad():
        actual = native(token_ids, positions)[0]
        expected = reference(
            token_ids,
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_text_loader_ignores_vision_and_optionally_mtp() -> None:
    target = {
        "model.language_model.embed_tokens.weight": __file__,
        "lm_head.weight": __file__,
        "mtp.fc.weight": __file__,
        "model.visual.blocks.0.attn.qkv.weight": __file__,
    }
    selected = _select_tensor_files(
        target,  # type: ignore[arg-type]
        {"model.language_model.embed_tokens.weight", "lm_head.weight"},
        include_mtp=False,
    )
    assert set(selected) == {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
