from __future__ import annotations

import json

import pytest
import torch
from test_inkling_config import _target

from resin.models.inkling.config import InklingConfig, YarnConfig
from resin_models_cuda.inkling.dspark import precompute_yarn_rope
from resin_models_cuda.inkling.model import (
    InklingVisionModel,
    ShortConvolution,
    select_convolution_state,
)
from resin_models_cuda.inkling.weights import _indexed_target_files


def _cutile_available() -> bool:
    try:
        import cuda.tile as ct
    except ImportError:
        return False
    return torch.cuda.is_available() and hasattr(ct, "Array")


requires_cutile = pytest.mark.skipif(
    not _cutile_available(),
    reason="requires CUDA and cuTile",
)


def test_short_convolution_split_matches_one_pass() -> None:
    torch.manual_seed(4)
    convolution = ShortConvolution(5)
    with torch.no_grad():
        convolution.weight.copy_(torch.randn_like(convolution.weight))
    hidden = torch.randn(2, 9, 5, dtype=torch.bfloat16)
    initial = torch.randn(2, 3, 5, dtype=torch.bfloat16)

    expected, expected_state = convolution(hidden, initial)
    first, first_state = convolution(hidden[:, :4], initial)
    second, actual_state = convolution(hidden[:, 4:], first_state)

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    torch.testing.assert_close(actual_state, expected_state)


@requires_cutile
def test_accepted_prefix_selects_exact_convolution_window() -> None:
    initial = torch.arange(2 * 3 * 3 * 256, device="cuda", dtype=torch.bfloat16)
    initial = initial.reshape(2, 3, 3, 256) % 128
    inputs = torch.arange(2 * 6 * 3 * 256, device="cuda", dtype=torch.bfloat16)
    inputs = 1_000 + inputs.reshape(2, 6, 3, 256) % 128
    processed = torch.tensor([1, 5], device="cuda")
    actual = select_convolution_state(initial, inputs, processed)

    for row, count in enumerate(processed.tolist()):
        history = torch.cat((initial[row].transpose(0, 1), inputs[row]), dim=0)
        expected = history[count : count + 3].transpose(0, 1)
        torch.testing.assert_close(actual[row], expected)


@requires_cutile
def test_all_six_verified_tokens_select_the_final_three_inputs() -> None:
    initial = torch.full((1, 2, 3, 256), -1.0, device="cuda")
    inputs = torch.arange(1 * 6 * 2 * 256, device="cuda").reshape(1, 6, 2, 256)
    out = torch.empty_like(initial)

    selected = select_convolution_state(
        initial,
        inputs,
        torch.tensor([6], device="cuda"),
        out=out,
    )

    assert selected.data_ptr() == out.data_ptr()
    torch.testing.assert_close(selected, inputs[:, -3:].transpose(1, 2))


def test_yarn_rope_applies_its_attention_factor() -> None:
    rope = YarnConfig(
        theta=8_000_000,
        factor=128.0,
        attention_factor=1.5,
        original_context_length=8192,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    cos, sin = precompute_yarn_rope(128, 4, rope)
    torch.testing.assert_close(cos[0], torch.full((64,), 1.5))
    torch.testing.assert_close(sin[0], torch.zeros(64))


def test_vision_hmlp_matches_pinned_checkpoint_shapes() -> None:
    config = InklingConfig.from_hugging_face(_target())
    with torch.device("meta"):
        vision = InklingVisionModel(config)

    assert [tuple(layer.projection.weight.shape) for layer in vision.layers] == [
        (128, 75),
        (320, 512),
        (4800, 5120),
        (4096, 9600),
    ]


def test_text_weight_index_does_not_require_unused_mtp_shard(tmp_path) -> None:
    target_shard = tmp_path / "model-00001-of-00001.safetensors"
    target_shard.touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.llm.embed.weight": target_shard.name,
                    "model.mtp.layers.0.input_proj.weight": "mtp.safetensors",
                }
            }
        )
    )

    files = _indexed_target_files(tmp_path, {"model.llm.embed.weight"})

    assert files == {"model.llm.embed.weight": target_shard}
    with pytest.raises(ValueError, match="missing tensors"):
        _indexed_target_files(tmp_path, {"model.llm.unembed.weight"})
