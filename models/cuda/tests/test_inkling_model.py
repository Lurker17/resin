from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import pytest
import torch
from test_inkling_config import _target

from resin.models.inkling.config import InklingConfig, YarnConfig
from resin_models_cuda.inkling.dspark import precompute_yarn_rope
from resin_models_cuda.inkling.model import (
    DecoderLayer,
    InklingVisionEncoderLayer,
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
@pytest.mark.parametrize("tokens", [1, 2, 127, 4096])
def test_cuda_prefill_convolution_matches_fp32_reference(tokens: int) -> None:
    torch.manual_seed(17)
    convolution = ShortConvolution(1024).cuda()
    with torch.no_grad():
        convolution.weight.uniform_(-0.25, 0.25)
    hidden = torch.randn(2, tokens, 1024, dtype=torch.bfloat16, device="cuda")
    initial = torch.randn(2, 3, 1024, dtype=torch.bfloat16, device="cuda")
    history = torch.cat((initial, hidden), dim=1)
    expected = torch.nn.functional.conv1d(
        history.transpose(1, 2).float(), convolution.weight, groups=1024
    ).transpose(1, 2)
    expected = (expected + hidden.float()).bfloat16()
    actual, state = convolution(hidden, initial)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.00003)
    torch.testing.assert_close(state, history[:, -3:], rtol=0, atol=0)


def test_prefill_ffn_shared_output_preserves_convolution_state_and_dspark_taps():
    from test_cuda_runner_graph_policy import cpu_prefill_regions

    class Attention(torch.nn.Module):
        def forward(self, hidden, *args, **_kwargs):
            k_state, v_state = args[-2:]
            keys, values = hidden[..., :1024], hidden[..., :1024] * 0.5
            return (
                hidden * 0.25,
                torch.cat((k_state, keys), dim=1)[:, -3:],
                torch.cat((v_state, values), dim=1)[:, -3:],
                keys,
                values,
            )

    torch.manual_seed(29)
    layers = []
    for _ in range(2):
        # Keep real layer/convolution math without allocating checkpoint-sized MLPs.
        layer = object.__new__(DecoderLayer)
        torch.nn.Module.__init__(layer)
        setattr(layer, "attn_norm", torch.nn.Identity())
        setattr(layer, "mlp_norm", torch.nn.Identity())
        setattr(layer, "attn", Attention())
        layer.attn_sconv = ShortConvolution(4096)
        layer.mlp_sconv = ShortConvolution(4096)
        setattr(layer, "mlp", torch.nn.SiLU())
        layer.prefill_ffn = None
        with torch.no_grad():
            layer.attn_sconv.weight.uniform_(-0.1, 0.1)
            layer.mlp_sconv.weight.uniform_(-0.1, 0.1)
        layers.append(layer)

    def forward(hidden, positions, states):
        taps, next_states = [], []
        for layer, state in zip(layers, states, strict=True):
            hidden, tap, next_state, _ = layer(
                hidden,
                positions,
                positions[:, -1] + 1,
                7,
                None,
                None,
                state,
                prefill=True,
                record_state_inputs=False,
            )
            taps.append(tap)
            next_states.append(next_state)
        return hidden, torch.cat(taps, dim=-1), next_states

    hidden = torch.randn(1, 7, 4096, dtype=torch.bfloat16)
    positions = torch.arange(7)[None]
    initial = [torch.zeros(1, 3, 10240, dtype=torch.bfloat16) for _ in layers]
    with torch.no_grad():
        expected, expected_taps, expected_states = forward(hidden, positions, initial)
        runner, callbacks, replays = cpu_prefill_regions(
            (layer.feed_forward for layer in layers), 4096
        )
        for layer, callback in zip(layers, callbacks, strict=True):
            layer.prefill_ffn = callback
        with runner._prefill_graphs():
            first, first_taps, states = forward(
                hidden[:, :3], positions[:, :3], initial
            )
            second, second_taps, states = forward(
                hidden[:, 3:], positions[:, 3:], states
            )

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    torch.testing.assert_close(
        torch.cat((first_taps, second_taps), dim=1), expected_taps
    )
    for state, expected_state in zip(states, expected_states, strict=True):
        torch.testing.assert_close(state, expected_state)
    assert replays == [(4, 1), (4, 2), (4, 1), (4, 2)]


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

    assert [
        tuple(layer.projection.weight.shape)
        for layer in cast(Sequence[InklingVisionEncoderLayer], vision.layers)
    ] == [
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
