"""The numerical diagnostic must distinguish nibble decoding and TP rounding."""

from types import SimpleNamespace

import torch

from deploy.gpt_oss_expert_probe import _dense_outputs, _dequantize


def test_mxfp4_dense_oracle_decodes_low_high_nibbles_and_e8m0_scales():
    packed = (
        torch.tensor(
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2,
            dtype=torch.uint8,
        )
        .reshape(1, 1, 16)
        .expand(2, 1, 16)
    )
    scales = torch.tensor([[127], [128]], dtype=torch.uint8)
    expected = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6] * 2
    )
    actual = _dequantize(packed, scales)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], 2 * expected)


def test_dense_tp_oracle_adds_expert_bias_once_before_routing_weights():
    args = SimpleNamespace(num_local_experts=1, intermediate_size=64, swiglu_limit=7.0)
    raw = {
        "gate_up_proj_blocks": torch.zeros(1, 128, 1, 16, dtype=torch.uint8),
        "gate_up_proj_scales": torch.full((1, 128, 1), 127, dtype=torch.uint8),
        "gate_up_proj_bias": torch.zeros(1, 128),
        "down_proj_blocks": torch.zeros(1, 32, 2, 16, dtype=torch.uint8),
        "down_proj_scales": torch.full((1, 32, 2), 127, dtype=torch.uint8),
        "down_proj_bias": torch.full((1, 32), 3.25),
    }
    hidden = torch.ones(2, 32, dtype=torch.bfloat16)
    indices = torch.zeros(2, 2, dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.125, 0.25]])
    outputs = _dense_outputs(raw, hidden, indices, weights, args, tensor_sizes=(2,))
    expected = torch.full((2, 32), 3.25) * weights.sum(1, keepdim=True)
    assert set(outputs) == {"fp32_fc2", "bf16_fc2", "bf16_fc2_tp2"}
    for output in outputs.values():
        torch.testing.assert_close(output, expected)
