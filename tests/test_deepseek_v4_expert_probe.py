import json

import torch
from safetensors.torch import save_file

from deploy.deepseek_v4_expert_probe import _checkpoint_weights


def test_checkpoint_probe_preserves_packed_bytes_and_default_scale_dtype(tmp_path):
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference/config.json").write_text(
        json.dumps(
            {
                "expert_dtype": "fp4",
                "n_activated_experts": 2,
                "n_routed_experts": 2,
            }
        )
    )
    tensors = {}
    for expert in range(2):
        for projection, shape in (("w1", (32, 32)), ("w2", (64, 16)), ("w3", (32, 32))):
            name = f"layers.0.ffn.experts.{expert}.{projection}"
            tensors[f"{name}.weight"] = torch.full(
                shape, expert + int(projection[-1]), dtype=torch.int8
            )
            tensors[f"{name}.scale"] = torch.full(
                (shape[0], shape[1] // 16), 127 + expert, dtype=torch.uint8
            )
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )

    weights, _, source = _checkpoint_weights(tmp_path, 2, "cpu")

    assert source["layer"] == "layers.0.ffn.experts"
    assert weights.up_gate.dtype == torch.float4_e2m1fn_x2
    assert weights.up_gate_scales.dtype == torch.float8_e8m0fnu
    for expert in range(2):
        for offset, projection in ((0, "w3"), (32, "w1")):
            prefix = f"layers.0.ffn.experts.{expert}.{projection}"
            torch.testing.assert_close(
                weights.up_gate[expert, offset : offset + 32].view(torch.int8),
                tensors[f"{prefix}.weight"],
            )
        torch.testing.assert_close(
            weights.down[expert].view(torch.int8),
            tensors[f"layers.0.ffn.experts.{expert}.w2.weight"],
        )
        assert (weights.down_scales[expert].view(torch.uint8) == 127 + expert).all()
