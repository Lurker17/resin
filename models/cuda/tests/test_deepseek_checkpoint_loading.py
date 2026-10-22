"""Checkpoint copies preserve values without materializing a device-sized source."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from resin_models_cuda.deepseek_v4.weights import load_deepseek_v4_weights
from resin_models_cuda.deepseek_v41.weights import load_weights


@pytest.fixture(params=[load_deepseek_v4_weights, load_weights], ids=["v4", "v41"])
def loader(request):
    return request.param


def checkpoint(tmp_path, *, columns=4):
    values = {
        "layers.0.engram.embed.weight": torch.arange(9 * columns)
        .reshape(9, columns)
        .float(),
        "packed": torch.arange(-4, 4, dtype=torch.int8).reshape(2, 4),
    }
    save_file(values, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in values}})
    )
    return values


def model():
    result = nn.Module()
    layer = nn.Module()
    engram, embedding = nn.Module(), nn.Module()
    embedding.register_buffer("weight", torch.full((9, 4), -1, dtype=torch.bfloat16))
    engram.add_module("embed", embedding)
    layer.add_module("engram", engram)
    result.layers = nn.ModuleList([layer])
    result.register_buffer("packed", torch.empty(2, 4, dtype=torch.float4_e2m1fn_x2))
    return result


def forbid_source_device_copy(monkeypatch):
    original = torch.Tensor.to

    def cast_only(tensor, *args, **kwargs):
        if "device" in kwargs or (args and isinstance(args[0], (str, torch.device))):
            raise AssertionError(
                "checkpoint must copy directly into its existing target"
            )
        result = original(tensor, *args, **kwargs)
        assert result.device.type == "cpu"
        return result

    monkeypatch.setattr(torch.Tensor, "to", cast_only)


def test_loader_copies_cpu_casts_and_fp4_bytes_into_existing_storage(
    loader, tmp_path, monkeypatch
):
    values, target = checkpoint(tmp_path), model()
    pointers = {name: tensor.data_ptr() for name, tensor in target.state_dict().items()}
    forbid_source_device_copy(monkeypatch)
    report = loader(tmp_path, target, rank=0, tensor_parallel_size=1, device="cpu")
    loaded = target.state_dict()
    assert pointers == {name: tensor.data_ptr() for name, tensor in loaded.items()}
    assert torch.equal(
        loaded["layers.0.engram.embed.weight"].float(),
        values["layers.0.engram.embed.weight"],
    )
    assert torch.equal(loaded["packed"].view(torch.int8), values["packed"])
    assert report == {"tensors": 2, "bytes": 80}


def test_loader_rejects_shape_mismatch_before_copy(loader, tmp_path, monkeypatch):
    checkpoint(tmp_path, columns=5)
    target = model()
    forbid_source_device_copy(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint shape.*does not match model"):
        loader(tmp_path, target, rank=0, tensor_parallel_size=1, device="cpu")
    assert target.layers[0].engram.embed.weight.eq(-1).all()
