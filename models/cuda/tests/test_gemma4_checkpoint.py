from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest
import torch
from safetensors.torch import save_file
from test_gemma4_assistant import _raw_config
from test_gemma4_config import gemma4_config_dict
from test_gemma4_parallel import _Parallel

from resin.backend.cuda.parallel import TensorParallelContext
from resin.models.gemma4.assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
)
from resin_models_cuda.gemma4 import assistant_checkpoint, checkpoint


@pytest.mark.parametrize("assistant", [False, True])
def test_checkpoint_shards_on_cpu_before_its_only_device_transfer(
    tmp_path, monkeypatch, assistant
):
    if assistant:
        raw = _raw_config()
        config = replace(
            Gemma4AssistantConfig.from_dict(raw),
            vocab_size=32,
            hidden_size=16,
            backbone_hidden_size=16,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_global_key_value_heads=2,
            head_dim=4,
            global_head_dim=8,
        )

        def build(_raw):
            with torch.device("meta"):
                return Gemma4AssistantForCausalLM(config), config

        monkeypatch.setattr(assistant_checkpoint, "build_meta_assistant", build)
        model, _ = build(raw)
        loader = assistant_checkpoint.load_assistant
    else:
        raw = gemma4_config_dict()
        raw["text_config"].update(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_global_key_value_heads=2,
            head_dim=4,
            global_head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
        )
        raw["vision_config"] = dict(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            position_embedding_size=16,
        )
        model, _ = checkpoint.build_meta_model(raw)
        loader = checkpoint.load_model
        monkeypatch.setattr(checkpoint, "prepare_nvfp4_modules", lambda model, _: model)
        monkeypatch.setattr(checkpoint, "select_nvfp4_backend", lambda *_: "unused")
    weights = {
        name: torch.zeros(value.shape, dtype=value.dtype)
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    transferred = []

    def transfer(module, device):
        assert torch.device(device).type == "cuda"
        tensors = [*module.parameters(), *module.buffers()]
        assert all(tensor.device.type == "cpu" for tensor in tensors)
        language = module.model if assistant else module.language_model
        for layer in language.layers:
            assert layer.self_attn.num_heads == 2
            assert layer.mlp.local.gate_proj.out_features == 64
            assert layer.mlp.local.down_proj.in_features == 64
        transferred.append(True)
        return module

    def premature_allocation(*_args, **_kwargs):
        raise AssertionError("the full checkpoint must not be allocated on CUDA")

    monkeypatch.setattr(type(model), "to", transfer)
    monkeypatch.setattr(type(model), "to_empty", premature_allocation)
    parallel = cast(TensorParallelContext, _Parallel(1, 2))
    loaded = loader(
        tmp_path,
        device="cuda",
        tensor_parallel=parallel,
        attention_parallel=parallel,
    )[0]
    assert transferred == [True]
    assert all(tensor.device.type == "cpu" for tensor in loaded.buffers())
