"""Materialize the original Gemma 4 E2B/E4B BF16 text backbones."""

from __future__ import annotations

from pathlib import Path

import torch

from resin.backend.loading import assign_model_tensor, read_config
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.torch import use_default_dtype
from resin.models.gemma4.edge_config import Gemma4EdgeConfig

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def load_edge_model(model_path: str | Path, *, device, **_kwargs):
    from resin_models_cuda.gemma4.edge_model import Gemma4EdgeForCausalLM

    root = Path(model_path).expanduser().resolve()
    raw = read_config(root)
    config = Gemma4EdgeConfig.from_dict(raw)
    with use_default_dtype(torch.bfloat16), torch.device("meta"):
        model = Gemma4EdgeForCausalLM(config)
    tensors = indexed_tensor_files(root)
    expected = set(model.state_dict())
    text_tensors = {key for key in tensors if key.startswith("model.language_model.")}
    # Audio/vision encoders are intentionally not materialized in this text
    # runner; they have no effect on language-only forward passes.
    if missing := expected - text_tensors:
        raise ValueError(
            f"E2B checkpoint is missing text tensors: {sorted(missing)[:8]}"
        )
    unused_shared = {
        f"model.language_model.layers.{index}.self_attn.{suffix}"
        for index in range(config.num_kv_producer_layers, config.num_hidden_layers)
        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight")
    }
    if unexpected := text_tensors - expected - unused_shared:
        raise ValueError(
            f"E2B checkpoint has unexpected text tensors: {sorted(unexpected)[:8]}"
        )
    with SafetensorReader() as reader:
        for key in sorted(expected):
            assign_model_tensor(
                model, key, reader.read(tensors[key], key), device=device
            )
    model.materialize_buffers(torch.device(device))
    model.prepare_attention()
    model.eval()
    return model, config, raw
