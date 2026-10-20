"""Resolve and stream NVIDIA's Gemma 4 NVFP4 checkpoint onto one GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resin.backend.loading import (
    MODEL_FILES,
    assign_model_tensor,
    read_config,
    resolve_model,
)
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.torch import use_default_dtype
from resin.models.gemma4.config import Gemma4Nvfp4Config
from resin_models_cuda.gemma4.model import Gemma4ForConditionalGeneration
from resin_models_cuda.nvfp4 import prepare_nvfp4_modules, select_nvfp4_backend

DEFAULT_MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"
DEFAULT_REVISION = "4135a98a9b728a548947683219633b25682223ac"

PROCESSOR_FILES = (
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*MODEL_FILES, *PROCESSOR_FILES],
        )
    )


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    selected = (
        DEFAULT_REVISION
        if revision is None and str(model) == DEFAULT_MODEL
        else revision
    )
    return resolve_model(model, selected, _download_model)


def build_meta_model(
    raw_config: dict[str, Any],
) -> tuple[Gemma4ForConditionalGeneration, Gemma4Nvfp4Config]:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    config = Gemma4Nvfp4Config.from_dict(raw_config)
    transformers_config = Gemma4Config.from_dict(raw_config)
    transformers_config.vision_config._attn_implementation = "sdpa"
    with use_default_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(config, transformers_config)
    return model, config


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    nvfp4_backend: str = "auto",
) -> tuple[Gemma4ForConditionalGeneration, Gemma4Nvfp4Config, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_model(raw_config)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Gemma 4 NVFP4 checkpoint can only be loaded on CUDA")
    model.to_empty(device=target)

    tensor_files = indexed_tensor_files(root, require_index=True)
    expected = set(model.state_dict())
    actual = set(tensor_files)
    ignored_missing = {
        "model.vision_tower.encoder.rotary_emb.inv_freq",
        "model.vision_tower.encoder.rotary_emb.original_inv_freq",
    }
    if missing := expected - actual - ignored_missing:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := actual - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            assign_model_tensor(model, key, reader.read(path, key), device=target)
    model.materialize_buffers(target)
    model.prepare_attention()
    model = prepare_nvfp4_modules(
        model, select_nvfp4_backend(nvfp4_backend, torch.backends.cudnn.version())
    )
    model.eval()
    return model, config, raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "build_meta_model",
    "load_model",
    "resolve_model_path",
]
