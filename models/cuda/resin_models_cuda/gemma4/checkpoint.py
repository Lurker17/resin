"""Resolve and stream NVIDIA's Gemma 4 NVFP4 checkpoint onto one GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.gemma4.config import Gemma4Nvfp4Config
from resin_models_cuda.gemma4.model import Gemma4ForConditionalGeneration

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
    selected = DEFAULT_REVISION if revision is None and str(model) == DEFAULT_MODEL else revision
    return resolve_model(model, selected, _download_model)


def _assign_tensor(model: torch.nn.Module, key: str, tensor: torch.Tensor, device) -> None:
    module_path, name = key.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError as error:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}") from error
    if name in module._parameters and module._parameters[name] is not None:
        existing = module._parameters[name]
        is_parameter = True
    elif name in module._buffers and module._buffers[name] is not None:
        existing = module._buffers[name]
        is_parameter = False
    else:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}")
    if tuple(existing.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {key!r}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(existing.shape)}"
        )
    moved = tensor.to(device=device)
    if is_parameter:
        module._parameters[name] = torch.nn.Parameter(moved, requires_grad=False)
    else:
        module._buffers[name] = moved


def build_meta_model(
    raw_config: dict[str, Any],
) -> tuple[Gemma4ForConditionalGeneration, Gemma4Nvfp4Config]:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    config = Gemma4Nvfp4Config.from_dict(raw_config)
    transformers_config = Gemma4Config.from_dict(raw_config)
    transformers_config.vision_config._attn_implementation = "sdpa"
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(config, transformers_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, config


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
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
            _assign_tensor(model, key, reader.read(path, key), target)
    model.materialize_buffers(target)
    model.eval()
    return model, config, raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "build_meta_model",
    "load_model",
    "resolve_model_path",
]
