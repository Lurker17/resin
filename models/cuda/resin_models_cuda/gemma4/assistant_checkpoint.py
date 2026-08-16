"""Resolve and stream Google's Gemma 4 31B assistant checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from resin.backend.loading import read_config, resolve_model
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.gemma4.assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
)
from resin_models_cuda.gemma4.checkpoint import _assign_tensor

DEFAULT_ASSISTANT = "google/gemma-4-31B-it-assistant"
DEFAULT_ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"


def _download_assistant(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )


def resolve_assistant_path(model: str | Path, revision: str | None = None) -> Path:
    selected = (
        DEFAULT_ASSISTANT_REVISION
        if revision is None and str(model) == DEFAULT_ASSISTANT
        else revision
    )
    return resolve_model(model, selected, _download_assistant)


def build_meta_assistant(
    raw_config: dict,
) -> tuple[Gemma4AssistantForCausalLM, Gemma4AssistantConfig]:
    config = Gemma4AssistantConfig.from_dict(raw_config)
    with torch.device("meta"):
        model = Gemma4AssistantForCausalLM(config)
    return model, config


def load_assistant(
    model_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[Gemma4AssistantForCausalLM, Gemma4AssistantConfig]:
    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_assistant(raw_config)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Gemma 4 assistant checkpoint can only be loaded on CUDA")
    model.to_empty(device=target)

    tensor_files = indexed_tensor_files(root)
    expected = set(model.state_dict())
    actual = set(tensor_files)
    if missing := expected - actual:
        raise ValueError(
            f"assistant checkpoint is missing tensors: {sorted(missing)[:8]}"
        )
    if unexpected := actual - expected:
        raise ValueError(
            f"assistant checkpoint has unexpected tensors: {sorted(unexpected)[:8]}"
        )
    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            _assign_tensor(model, key, reader.read(path, key), target)
    model.materialize_buffers(target)
    model.eval()
    return model, config


__all__ = [
    "DEFAULT_ASSISTANT",
    "DEFAULT_ASSISTANT_REVISION",
    "build_meta_assistant",
    "load_assistant",
    "resolve_assistant_path",
]
