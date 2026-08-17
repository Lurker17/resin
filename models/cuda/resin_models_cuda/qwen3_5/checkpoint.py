"""Resolve Qwen3.5-family checkpoints without loading their vision tower."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.torch import use_default_dtype
from resin_models_cuda.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5.model import Qwen3_5ForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

TOKENIZER_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*MODEL_FILES, *TOKENIZER_FILES],
        )
    )


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    """Resolve a local directory or download a pinned Hub checkpoint."""

    selected = (
        DEFAULT_REVISION
        if revision is None and str(model) == DEFAULT_MODEL
        else revision
    )
    return resolve_model(model, selected, _download_model)


def load_config(model_path: str | Path) -> tuple[Qwen35Config, dict]:
    raw_config = read_config(Path(model_path))
    return Qwen35Config.from_hugging_face(raw_config), raw_config


def build_meta_model(
    raw_config: dict[str, Any], *, include_mtp: bool
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen35Config]:
    config = Qwen35Config.from_hugging_face(raw_config)
    if include_mtp:
        config.require_mtp()
    with use_default_dtype(torch.bfloat16), torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config, include_mtp=include_mtp)
    return model, config


def _assign_tensor(
    model: torch.nn.Module, key: str, tensor: torch.Tensor, device: torch.device
) -> None:
    module_path, name = key.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError as error:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}") from error
    parameter = module._parameters.get(name)
    if parameter is None:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}")
    if tuple(parameter.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {key!r}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(parameter.shape)}"
        )
    module._parameters[name] = torch.nn.Parameter(
        tensor.to(device=device), requires_grad=False
    )


def _select_tensor_files(
    tensor_files: dict[str, Path], expected: set[str], *, include_mtp: bool
) -> dict[str, Path]:
    checkpoint_scope = {
        key
        for key in tensor_files
        if key.startswith("model.language_model.")
        or key.startswith("lm_head.")
        or (include_mtp and key.startswith("mtp."))
    }
    if missing := expected - checkpoint_scope:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := checkpoint_scope - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")
    return {key: tensor_files[key] for key in expected}


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    include_mtp: bool,
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen35Config, dict[str, Any]]:
    """Stream the text target and, when selected, its bundled MTP weights."""

    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_model(raw_config, include_mtp=include_mtp)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Qwen3.5 CUDA runner can only load weights on CUDA")
    model.to_empty(device=target)

    expected = set(model.state_dict())
    tensor_files = _select_tensor_files(
        indexed_tensor_files(root, require_index=True),
        expected,
        include_mtp=include_mtp,
    )
    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            _assign_tensor(model, key, reader.read(path, key), target)
    model.language_model.materialize_buffers(target)
    model.eval()
    return model, config, raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "build_meta_model",
    "load_config",
    "load_model",
    "resolve_model_path",
]
