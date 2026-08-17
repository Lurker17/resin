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

DEFAULT_MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
DEFAULT_REVISION = "554ebba9b5f1b79dc11246341960360e6ef05ef4"

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


def is_modelopt_nvfp4(raw_config: dict[str, Any]) -> bool:
    """Recognize RadixArk's mixed FP8-projection/NVFP4-MLP contract."""

    quantization = raw_config.get("quantization_config") or {}
    groups = quantization.get("config_groups") or {}
    fp8 = groups.get("group_0") or {}
    nvfp4 = groups.get("group_1") or {}
    fp8_weights = fp8.get("weights") or {}
    nvfp4_weights = nvfp4.get("weights") or {}
    return (
        quantization.get("quant_method") == "modelopt"
        and quantization.get("quant_algo") == "MIXED_PRECISION"
        and fp8_weights.get("num_bits") == 8
        and fp8_weights.get("type") == "float"
        and nvfp4_weights.get("num_bits") == 4
        and nvfp4_weights.get("type") == "float"
        and nvfp4_weights.get("group_size") == 16
    )


def build_meta_model(
    raw_config: dict[str, Any], *, include_mtp: bool
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen35Config]:
    config = Qwen35Config.from_hugging_face(raw_config)
    if include_mtp:
        config.require_mtp()
    quantized = is_modelopt_nvfp4(raw_config)
    if raw_config.get("quantization_config") and not quantized:
        raise ValueError("unsupported Qwen3.5 checkpoint quantization contract")
    with use_default_dtype(torch.bfloat16), torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(
            config, include_mtp=include_mtp, quantized=quantized
        )
    return model, config


def _assign_tensor(
    model: torch.nn.Module, key: str, tensor: torch.Tensor, device: torch.device
) -> None:
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
    "is_modelopt_nvfp4",
    "load_config",
    "load_model",
    "resolve_model_path",
]
