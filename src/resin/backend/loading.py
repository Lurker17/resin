"""Lazy, framework-neutral model loading helpers."""

from __future__ import annotations

import glob
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_FILES = ("config.json", "model*.safetensors", "model.safetensors.index.json")


def require_torch(runner_name: str):
    try:
        import torch
    except ImportError as error:
        raise ImportError(f"{runner_name} requires PyTorch") from error
    return torch


def default_device(torch, *, require_cuda: bool = False) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            "this backend requires CUDA; pass device='cpu' only for explicit "
            "test or reference execution"
        )
    return "cpu"


def resolve_dtype(torch, dtype):
    if dtype is None or not isinstance(dtype, str):
        return dtype
    resolved = getattr(torch, dtype.removeprefix("torch."), None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype!r}")
    return resolved


def resolve_model(
    model: str | Path,
    revision: str | None,
    download: Callable[[str, str | None], Path],
) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        raise ValueError(f"model path is not a directory: {candidate}")
    return download(str(model), revision)


def read_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    return json.loads(config_path.read_text())


def load_safetensor_shards(model_path: Path, *, device, backend: str) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError(
            f"loading {backend} weights requires safetensors; install safetensors"
        ) from error

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"no model safetensors found in {model_path}")
    weights: dict[str, Any] = {}
    for weight_file in weight_files:
        shard = load_file(weight_file, device=str(device))
        duplicates = weights.keys() & shard.keys()
        if duplicates:
            raise ValueError(f"duplicate tensors across weight shards: {duplicates}")
        weights.update(shard)
    return weights
