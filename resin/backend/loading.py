"""Shared PyTorch model loading helpers."""

from __future__ import annotations

import glob
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

MODEL_FILES = ("config.json", "model*.safetensors", "model.safetensors.index.json")


def default_device(*, require_cuda: bool = False) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            "this backend requires CUDA; pass device='cpu' only for explicit "
            "test or reference execution"
        )
    return "cpu"


def resolve_dtype(dtype):
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
    from safetensors.torch import load_file

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


def materialize_safetensor_model(
    model_path: Path,
    *,
    device,
    dtype,
    backend: str,
    build: Callable[[], Any],
    after_cast: Callable[[Any], None] | None = None,
):
    """Build a meta model, assign safetensors, and finalize inference buffers."""

    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    with torch.device("meta"):
        model = build()
    weights = load_safetensor_shards(model_path, device=load_target, backend=backend)
    model.load_state_dict(weights, strict=True, assign=True)
    model.materialize_buffers(load_target)
    if target != load_target:
        model.to(target)

    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    if after_cast is not None:
        after_cast(model)
    # Module.to(dtype=...) casts floating-point buffers too. Each model's
    # materializer restores computed buffers such as RoPE frequencies.
    model.materialize_buffers(target)
    model.eval()
    return model


def assign_model_tensor(
    model,
    key: str,
    tensor,
    *,
    device,
    dtype=None,
) -> None:
    """Validate and assign one parameter or buffer into a materialized module."""

    module_path, name = key.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError as error:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}") from error
    if name in module._parameters and module._parameters[name] is not None:
        existing, parameter = module._parameters[name], True
    elif name in module._buffers and module._buffers[name] is not None:
        existing, parameter = module._buffers[name], False
    else:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}")
    if tuple(existing.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {key!r}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(existing.shape)}"
        )
    moved = tensor.to(
        device=device,
        dtype=dtype if dtype is not None and tensor.is_floating_point() else None,
    )
    if parameter:
        module._parameters[name] = torch.nn.Parameter(
            moved, requires_grad=existing.requires_grad
        )
    else:
        module._buffers[name] = moved
