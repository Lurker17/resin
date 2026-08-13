"""Stream an official GPT-OSS checkpoint into a materialized model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, Self

import torch

from resin.backend.loading import resolve_dtype

_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\.(?P<name>.+)$"
)


def _safe_open():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError(
            "loading GPT-OSS weights requires safetensors; install safetensors"
        ) from error
    return safe_open


def _indexed_tensor_files(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        model_file = root / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"missing checkpoint weights: {model_file}")
        with _safe_open()(str(model_file), framework="pt", device="cpu") as handle:
            return {key: model_file for key in handle.keys()}

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {index_path}: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path.name} must contain a non-empty weight_map")

    tensor_files = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint index contains an invalid tensor name")
        if not isinstance(filename, str):
            raise ValueError("checkpoint index contains an invalid shard name")
        relative = Path(filename)
        if (
            not filename
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"invalid checkpoint shard name: {filename!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {path}")
        tensor_files[key] = path
    return tensor_files


class _TensorReader:
    """Keep at most one memory-mapped shard open while loading tensors."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._context: Any | None = None
        self._handle: Any | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._path = None
        self._context = None
        self._handle = None

    def read(self, path: Path, key: str):
        if path != self._path:
            self.close()
            self._context = _safe_open()(str(path), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._path = path
        return self._handle.get_tensor(key)


def _assign_tensor(model: Any, key: str, tensor: Any, *, device, dtype) -> None:
    module_path, name = key.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError as error:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}") from error

    if name in module._parameters and module._parameters[name] is not None:
        existing = module._parameters[name]
        parameter = True
    elif name in module._buffers and module._buffers[name] is not None:
        existing = module._buffers[name]
        parameter = False
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


def load_gpt_oss_weights(
    model_path: str | Path,
    model: Any,
    *,
    device: Any,
    num_hidden_layers: int,
    expert_tensor_names: Collection[str],
    install_expert_layer: Callable[[int, Mapping[str, Any]], None],
    dtype: Any = None,
) -> None:
    """Load ordinary tensors and install one packed expert layer at a time."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("GPT-OSS weights must be loaded onto a material device")
    resolved_dtype = resolve_dtype(torch, dtype)
    if resolved_dtype is not None and not resolved_dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")

    tensor_files = _indexed_tensor_files(root)
    ordinary_keys = tuple(
        key for key in model.state_dict() if _EXPERT_KEY.fullmatch(key) is None
    )
    expert_names = tuple(expert_tensor_names)
    expected_experts = {
        f"model.layers.{layer}.mlp.experts.{name}"
        for layer in range(num_hidden_layers)
        for name in expert_names
    }
    expected = set(ordinary_keys) | expected_experts
    actual = set(tensor_files)
    if missing := expected - actual:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := actual - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    with _TensorReader() as reader:
        for key in ordinary_keys:
            tensor = reader.read(tensor_files[key], key)
            _assign_tensor(
                model,
                key,
                tensor,
                device=target,
                dtype=resolved_dtype,
            )

        for layer in range(num_hidden_layers):
            tensors = {}
            for name in expert_names:
                key = f"model.layers.{layer}.mlp.experts.{name}"
                tensor = reader.read(tensor_files[key], key)
                tensors[name] = tensor.to(device=target)
            install_expert_layer(layer, tensors)


__all__ = ["load_gpt_oss_weights"]
