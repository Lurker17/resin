"""Stream an official GPT-OSS checkpoint into a materialized model."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import torch

from resin.backend.loading import assign_model_tensor, resolve_dtype
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files

_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\.(?P<name>.+)$"
)


def _identity_tensor(_key: str, tensor: Any) -> Any:
    return tensor


def load_gpt_oss_weights(
    model_path: str | Path,
    model: Any,
    *,
    device: Any,
    num_hidden_layers: int,
    expert_tensor_names: Collection[str],
    install_expert_layer: Callable[[int, Mapping[str, Any]], None],
    transform_tensor: Callable[[str, Any], Any] | None = None,
    dtype: Any = None,
) -> None:
    """Load ordinary tensors and install one packed expert layer at a time."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("GPT-OSS weights must be loaded onto a material device")
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None and not resolved_dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")

    tensor_files = indexed_tensor_files(root)
    transform_tensor = transform_tensor or _identity_tensor
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

    with SafetensorReader() as reader:
        for key in ordinary_keys:
            tensor = reader.read(tensor_files[key], key)
            tensor = transform_tensor(key, tensor)
            assign_model_tensor(
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
                tensor = transform_tensor(key, tensor)
                tensors[name] = tensor.to(device=target)
            install_expert_layer(layer, tensors)


__all__ = ["load_gpt_oss_weights"]
