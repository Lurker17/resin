"""Stream and tensor-parallel shard the official DeepSeek-V4 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def _indexed_tensor_files(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing checkpoint index: {index_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid checkpoint index: {index_path}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index must contain a non-empty weight_map")

    files: dict[str, Path] = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint index contains an invalid tensor name")
        if not isinstance(filename, str):
            raise ValueError("checkpoint index contains an invalid shard name")
        relative = Path(filename)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"invalid checkpoint shard name: {filename!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {path}")
        files[name] = path
    return files


def tensor_parallel_axis(name: str) -> int | None:
    """Return the axis used by DeepSeek's published TP=4 conversion."""

    if name in {"embed.weight", "head.weight"} or name.endswith(".attn_sink"):
        return 0
    if any(
        f".{component}." in name
        for component in ("wq_b", "wo_a", "weights_proj", "markov_w1", "markov_w2")
    ):
        return 0
    if ".wo_b." in name:
        return 1
    return None


def _shard(tensor: Any, name: str, rank: int, size: int):
    axis = tensor_parallel_axis(name)
    if axis is None or size == 1:
        return tensor
    width = tensor.shape[axis]
    if width % size:
        raise ValueError(
            f"checkpoint tensor {name!r} axis {axis} is not divisible by TP={size}"
        )
    local_width = width // size
    return tensor.narrow(axis, rank * local_width, local_width).contiguous()


def _dequantize_wo_a(weight: Any, scale: Any):
    """Match the release's conversion of grouped FP8 ``wo_a``."""

    weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
    weight = weight.float() * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def _coerce_for_target(torch: Any, tensor: Any, target: Any):
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is not None and target.dtype == fp4 and tensor.dtype == torch.int8:
        tensor = tensor.view(fp4)
    elif tensor.dtype != target.dtype:
        tensor = tensor.to(dtype=target.dtype)
    if tuple(tensor.shape) != tuple(target.shape):
        raise ValueError(
            f"checkpoint shape {tuple(tensor.shape)} does not match model "
            f"shape {tuple(target.shape)}"
        )
    return tensor


def load_deepseek_v4_weights(
    model_path: str | Path,
    model: Any,
    *,
    rank: int,
    tensor_parallel_size: int,
    device: Any,
) -> dict[str, int]:
    """Load one TP rank directly from the 48 original safetensor shards.

    This avoids materializing DeepSeek's four converted checkpoint copies.
    Only the experts owned by ``model`` are read; replicated tensors and the
    conventional tensor-parallel axes follow the official ``convert.py``.
    """

    import torch

    if not 0 <= rank < tensor_parallel_size:
        raise ValueError("rank must be in [0, tensor_parallel_size)")
    root = Path(model_path).expanduser().resolve()
    tensor_files = _indexed_tensor_files(root)
    targets = dict(model.state_dict())
    if missing := set(targets) - set(tensor_files):
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")

    ordered = sorted(targets, key=lambda name: (tensor_files[name].name, name))
    current_path: Path | None = None
    context = None
    handle = None
    loaded_bytes = 0
    loaded_tensors = 0
    try:
        for name in ordered:
            path = tensor_files[name]
            if path != current_path:
                if context is not None:
                    context.__exit__(None, None, None)
                context = safe_open(str(path), framework="pt", device="cpu")
                handle = context.__enter__()
                current_path = path

            tensor = _shard(handle.get_tensor(name), name, rank, tensor_parallel_size)
            if name.endswith(".wo_a.weight"):
                scale_name = name.removesuffix("weight") + "scale"
                scale_path = tensor_files.get(scale_name)
                if scale_path != path:
                    raise ValueError("wo_a weight and scale must occupy the same shard")
                scale = _shard(
                    handle.get_tensor(scale_name),
                    scale_name,
                    rank,
                    tensor_parallel_size,
                )
                tensor = _dequantize_wo_a(tensor, scale)
            tensor = _coerce_for_target(torch, tensor, targets[name])
            loaded_bytes += tensor.numel() * tensor.element_size()
            loaded_tensors += 1
            with torch.no_grad():
                targets[name].copy_(tensor.to(device=device))
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    return {"tensors": loaded_tensors, "bytes": loaded_bytes}


__all__ = ["load_deepseek_v4_weights", "tensor_parallel_axis"]
