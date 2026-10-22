"""Stream checkpoint-shaped tensors into prepared V4.1 attention/EP shards."""

from pathlib import Path

import torch

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.deepseek_v4.weights import (
    _coerce_for_target,
    _shard_routed_expert,
    tensor_parallel_axis,
)


def _attention_shard(tensor, name, rank, size):
    axis = (
        0
        if name.endswith(("markov_head.embed.weight", "markov_head.head.weight"))
        else tensor_parallel_axis(name)
    )
    if axis is None or size == 1:
        return tensor
    if tensor.shape[axis] % size:
        raise ValueError(f"{name} does not divide attention TP={size}")
    width = tensor.shape[axis] // size
    return tensor.narrow(axis, rank * width, width)


def _dequantize_output(weight, scales):
    """The published converter materializes grouped wo_a in BF16."""
    return (
        (
            weight.float().unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))
            * scales.float()[:, None, :, None]
        )
        .flatten(2, 3)
        .flatten(0, 1)
        .bfloat16()
    )


def load_weights(model_path, model, *, rank, tensor_parallel_size, device):
    if not 0 <= rank < tensor_parallel_size:
        raise ValueError("attention rank is outside its group")
    files = indexed_tensor_files(Path(model_path), require_index=True)
    targets = dict(model.state_dict())
    if missing := set(targets) - files.keys():
        raise ValueError(f"checkpoint is missing V4.1 tensors: {sorted(missing)[:8]}")
    count = total = 0
    with SafetensorReader() as reader, torch.no_grad():
        for name in sorted(targets, key=lambda key: (files[key].name, key)):
            tensor = reader.read(files[name], name)
            target = targets[name]
            if ".engram.embed." in name:
                width = (
                    tensor.shape[0] + tensor_parallel_size - 1
                ) // tensor_parallel_size
                tensor = tensor[rank * width : (rank + 1) * width]
                # Last row-shard is padded exactly as in the official converter.
                if tensor.shape[0] < width:
                    tensor = torch.cat(
                        (
                            tensor,
                            tensor.new_full(
                                (width - tensor.shape[0], tensor.shape[1]),
                                1 if name.endswith("scale") else 0,
                            ),
                        )
                    )
            else:
                tensor = _attention_shard(tensor, name, rank, tensor_parallel_size)
                tensor = _shard_routed_expert(tensor, name, model)
            if name.endswith(".wo_a.weight"):
                scale_name = name.removesuffix("weight") + "scale"
                scale = _attention_shard(
                    reader.read(files[scale_name], scale_name),
                    scale_name,
                    rank,
                    tensor_parallel_size,
                )
                tensor = _dequantize_output(tensor, scale)
            tensor = _coerce_for_target(tensor, target)
            target.copy_(tensor.to(device=device))
            total += target.numel() * target.element_size()
            count += 1
    return {"tensors": count, "bytes": total}
