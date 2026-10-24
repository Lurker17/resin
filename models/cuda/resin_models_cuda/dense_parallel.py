"""Intermediate-channel tensor parallelism for dense gated MLPs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from resin.backend.cuda.parallel import TensorParallelContext, TensorShard
from resin_models_cuda.nvfp4 import Nvfp4LinearWeights


class TensorParallelSum(nn.Module):
    """Sum local partials before the model applies residuals or normalization."""

    def __init__(self, local: nn.Module, parallel: TensorParallelContext):
        super().__init__()
        self.local = local
        self.parallel = parallel

    def forward(self, hidden: Tensor) -> Tensor:
        return self.parallel.sum(self.local(hidden))


def _slice(tensor: Tensor, axis: int, shard: TensorShard) -> Tensor:
    # A contiguous row view still retains the whole checkpoint allocation.
    return tensor.narrow(axis, shard.start, shard.size).clone(
        memory_format=torch.contiguous_format
    )


def shard_linear(
    source: nn.Linear | Nvfp4LinearWeights,
    shard: TensorShard,
    *,
    output: bool,
    rank: int,
) -> nn.Module:
    """Slice a checkpoint projection; assign an input-sharded bias only to rank zero."""
    axis = 0 if output else 1
    extent = source.out_features if output else source.in_features
    if not 0 <= shard.start < shard.stop <= extent:
        raise ValueError("linear shard must be nonempty and within the projection axis")
    if (
        isinstance(source, Nvfp4LinearWeights)
        and not output
        and (shard.start % source.group_size or shard.stop % source.group_size)
    ):
        raise ValueError(
            "NVFP4 input shard boundaries must align to quantization groups"
        )
    in_features = source.in_features if output else shard.size
    out_features = shard.size if output else source.out_features
    if isinstance(source, nn.Linear):
        linear = nn.Linear(
            in_features,
            out_features,
            bias=source.bias is not None,
            device="meta",
            dtype=source.weight.dtype,
        )
        linear.weight = nn.Parameter(
            _slice(source.weight, axis, shard),
            requires_grad=source.weight.requires_grad,
        )
        if source.bias is not None:
            bias = (
                _slice(source.bias, 0, shard)
                if output
                else (
                    source.bias.clone() if rank == 0 else torch.zeros_like(source.bias)
                )
            )
            linear.bias = nn.Parameter(bias, requires_grad=source.bias.requires_grad)
    else:
        with torch.device("meta"):
            linear = Nvfp4LinearWeights(in_features, out_features, source.group_size)
        weight_shard = (
            shard if output else TensorShard(shard.start // 2, shard.stop // 2)
        )
        scale_shard = (
            shard
            if output
            else TensorShard(
                shard.start // source.group_size, shard.stop // source.group_size
            )
        )
        linear.weight = _slice(source.weight, axis, weight_shard)
        linear.weight_scale = _slice(source.weight_scale, axis, scale_shard)
        linear.input_scale = source.input_scale
        linear.weight_scale_2 = source.weight_scale_2
    linear.train(source.training)
    return linear


def shard_dense_mlp(
    module: nn.Module, tensor_parallel: TensorParallelContext
) -> nn.Module:
    """Shard loaded checkpoint tensors, then wrap the original MLP semantics.

    Call after checkpoint assignment and before NVFP4 layout preparation. Only
    gate/up output rows and down input columns are partitioned; attention and
    embedding tensors remain replicated. A down-projection bias belongs to rank
    zero so it is added exactly once by the reduction.
    """
    parallel = tensor_parallel
    if not parallel.enabled:
        return module

    def projection(name: str) -> nn.Linear | Nvfp4LinearWeights:
        value = getattr(module, name, None)
        if not isinstance(value, (nn.Linear, Nvfp4LinearWeights)):
            raise TypeError(
                "dense TP requires unprepared Linear or NVFP4 gate/up/down projections"
            )
        return value

    projections = tuple(
        projection(name) for name in ("gate_proj", "up_proj", "down_proj")
    )
    gate, up, down = projections
    intermediate = gate.out_features
    if (up.in_features, up.out_features, down.in_features, down.out_features) != (
        gate.in_features,
        intermediate,
        intermediate,
        gate.in_features,
    ):
        raise ValueError("dense MLP projections have incompatible dimensions")
    shard = parallel.layout.even_shard(intermediate, name="MLP intermediate_size")
    if any(isinstance(value, Nvfp4LinearWeights) for value in projections):
        # CuTe W4A16 packs output rows in groups of 64; down-projection input
        # shards must also preserve the checkpoint's 16-value quantization groups.
        if shard.size % 64 or any(
            isinstance(value, Nvfp4LinearWeights) and shard.size % value.group_size
            for value in projections
        ):
            raise ValueError(
                "NVFP4 dense TP requires intermediate shards aligned to 64"
            )
    # Validate the complete geometry before replacing any checkpoint module.
    for name, source, output in zip(
        ("gate_proj", "up_proj", "down_proj"),
        projections,
        (True, True, False),
        strict=True,
    ):
        setattr(
            module,
            name,
            shard_linear(source, shard, output=output, rank=parallel.rank),
        )
    if hasattr(module, "intermediate_size"):
        setattr(module, "intermediate_size", shard.size)
    wrapped = TensorParallelSum(module, parallel)
    wrapped.train(module.training)
    return wrapped
