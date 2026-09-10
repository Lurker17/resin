"""GPT-OSS MXFP4 expert construction, sharding, and checkpoint adaptation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout, TensorShard
from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernelConfig,
    MoERouting,
)

_CHECKPOINT_TO_STANDARD = {
    "gate_up_proj_blocks": "gate_up",
    "gate_up_proj_scales": "gate_up_scales",
    "gate_up_proj_bias": "gate_up_bias",
    "down_proj_blocks": "down",
    "down_proj_scales": "down_scales",
    "down_proj_bias": "down_bias",
}
EXPERT_CHECKPOINT_NAMES = tuple(_CHECKPOINT_TO_STANDARD)


def _slice_axis(tensor, axis: int, shard: TensorShard):
    selection = [slice(None)] * tensor.ndim
    selection[axis] = shard.as_slice()
    return tensor[tuple(selection)].contiguous()


def shard_model_config(args, tensor_parallel: TensorParallelLayout):
    query_heads = tensor_parallel.even_shard(
        args.num_attention_heads, name="num_attention_heads"
    )
    kv_heads = tensor_parallel.even_shard(
        args.num_key_value_heads, name="num_key_value_heads"
    )
    intermediate = tensor_parallel.aligned_shard(
        args.intermediate_size,
        alignment=32,
        name="intermediate_size",
    )
    return replace(
        args,
        num_attention_heads=query_heads.size,
        num_key_value_heads=kv_heads.size,
        intermediate_size=intermediate.size,
    )


def shard_checkpoint_tensor(
    key: str,
    tensor,
    *,
    args,
    tensor_parallel: TensorParallelLayout,
):
    """Return one rank's canonical contiguous checkpoint shard."""

    if not tensor_parallel.enabled:
        return tensor

    query_heads = tensor_parallel.even_shard(
        args.num_attention_heads, name="num_attention_heads"
    )
    kv_heads = tensor_parallel.even_shard(
        args.num_key_value_heads, name="num_key_value_heads"
    )
    intermediate = tensor_parallel.aligned_shard(
        args.intermediate_size,
        alignment=32,
        name="intermediate_size",
    )
    query_width = TensorShard(
        query_heads.start * args.head_dim,
        query_heads.stop * args.head_dim,
    )
    kv_width = TensorShard(
        kv_heads.start * args.head_dim,
        kv_heads.stop * args.head_dim,
    )

    if key.endswith(".self_attn.q_proj.weight") or key.endswith(
        ".self_attn.q_proj.bias"
    ):
        return _slice_axis(tensor, 0, query_width)
    if key.endswith(".self_attn.k_proj.weight") or key.endswith(
        ".self_attn.k_proj.bias"
    ):
        return _slice_axis(tensor, 0, kv_width)
    if key.endswith(".self_attn.v_proj.weight") or key.endswith(
        ".self_attn.v_proj.bias"
    ):
        return _slice_axis(tensor, 0, kv_width)
    if key.endswith(".self_attn.o_proj.weight"):
        return _slice_axis(tensor, 1, query_width)
    if key.endswith(".self_attn.sinks"):
        return _slice_axis(tensor, 0, query_heads)
    if key.endswith(".self_attn.o_proj.bias"):
        return tensor if tensor_parallel.rank == 0 else torch.zeros_like(tensor)

    expert_name = key.rsplit(".", 1)[-1]
    if expert_name in {
        "gate_up_proj_blocks",
        "gate_up_proj_scales",
        "gate_up_proj_bias",
    }:
        paired = TensorShard(2 * intermediate.start, 2 * intermediate.stop)
        return _slice_axis(tensor, 1, paired)
    if expert_name in {"down_proj_blocks", "down_proj_scales"}:
        grouped = TensorShard(intermediate.start // 32, intermediate.stop // 32)
        return _slice_axis(tensor, 2, grouped)
    if expert_name == "down_proj_bias":
        return tensor if tensor_parallel.rank == 0 else torch.zeros_like(tensor)
    return tensor


def cuda_architecture(device) -> str:
    if device.type != "cuda":
        raise ValueError("packed GPT-OSS MXFP4 execution requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def moe_kernel_config(args, architecture: str) -> MoEKernelConfig:
    return MoEKernelConfig(
        architecture=architecture,
        quantization="mxfp4",
        activation="swiglu",
        top_k=args.num_experts_per_tok,
        num_experts=args.num_local_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        group_size=32,
        expert_bias=True,
        activation_alpha=1.702,
        activation_beta=1.0,
        activation_limit=args.swiglu_limit,
        gate_up_layout="interleaved_gate_up",
    )


def kernel_expert_factory(kernel, config: MoEKernelConfig):
    class KernelExperts(torch.nn.Module):
        storage_format = config.quantization

        def __init__(self, args) -> None:
            super().__init__()
            actual = (
                args.num_local_experts,
                args.hidden_size,
                args.intermediate_size,
                args.num_experts_per_tok,
            )
            expected = (
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
                config.top_k,
            )
            if actual != expected:
                raise ValueError(
                    f"expert module dimensions {actual} do not match kernel {expected}"
                )
            self._prepared_weights = None

        def install_prepared_weights(self, weights) -> None:
            if self._prepared_weights is not None:
                raise RuntimeError("packed expert weights are already installed")
            self._prepared_weights = weights

        @property
        def is_prepared(self) -> bool:
            return self._prepared_weights is not None

        def forward(self, hidden, indices, weights, logits):
            if self._prepared_weights is None:
                raise RuntimeError("packed expert weights have not been loaded")
            routing = MoERouting(
                router_logits=logits,
                selected_experts=indices,
                expert_weights=weights,
            )
            return kernel.forward(hidden, routing, self._prepared_weights, config)

    return lambda args: KernelExperts(args)


def install_expert_layer(
    model,
    kernel,
    config: MoEKernelConfig,
    layer_index: int,
    checkpoint_tensors: dict[str, Any],
) -> None:
    raw_tensors = {}
    for name, tensor in checkpoint_tensors.items():
        standard_name = _CHECKPOINT_TO_STANDARD[name]
        if name.endswith("_blocks"):
            if tensor.ndim != 4 or tensor.shape[-1] != 16:
                raise ValueError(
                    f"{name} must have checkpoint shape [E, N, K/32, 16], "
                    f"got {tuple(tensor.shape)}"
                )
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        raw_tensors[standard_name] = tensor

    raw = MoEExpertWeights(
        gate_up=raw_tensors["gate_up"],
        down=raw_tensors["down"],
        gate_up_scales=raw_tensors["gate_up_scales"],
        down_scales=raw_tensors["down_scales"],
        gate_up_bias=raw_tensors["gate_up_bias"],
        down_bias=raw_tensors["down_bias"],
    )
    prepared = kernel.prepare_weights(raw, config)
    model.model.layers[layer_index].mlp.experts.install_prepared_weights(prepared)
    # Released raw tensors become unreachable after this call, allowing the
    # allocator to reuse their storage while the next layer is streamed.


__all__ = [
    "EXPERT_CHECKPOINT_NAMES",
    "cuda_architecture",
    "install_expert_layer",
    "kernel_expert_factory",
    "moe_kernel_config",
    "shard_checkpoint_tensor",
    "shard_model_config",
]
