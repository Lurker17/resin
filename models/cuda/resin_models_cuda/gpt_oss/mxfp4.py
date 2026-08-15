"""Kernel-neutral bridge for streaming GPT-OSS MXFP4 checkpoints.

The model remains ordinary Resin-owned PyTorch code.  This module replaces only
the expert implementation with the small :mod:`resin.moe_kernels` contract and
streams official packed tensors through the selected adapter one layer at a
time.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout, TensorShard
from resin.backend.loading import resolve_dtype
from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernelConfig,
    MoERouting,
)
from resin_models_cuda.gpt_oss.weights import load_gpt_oss_weights

_CHECKPOINT_TO_STANDARD = {
    "gate_up_proj_blocks": "gate_up",
    "gate_up_proj_scales": "gate_up_scales",
    "gate_up_proj_bias": "gate_up_bias",
    "down_proj_blocks": "down",
    "down_proj_scales": "down_scales",
    "down_proj_bias": "down_bias",
}
_EXPERT_CHECKPOINT_NAMES = tuple(_CHECKPOINT_TO_STANDARD)


def _slice_axis(tensor, axis: int, shard: TensorShard):
    selection = [slice(None)] * tensor.ndim
    selection[axis] = shard.as_slice()
    return tensor[tuple(selection)].contiguous()


def _local_args(args, tensor_parallel: TensorParallelLayout):
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


def _checkpoint_tensor_for_rank(
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


def _cuda_architecture(torch, device) -> str:
    if device.type != "cuda":
        raise ValueError("packed GPT-OSS MXFP4 execution requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def _moe_config(args, architecture: str) -> MoEKernelConfig:
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


def _make_kernel_expert_factory(torch, kernel, config: MoEKernelConfig):
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


def _install_expert_layer(
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
    # Let PyTorch reuse released raw/padding allocations for the next layer.
    # Emptying the CUDA cache here forces a sync and fresh allocations once per
    # layer without making any additional memory available to this process.


def load_mxfp4_model(
    model_path: str | Path,
    *,
    device,
    kernel,
    dtype=None,
    tensor_parallel: TensorParallelLayout | None = None,
):
    """Load an official packed checkpoint without Transformers or tokenizers."""

    from resin_models_cuda.gpt_oss.config import GptOssConfig
    from resin_models_cuda.gpt_oss.model import GptOssForCausalLM

    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    args = GptOssConfig.from_dict(config)
    tensor_parallel = tensor_parallel or TensorParallelLayout()
    quantization = config.get("quantization_config") or config.get("quantization")
    if not quantization or quantization.get("quant_method") != "mxfp4":
        raise ValueError("load_mxfp4_model requires an MXFP4 GPT-OSS checkpoint")

    target = torch.device(device)
    resolved_dtype = resolve_dtype(torch, dtype)
    if resolved_dtype not in {None, torch.bfloat16}:
        raise ValueError("the packed SM90 GPT-OSS path requires bfloat16 activations")
    architecture = _cuda_architecture(torch, target)
    local_args = _local_args(args, tensor_parallel)
    moe_config = _moe_config(local_args, architecture)
    kernel.validate(moe_config)

    expert_factory = _make_kernel_expert_factory(torch, kernel, moe_config)
    with torch.device("meta"):
        model = GptOssForCausalLM(local_args, expert_factory)

    load_gpt_oss_weights(
        path,
        model,
        device=target,
        dtype=torch.bfloat16,
        num_hidden_layers=args.num_hidden_layers,
        expert_tensor_names=_EXPERT_CHECKPOINT_NAMES,
        transform_tensor=lambda key, tensor: _checkpoint_tensor_for_rank(
            key,
            tensor,
            args=args,
            tensor_parallel=tensor_parallel,
        ),
        install_expert_layer=lambda layer, tensors: _install_expert_layer(
            model, kernel, moe_config, layer, dict(tensors)
        ),
    )

    model.materialize_buffers(target)
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm":
            module.float()
    unprepared = [
        index
        for index, layer in enumerate(model.model.layers)
        if not layer.mlp.experts.is_prepared
    ]
    if unprepared:
        raise RuntimeError(f"packed expert layers were not prepared: {unprepared}")
    remaining_meta = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            "ordinary model parameters remain on meta: " + ", ".join(remaining_meta[:8])
        )
    model.eval()
    return torch, model, config


__all__ = ["load_mxfp4_model"]
