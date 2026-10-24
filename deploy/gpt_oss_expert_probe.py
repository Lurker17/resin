"""Isolate checkpoint MXFP4 sharding from full-model rounding amplification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import read_config
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.flashinfer import (
    FlashInferSM90MXFP4Kernel,
    MXFP4Config,
    MXFP4Weights,
)
from resin_models_cuda.gpt_oss.mxfp4 import (
    EXPERT_CHECKPOINT_NAMES,
    shard_checkpoint_tensor,
)

_RAW_FIELDS = {
    "gate_up_proj_blocks": "gate_up",
    "gate_up_proj_scales": "gate_up_scales",
    "gate_up_proj_bias": "gate_up_bias",
    "down_proj_blocks": "down",
    "down_proj_scales": "down_scales",
    "down_proj_bias": "down_bias",
}


def _dequantize(blocks, scales):
    """Decode canonical MXFP4 low/high nibbles with one E8M0 scale per 32."""

    packed = blocks.reshape(*blocks.shape[:-2], -1)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6), device=blocks.device, dtype=torch.float32
    )
    values = levels[(nibbles & 7).long()] * torch.where(nibbles < 8, 1, -1)
    return (
        values.reshape(*scales.shape, 32)
        * torch.exp2(scales.float() - 127).unsqueeze(-1)
    ).flatten(-2)


def _error(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "rms": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
        "relative_l2": float(
            difference.norm() / expected.float().norm().clamp_min(1e-12)
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _dense_outputs(raw, hidden, indices, routing_weights, args, *, tensor_sizes=(2, 4)):
    """Independent dense reference with explicit FC2 rounding boundaries.

    Every variant shares one FP32 FC1 calculation and its BF16 activation.
    Only the placement of the FC2 BF16 rounding differs. Expert bias and
    routing weights are applied in FP32 after complete expert contributions.
    """

    outputs = {
        name: torch.zeros_like(hidden, dtype=torch.float32)
        for name in (
            "fp32_fc2",
            "bf16_fc2",
            *(f"bf16_fc2_tp{size}" for size in tensor_sizes),
        )
    }
    for expert in range(args.num_local_experts):
        rows, slots = torch.where(indices == expert)
        if not rows.numel():
            continue
        gate_up = _dequantize(
            raw["gate_up_proj_blocks"][expert], raw["gate_up_proj_scales"][expert]
        )
        down = _dequantize(
            raw["down_proj_blocks"][expert], raw["down_proj_scales"][expert]
        )
        paired = F.linear(
            hidden[rows].float(),
            gate_up,
            raw["gate_up_proj_bias"][expert].bfloat16().float(),
        )
        gate = paired[:, ::2].clamp(max=args.swiglu_limit)
        up = paired[:, 1::2].clamp(-args.swiglu_limit, args.swiglu_limit)
        activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).bfloat16().float()
        projections = {"fp32_fc2": F.linear(activated, down)}
        projections["bf16_fc2"] = projections["fp32_fc2"].bfloat16().float()
        for size in tensor_sizes:
            partial = torch.zeros_like(projections["fp32_fc2"])
            for rank in range(size):
                shard = (
                    TensorParallelLayout(rank=rank, size=size)
                    .aligned_shard(
                        args.intermediate_size, alignment=32, name="intermediate_size"
                    )
                    .as_slice()
                )
                partial += (
                    F.linear(activated[:, shard], down[:, shard]).bfloat16().float()
                )
            projections[f"bf16_fc2_tp{size}"] = partial
        bias = raw["down_proj_bias"][expert].bfloat16().float()
        probabilities = routing_weights[rows, slots].float()[:, None]
        for name, projection in projections.items():
            outputs[name].index_add_(0, rows, (projection + bias) * probabilities)
    return outputs


def _weights(raw):
    tensors = {}
    for name, value in raw.items():
        if name.endswith("_blocks"):
            value = value.flatten(-2)
        tensors[_RAW_FIELDS[name]] = value
    return MXFP4Weights(**tensors)


@torch.inference_mode()
def run_expert_probe(model_path, reference_path, *, device="cuda:0"):
    """Compare one real layer's full, EP, TP, and independently decoded weights."""

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = GptOssConfig.from_dict(read_config(model_path))
    reference = torch.load(reference_path, map_location=device, weights_only=True)
    hidden = reference["hidden"].to(device=device, dtype=torch.bfloat16)
    indices = reference["indices"].to(device=device, dtype=torch.int32)
    probabilities = reference["weights"].to(device=device, dtype=torch.float32)
    paths = indexed_tensor_files(Path(model_path))
    raw = {}
    with SafetensorReader() as reader:
        for name in EXPERT_CHECKPOINT_NAMES:
            key = f"model.layers.0.mlp.experts.{name}"
            raw[name] = reader.read(paths[key], key).to(device)
    kernel = FlashInferSM90MXFP4Kernel()
    config = MXFP4Config.from_model(args)
    prepared = kernel.prepare_weights(_weights(raw), config)
    full = kernel.forward(hidden, indices, probabilities, prepared)
    full_partial = kernel.forward_partial(hidden, indices, probabilities, prepared)
    del prepared
    report = {
        "gpu": torch.cuda.get_device_name(),
        "rows": hidden.shape[0],
        "reference_path": str(reference_path),
        "full_kernel_vs_captured": _error(full, reference["output"].to(device)),
        "full_partial_vs_full_kernel": _error(full_partial.bfloat16(), full),
        "shards": {},
    }
    dense = _dense_outputs(raw, hidden, indices, probabilities, args)
    report["dense_vs_full_kernel"] = {
        name: _error(output.bfloat16(), full) for name, output in dense.items()
    }
    for axis, sizes in (("expert_tp", (2, 4)), ("ep", (2, 4))):
        for size in sizes:
            result = torch.zeros_like(hidden, dtype=torch.float32)
            for rank in range(size):
                layout = TensorParallelLayout(rank=rank, size=size)
                options = {
                    "expert_tensor_parallel"
                    if axis == "expert_tp"
                    else "expert_parallel": layout
                }
                shard = {
                    name: shard_checkpoint_tensor(
                        f"model.layers.0.mlp.experts.{name}",
                        value,
                        args=args,
                        tensor_parallel=TensorParallelLayout(),
                        **options,
                    )
                    for name, value in raw.items()
                }
                if axis == "expert_tp":
                    intermediate = layout.aligned_shard(
                        args.intermediate_size, alignment=32, name="intermediate_size"
                    )
                    local = replace(config, intermediate_size=intermediate.size)
                    selected, selected_weights = indices, probabilities
                else:
                    experts = layout.even_shard(
                        args.num_local_experts, name="num_experts"
                    )
                    local = replace(config, num_experts=experts.size)
                    owned = (indices >= experts.start) & (indices < experts.stop)
                    selected = torch.where(owned, indices - experts.start, experts.size)
                    selected_weights = probabilities.masked_fill(~owned, 0)
                prepared = kernel.prepare_weights(_weights(shard), local)
                result += kernel.forward_partial(
                    hidden, selected, selected_weights, prepared
                )
                del prepared, shard
            name = f"{axis}{size}"
            oracle = dense[f"bf16_fc2_tp{size}" if axis == "expert_tp" else "bf16_fc2"]
            report["shards"][name] = {
                "vs_full_kernel": _error(result.bfloat16(), full),
                "vs_same_shard_dense": _error(result.bfloat16(), oracle.bfloat16()),
                "dense_shard_vs_dense_full": _error(
                    oracle.bfloat16(), dense["bf16_fc2"].bfloat16()
                ),
                "vs_fp32_fc2_dense": _error(
                    result.bfloat16(), dense["fp32_fc2"].bfloat16()
                ),
            }
    torch.cuda.synchronize()
    return report
