"""FlashInfer's prepared FP4 routed-expert kernel for DeepSeek V4.1."""

from __future__ import annotations

import torch
from torch import Tensor


def fp4_fused_moe(
    left: Tensor,
    left_scales: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    w13: Tensor,
    w13_scales: Tensor,
    w2: Tensor,
    w2_scales: Tensor,
    output_scales: Tensor,
    clamp_limits: Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    scale_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Run FlashInfer's fused MXFP8×MXFP4 routed MoE kernel."""

    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType

    if scale_dtype != torch.float8_e8m0fnu:
        raise ValueError("DeepSeek-V4.1 fused MoE scales must use float8_e8m0fnu")
    if left.dtype != torch.float8_e4m3fn:
        raise TypeError("DeepSeek-V4.1 MoE activations must use float8_e4m3fn")
    if w13.dtype != torch.float4_e2m1fn_x2 or w2.dtype != torch.float4_e2m1fn_x2:
        raise TypeError("DeepSeek-V4.1 MoE weights must use packed float4_e2m1fn_x2")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("fused MoE weights must include a local-expert dimension")
    depth = left.shape[-1]
    rows = left.numel() // depth
    local_experts, doubled_intermediate, packed_depth = w13.shape
    if packed_depth != depth // 2 or w2.shape[:2] != (local_experts, depth):
        raise ValueError("fused MoE weights have incompatible hidden dimensions")
    intermediate = doubled_intermediate // 2
    if doubled_intermediate % 2 or w2.shape[2] != intermediate // 2:
        raise ValueError("fused MoE weights have incompatible intermediate dimensions")
    if left_scales.dtype != torch.float8_e4m3fn or left_scales.shape != (
        rows,
        depth // 32,
    ):
        raise ValueError(
            "FlashInfer activation scales must contain one repeated UE8M0 byte "
            "per 32 inputs"
        )
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != rows:
        raise ValueError("top-k expert ids and weights must have shape (tokens, top_k)")
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.bfloat16:
        raise TypeError("FlashInfer routed MoE requires int32 ids and bfloat16 weights")
    if w13_scales.shape != (local_experts, doubled_intermediate, depth // 32):
        raise ValueError("W13 scales have an incompatible shape")
    if w2_scales.shape != (local_experts, depth, intermediate // 32):
        raise ValueError("W2 scales have an incompatible shape")
    if output_scales.shape != (local_experts,) or clamp_limits.shape != (
        local_experts,
    ):
        raise ValueError("fused MoE scalar parameters must have one value per expert")

    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(topk_ids, topk_weights),
        routing_bias=None,
        hidden_states=left.view(rows, depth),
        hidden_states_scale=left_scales,
        gemm1_weights=w13.view(torch.uint8),
        gemm1_weights_scale=w13_scales.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp_limits,
        gemm2_weights=w2.view(torch.uint8),
        gemm2_weights_scale=w2_scales.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=output_scales,
        output1_scale_gate_scalar=output_scales,
        output2_scale_scalar=output_scales,
        num_experts=num_experts,
        top_k=topk_ids.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_experts,
        routed_scaling_factor=None,
        routing_method_type=5,
        do_finalize=True,
        enable_pdl=True,
        activation_type=ActivationType.Swiglu.value,
        tune_max_num_tokens=8192,
    )
    return result[0]


def prepare_fp4_fused_moe_weights(
    w13: Tensor,
    w13_scales: Tensor,
    w2: Tensor,
    w2_scales: Tensor,
) -> None:
    """Convert checkpoint-linear expert tensors to FlashInfer's kernel layout."""

    from flashinfer.fused_moe import reorder_rows_for_gated_act_gemm
    from flashinfer.quantization import block_scale_interleave, shuffle_matrix_a

    tensors = (w13, w13_scales, w2, w2_scales)
    if all(
        getattr(tensor, "_resin_flashinfer_moe_layout", False) for tensor in tensors
    ):
        return
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("fused MoE weights must have matching expert dimensions")
    if w13_scales.shape[:2] != w13.shape[:2] or w2_scales.shape[:2] != w2.shape[:2]:
        raise ValueError("fused MoE scales must match their weight row dimensions")

    epilogue_tile = 128

    def row_permutation(rows: int, *, gated: bool) -> Tensor:
        probe = torch.arange(rows, dtype=torch.long).view(rows, 1)
        if gated:
            probe = reorder_rows_for_gated_act_gemm(probe)
        return shuffle_matrix_a(probe, epilogue_tile).flatten().to(w13.device)

    w13_permutation = row_permutation(w13.shape[1], gated=True)
    w2_permutation = row_permutation(w2.shape[1], gated=False)
    with torch.no_grad():
        w13_bytes = w13.view(torch.uint8)
        w2_bytes = w2.view(torch.uint8)
        w13_bytes.copy_(w13_bytes.index_select(1, w13_permutation))
        w2_bytes.copy_(w2_bytes.index_select(1, w2_permutation))
        w13_scale_bytes = w13_scales.view(torch.uint8)
        w2_scale_bytes = w2_scales.view(torch.uint8)
        w13_scale_bytes.copy_(
            block_scale_interleave(
                w13_scale_bytes.index_select(1, w13_permutation)
            ).reshape_as(w13_scale_bytes)
        )
        w2_scale_bytes.copy_(
            block_scale_interleave(
                w2_scale_bytes.index_select(1, w2_permutation)
            ).reshape_as(w2_scale_bytes)
        )
    for tensor in tensors:
        setattr(tensor, "_resin_flashinfer_moe_layout", True)
