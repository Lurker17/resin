"""Prepare V4.1 checkpoint tensors for the selected complete FFN backend."""

from functools import partial
from typing import cast

import torch
from torch import Tensor

from resin.backend.cuda.mega_moe import MegaMoe, prepare_weights
from resin.backend.torch import benchmark_expert_indices
from resin_models_cuda.deepseek_v41.kernels import act_quant
from resin_models_cuda.deepseek_v41.linear import weight_scale_rows


def mega_gate(gate, hidden, input_ids, active, out, *, ep_rank):
    """Fuse the released gate into the FFN workspace, retaining BF16 weights."""
    import deep_gemm

    fixed = gate.benchmark_balanced_experts
    image_mask = (
        input_ids == gate.image_token_id
        if input_ids is not None and gate.bias_vl is not None
        else None
    )
    deep_gemm.bf16_mega_gate(
        hidden,
        gate.weight,
        gate.topk,
        False,
        0,
        gate.route_scale,
        ep_rank,
        mask=active,
        bias=gate.bias,
        image_bias=gate.bias_vl if image_mask is not None else None,
        image_token_mask=image_mask,
        fix_routing_mask=(
            torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
            if fixed
            else None
        ),
        unmapped_topk_idx=(
            benchmark_expert_indices(
                gate.weight[:, 0].expand(len(hidden), -1),
                gate.topk,
                gate.benchmark_replica,
                gate.benchmark_expert_parallel_size,
            )
            if fixed
            else None
        ),
        out=out,
    )


@torch.no_grad()
def prepare_mega_moe(source, collective):
    import deep_gemm

    gate = source.gate
    if (
        gate.temperature != 1
        or gate.score_func != "sqrtsoftplus"
        or not gate.normalized
        or gate.topk < 2
    ):
        raise ValueError(
            "DeepGEMM requires the normalized sqrtsoftplus gate with temperature 1 and top-k > 1"
        )
    intermediate = source.routed_intermediate_size
    # The checkpoint container shares FlashInfer's [up | gate] backing buffer.
    # DeepGEMM accepts [gate | up] before its eight-row interleaving transform.
    routed_weight = torch.cat(
        (
            source._fused_w13_weight[:, intermediate:].view(torch.int8),
            source._fused_w13_weight[:, :intermediate].view(torch.int8),
        ),
        dim=1,
    )
    routed_scale = torch.cat(
        (
            source._fused_w13_scale[:, intermediate:].float(),
            source._fused_w13_scale[:, :intermediate].float(),
        ),
        dim=1,
    )
    routed = cast(
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        deep_gemm.transform_weights_for_mega_moe(
            prepare_weights(routed_weight, routed_scale),
            prepare_weights(
                source._fused_w2_weight.view(torch.int8), source._fused_w2_scale
            ),
        ),
    )
    shared = source.shared_experts
    shared_weight = torch.cat((shared.w1.weight, shared.w3.weight))
    shared_scale = torch.cat(
        (weight_scale_rows(shared.w1), weight_scale_rows(shared.w3))
    )
    shared_weights = cast(
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        deep_gemm.transform_weights_for_mega_moe(
            prepare_weights(shared_weight, shared_scale),
            prepare_weights(shared.w2.weight, weight_scale_rows(shared.w2)),
        ),
    )
    return MegaMoe(
        source.gate,
        collective,
        lambda hidden: act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu),
        route=partial(mega_gate, gate, ep_rank=collective.group.local_rank),
        gate_up=routed[0],
        down=routed[1],
        shared_gate_up=shared_weights[0],
        shared_down=shared_weights[1],
        experts=source.n_routed_experts,
        topk=source.n_activated_experts,
        hidden=source.dim,
        intermediate=intermediate,
        clamp=source.swiglu_limit,
    )
