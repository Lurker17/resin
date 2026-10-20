"""Prepared local experts and eager expert-parallel dispatch/combine.

The architectural MoE owns gating and shared experts. This module owns local
expert weights, their kernel layout and communication across expert owners.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from resin.backend.expert_parallel import ExpertParallelDispatch

if TYPE_CHECKING:
    from resin.backend.topology import RankGroup
    from resin_models_cuda.deepseek_v4.model import MoEWeights


class RoutedExperts(nn.Module):
    """Execute a prepared expert shard and return complete local token rows.

    Legacy TP receives replicated rows and reduces expert-shard outputs. EP
    instead dispatches independent local rows to expert owners and combines the
    returned contributions without reducing unrelated attention-DP requests.
    """

    def __init__(
        self,
        source: MoEWeights,
        *,
        reduce: Callable[[Tensor], Tensor],
        block_size: int,
        scale_fmt: str | None,
        scale_dtype: torch.dtype,
        expert_parallel: RankGroup | None = None,
        replication: RankGroup | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = source.n_routed_experts
        self.start = source.experts_start_idx
        self.stop = source.experts_end_idx
        self.reduce = reduce
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.scale_dtype = scale_dtype
        self.fused = source._fused_fp4
        self.dispatch = (
            ExpertParallelDispatch(
                self.local,
                num_experts=self.num_experts,
                group=expert_parallel,
                replication=replication,
            )
            if expert_parallel is not None
            else None
        )
        if self.fused:
            from resin_models_cuda.deepseek_v4.gemm import (
                prepare_fp4_fused_moe_weights,
            )

            # Adopt the loaded storage; checkpoint Expert wrappers are aliases
            # into these buffers and need not survive runtime preparation.
            for name in ("w13_weight", "w13_scale", "w2_weight", "w2_scale"):
                self.register_buffer(name, getattr(source, f"_fused_{name}"))
            prepare_fp4_fused_moe_weights(
                self.w13_weight, self.w13_scale, self.w2_weight, self.w2_scale
            )
            self.register_buffer("output_scales", source._fused_output_scales)
            self.register_buffer("clamp_limits", source._fused_clamp_limits)
        else:
            self.experts = nn.ModuleList(source.experts[self.start : self.stop])

    def forward(self, hidden: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        if self.dispatch is not None:
            return self.dispatch(hidden, indices, weights)
        return self.reduce(self.local(hidden, indices, weights))

    def local(self, hidden: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        if hidden.shape[0] == 0:
            return torch.zeros_like(hidden)
        if self.fused:
            from resin_models_cuda.deepseek_v4.cutile import moe_act_quant
            from resin_models_cuda.deepseek_v4.gemm import fp4_fused_moe

            quantized, scales = moe_act_quant(
                hidden, self.block_size, self.scale_fmt, self.scale_dtype
            )
            partial = fp4_fused_moe(
                quantized,
                scales,
                indices,
                weights,
                self.w13_weight,
                self.w13_scale,
                self.w2_weight,
                self.w2_scale,
                self.output_scales,
                self.clamp_limits,
                num_experts=self.num_experts,
                local_expert_offset=self.start,
                scale_dtype=self.scale_dtype,
            )
        else:
            partial = torch.zeros_like(hidden, dtype=torch.bfloat16)
            counts = torch.bincount(
                indices.flatten(), minlength=self.num_experts
            ).tolist()
            for local, expert in enumerate(self.experts):
                expert_id = self.start + local
                if counts[expert_id]:
                    rows, slots = torch.where(indices == expert_id)
                    partial[rows] += expert(hidden[rows], weights[rows, slots, None])
        return partial
