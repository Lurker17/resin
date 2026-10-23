"""DeepInfer's lazy shifted-HC transitions using the pinned DeepGEMM ABI.

Adapted from DeepInfer's v4_modules/hc.py::_mega_mhc and hc_lazy.py (Apache-2.0).
Revision: 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01.
Engram and DSpark taps materialize the residual before observing its streams.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HCLazyPost:
    """Pending sublayer output update and the pre-mix for its successor."""

    residual: Tensor
    pre: Tensor
    post: Tensor
    comb: Tensor


def materialize(output: Tensor, state: HCLazyPost) -> Tensor:
    """Apply the pending post-mix at an Engram, feature tap, or model boundary."""
    return (
        state.post[..., None] * output[..., None, :]
        + (state.comb[..., None] * state.residual.unsqueeze(-2)).sum(-3)
    ).to(output.dtype)


def mega_transition(
    block,
    output: Tensor,
    state: HCLazyPost,
    name: str,
    *,
    prepare_ffn: bool = False,
) -> tuple[Tensor, HCLazyPost]:
    """Fuse previous post, current mixes, shifted pre, RMSNorm and optional FP8.

    The optional FP8 outputs go straight into mega-MoE's symmetric workspace;
    the returned BF16 output still feeds the router, as it does in DeepInfer.
    """
    import deep_gemm

    args = block.args
    dim, hc = args.dim, args.hc_mult
    updated = torch.empty_like(state.residual)
    next_pre, next_post, next_comb = (
        torch.empty_like(state.pre),
        torch.empty_like(state.post),
        torch.empty_like(state.comb),
    )
    normalized = torch.empty_like(output)
    cast_kwargs = (
        block.ffn.fused_norm_cast_kwargs(output.numel() // dim) if prepare_ffn else {}
    )
    deep_gemm.mega_mhc(
        output.reshape(-1, dim),
        state.residual.reshape(-1, hc, dim),
        state.pre.reshape(-1, hc, 1),
        state.post.reshape(-1, hc, 1),
        state.comb.reshape(-1, hc, hc),
        fn=getattr(block, f"hc_{name}_fn"),
        mix_scales=getattr(block, f"hc_{name}_scale"),
        mix_bases=getattr(block, f"hc_{name}_base"),
        hc_mult=hc,
        hc_norm_eps=args.norm_eps,
        hc_pre_eps=args.hc_eps,
        hc_post_scale=2,
        sinkhorn_eps=args.hc_eps,
        num_sinkhorn_iters=args.hc_sinkhorn_iters,
        rmsnorm_weight=getattr(block, f"{name}_norm").weight,
        rmsnorm_eps=args.norm_eps,
        rmsnorm_scale=1,
        new_residual=updated.reshape(-1, hc, dim),
        new_prev_mix=next_pre.reshape(-1, hc, 1),
        new_post_mix=next_post.reshape(-1, hc, 1),
        new_comb_res_mix=next_comb.reshape(-1, hc, hc),
        y_bf16=normalized.reshape(-1, dim),
        **cast_kwargs,
    )
    return normalized, HCLazyPost(updated, next_pre, next_post, next_comb)
