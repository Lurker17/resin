"""DeepInfer's lazy shifted-HC transitions using the pinned DeepGEMM ABI.

Adapted from DeepInfer's v4_modules/hc.py::_mega_mhc, hc_lazy.py, and
mhc_ops.py::mhc_pre_norm_fn (Apache-2.0).
Revision: 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01.
Engram and DSpark taps materialize the residual before observing its streams.
"""

from dataclasses import dataclass
from functools import cache
from importlib import import_module

import torch
from torch import Tensor


@dataclass(frozen=True)
class HCLazyPost:
    """Pending sublayer output update and the pre-mix for its successor."""

    residual: Tensor
    pre: Tensor
    post: Tensor
    comb: Tensor


@cache
def _post_kernel():
    # Load the shared TileLang compatibility adapter before TileKernels 1.0.
    import_module("resin_models_cuda.deepseek_v41.tilelang_kernels")
    return import_module("tile_kernels.mhc.post_kernel").mhc_post_fwd


def _materialize_native(output: Tensor, state: HCLazyPost) -> Tensor:
    """Use DeepInfer's [1, tokens, ...] post ABI, with comb[input, output]."""
    hc, dim = state.residual.shape[-2:]
    return _post_kernel()(
        output.reshape(1, -1, dim),
        state.residual.reshape(1, -1, hc, dim),
        state.post.reshape(1, -1, hc, 1),
        state.comb.reshape(1, -1, hc, hc),
    ).view_as(state.residual)


def materialize(output: Tensor, state: HCLazyPost) -> Tensor:
    """Apply the pending post-mix at an Engram, feature tap, or model boundary."""
    if output.is_cuda:
        return _materialize_native(output, state)
    return (
        state.post[..., None] * output[..., None, :]
        + (state.comb[..., None] * state.residual.unsqueeze(-2)).sum(-3)
    ).to(output.dtype)


@cache
def _prenorm_kernel():
    import_module("resin_models_cuda.deepseek_v41.tilelang_kernels")
    # TileKernels 1.0 exposes the reducer builder; newer releases wrap this as
    # mhc_reduce_partials_and_rmsnorm_fwd. Keep the pinned ABI in this adapter.
    return import_module("tile_kernels.mhc.norm_fn_kernel")._mhc_pre_norm_fn_fwd_norm


def _prenorm_native(hidden: Tensor, fn: Tensor, eps: float) -> Tensor:
    """DeepInfer's fixed 16-way HC prenorm reduction, independent of row count."""
    import deep_gemm

    mixes, width = fn.shape
    flat = hidden.reshape(-1, width)
    rows, splits = flat.shape[0], 16
    partials = torch.empty(
        (splits, rows, 1, mixes), device=hidden.device, dtype=torch.float32
    )
    squares = torch.empty((splits, rows, 1), device=hidden.device, dtype=torch.float32)
    deep_gemm.tf32_hc_prenorm_gemm(
        flat, fn, partials.view(splits, rows, mixes), squares.view(splits, rows), splits
    )
    output = torch.empty_like(partials[0, :, 0])
    _prenorm_kernel()(mixes, 1, width, eps, splits)(
        partials,
        squares,
        torch.empty_like(partials[0]),
        torch.empty_like(squares[0]),
        output,
    )
    return output.view(*hidden.shape[:-2], mixes)


def prenorm(hidden: Tensor, fn: Tensor, eps: float) -> Tensor:
    """Compute HC mixes at materialized boundaries (embedding, Engram, taps)."""
    if hidden.is_cuda:
        return _prenorm_native(hidden, fn, eps)
    flat = hidden.flatten(-2).float()
    return torch.nn.functional.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + eps
    )


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
