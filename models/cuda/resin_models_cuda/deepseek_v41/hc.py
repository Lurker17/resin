"""The attention-to-FFN shifted-HC transition, fused by pinned DeepGEMM."""

import torch


def mega_ffn_transition(block, output, residual, pre, post, comb):
    """Keep full residuals at Block boundaries, including Engram/DSpark taps."""
    import deep_gemm

    args = block.args
    dim, hc = args.dim, args.hc_mult
    updated = torch.empty_like(residual)
    next_pre, next_post, next_comb = (
        torch.empty_like(pre),
        torch.empty_like(post),
        torch.empty_like(comb),
    )
    normalized = torch.empty_like(output)
    deep_gemm.mega_mhc(
        output.reshape(-1, dim),
        residual.reshape(-1, hc, dim),
        pre.reshape(-1, hc, 1),
        post.reshape(-1, hc, 1),
        comb.reshape(-1, hc, hc),
        fn=block.hc_ffn_fn,
        mix_scales=block.hc_ffn_scale,
        mix_bases=block.hc_ffn_base,
        hc_mult=hc,
        hc_norm_eps=args.norm_eps,
        hc_pre_eps=args.hc_eps,
        hc_post_scale=2,
        sinkhorn_eps=args.hc_eps,
        num_sinkhorn_iters=args.hc_sinkhorn_iters,
        rmsnorm_weight=block.ffn_norm.weight,
        rmsnorm_eps=args.norm_eps,
        rmsnorm_scale=1,
        new_residual=updated.reshape(-1, hc, dim),
        new_prev_mix=next_pre.reshape(-1, hc, 1),
        new_post_mix=next_post.reshape(-1, hc, 1),
        new_comb_res_mix=next_comb.reshape(-1, hc, hc),
        y_bf16=normalized.reshape(-1, dim),
    )
    return normalized, updated, next_pre, next_post, next_comb
