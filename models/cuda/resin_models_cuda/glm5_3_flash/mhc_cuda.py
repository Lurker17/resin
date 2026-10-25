"""Fused GLM mHC readout and RMS normalization with runtime token count.

Adapted from SGLang (Apache-2.0), revision
baa9befa381511cd4d56f83980951e5a45fffc6d,
python/sglang/kernels/ops/layernorm/mhc.py::mhc_pre_big_fuse_with_norm_tilelang.
PDL scheduling is disabled; the arithmetic and BF16 boundaries are preserved.
The RMS sum uses the FP32 collapsed stream before its BF16 store, as in the
serving reference. Splits are fixed by the caller, never by request length.
"""

import math
from typing import Any

import tilelang
import tilelang.language as _language

# TileLang's function body is a GPU DSL: its buffers and shape annotations are
# interpreted by the compiler, rather than Python's type system.
T: Any = _language


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def pre_with_norm(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 64,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.

    Identical to mhc_pre_big_fuse_tilelang for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + norm_eps) * norm_weight before writing to HBM.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]

    ENABLE_PDL = False
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Stash unnormalized weighted-sum output in shared memory as bf16
            # (matches the rounding the reference path does when RMSNorm reads bf16).
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()
