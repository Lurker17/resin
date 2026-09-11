"""TileKernels quantization and fused mHC adapted from SGLang."""

from __future__ import annotations

import math
import sys

import torch
from torch import Tensor

if sys.platform == "linux":
    import tilelang
    import tilelang.language as T
    from tilelang.backend import target

    # TileKernels 1.0.0 uses the path renamed in TileLang 0.1.14. Remove this
    # compatibility alias when upgrading to a release with the corrected import.
    sys.modules.setdefault("tilelang.utils.target", target)
    from tile_kernels.quant import per_token_cast

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def _mhc_pre_norm_kernel(
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
        rms_epsilon: float,
        pre_epsilon: float,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
        norm_epsilon: float,
        num_splits: int,
        hidden_copies: int = 4,
        mix_width: int = 24,
    ):
        """Run MHC control work and hidden-vector mixing on separate warps."""

        num_tokens = T.dynamic("num_tokens")
        hidden_block = math.gcd(1024, hidden_size)

        gemm_out_mul: T.Tensor[[num_splits, num_tokens, mix_width], T.float32]
        gemm_out_sqrsum: T.Tensor[[num_splits, num_tokens], T.float32]
        hc_scale: T.Tensor[[3], T.float32]
        hc_base: T.Tensor[[mix_width], T.float32]
        residual: T.Tensor[[num_tokens, hidden_copies, hidden_size], T.bfloat16]
        post_mix: T.Tensor[[num_tokens, hidden_copies], T.float32]
        comb_mix: T.Tensor[[num_tokens, hidden_copies * hidden_copies], T.float32]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
        norm_weight: T.Tensor[[hidden_size], T.bfloat16]

        enable_pdl = torch.cuda.get_device_capability()[0] >= 9
        with T.Kernel(num_tokens, threads=96) as token:
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(mix_width, T.float32)
            T.clear(mixes)
            rms[0] = 0

            if enable_pdl:
                T.pdl_sync()

            for split in T.serial(num_splits):
                rms[0] += gemm_out_sqrsum[split, token]
            rms[0] = T.rsqrt(rms[0] / (hidden_copies * hidden_size) + rms_epsilon)
            for column in T.Parallel(mix_width):
                mixes[column] = 0
                for split in T.serial(num_splits):
                    mixes[column] += gemm_out_mul[split, token, column]
                mixes[column] *= rms[0]
            mixes_shared = T.alloc_shared(mix_width, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                combination = T.alloc_fragment(
                    (hidden_copies, hidden_copies), T.float32
                )
                for column in T.Parallel(hidden_copies):
                    post_mix[token, column] = 2.0 * T.sigmoid(
                        mixes_shared[column + hidden_copies] * hc_scale[1]
                        + hc_base[column + hidden_copies]
                    )
                for row, column in T.Parallel(hidden_copies, hidden_copies):
                    index = row * hidden_copies + column + hidden_copies * 2
                    combination[row, column] = (
                        mixes_shared[index] * hc_scale[2] + hc_base[index]
                    )

                row_sum = T.alloc_fragment(hidden_copies, T.float32)
                column_sum = T.alloc_fragment(hidden_copies, T.float32)
                row_max = T.alloc_fragment(hidden_copies, T.float32)
                T.reduce_max(combination, row_max, dim=1)
                for row, column in T.Parallel(hidden_copies, hidden_copies):
                    combination[row, column] = T.exp(
                        combination[row, column] - row_max[row]
                    )
                T.reduce_sum(combination, row_sum, dim=1)
                for row, column in T.Parallel(hidden_copies, hidden_copies):
                    combination[row, column] = (
                        combination[row, column] / row_sum[row] + sinkhorn_epsilon
                    )

                T.reduce_sum(combination, column_sum, dim=0)
                for row, column in T.Parallel(hidden_copies, hidden_copies):
                    combination[row, column] /= column_sum[column] + sinkhorn_epsilon

                for _ in T.serial(sinkhorn_iterations - 1):
                    T.reduce_sum(combination, row_sum, dim=1)
                    for row, column in T.Parallel(hidden_copies, hidden_copies):
                        combination[row, column] /= row_sum[row] + sinkhorn_epsilon
                    T.reduce_sum(combination, column_sum, dim=0)
                    for row, column in T.Parallel(hidden_copies, hidden_copies):
                        combination[row, column] /= (
                            column_sum[column] + sinkhorn_epsilon
                        )

                for row, column in T.Parallel(hidden_copies, hidden_copies):
                    comb_mix[token, row * hidden_copies + column] = combination[
                        row, column
                    ]
            else:
                pre_shared = T.alloc_shared(hidden_copies, T.float32)
                for copy in T.Parallel(hidden_copies):
                    pre_shared[copy] = (
                        T.sigmoid(mixes_shared[copy] * hc_scale[0] + hc_base[copy])
                        + pre_epsilon
                    )

                output_shared = T.alloc_shared(hidden_size, T.bfloat16)
                sumsq_per_position = T.alloc_fragment(hidden_block, T.float32)
                T.clear(sumsq_per_position)

                for block in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                    residual_shared = T.alloc_shared(
                        (hidden_copies, hidden_block), T.bfloat16
                    )
                    residual_local = T.alloc_fragment(
                        (hidden_copies, hidden_block), T.float32
                    )
                    T.copy(residual[token, 0, block * hidden_block], residual_shared)
                    T.copy(residual_shared, residual_local)
                    output_local = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(output_local)
                    for copy in T.serial(hidden_copies):
                        pre = pre_shared[copy]
                        for offset in T.Parallel(hidden_block):
                            output_local[offset] += pre * residual_local[copy, offset]
                    for offset in T.Parallel(hidden_block):
                        sumsq_per_position[offset] += (
                            output_local[offset] * output_local[offset]
                        )
                        output_shared[block * hidden_block + offset] = T.bfloat16(
                            output_local[offset]
                        )

                sumsq = T.alloc_fragment(1, T.float32)
                T.reduce_sum(sumsq_per_position, sumsq, dim=0)
                norm_scale = T.alloc_fragment(1, T.float32)
                norm_scale[0] = T.rsqrt(sumsq[0] / hidden_size + norm_epsilon)

                for block in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    weight_shared = T.alloc_shared(hidden_block, T.bfloat16)
                    weight_local = T.alloc_fragment(hidden_block, T.float32)
                    T.copy(norm_weight[block * hidden_block], weight_shared)
                    T.copy(weight_shared, weight_local)
                    output_local = T.alloc_fragment(hidden_block, T.float32)
                    for offset in T.Parallel(hidden_block):
                        output_local[offset] = (
                            output_shared[block * hidden_block + offset]
                            * norm_scale[0]
                            * weight_local[offset]
                        )
                    T.copy(output_local, layer_input[token, block * hidden_block])

            if enable_pdl:
                T.pdl_trigger()

else:
    _mhc_pre_norm_kernel = None
    per_token_cast = None


def mhc_pre_norm(
    dot_mix: Tensor,
    sqrsum: Tensor,
    residual: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    norm_weight: Tensor,
    *,
    rms_epsilon: float,
    norm_epsilon: float,
    pre_epsilon: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse MHC split reduction, weighted residual mixing, and RMSNorm."""

    num_splits, rows, mix_width = dot_mix.shape
    hidden_copies, hidden_size = residual.shape[-2:]
    if (
        mix_width != 24
        or hidden_copies != 4
        or residual.shape[0] != rows
        or sqrsum.shape != (num_splits, rows)
        or norm_weight.shape != (hidden_size,)
        or norm_weight.dtype != torch.bfloat16
    ):
        raise ValueError("optimized HC inputs have incompatible shapes")
    if _mhc_pre_norm_kernel is None:
        raise RuntimeError("the optimized HC kernel requires TileLang on Linux")

    post = torch.empty((rows, 4), dtype=torch.float32, device=residual.device)
    comb = torch.empty((rows, 16), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        (rows, hidden_size), dtype=residual.dtype, device=residual.device
    )
    _mhc_pre_norm_kernel(
        dot_mix,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        hidden_size,
        rms_epsilon,
        pre_epsilon,
        sinkhorn_epsilon,
        sinkhorn_iterations,
        norm_epsilon,
        num_splits,
        hidden_copies,
        mix_width,
    )
    return post, comb.view(rows, 4, 4), layer_input


__all__ = ["mhc_pre_norm", "per_token_cast"]
