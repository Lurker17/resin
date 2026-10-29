"""Gated DeltaNet decode and speculative verification on the slab-resident state.

The general path gathers each row's recurrent state out of the cache arena,
clones it, runs FLA's fused recurrence and scatters the stacked result back:
four extra passes over 50 MB per row per step for chandra-2. These kernels read
the state at its source checkpoint slot and write the updated state to its
destination slot (the same slot except when a decode crosses a state block).
The math follows FLA's fused_recurrent_gated_delta_rule with in-kernel q/k L2
norm, including torch's BF16 rounding of sigmoid(beta).

Speculative verification runs T tokens per row. The state after the first
token is written in place; tokens 2..T only save their small update vectors
(normalized key, value, decay, beta). After acceptance is known, gdn_commit
applies the accepted updates to the stored state, so state traffic per
committed token matches ordinary decode and no per-step state copies exist.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

from resin_models_cuda.qwen3_5.cache import RecurrentStatePool


@triton.jit
def _gdn_decode_kernel(
    mixed,
    mixed_stride_b,
    mixed_stride_t,
    a,
    b,
    ab_stride_b,
    ab_stride_t,
    A_log,
    dt_bias,
    states,
    source,
    destination,
    layer_offset,
    pending,
    out,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    head = i_hv * K * V
    tile = o_k[:, None] * V + o_v[None, :]
    src = tl.load(source + i_n) + layer_offset + head
    dst = tl.load(destination + i_n) + layer_offset + head
    b_h = tl.load(states + src + tile)
    A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)

    for t in tl.static_range(T):
        row = mixed + i_n.to(tl.int64) * mixed_stride_b + t * mixed_stride_t
        b_q = tl.load(row + i_h * K + o_k).to(tl.float32)
        b_k = tl.load(row + H * K + i_h * K + o_k).to(tl.float32)
        b_v = tl.load(row + 2 * H * K + i_hv * V + o_v).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        gate_row = i_n.to(tl.int64) * ab_stride_b + t * ab_stride_t + i_hv
        b_beta = (
            tl.sigmoid(tl.load(b + gate_row).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        x = tl.load(a + gate_row).to(tl.float32) + bias
        b_g = A * tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        if t > 0:
            # Save this token's update for the post-acceptance commit.
            slot = pending + ((i_n * (T - 1) + (t - 1)) * HV + i_hv).to(tl.int64) * (
                K + V + 2
            )
            if i_v == 0:
                tl.store(slot + o_k, b_k)
                tl.store(slot + K + V, b_g)
                tl.store(slot + K + V + 1, b_beta)
            tl.store(slot + K + o_v, b_v)
        b_h *= tl.exp(b_g)
        b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            out + ((i_n.to(tl.int64) * T + t) * HV + i_hv) * V + o_v,
            b_o.to(out.dtype.element_ty),
        )
        if t == 0:
            tl.store(states + dst + tile, b_h)


@triton.jit
def _gdn_commit_kernel(
    states,
    stored,
    committed,
    accepted,
    layer_offset,
    pending,
    T: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    count = tl.load(accepted + i_n)
    if count > 0:
        o_k = tl.arange(0, K)
        o_v = i_v * BV + tl.arange(0, BV)
        head = i_hv * K * V
        tile = o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(states + tl.load(stored + i_n) + layer_offset + head + tile)
        for t in tl.static_range(T - 1):
            if t < count:
                slot = pending + ((i_n * (T - 1) + t) * HV + i_hv).to(tl.int64) * (
                    K + V + 2
                )
                b_k = tl.load(slot + o_k)
                b_v = tl.load(slot + K + o_v)
                b_g = tl.load(slot + K + V)
                b_beta = tl.load(slot + K + V + 1)
                b_h *= tl.exp(b_g)
                b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_beta
                b_h += b_k[:, None] * b_v[None, :]
        tl.store(states + tl.load(committed + i_n) + layer_offset + head + tile, b_h)


@triton.jit
def _conv_decode_kernel(
    x,
    weight,
    states,
    source,
    destination,
    layer_offset,
    out,
    C: tl.constexpr,
    W: tl.constexpr,
    BC: tl.constexpr,
):
    """One causal depthwise-conv step + SiLU; shifts the slot's window in place."""
    i_c = tl.program_id(0)
    i_n = tl.program_id(1)
    o_c = i_c * BC + tl.arange(0, BC)
    valid = o_c < C
    src = states + tl.load(source + i_n) + layer_offset + o_c * (W - 1)
    dst = states + tl.load(destination + i_n) + layer_offset + o_c * (W - 1)
    tl.static_assert(W == 4, "chandra-2 uses a width-4 convolution")
    b_x = tl.load(x + i_n.to(tl.int64) * C + o_c, valid, other=0)
    s0 = tl.load(src, valid, other=0)
    s1 = tl.load(src + 1, valid, other=0)
    s2 = tl.load(src + 2, valid, other=0)
    acc = s0.to(tl.float32) * tl.load(weight + o_c * W, valid, other=0).to(tl.float32)
    acc += s1.to(tl.float32) * tl.load(weight + o_c * W + 1, valid, other=0).to(
        tl.float32
    )
    acc += s2.to(tl.float32) * tl.load(weight + o_c * W + 2, valid, other=0).to(
        tl.float32
    )
    acc += b_x.to(tl.float32) * tl.load(weight + o_c * W + 3, valid, other=0).to(
        tl.float32
    )
    tl.debug_barrier()
    tl.store(dst, s1, valid)
    tl.store(dst + 1, s2, valid)
    tl.store(dst + 2, b_x, valid)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + i_n.to(tl.int64) * C + o_c,
        (acc * tl.sigmoid(acc)).to(out.dtype.element_ty),
        valid,
    )


def conv_decode(
    x: Tensor, weight: Tensor, pool: RecurrentStatePool, linear_layer: int
) -> Tensor:
    """x: [B, C] projected inputs; weight: [C, 1, W]. Returns silu(conv) as [B, C]."""
    batch, channels = x.shape
    width = weight.shape[-1]
    x = x.contiguous()
    out = torch.empty((batch, channels), dtype=x.dtype, device=x.device)
    block = 1024
    cast(Any, _conv_decode_kernel)[(triton.cdiv(channels, block), batch)](
        x,
        weight,
        pool.convolution_states,
        pool.convolution_source,
        pool.convolution_destination,
        linear_layer * pool.convolution_layer_stride,
        out,
        C=channels,
        W=width,
        BC=block,
        num_warps=4,
    )
    return out


def gdn_decode(
    mixed: Tensor,
    a: Tensor,
    b: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    pool: RecurrentStatePool,
    linear_layer: int,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    block_v: int = 32,
) -> Tensor:
    """mixed: [B, T, 2*H*K + HV*V] post-conv activations; a, b: [B, T, HV] raw projections.

    Returns [B, T, HV, V]. Writes the state after token one to the destination.
    """

    batch, tokens = mixed.shape[:2]
    out = torch.empty(
        (batch, tokens, num_value_heads, value_dim),
        dtype=mixed.dtype,
        device=mixed.device,
    )
    source, destination, layer_offset = pool.for_layer(linear_layer)
    pending = out
    if tokens > 1:
        if pool.pending is None:
            raise ValueError("multi-token GDN decode requires a pending buffer")
        pending = pool.pending[linear_layer]
    grid = (value_dim // block_v, batch * num_value_heads)
    cast(Any, _gdn_decode_kernel)[grid](
        mixed,
        mixed.stride(0),
        mixed.stride(1),
        a,
        b,
        a.stride(0),
        a.stride(1),
        A_log,
        dt_bias,
        pool.states,
        source,
        destination,
        layer_offset,
        pending,
        out,
        key_dim**-0.5,
        T=tokens,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BV=block_v,
        num_warps=4,
    )
    return out


def gdn_commit(
    pool: RecurrentStatePool,
    linear_layers: int,
    stored: Tensor,
    committed: Tensor,
    accepted: Tensor,
    *,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    block_v: int = 32,
) -> None:
    """Apply each row's first `accepted` pending updates to its stored state.

    stored/committed: [B] element offsets of the state written by verification
    and of the committed position's slot. Rows with zero accepted drafts keep
    the verified state where it is.
    """

    if pool.pending is None:
        raise ValueError("gdn_commit requires a pending buffer")
    batch = accepted.shape[0]
    tokens = pool.pending.shape[2] + 1
    grid = (value_dim // block_v, batch * num_value_heads)
    for layer in range(linear_layers):
        cast(Any, _gdn_commit_kernel)[grid](
            pool.states,
            stored,
            committed,
            accepted,
            layer * pool.layer_stride,
            pool.pending[layer],
            T=tokens,
            HV=num_value_heads,
            K=key_dim,
            V=value_dim,
            BV=block_v,
            num_warps=4,
        )


__all__ = ["RecurrentStatePool", "conv_decode", "gdn_commit", "gdn_decode"]
