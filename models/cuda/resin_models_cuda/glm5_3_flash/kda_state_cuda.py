# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted from flash-linear-attention 0.5.2, fla/ops/kda/fused_recurrent.py.
# Original code is MIT licensed; see LICENSE.fla in this directory.
"""Accepted-prefix KDA state updates without discarded attention outputs.

Adapted from FLA 0.5.2's fused recurrent kernel. Keep its [K,V] reduction
layout and FP32 recurrence; integrate Resin's exact FP32 beta sigmoid.
Accepted counts, packed offsets and strides remain device/runtime inputs.
"""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from triton.language.extra.cuda import libdevice


@triton.jit(
    do_not_specialize=[
        "k0",
        "k1",
        "v0",
        "v1",
        "g0",
        "g1",
        "b0",
        "b1",
        "c0",
        "lower_bound",
    ]
)
def _accepted_state(
    Key,
    Value,
    Gate,
    Beta,
    A,
    Bias,
    Initial,
    Final,
    Offsets,
    Counts,
    k0,
    k1,
    v0,
    v1,
    g0,
    g1,
    b0,
    b1,
    c0,
    lower_bound,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BOUNDED: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, nh = pid % nv, pid // nv
    row, head = nh // H, nh % H
    start = tl.load(Offsets + row).to(tl.int64)
    end = tl.load(Offsets + row + 1).to(tl.int64)
    accepted = tl.load(Counts + row * c0).to(tl.int64)
    length = end - start
    key = tl.arange(0, BK)
    value = iv * BV + tl.arange(0, BV)
    valid = (key[:, None] < K) & (value[None, :] < V)
    address = (row * H + head) * K * V + key[:, None] * V + value[None, :]
    initial = tl.load(Initial + address, valid, other=0).to(tl.float32)
    if length == 0:
        # Pinned FLA returns without writing; Resin preserves the original row.
        tl.store(Final + address, initial, valid)
    else:
        state = tl.full((BK, BV), 0, tl.float32) + initial
        limit = tl.minimum(accepted, length)
        for token in tl.range(0, limit, num_stages=2):
            current = start + token
            k = tl.load(
                Key + current * k0 + head * k1 + key,
                key < K,
                other=0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            v = tl.load(
                Value + current * v0 + head * v1 + value,
                value < V,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
            raw_gate = tl.load(
                Gate + current * g0 + head * g1 + key,
                key < K,
                other=0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            raw_beta = tl.load(
                Beta + current * b0 + head * b1, eviction_policy="evict_last"
            ).to(tl.float32)
            a = tl.load(A + head).to(tl.float32)
            bias = tl.load(Bias + head * K + key, key < K, other=0).to(tl.float32)
            raw_gate = raw_gate + bias
            if BOUNDED:
                gate = lower_bound * tl.sigmoid(exp(a) * raw_gate)
            else:
                gate = -exp(a) * softplus(raw_gate)
            state *= exp(gate[:, None])
            v -= tl.sum(state * k[:, None], 0)
            # Same FP32 boundary as Resin's separate beta_sigmoid kernel.
            beta = tl.div_rn(1.0, 1.0 + libdevice.exp(-raw_beta))
            v *= beta
            state += k[:, None] * v[None, :]
        tl.store(Final + address, state, valid)


def accepted_kda_state(
    key,
    value,
    gate,
    beta,
    a,
    bias,
    initial,
    offsets,
    counts,
    lower_bound,
    *,
    out=None,
    initial_is_owned: bool = False,
):
    """Return state for counts in [1, verification width].

    Counts are supplied by speculative acceptance. Empty sequence intervals copy
    their original state, including idle/padded attention replicas.

    The default owns independent output storage. A tape that has proved its
    initial state is disjoint, owned gather scratch may explicitly overwrite it:
    ``out=initial, initial_is_owned=True``. Engine page views are never valid
    overwrite destinations. Each program reads its entire disjoint value tile
    before storing; it does not read another program's value columns.
    """
    if not initial.is_cuda or initial.ndim != 4:
        raise ValueError("KDA rollback requires a four-dimensional CUDA state")
    rows, heads, kd, vd = initial.shape
    if not initial.is_contiguous() or initial.dtype != torch.float32:
        raise ValueError("KDA rollback requires the existing packed FP32 initial state")
    if key.shape != gate.shape or key.shape[1:] != (heads, kd):
        raise ValueError("key/gate geometry differs")
    if value.shape != (key.shape[0], heads, vd) or beta.shape != key.shape[:2]:
        raise ValueError("value/beta geometry differs")
    if any(x.stride(-1) != 1 for x in (key, value, gate)):
        raise ValueError("KDA rollback requires unit channel stride")
    if (
        counts.shape != (rows,)
        or offsets.shape != (rows + 1,)
        or counts.dtype not in (torch.int32, torch.int64)
        or offsets.dtype not in (torch.int32, torch.int64)
        or not offsets.is_contiguous()
    ):
        raise ValueError("KDA rollback requires row-aligned counts and packed offsets")
    if a.numel() != heads or bias.numel() != heads * kd:
        raise ValueError("KDA rollback gate parameters differ from head geometry")
    if any(
        x.device != initial.device
        for x in (key, value, gate, beta, a, bias, offsets, counts)
    ):
        raise ValueError("KDA rollback inputs must share the state device")
    if not a.is_contiguous() or not bias.is_contiguous():
        raise ValueError("KDA rollback gate parameters must be contiguous")
    if out is None:
        final = torch.empty_like(initial)
    else:
        if not initial_is_owned or out is not initial:
            raise ValueError(
                "KDA output reuse requires the explicitly owned initial tensor"
            )
        final = out
    if not rows:
        return final
    with torch.cuda.device(initial.device):
        cast(Any, _accepted_state)[(triton.cdiv(vd, 32) * rows * heads,)](
            key,
            value,
            gate,
            beta,
            a,
            bias,
            initial,
            final,
            offsets,
            counts,
            *key.stride()[:2],
            *value.stride()[:2],
            *gate.stride()[:2],
            *beta.stride(),
            counts.stride(0),
            0.0 if lower_bound is None else lower_bound,
            heads,
            kd,
            vd,
            triton.next_power_of_2(kd),
            32,
            lower_bound is not None,
            num_warps=4,
            num_stages=2,
        )
    return final
