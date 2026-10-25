"""Fused GLM RMS normalization; token count is runtime data, never a JIT key."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _normalize(
    Source,
    Weight,
    Gate,
    Output,
    rows,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GATED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, BLOCK)
    offset = row[:, None] * WIDTH + column[None, :]
    valid = (row[:, None] < rows) & (column[None, :] < WIDTH)
    value = tl.load(Source + offset, valid, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(value * value, 1) / WIDTH + EPS)
    normalized = value * inverse[:, None]
    weight = tl.load(Weight + column, column < WIDTH, other=0).to(tl.float32)
    if GATED:
        gate = tl.load(Gate + offset, valid, other=0).to(tl.float32)
        result = (normalized * weight[None, :]) * tl.sigmoid(gate)
    else:
        result = normalized * weight[None, :]
    tl.store(Output + offset, result, valid)


def normalize(value, weight, eps, *, gate=None):
    value = value.contiguous()
    if gate is not None:
        gate = gate.contiguous()
        if gate.shape != value.shape:
            raise ValueError("normalization gate must match the input shape")
    width = value.shape[-1]
    if weight.numel() != width:
        raise ValueError("normalization weight must match the input width")
    rows = value.numel() // width
    result = torch.empty_like(value)
    if rows:
        block_rows = 4 if width <= 256 else 1
        cast(Any, _normalize)[(triton.cdiv(rows, block_rows),)](
            value,
            weight.contiguous(),
            gate if gate is not None else value,
            result,
            rows,
            eps,
            width,
            triton.next_power_of_2(width),
            block_rows,
            gate is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result
