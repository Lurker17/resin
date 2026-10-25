"""KDA input preparation and preservation of unwritten idle final-state rows."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["rows", "stride"])
def _beta_sigmoid(
    Source, Output, rows, stride, HEADS: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, head = index // HEADS, index % HEADS
    value = tl.load(Source + row * stride + head, row < rows, other=0).to(tl.float32)
    # Match the separate Torch FP32 sigmoid, including its division rounding.
    result = tl.div_rn(1.0, 1.0 + libdevice.exp(-value))
    tl.store(Output + index, result, row < rows)


def beta_sigmoid(value):
    if value.ndim != 2 or value.stride(1) != 1:
        raise ValueError("KDA beta must be a row-strided token/head matrix")
    output = torch.empty(value.shape, dtype=torch.float32, device=value.device)
    if value.numel():
        cast(Any, _beta_sigmoid)[(triton.cdiv(value.numel(), 256),)](
            value,
            output,
            value.shape[0],
            value.stride(0),
            value.shape[1],
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


@triton.jit
def _preserve_idle(
    Initial,
    Final,
    Offsets,
    INITIAL_STRIDES: tl.constexpr,
    FINAL_STRIDES: tl.constexpr,
    HEADS: tl.constexpr,
    KEY: tl.constexpr,
    VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(Offsets + row)
    end = tl.load(Offsets + row + 1)
    # FLA already wrote every active row. Do not read or rewrite that state.
    if start == end:
        index = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        head = index // (KEY * VALUE)
        key = index // VALUE % KEY
        value = index % VALUE
        source = (
            row * INITIAL_STRIDES[0]
            + head * INITIAL_STRIDES[1]
            + key * INITIAL_STRIDES[2]
            + value * INITIAL_STRIDES[3]
        )
        destination = (
            row * FINAL_STRIDES[0]
            + head * FINAL_STRIDES[1]
            + key * FINAL_STRIDES[2]
            + value * FINAL_STRIDES[3]
        )
        data = tl.load(Initial + source, index < HEADS * KEY * VALUE, other=0)
        tl.store(Final + destination, data, index < HEADS * KEY * VALUE)


def preserve_idle(initial, final, offsets):
    """Fill only the independent FLA output's unwritten rows; never mutate input."""
    if initial.shape != final.shape or initial.ndim != 4:
        raise ValueError(
            "KDA initial/final states must have matching four-dimensional shapes"
        )
    if initial.untyped_storage().data_ptr() == final.untyped_storage().data_ptr():
        raise ValueError("KDA final state must own storage independent of its input")
    rows, heads, key, value = initial.shape
    if rows:
        block = 16384
        cast(Any, _preserve_idle)[(rows, triton.cdiv(heads * key * value, block))](
            initial,
            final,
            offsets,
            initial.stride(),
            final.stride(),
            heads,
            key,
            value,
            block,
            num_warps=8,
        )
    return final
