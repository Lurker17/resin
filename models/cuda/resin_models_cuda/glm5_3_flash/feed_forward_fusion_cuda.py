"""Fused GLM SwiGLU and routing arithmetic, preserving FP32 operation order.

Top-k stays in PyTorch. The eight-value normalization sum follows CUDA ATen's
Reduce.cuh block_x_reduce: shuffle-down offsets 4, 2, 1. Explicit libdevice exp
and round-to-nearest division preserve the original sigmoid/SiLU operations;
approximate Triton sigmoid and reciprocal are deliberately not substituted.
"""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["rows", "gate_row", "gate_col", "up_row", "up_col"])
def _swiglu(
    Gate,
    Up,
    Output,
    rows,
    gate_row,
    gate_col,
    up_row,
    up_col,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // WIDTH, index % WIDTH
    valid = row < rows
    gate = tl.load(Gate + row * gate_row + col * gate_col, valid, other=0).to(
        tl.float32
    )
    up = tl.load(Up + row * up_row + col * up_col, valid, other=0).to(tl.float32)
    # torch.clamp converts its scalar bounds to the input dtype before the
    # operation. This matters for nonrepresentable BF16/FP16 bounds too.
    high = tl.full((), LIMIT, tl.float32).to(Gate.dtype.element_ty).to(tl.float32)
    gate = tl.where(gate > high, high, gate)
    up = tl.where(up < -high, -high, up)
    up = tl.where(up > high, high, up)
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(Output + index, activated * up, valid)


@triton.jit(do_not_specialize=["rows", "row_stride", "col_stride", "bias_stride"])
def _route_scores(
    Logits,
    Bias,
    Scores,
    Corrected,
    rows,
    row_stride,
    col_stride,
    bias_stride,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // EXPERTS, index % EXPERTS
    valid = row < rows
    value = tl.load(Logits + row * row_stride + col * col_stride, valid, other=0)
    bias = tl.load(Bias + col * bias_stride, valid, other=0)
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-value))
    tl.store(Scores + index, score, valid)
    tl.store(Corrected + index, score + bias, valid)


@triton.jit(do_not_specialize=["rows", "score_row", "index_row"])
def _route_weights(
    Scores,
    Indices,
    OutputIds,
    OutputWeights,
    rows,
    score_row,
    index_row,
    NORMALIZE: tl.constexpr,
    SCALING: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, 8)
    ids = tl.load(
        Indices + row[:, None] * index_row + col[None, :], row[:, None] < rows, other=0
    )
    weights = tl.load(
        Scores + row[:, None] * score_row + ids, row[:, None] < rows, other=0
    )
    if NORMALIZE:
        # Each CUDA ATen reduction lane initially owns one of the eight values.
        # Its 4/2/1 shuffle order is explicit here, not a compiler-chosen sum.
        v0 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 0, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v1 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 1, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v2 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 2, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v3 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 3, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v4 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 4, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v5 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 5, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v6 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 6, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        v7 = tl.gather(weights, tl.full((BLOCK_ROWS, 1), 7, tl.int32), 1).reshape(
            (BLOCK_ROWS,)
        )
        total = ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7))
        weights = tl.div_rn(weights, total[:, None] + 1e-20)
    tl.store(OutputIds + row[:, None] * 8 + col[None, :], ids, row[:, None] < rows)
    tl.store(
        OutputWeights + row[:, None] * 8 + col[None, :],
        weights * SCALING,
        row[:, None] < rows,
    )


def swiglu(gate, up, limit):
    rows, width = gate.shape
    output = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    if rows and width:
        cast(Any, _swiglu)[(triton.cdiv(rows * width, 256),)](
            gate,
            up,
            output,
            rows,
            *gate.stride(),
            *up.stride(),
            width,
            limit,
            256,
            enable_fp_fusion=False,
        )
    return output


def route_scores(logits, bias):
    rows, experts = logits.shape
    scores = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
    corrected = torch.empty_like(scores)
    if rows:
        cast(Any, _route_scores)[(triton.cdiv(rows * experts, 256),)](
            logits,
            bias,
            scores,
            corrected,
            rows,
            *logits.stride(),
            bias.stride(0),
            experts,
            256,
            enable_fp_fusion=False,
        )
    return scores, corrected


def route_weights(scores, indices, normalize, scaling):
    rows = scores.shape[0]
    output_ids = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    output_weights = torch.empty(
        indices.shape, dtype=torch.float32, device=indices.device
    )
    if rows:
        cast(Any, _route_weights)[(triton.cdiv(rows, 16),)](
            scores,
            indices,
            output_ids,
            output_weights,
            rows,
            scores.stride(0),
            indices.stride(0),
            normalize,
            scaling,
            16,
            enable_fp_fusion=False,
        )
    return output_ids, output_weights
