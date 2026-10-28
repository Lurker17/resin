"""Triton implementations for Gemma Edge; excluded from CPU import paths."""

from __future__ import annotations

from typing import Any, cast

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _sparse_gather_dot(
    hidden,
    weights,
    tokens,
    output,
    WIDTH: tl.constexpr,
    CANDIDATES: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    TOKEN_ROW_STRIDE: tl.constexpr,
    TOKEN_COL_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    row = tl.program_id(0)
    candidates = tl.program_id(1) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    ids = tl.load(
        tokens + row * TOKEN_ROW_STRIDE + candidates * TOKEN_COL_STRIDE,
        candidates < CANDIDATES,
        other=0,
    )
    vector = tl.load(
        hidden + row * HIDDEN_STRIDE + channels,
        channels < WIDTH,
        other=0,
    ).to(tl.float32)
    matrix = tl.load(
        weights + ids[:, None] * WEIGHT_STRIDE + channels[None, :],
        (candidates[:, None] < CANDIDATES) & (channels[None, :] < WIDTH),
        other=0,
    ).to(tl.float32)
    scores = tl.sum(matrix * vector[None, :], axis=1)
    tl.store(output + row * CANDIDATES + candidates, scores, candidates < CANDIDATES)


def launch_sparse_gather_dot(hidden, weights, tokens, output):
    width = hidden.shape[1]
    candidates = tokens.shape[1]
    block_tokens = 16
    cast(Any, _sparse_gather_dot)[(len(hidden), triton.cdiv(candidates, block_tokens))](
        hidden,
        weights,
        tokens,
        output,
        WIDTH=width,
        CANDIDATES=candidates,
        HIDDEN_STRIDE=hidden.stride(0),
        WEIGHT_STRIDE=weights.stride(0),
        TOKEN_ROW_STRIDE=tokens.stride(0),
        TOKEN_COL_STRIDE=tokens.stride(1),
        BLOCK_TOKENS=block_tokens,
        BLOCK_CHANNELS=triton.next_power_of_2(width),
        num_warps=4,
    )


@triton.jit
def _gelu_tanh_mul(gate, up, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(gate + offsets, offsets < elements, other=0).to(tl.float32)
    y = tl.load(up + offsets, offsets < elements, other=0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    activated = (0.5 * x) * (1.0 + libdevice.tanh(inner))
    rounded = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, rounded * y, offsets < elements)


def launch_gelu_tanh_mul(gate, up, output):
    elements = gate.numel()
    cast(Any, _gelu_tanh_mul)[(triton.cdiv(elements, 1024),)](
        gate,
        up,
        output,
        elements,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit(
    do_not_specialize=[
        "query_token_stride",
        "query_head_stride",
        "key_token_stride",
        "key_head_stride",
        "cos_token_stride",
        "cos_channel_stride",
        "sin_token_stride",
        "sin_channel_stride",
        "eps",
    ],
    do_not_specialize_on_alignment=[
        "query",
        "key",
        "query_weight",
        "key_weight",
        "cos",
        "sin",
        "query_output",
        "key_output",
    ],
)
def _strided_qk_rmsnorm_rope(
    query,
    key,
    query_weight,
    key_weight,
    cos,
    sin,
    query_output,
    key_output,
    query_token_stride,
    query_head_stride,
    key_token_stride,
    key_head_stride,
    cos_token_stride,
    cos_channel_stride,
    sin_token_stride,
    sin_channel_stride,
    eps,
    QUERY_HEADS: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // (QUERY_HEADS + KEY_HEADS)
    head = program % (QUERY_HEADS + KEY_HEADS)
    channel = tl.arange(0, WIDTH)
    partner = (channel + WIDTH // 2) % WIDTH
    if KEY_HEADS:
        is_query = head < QUERY_HEADS
        source = tl.where(
            is_query,
            query + token * query_token_stride + head * query_head_stride,
            key + token * key_token_stride + (head - QUERY_HEADS) * key_head_stride,
        )
        weight = tl.where(is_query, query_weight, key_weight)
        target = tl.where(
            is_query,
            query_output + (token * QUERY_HEADS + head) * WIDTH,
            key_output + (token * KEY_HEADS + head - QUERY_HEADS) * WIDTH,
        )
    else:
        source = query + token * query_token_stride + head * query_head_stride
        weight = query_weight
        target = query_output + (token * QUERY_HEADS + head) * WIDTH
    x = tl.load(source + channel).to(tl.float32)
    squared_sum = tl.sum(x * x, axis=0)
    inverse = tl.rsqrt(squared_sum / WIDTH + eps)
    w = tl.load(weight + channel).to(tl.float32)
    normalized = ((x * inverse) * w).to(tl.bfloat16)
    # Re-read the opposite half rather than gathering registers across warps.
    # Both channels use the same per-head reduction and multiplicative scale.
    opposite = tl.load(source + partner).to(tl.float32)
    opposite_weight = tl.load(weight + partner).to(tl.float32)
    normalized_opposite = ((opposite * inverse) * opposite_weight).to(tl.bfloat16)
    rotated = tl.where(channel < WIDTH // 2, -normalized_opposite, normalized_opposite)
    cosine = tl.load(cos + token * cos_token_stride + channel * cos_channel_stride).to(
        tl.float32
    )
    sine = tl.load(sin + token * sin_token_stride + channel * sin_channel_stride).to(
        tl.float32
    )
    # PyTorch materializes both BF16 products before adding them. Keep these
    # explicit conversions and disable FMA at launch: a single FP32 FMA would
    # silently change the model's arithmetic contract.
    first = (normalized.to(tl.float32) * cosine).to(tl.bfloat16)
    second = (rotated.to(tl.float32) * sine).to(tl.bfloat16)
    result = (first.to(tl.float32) + second.to(tl.float32)).to(tl.bfloat16)
    tl.store(target + channel, result)


def launch_norm_rope(
    query, key, query_weight, key_weight, cos, sin, query_output, key_output, eps
):
    key_heads = 0 if key is None else key.shape[1]
    cast(Any, _strided_qk_rmsnorm_rope)[
        (query.shape[0] * (query.shape[1] + key_heads),)
    ](
        query,
        query if key is None else key,
        query_weight,
        query_weight if key_weight is None else key_weight,
        cos,
        sin,
        query_output,
        query_output if key_output is None else key_output,
        query.stride(0),
        query.stride(1),
        0 if key is None else key.stride(0),
        0 if key is None else key.stride(1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        eps,
        QUERY_HEADS=query.shape[1],
        KEY_HEADS=key_heads,
        WIDTH=query.shape[2],
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit(
    do_not_specialize=["hidden_stride", "residual_stride", "eps"],
    do_not_specialize_on_alignment=["hidden", "weight", "residual", "scale", "output"],
)
def _norm_residual(
    hidden,
    weight,
    residual,
    scale,
    output,
    hidden_stride,
    residual_stride,
    eps,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    values = tl.load(hidden + row * hidden_stride + channel, valid, other=0).to(
        tl.float32
    )
    squared_sum = tl.sum(values * values, axis=0)
    inverse = tl.rsqrt(squared_sum / WIDTH + eps)
    weights = tl.load(weight + channel, valid, other=0).to(tl.float32)
    normalized = ((values * inverse) * weights).to(tl.bfloat16)
    previous = tl.load(residual + row * residual_stride + channel, valid, other=0).to(
        tl.float32
    )
    added = (normalized.to(tl.float32) + previous).to(tl.bfloat16)
    if HAS_SCALE:
        multiplier = tl.load(scale).to(tl.float32)
        added = (added.to(tl.float32) * multiplier).to(tl.bfloat16)
    tl.store(output + row * WIDTH + channel, added, valid)


def launch_norm_residual(hidden, weight, residual, scale, output, eps):
    width = hidden.shape[1]
    cast(Any, _norm_residual)[(hidden.shape[0],)](
        hidden,
        weight,
        residual,
        weight if scale is None else scale,
        output,
        hidden.stride(0),
        residual.stride(0),
        eps,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        HAS_SCALE=scale is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit
def _reserved_metadata(
    positions,
    local_first,
    assistant_local_indptr,
    assistant_global_indptr,
    verify_local_indptr,
    verify_global_indptr,
    assistant_local_last,
    assistant_global_last,
    verify_local_last,
    verify_global_last,
    sampling_offsets,
    ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    SAMPLE_STRIDE: tl.constexpr,
):
    row = tl.arange(0, BLOCK_ROWS)
    valid = row < ROWS
    position = tl.load(positions + row, valid, other=1)
    first = tl.load(local_first + row, valid, other=0)
    assistant_first = tl.maximum(0, position - WINDOW) // 64
    assistant_end = (position + 63) // 64
    verify_end = (position + QUERY_WIDTH + 63) // 64
    al = tl.where(valid, assistant_end - assistant_first, 0).to(tl.int32)
    ag = tl.where(valid, assistant_end, 0).to(tl.int32)
    vl = tl.where(valid, verify_end - first, 0).to(tl.int32)
    vg = tl.where(valid, verify_end, 0).to(tl.int32)
    als, ags = tl.cumsum(al), tl.cumsum(ag)
    vls, vgs = tl.cumsum(vl), tl.cumsum(vg)
    tl.store(assistant_local_indptr + row, als - al, valid)
    tl.store(assistant_global_indptr + row, ags - ag, valid)
    tl.store(verify_local_indptr + row, vls - vl, valid)
    tl.store(verify_global_indptr + row, vgs - vg, valid)
    tl.store(assistant_local_indptr + ROWS, tl.sum(al, 0))
    tl.store(assistant_global_indptr + ROWS, tl.sum(ag, 0))
    tl.store(verify_local_indptr + ROWS, tl.sum(vl, 0))
    tl.store(verify_global_indptr + ROWS, tl.sum(vg, 0))
    assistant_last = ((position - 1) % 64 + 1).to(tl.int32)
    verify_last = ((position + QUERY_WIDTH - 1) % 64 + 1).to(tl.int32)
    tl.store(assistant_local_last + row, assistant_last, valid)
    tl.store(assistant_global_last + row, assistant_last, valid)
    tl.store(verify_local_last + row, verify_last, valid)
    tl.store(verify_global_last + row, verify_last, valid)
    tl.store(sampling_offsets + row, (position + 1) * SAMPLE_STRIDE, valid)


@triton.jit
def _reserved_page_rows(
    positions,
    first_pages,
    page_table,
    assistant_indptr,
    verify_indptr,
    assistant_indices,
    verify_indices,
    append_pages,
    append_offsets,
    TABLE_WIDTH: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    LOCAL: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    position = tl.load(positions + row)
    first = tl.load(first_pages + row)
    assistant_first = tl.maximum(0, position - WINDOW) // 64 if LOCAL else 0
    assistant_start = tl.load(assistant_indptr + row)
    assistant_count = tl.load(assistant_indptr + row + 1) - assistant_start
    verify_start = tl.load(verify_indptr + row)
    verify_count = tl.load(verify_indptr + row + 1) - verify_start
    columns = tl.arange(0, BLOCK_PAGES)
    assistant_pages = tl.load(
        page_table + row * TABLE_WIDTH + columns + assistant_first - first,
        columns < assistant_count,
        other=0,
    )
    verify_pages = tl.load(
        page_table + row * TABLE_WIDTH + columns,
        columns < verify_count,
        other=0,
    )
    tl.store(
        assistant_indices + assistant_start + columns,
        assistant_pages,
        columns < assistant_count,
    )
    tl.store(
        verify_indices + verify_start + columns, verify_pages, columns < verify_count
    )
    query = tl.arange(0, triton.next_power_of_2(QUERY_WIDTH))
    absolute = position + query
    pages = tl.load(
        page_table + row * TABLE_WIDTH + absolute // 64 - first,
        query < QUERY_WIDTH,
        other=0,
    )
    tl.store(append_pages + row * QUERY_WIDTH + query, pages, query < QUERY_WIDTH)
    tl.store(
        append_offsets + row * QUERY_WIDTH + query, absolute % 64, query < QUERY_WIDTH
    )


def launch_reserved_paged_metadata(inputs, query_width, window, sampling_stride):
    positions = inputs["positions"]
    rows = positions.numel()
    cast(Any, _reserved_metadata)[(1,)](
        positions,
        inputs["envelope_local_first_pages"],
        inputs["assistant_local_kv_indptr"],
        inputs["assistant_global_kv_indptr"],
        inputs["verify_local_kv_indptr"],
        inputs["verify_global_kv_indptr"],
        inputs["assistant_local_last_page_lengths"],
        inputs["assistant_global_last_page_lengths"],
        inputs["verify_local_last_page_lengths"],
        inputs["verify_global_last_page_lengths"],
        inputs["sampling_offsets"],
        rows,
        triton.next_power_of_2(rows),
        query_width,
        window,
        sampling_stride,
        num_warps=4,
    )
    for kind in ("local", "global"):
        table = inputs[f"envelope_{kind}_page_table"]
        cast(Any, _reserved_page_rows)[(rows,)](
            positions,
            inputs[f"envelope_{kind}_first_pages"],
            table,
            inputs[f"assistant_{kind}_kv_indptr"],
            inputs[f"verify_{kind}_kv_indptr"],
            inputs[f"assistant_{kind}_page_indices"],
            inputs[f"verify_{kind}_page_indices"],
            inputs[f"verify_{kind}_append_pages"],
            inputs[f"verify_{kind}_append_offsets"],
            table.shape[1],
            query_width,
            window,
            kind == "local",
            triton.next_power_of_2(table.shape[1]),
            num_warps=4,
        )
