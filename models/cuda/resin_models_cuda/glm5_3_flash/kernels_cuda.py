"""Runtime-length CUDA page and conv adapters for GLM-5.3-Flash.

All sequence lengths and page table widths are runtime values. Compilation keys
contain model geometry and tile choices, not request context or sampling values.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _l2norm(
    Source,
    Output,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, BLOCK_WIDTH)
    valid = (row[:, None] < rows) & (column[None, :] < WIDTH)
    value = tl.load(Source + row[:, None] * WIDTH + column[None, :], valid, other=0).to(
        tl.float32
    )
    inverse = 1.0 / tl.sqrt(tl.sum(value * value, 1) + 1e-6)
    tl.store(
        Output + row[:, None] * WIDTH + column[None, :],
        value * inverse[:, None],
        valid,
    )


def l2norm_cuda(source):
    """FLA-compatible BF16 normalization without a token-count tuning key."""
    source = source.contiguous()
    output = torch.empty_like(source)
    width, rows = source.shape[-1], source.numel() // source.shape[-1]
    cast(Any, _l2norm)[(triton.cdiv(rows, 16),)](
        source, output, rows, width, 16, triton.next_power_of_2(width), num_warps=4
    )
    return output


@triton.jit(
    do_not_specialize=[
        "state_stride",
        "final_stride",
        "qkv_row_stride",
        "qkv_group_stride",
        "qkv_channel_stride",
        "out_row_stride",
        "out_group_stride",
    ]
)
def _conv(
    QKV,
    Weight,
    Initial,
    Final,
    Offsets,
    Sequences,
    Out,
    state_stride,
    final_stride,
    qkv_row_stride,
    qkv_group_stride,
    qkv_channel_stride,
    out_row_stride,
    out_group_stride,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    channel = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(Sequences + token).to(tl.int64)
    start = tl.load(Offsets + seq).to(tl.int64)
    end = tl.load(Offsets + seq + 1).to(tl.int64)
    valid = channel < CHANNELS
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(4):
        source = token - 3 + j
        current = tl.load(
            QKV
            + source * qkv_row_stride
            + group * qkv_group_stride
            + channel * qkv_channel_stride,
            valid & (source >= start),
            other=0,
        ).to(tl.float32)
        previous = tl.load(
            Initial
            + seq * state_stride
            + (group * CHANNELS + channel) * 3
            + source
            - start
            + 3,
            valid & (source < start),
            other=0,
        ).to(tl.float32)
        weight = tl.load(Weight + (group * CHANNELS + channel) * 4 + j, valid, other=0)
        # The qualified compact kernel contracts each tap into one RN FMA.
        # Runtime strides can inhibit automatic contraction (separate mul/add
        # then changes BF16 outputs). Preserve the original rounding explicitly.
        acc = tl.fma(current + previous, weight, acc)
    value = acc * tl.sigmoid(acc)
    tl.store(
        Out + token * out_row_stride + group * out_group_stride + channel, value, valid
    )
    if token == end - 1:
        for j in tl.static_range(3):
            source = end - 3 + j
            final_current = tl.load(
                QKV
                + source * qkv_row_stride
                + group * qkv_group_stride
                + channel * qkv_channel_stride,
                valid & (source >= start),
                other=0,
            )
            final_previous = tl.load(
                Initial
                + seq * state_stride
                + (group * CHANNELS + channel) * 3
                + source
                - start
                + 3,
                valid & (source < start),
                other=0,
            )
            tl.store(
                Final + seq * final_stride + (group * CHANNELS + channel) * 3 + j,
                final_current + final_previous,
                valid,
            )


def convolution_cuda(qkv, weight, cache, *, planar=False):
    # The packed input projection contains beta/forget/output-gate columns
    # after QKV. Read its row stride directly; do not repack the large tensor.
    # Plane output makes each Q/K/V input to FLA contiguous without three copies.
    tokens, _, channels = qkv.shape
    shape = (3, tokens, channels) if planar else (tokens, 3, channels)
    out = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    # Empty sequences must preserve their initial state.
    final = cache.convolution.clone()
    if not tokens:
        return out, final
    cast(Any, _conv)[(qkv.shape[0], 3, triton.cdiv(channels, 256))](
        qkv,
        weight,
        cache.convolution,
        final,
        cache.cu_seqlens,
        cache.sequence_indices,
        out,
        cache.convolution.stride(0),
        final.stride(0),
        *qkv.stride(),
        out.stride(1 if planar else 0),
        out.stride(0 if planar else 1),
        channels,
        256,
    )
    return out, final


@triton.jit(do_not_specialize=["table_stride", "cache_stride"])
def _store(
    Source,
    Positions,
    Sequences,
    Table,
    Cache,
    table_stride,
    cache_stride,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    ORIGIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(Sequences + token).to(tl.int64)
    logical = tl.load(Positions + token)
    pos = logical + ORIGIN
    # Page IDs fit int32; their element offsets do not. Widen before the
    # multiplication: a 59,392-element BF16 stride overflows at page 36,158.
    page = tl.load(
        Table + seq * table_stride + pos // PAGE,
        (seq >= 0) & (logical >= 0) & (pos // PAGE < table_stride),
        other=0,
    ).to(tl.int64)
    value = tl.load(Source + token * WIDTH + col, col < WIDTH, other=0)
    tl.store(
        Cache + page * cache_stride + pos % PAGE * WIDTH + col,
        value,
        (col < WIDTH) & (page > 0),
    )


@triton.jit(
    do_not_specialize=["table_stride", "key_stride", "gate_stride", "pool_stride"]
)
def _compress(
    Positions,
    Sequences,
    Table,
    Keys,
    Gates,
    Ape,
    Pooled,
    table_stride,
    key_stride,
    gate_stride,
    pool_stride,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    ORIGIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pos = tl.load(Positions + token)
    if (pos >= 0) & ((pos + 1) % 4 == 0):
        seq = tl.load(Sequences + token).to(tl.int64)
        pool_slot = pos // 4 + ORIGIN
        page = tl.load(
            Table + seq * table_stride + pool_slot // (PAGE // 4),
            (seq >= 0) & (pool_slot // (PAGE // 4) < table_stride),
            other=0,
        ).to(tl.int64)
        channels = tl.arange(0, BLOCK)
        group = tl.arange(0, 4)
        source = pos - 3 + group + ORIGIN
        source_pages = tl.load(
            Table + seq * table_stride + source // PAGE,
            (seq >= 0) & (source // PAGE < table_stride),
            other=0,
        ).to(tl.int64)
        offsets = (source[:, None] % PAGE) * WIDTH + channels[None, :]
        valid = (source_pages[:, None] > 0) & (channels[None, :] < WIDTH)
        keys = tl.load(
            Keys + source_pages[:, None] * key_stride + offsets, valid, other=0
        )
        logits = tl.load(
            Gates + source_pages[:, None] * gate_stride + offsets, valid, other=0
        ).to(tl.float32)
        logits += tl.load(
            Ape + group[:, None] * WIDTH + channels[None, :],
            channels[None, :] < WIDTH,
            other=0,
        ).to(tl.float32)
        weights = tl.exp(logits - tl.max(logits, 0)[None, :])
        weights = weights / tl.sum(weights, 0)[None, :]
        # HF rounds softmax to the key dtype, then the product, before reduction.
        product = (weights.to(keys.dtype).to(tl.float32) * keys.to(tl.float32)).to(
            keys.dtype
        )
        pooled = tl.sum(product.to(tl.float32), 0)
        tl.store(
            Pooled + page * pool_stride + pool_slot % (PAGE // 4) * WIDTH + channels,
            pooled,
            (page > 0) & (channels < WIDTH),
        )


@triton.jit(
    do_not_specialize=["table_stride", "key_stride", "gate_stride", "pool_stride"]
)
def _compress_tail(
    Positions,
    Sequences,
    Offsets,
    Table,
    Keys,
    Gates,
    TailKeys,
    TailGates,
    TailEnds,
    Ape,
    Pooled,
    table_stride,
    key_stride,
    gate_stride,
    pool_stride,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    ORIGIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pos = tl.load(Positions + token)
    if (pos >= 0) & ((pos + 1) % 4 == 0):
        seq = tl.load(Sequences + token).to(tl.int64)
        pool_slot = pos // 4 + ORIGIN
        page = tl.load(
            Table + seq * table_stride + pool_slot // (PAGE // 4),
            (seq >= 0) & (pool_slot // (PAGE // 4) < table_stride),
            other=0,
        ).to(tl.int64)
        begin, end = tl.load(Offsets + seq), tl.load(Offsets + seq + 1)
        first = tl.load(Positions + begin, begin < end, other=-1)
        tail_end = tl.load(TailEnds + seq)
        channels, group = tl.arange(0, BLOCK), tl.arange(0, 4)
        source = pos - 3 + group
        row = begin + source - first
        current = (source >= first) & (row < end)
        prior = source - (tail_end - 3)
        prior_valid = (~current) & (prior >= 0) & (prior < 4)
        mask = channels[None, :] < WIDTH
        keys = tl.load(
            Keys + row[:, None].to(tl.int64) * key_stride + channels[None, :],
            current[:, None] & mask,
            other=0,
        ).to(tl.bfloat16)
        old_keys = tl.load(
            TailKeys + seq * 4 * WIDTH + prior[:, None] * WIDTH + channels[None, :],
            prior_valid[:, None] & mask,
            other=0,
        )
        keys = tl.where(current[:, None], keys, old_keys)
        logits = tl.load(
            Gates + row[:, None].to(tl.int64) * gate_stride + channels[None, :],
            current[:, None] & mask,
            other=0,
        ).to(tl.float32)
        old_logits = tl.load(
            TailGates + seq * 4 * WIDTH + prior[:, None] * WIDTH + channels[None, :],
            prior_valid[:, None] & mask,
            other=0,
        )
        logits = tl.where(current[:, None], logits, old_logits)
        logits += tl.load(
            Ape + group[:, None] * WIDTH + channels[None, :], mask, other=0
        ).to(tl.float32)
        weights = tl.exp(logits - tl.max(logits, 0)[None, :])
        weights = weights / tl.sum(weights, 0)[None, :]
        product = (weights.to(keys.dtype).to(tl.float32) * keys.to(tl.float32)).to(
            keys.dtype
        )
        pooled = tl.sum(product.to(tl.float32), 0)
        tl.store(
            Pooled + page * pool_stride + pool_slot % (PAGE // 4) * WIDTH + channels,
            pooled,
            (page > 0) & (channels < WIDTH),
        )


def store_sparse_cuda(latent, keys, gates, ape, cache):
    if getattr(cache, "index_tail", None) is not None:
        from resin_models_cuda.glm5_3_flash.index_tail import (
            finish_update,
            record_update,
        )

        update = record_update(cache, keys, gates)
        tail = update.initial
        cast(Any, _store)[(latent.shape[0], triton.cdiv(latent.shape[-1], 256))](
            latent,
            cache.positions,
            cache.sequence_indices,
            cache.page_table,
            cache.latent,
            cache.page_table.stride(0),
            cache.latent.stride(0),
            latent.shape[-1],
            cache.page_tokens,
            getattr(cache, "storage_origin", 0),
            256,
        )
        cast(Any, _compress_tail)[(latent.shape[0],)](
            cache.positions,
            cache.sequence_indices,
            cache.cu_seqlens,
            cache.page_table,
            keys,
            gates,
            tail.keys,
            tail.gates,
            tail.end,
            ape,
            cache.pooled_keys,
            cache.page_table.stride(0),
            keys.stride(0),
            gates.stride(0),
            cache.pooled_keys.stride(0),
            keys.shape[-1],
            cache.page_tokens,
            getattr(cache, "storage_origin", 0),
            triton.next_power_of_2(keys.shape[-1]),
        )
        finish_update(cache, update)
        return
    for source, target in (
        (latent, cache.latent),
        (keys, cache.index_keys),
        (gates, cache.index_gates),
    ):
        width = source.shape[-1]
        cast(Any, _store)[(source.shape[0], triton.cdiv(width, 256))](
            source,
            cache.positions,
            cache.sequence_indices,
            cache.page_table,
            target,
            cache.page_table.stride(0),
            target.stride(0),
            width,
            cache.page_tokens,
            getattr(cache, "storage_origin", 0),
            256,
        )
    cast(Any, _compress)[(latent.shape[0],)](
        cache.positions,
        cache.sequence_indices,
        cache.page_table,
        cache.index_keys,
        cache.index_gates,
        ape,
        cache.pooled_keys,
        cache.page_table.stride(0),
        cache.index_keys.stride(0),
        cache.index_gates.stride(0),
        cache.pooled_keys.stride(0),
        keys.shape[-1],
        cache.page_tokens,
        getattr(cache, "storage_origin", 0),
        triton.next_power_of_2(keys.shape[-1]),
    )


@triton.jit(do_not_specialize=["table_stride", "pool_stride", "pool_count"])
def _score(
    Query,
    Weights,
    Positions,
    Sequences,
    Table,
    Pooled,
    Scores,
    table_stride,
    pool_stride,
    pool_count,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    ORIGIN: tl.constexpr,
    BH: tl.constexpr,
    BP: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pools = tl.program_id(1) * BP + tl.arange(0, BP)
    heads = tl.arange(0, BH)
    dim = tl.arange(0, WIDTH)
    seq = tl.load(Sequences + token).to(tl.int64)
    pos = tl.load(Positions + token)
    pool_slots = pools + ORIGIN
    pages = tl.load(
        Table + seq * table_stride + pool_slots // (PAGE // 4),
        (pool_slots < pool_count) & (seq >= 0),
        other=0,
    ).to(tl.int64)
    valid = (pools < pool_count) & ((pools + 1) * 4 <= pos + 1) & (pages > 0)
    q = tl.load(
        Query + (token * HEADS + heads[:, None]) * WIDTH + dim[None, :],
        heads[:, None] < HEADS,
        other=0,
    )
    k = tl.load(
        Pooled
        + pages[None, :] * pool_stride
        + (pool_slots[None, :] % (PAGE // 4)) * WIDTH
        + dim[:, None],
        valid[None, :],
        other=0,
    )
    dots = tl.dot(q, k).to(tl.float32) * (WIDTH**-0.5)
    weights = tl.load(Weights + token * HEADS + heads, heads < HEADS, other=0).to(
        tl.float32
    ) * (HEADS**-0.5)
    score = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], 0)
    tl.store(
        Scores + token * pool_count + pools,
        tl.where(valid, score, -float("inf")),
        pools < pool_count,
    )


@triton.jit(do_not_specialize=["table_stride", "latent_stride", "selected_count"])
def _expand(
    Selected,
    Positions,
    Sequences,
    Table,
    Output,
    table_stride,
    latent_stride,
    selected_count,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    ORIGIN: tl.constexpr,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    seq = tl.load(Sequences + token).to(tl.int64)
    pos = tl.load(Positions + token)
    pool = tl.load(
        Selected + token * selected_count + col // 4,
        (col < selected_count * 4),
        other=-1,
    ).to(tl.int64)
    logical = pool * 4 + col % 4
    selected_valid = (
        (col < selected_count * 4) & (pool >= 0) & ((pool + 1) * 4 <= pos + 1)
    )
    # Fixed tail location keeps output shape stable when context capacity changes.
    tail = (pos + 1) // 4 * 4 + col - TOPK
    use_tail = (col >= TOPK) & (col < TOPK + 3) & (tail <= pos)
    logical = tl.where(use_tail, tail, logical)
    storage = logical + ORIGIN
    valid = (
        (selected_valid | use_tail) & (logical >= 0) & (storage // PAGE < table_stride)
    )
    page = tl.load(
        Table + seq * table_stride + storage // PAGE, valid & (seq >= 0), other=0
    ).to(tl.int64)
    physical = page * latent_stride + storage % PAGE
    tl.store(
        Output + token * OUT + col,
        tl.where(valid & (page > 0), physical, -1),
        col < OUT,
    )


def select_indices_cuda(query, weights, positions, sequences, cache, topk):
    pools = cache.page_table.shape[1] * cache.page_tokens // 4
    scores = torch.empty(
        (query.shape[0], pools), dtype=torch.float32, device=query.device
    )
    cast(Any, _score)[(query.shape[0], triton.cdiv(pools, 64))](
        query.contiguous(),
        weights.contiguous(),
        positions,
        sequences,
        cache.page_table,
        cache.pooled_keys,
        scores,
        cache.page_table.stride(0),
        cache.pooled_keys.stride(0),
        pools,
        query.shape[1],
        query.shape[2],
        cache.page_tokens,
        getattr(cache, "storage_origin", 0),
        max(16, triton.next_power_of_2(query.shape[1])),
        64,
    )
    chosen = scores.topk(min(topk // 4, pools), dim=-1, sorted=False).indices
    width = triton.cdiv(topk + 3, 8) * 8
    out = torch.empty((query.shape[0], width), dtype=torch.int32, device=query.device)
    cast(Any, _expand)[(query.shape[0],)](
        chosen,
        positions,
        sequences,
        cache.page_table,
        out,
        cache.page_table.stride(0),
        cache.latent.stride(0) // cache.latent.shape[-1],
        chosen.shape[1],
        topk,
        cache.page_tokens,
        getattr(cache, "storage_origin", 0),
        width,
        triton.next_power_of_2(width),
    )
    return out
