"""FlashAttention-4 paged-cache helpers for Qwen3.5 full-attention layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from resin_models_cuda.qwen3_5.cache import PAGE_TOKENS, DeviceCache


def append_kv(
    cache: DeviceCache,
    *,
    group_index: int,
    layer: int,
    layer_count: int,
    positions: Tensor,
    keys: Tensor,
    values: Tensor,
) -> None:
    """Write a rectangular batch into one layer's slab-addressed pages."""

    rows = cache.rows(group_index, layer, layer_count, positions)
    pages = rows.div(PAGE_TOKENS, rounding_mode="floor")
    offsets = rows.remainder(PAGE_TOKENS)
    cache.keys[pages, offsets] = keys
    cache.values[pages, offsets] = values


def paged_attention(
    queries: Tensor,
    *,
    cache: DeviceCache,
    group_index: int,
    layer: int,
    layer_count: int,
    sequence_lengths: Tensor,
    max_sequence_length: int,
    softmax_scale: float,
) -> Tensor:
    """Run causal Qwen GQA against BF16 paged KV with FA4."""

    from flash_attn.cute import flash_attn_varlen_func

    if queries.ndim != 4:
        raise ValueError("Qwen3.5 attention expects [batch, tokens, heads, dim]")
    batch, tokens = queries.shape[:2]
    page_table = cache.pages(group_index, layer, layer_count)
    group = cache.groups[group_index]
    retained_starts = group.first_blocks * PAGE_TOKENS
    used_lengths = (sequence_lengths - retained_starts).to(torch.int32)
    max_k = min(max_sequence_length, page_table.shape[1] * PAGE_TOKENS)
    cu_q = (
        torch.arange(batch + 1, dtype=torch.int32, device=queries.device) * tokens
    )
    kwargs: dict[str, Any] = {
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": tokens,
        "max_seqlen_k": max_k,
        "min_seqlen_k": 1,
        "seqused_k": used_lengths,
        "page_table": page_table,
        "softmax_scale": softmax_scale,
        "causal": True,
        "window_size": (None, None),
        "num_splits": 1,
        "pack_gqa": True,
    }
    output = flash_attn_varlen_func(
        queries.reshape(batch * tokens, *queries.shape[2:]),
        cache.keys,
        cache.values,
        **kwargs,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.view(batch, tokens, *output.shape[1:])


__all__ = ["append_kv", "paged_attention"]
