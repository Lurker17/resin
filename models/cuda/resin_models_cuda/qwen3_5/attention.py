"""FlashInfer paged-cache helpers for Qwen3.5 full-attention layers."""

from __future__ import annotations

from dataclasses import dataclass
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
    cache.keys[pages, offsets] = keys.to(cache.keys.dtype)
    cache.values[pages, offsets] = values.to(cache.values.dtype)


@dataclass(slots=True)
class PreparedPagedAttention:
    """One FlashInfer plan shared by every physical layer in a cache group."""

    wrapper: Any
    cache: DeviceCache
    logical_pages: Tensor
    physical_pages: Tensor
    layer_count: int
    tokens: int

    def run(self, queries: Tensor, layer: int) -> Tensor:
        if not 0 <= layer < self.layer_count:
            raise ValueError("Qwen3.5 attention layer lies outside its cache group")
        self.physical_pages.copy_(self.logical_pages * self.layer_count + layer)
        query = queries[:, 0] if self.tokens == 1 else queries.flatten(0, 1)
        output = self.wrapper.run(query, (self.cache.keys, self.cache.values))
        return output.view_as(queries)


class PagedAttentionRuntime:
    """Reusable FlashInfer planning workspace for eager Qwen attention."""

    def __init__(
        self,
        device: torch.device,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace_bytes: int,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.prefill = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="auto"
        )
        self.decode = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_tensor_cores=True,
            backend="auto",
        )

    def prepare(
        self,
        cache: DeviceCache,
        *,
        group_index: int,
        layer_count: int,
        sequence_lengths: Tensor,
        tokens: int,
        softmax_scale: float,
    ) -> PreparedPagedAttention:
        group = cache.groups[group_index]
        retained_starts = group.first_blocks * PAGE_TOKENS
        used_lengths = (sequence_lengths - retained_starts).to(torch.int32)
        if bool(torch.any(used_lengths <= 0)):
            raise ValueError("Qwen3.5 paged attention requires retained context")
        required_pages = torch.div(
            used_lengths + PAGE_TOKENS - 1,
            PAGE_TOKENS,
            rounding_mode="floor",
        )
        width = group.slab_ids.shape[1]
        if bool(torch.any(required_pages > width)):
            raise ValueError("Qwen3.5 paged attention exceeds its cache lease")

        pages_per_slab = cache.geometry.slab_units // layer_count
        page_table = group.slab_ids * pages_per_slab + group.slotnums
        columns = torch.arange(width, device=self.device)[None]
        logical_pages = page_table[columns < required_pages[:, None]].to(torch.int32)
        physical_pages = logical_pages * layer_count
        kv_indptr = torch.zeros(
            len(used_lengths) + 1, dtype=torch.int32, device=self.device
        )
        kv_indptr[1:] = required_pages.cumsum(0)
        last_page_lengths = (used_lengths - 1).remainder(PAGE_TOKENS) + 1
        common = {
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.bfloat16,
            "kv_data_type": cache.keys.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": softmax_scale,
        }
        if tokens == 1:
            wrapper = self.decode
            wrapper.plan(
                kv_indptr,
                physical_pages,
                last_page_lengths,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                PAGE_TOKENS,
                **common,
            )
        else:
            wrapper = self.prefill
            qo_indptr = (
                torch.arange(
                    len(used_lengths) + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                * tokens
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                physical_pages,
                last_page_lengths,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                PAGE_TOKENS,
                causal=True,
                **common,
            )
        return PreparedPagedAttention(
            wrapper,
            cache,
            logical_pages,
            physical_pages,
            layer_count,
            tokens,
        )


__all__ = [
    "PagedAttentionRuntime",
    "PreparedPagedAttention",
    "append_kv",
]
