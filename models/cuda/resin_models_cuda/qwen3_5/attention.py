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


@dataclass(frozen=True, slots=True)
class GraphAttentionSpec:
    """One statically ordered attention call inside a Qwen decode graph."""

    group_index: int
    layer_count: int
    tokens: int
    sequence_offset: int


@dataclass(slots=True)
class _GraphAttentionCall:
    spec: GraphAttentionSpec
    prepared: PreparedPagedAttention
    kv_indptr: Tensor
    last_page_lengths: Tensor
    qo_indptr: Tensor | None


class GraphPagedAttentionRuntime:
    """Fixed-buffer FlashInfer calls used by one captured decode bucket."""

    def __init__(
        self,
        device: torch.device,
        cache: DeviceCache,
        specs: tuple[GraphAttentionSpec, ...],
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace: Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.cache = cache
        self.specs = specs
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim**-0.5
        self.workspace = workspace
        self._cursor = 0
        calls = []
        batch_size = cache.groups[0].slab_ids.shape[0]
        for spec in specs:
            group = cache.groups[spec.group_index]
            page_capacity = batch_size * group.slab_ids.shape[1]
            logical_pages = torch.empty(page_capacity, dtype=torch.int32, device=device)
            physical_pages = torch.empty_like(logical_pages)
            kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
            last = torch.empty(batch_size, dtype=torch.int32, device=device)
            common = {
                "pos_encoding_mode": "NONE",
                "q_data_type": torch.bfloat16,
                "kv_data_type": cache.keys.dtype,
                "o_data_type": torch.bfloat16,
                "sm_scale": self.softmax_scale,
                "disable_split_kv": True,
                "non_blocking": False,
            }
            initial_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            )
            initial_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
            initial_last = torch.ones(batch_size, dtype=torch.int32, device=device)
            qo_indptr = None
            if spec.tokens == 1:
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=kv_indptr,
                    paged_kv_indices_buffer=physical_pages,
                    paged_kv_last_page_len_buffer=last,
                    backend="fa2",
                )
                wrapper.plan(
                    initial_indptr,
                    initial_indices,
                    initial_last,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    PAGE_TOKENS,
                    **common,
                )
            else:
                qo_indptr = torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=physical_pages,
                    paged_kv_last_page_len_buf=last,
                    backend="auto",
                )
                initial_qo = initial_indptr * spec.tokens
                wrapper.plan(
                    initial_qo,
                    initial_indptr,
                    initial_indices,
                    initial_last,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    PAGE_TOKENS,
                    causal=True,
                    **common,
                )
            calls.append(
                _GraphAttentionCall(
                    spec,
                    PreparedPagedAttention(
                        wrapper,
                        cache,
                        logical_pages,
                        physical_pages,
                        spec.layer_count,
                        spec.tokens,
                    ),
                    kv_indptr,
                    last,
                    qo_indptr,
                )
            )
        self.calls = tuple(calls)

    def reset(self) -> None:
        self._cursor = 0

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
        del sequence_lengths
        if cache is not self.cache or self._cursor >= len(self.calls):
            raise RuntimeError("Qwen3.5 graph attention call order changed")
        call = self.calls[self._cursor]
        self._cursor += 1
        expected = call.spec
        if (
            expected.group_index != group_index
            or expected.layer_count != layer_count
            or expected.tokens != tokens
            or softmax_scale != self.softmax_scale
        ):
            raise RuntimeError("Qwen3.5 graph attention signature changed")
        return call.prepared

    def graph_inputs(self) -> dict[str, Tensor]:
        return {
            f"attention.{index}.logical_pages": call.prepared.logical_pages
            for index, call in enumerate(self.calls)
        }

    @staticmethod
    def _host_pages(
        group: Any,
        *,
        rows: int,
        graph_rows: int,
        graph_width: int,
        sequence_lengths: Any,
        layer_count: int,
        slab_units: int,
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        logical = np.zeros(graph_rows * graph_width, dtype=np.int32)
        indptr = np.zeros(graph_rows + 1, dtype=np.int32)
        last = np.ones(graph_rows, dtype=np.int32)
        cursor = 0
        pages_per_slab = slab_units // layer_count
        for row in range(graph_rows):
            if row < rows:
                retained = int(group.block_table_start_indices[row]) * PAGE_TOKENS
                used = int(sequence_lengths[row]) - retained
                if used <= 0:
                    raise ValueError(
                        "Qwen3.5 graph attention requires retained context"
                    )
                count = (used + PAGE_TOKENS - 1) // PAGE_TOKENS
                slabs = group.block_slabs(row)
                slots = group.block_slots(row)
                if count > len(slabs) or count > graph_width:
                    raise ValueError("Qwen3.5 graph attention exceeds its cache lease")
                logical[cursor : cursor + count] = slabs[:count].astype(
                    np.int64
                ) * pages_per_slab + slots[:count].astype(np.int64)
                last[row] = (used - 1) % PAGE_TOKENS + 1
            else:
                count = 1
            cursor += count
            indptr[row + 1] = cursor
        return logical, indptr, last

    def plan(self, cache_plan: Any, positions: Any) -> dict[str, object]:
        """Plan every wrapper for current host rows before graph replay."""

        import numpy as np

        positions = np.asarray(positions, dtype=np.int64)
        rows = cache_plan.rows
        graph_rows = self.cache.groups[0].slab_ids.shape[0]
        inputs: dict[str, object] = {}
        for index, call in enumerate(self.calls):
            spec = call.spec
            group = cache_plan.groups[spec.group_index]
            graph_width = self.cache.groups[spec.group_index].slab_ids.shape[1]
            lengths = positions + spec.sequence_offset
            logical, indptr, last = self._host_pages(
                group,
                rows=rows,
                graph_rows=graph_rows,
                graph_width=graph_width,
                sequence_lengths=lengths,
                layer_count=spec.layer_count,
                slab_units=self.cache.geometry.slab_units,
            )
            physical = logical * spec.layer_count
            common = {
                "pos_encoding_mode": "NONE",
                "q_data_type": torch.bfloat16,
                "kv_data_type": self.cache.keys.dtype,
                "o_data_type": torch.bfloat16,
                "sm_scale": self.softmax_scale,
                "disable_split_kv": True,
                "non_blocking": False,
            }
            page_count = int(indptr[-1])
            device_indptr = torch.as_tensor(
                indptr, dtype=torch.int32, device=self.device
            )
            device_pages = torch.as_tensor(
                physical[:page_count], dtype=torch.int32, device=self.device
            )
            device_last = torch.as_tensor(last, dtype=torch.int32, device=self.device)
            if spec.tokens == 1:
                call.prepared.wrapper.plan(
                    device_indptr,
                    device_pages,
                    device_last,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    PAGE_TOKENS,
                    **common,
                )
            else:
                qo = np.arange(
                    0,
                    (graph_rows + 1) * spec.tokens,
                    spec.tokens,
                    dtype=np.int32,
                )
                call.prepared.wrapper.plan(
                    torch.as_tensor(qo, dtype=torch.int32, device=self.device),
                    device_indptr,
                    device_pages,
                    device_last,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    PAGE_TOKENS,
                    causal=True,
                    **common,
                )
            inputs[f"attention.{index}.logical_pages"] = logical
        self.reset()
        return inputs


__all__ = [
    "GraphAttentionSpec",
    "GraphPagedAttentionRuntime",
    "PagedAttentionRuntime",
    "PreparedPagedAttention",
    "append_kv",
]
