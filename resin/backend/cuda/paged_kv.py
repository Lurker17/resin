"""Physical paged-KV storage and common device metadata views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resin.backend.resources import CachePoolBinding


class PagedKvArena:
    """One zero-initialized K/V allocation backed by a native cache pool."""

    def __init__(
        self,
        torch,
        device,
        layout,
        *,
        layer_count: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype,
        resource: str = "paged KV",
    ) -> None:
        self.capacity = layout.parent_capacity
        self.page_size = page_size
        self.binding = CachePoolBinding(layout, resource=resource)
        shape = (2, layer_count, self.capacity + 1, page_size, num_kv_heads, head_dim)
        self.storage = torch.zeros(shape, dtype=dtype, device=device)
        self.keys = self.storage[0]
        self.values = self.storage[1]

    def check_pages(self, page_ids) -> None:
        if any(page < 0 or page > self.capacity for page in page_ids):
            raise ValueError("BatchPlan cache block ID is outside the physical slab")


@dataclass(frozen=True, slots=True)
class PagedKvTable:
    """Dense page table, logical lengths, and append slots for one KV group."""

    page_table: Any
    lengths: Any
    append_slots: Any
    min_length: int
    max_length: int


@dataclass(frozen=True, slots=True)
class PagedKvCsr:
    """CSR page metadata shared by paged append and decode kernels."""

    indptr: Any
    page_ids: Any
    last_page_lengths: Any
    append_batch_indices: Any
    append_positions: Any
