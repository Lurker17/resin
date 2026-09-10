"""Cross-model CUDA operations over Resin's slab-paged KV protocol."""

from resin.backend.cuda.paged_kv import SlabPagedKvCache


def append_slab_kv(
    cache: SlabPagedKvCache,
    *,
    group_index: int,
    layer: int,
    layer_count: int,
    positions,
    keys,
    values,
) -> None:
    rows = cache.rows(group_index, layer, layer_count, positions)
    pages = rows.div(cache.page_tokens, rounding_mode="floor")
    offsets = rows.remainder(cache.page_tokens)
    cache.keys[pages, offsets] = keys.to(cache.keys.dtype)
    cache.values[pages, offsets] = values.to(cache.values.dtype)


__all__ = ["append_slab_kv"]
