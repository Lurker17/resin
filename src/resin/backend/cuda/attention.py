"""Pinned accelerator call adapters shared by CUDA model runners."""

from __future__ import annotations

_UNSET = object()


def paged_fa4(
    kernel,
    query,
    key_cache,
    value_cache,
    *,
    cu_seqlens_q,
    max_seqlen_q: int,
    page_table,
    seqused_k,
    min_seqlen_k: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink=_UNSET,
):
    """Invoke the exact paged-varlen FA4 surface used by Resin runners."""

    options = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": None,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "min_seqlen_k": min_seqlen_k,
        "seqused_q": None,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "softmax_scale": softmax_scale,
        "causal": True,
        "window_size": window_size,
        "softcap": 0.0,
        # FA4's SM90 path does not support split-KV. Its automatic heuristic
        # can otherwise choose an unsupported split count for long decode rows.
        "num_splits": 1,
        "pack_gqa": True,
        "deterministic": False,
        "return_lse": False,
    }
    if learnable_sink is not _UNSET:
        options["learnable_sink"] = learnable_sink
    result = kernel(query, key_cache, value_cache, **options)
    return result[0] if isinstance(result, tuple) else result


__all__ = ["paged_fa4"]
