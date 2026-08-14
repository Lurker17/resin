"""FlashInfer attention adapters for DeepSeek-V4 decode."""

from __future__ import annotations

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from torch import Tensor


def sparse_decode_attention(
    query: Tensor,
    window_cache: Tensor,
    compressed_cache: Tensor,
    sinks: Tensor,
    indices: Tensor,
    sparse_lengths: Tensor,
    sequence_lengths: Tensor,
    workspace: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Run FlashInfer's DeepSeek-V4 sparse MLA decode kernel."""

    if query.ndim != 4 or query.shape[:2] != (1, 1):
        raise ValueError("DeepSeek-V4 graph decode requires one query token")
    if window_cache.ndim != 3 or window_cache.shape[0] != 1:
        raise ValueError("DeepSeek-V4 graph decode requires one window KV pool")
    if compressed_cache.ndim != 3 or compressed_cache.shape[0] != 1:
        raise ValueError("DeepSeek-V4 graph decode requires one compressed KV pool")
    if indices.shape[:2] != (1, 1):
        raise ValueError("DeepSeek-V4 sparse indices must contain one query row")
    if query.shape[-1] != window_cache.shape[-1]:
        raise ValueError("query and latent KV dimensions must match")
    if sinks.shape != (query.shape[2],):
        raise ValueError("learnable sinks must contain one value per query head")

    return trtllm_batch_decode_sparse_mla_dsv4(
        query,
        window_cache,
        workspace,
        indices.view(1, -1),
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_lengths,
        seq_lens=sequence_lengths,
        bmm1_scale=softmax_scale,
        sinks=sinks,
        kv_layout="HND",
        enable_pdl=True,
    )


__all__ = ["sparse_decode_attention"]
