"""FlashInfer attention adapters for DeepSeek-V4 decode."""

from __future__ import annotations

import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from torch import Tensor
from torch.nn import functional as F


def sparse_topk_lengths(
    compressed_complete: Tensor, compressed_index_width: int, window_size: int
) -> Tensor:
    """Bound FlashInfer's active length by the physical sparse-index row."""

    if compressed_index_width < 0 or window_size <= 0:
        raise ValueError("sparse attention widths must be nonnegative")
    return (compressed_complete.clamp_max(compressed_index_width) + window_size).to(
        torch.int32
    )


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

    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("DeepSeek-V4 graph decode requires one query per batch row")
    batch_size = query.shape[0]
    if window_cache.ndim != 3 or compressed_cache.ndim != 3:
        raise ValueError("DeepSeek-V4 physical KV pools must be paged tensors")
    if not window_cache.is_contiguous() or not compressed_cache.is_contiguous():
        raise ValueError("DeepSeek-V4 physical KV pools must be contiguous")
    if indices.shape[:2] != (batch_size, 1):
        raise ValueError("DeepSeek-V4 sparse indices must contain one row per query")
    if (
        query.shape[-1] != window_cache.shape[-1]
        or query.shape[-1] != compressed_cache.shape[-1]
    ):
        raise ValueError("query and latent KV dimensions must match")
    if sinks.shape != (query.shape[2],):
        raise ValueError("learnable sinks must contain one value per query head")

    padding = -indices.shape[-1] % 4
    if padding:
        indices = F.pad(indices, (0, padding), value=-1)

    return trtllm_batch_decode_sparse_mla_dsv4(
        query,
        window_cache,
        workspace,
        indices.view(batch_size, -1),
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_lengths,
        seq_lens=sequence_lengths,
        bmm1_scale=softmax_scale,
        sinks=sinks,
        kv_layout="HND",
        enable_pdl=True,
    )


def sparse_verify_attention(
    query: Tensor,
    window_cache: Tensor,
    compressed_cache: Tensor,
    sinks: Tensor,
    indices: Tensor,
    sparse_lengths: Tensor,
    sequence_lengths: Tensor,
    workspace: Tensor,
    softmax_scale: float,
    *,
    enable_pdl: bool = True,
) -> Tensor:
    """Verify a short query block with one sparse-MLA kernel invocation."""

    if query.ndim != 4 or query.shape[1] == 0:
        raise ValueError("DeepSeek-V4 verification requires a nonempty query block")
    batch_size, block_length, heads, dimension = query.shape
    if window_cache.ndim != 3 or compressed_cache.ndim != 3:
        raise ValueError("DeepSeek-V4 physical KV pools must be three-dimensional")
    if not window_cache.is_contiguous() or not compressed_cache.is_contiguous():
        raise ValueError("DeepSeek-V4 physical KV pools must be contiguous")
    if indices.shape[:2] != (batch_size, block_length):
        raise ValueError("DeepSeek-V4 sparse indices must match the query block")
    if sparse_lengths.shape != (batch_size, block_length):
        raise ValueError("DeepSeek-V4 sparse lengths must match the query block")
    if sequence_lengths.shape != (batch_size, block_length):
        raise ValueError("DeepSeek-V4 sequence lengths must match the query block")
    if dimension != window_cache.shape[-1] or dimension != compressed_cache.shape[-1]:
        raise ValueError("query and latent KV dimensions must match")
    if sinks.shape != (heads,):
        raise ValueError("learnable sinks must contain one value per query head")

    padding = -indices.shape[-1] % 4
    if padding:
        indices = F.pad(indices, (0, padding), value=-1)
    query_rows = batch_size * block_length
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query.view(query_rows, 1, heads, dimension),
        window_cache,
        workspace,
        indices.view(query_rows, -1),
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_lengths.flatten(),
        seq_lens=sequence_lengths.flatten(),
        bmm1_scale=softmax_scale,
        sinks=sinks,
        kv_layout="HND",
        enable_pdl=enable_pdl,
    )
    return output.view(batch_size, block_length, heads, dimension)


__all__ = [
    "sparse_decode_attention",
    "sparse_topk_lengths",
    "sparse_verify_attention",
]
