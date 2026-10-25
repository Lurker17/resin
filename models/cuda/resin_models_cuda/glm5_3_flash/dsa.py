"""Raw E4M3 NoPE-512 sparse MLA using FlashInfer's TRT-LLM backend.

The Q/K cast and scalar-one scale contract follow SGLang baa9befa3815,
``dsa_backend._forward_trtllm`` / ``mla_quantize_for_fp8_no_rope`` (Apache-2.0).
This is the raw 512-byte format, not FlashMLA's block-scaled 528-byte format.
Sparse indices are *storage-row offsets*, including gaps between Engine pages;
FlashInfer v0.7.1's sparse TMA descriptor has a fixed 512-element row stride.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

MAX_QUERY_ROWS = 128
WORKSPACE_BYTES = 128 * 1024**2


class DsaWorkspace:
    """Arena-owned scratch for ordered launches on the runner's CUDA stream.

    One instance serves every DSA layer and native MTP. Its tensors stay alive
    with the cache arena through every captured replay. Concurrent streams need
    separate instances; neither the scratch nor counters may be resized.
    """

    def __init__(self, device):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        required = {"sparse_mla_top_k_lens", "multi_ctas_kv_counter_buffer"}
        if not required.issubset(
            inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
        ):
            raise RuntimeError("GLM FP8 NoPE MLA requires FlashInfer 0.7.1")
        device = torch.device(device)
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.buffer = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.counters = torch.zeros(
            get_trtllm_gen_multi_ctas_kv_counter_bytes(MAX_QUERY_ROWS, 64, sm_count),
            dtype=torch.uint8,
            device=device,
        )
        # Local development stubs can predate the validated 0.7.1 keyword API.
        # This adapter always uses the default return_lse=False, returning Tensor.
        self.attention = cast(
            Callable[..., Tensor], trtllm_batch_decode_with_kv_cache_mla
        )


def sparse_metadata(indices: Tensor, sequence_lengths: Tensor):
    """Pack valid indices stably and provide safe lengths for empty graph rows.

    Pool selection can leave holes before its causal tail. Native H512 TRT needs
    a compact valid prefix, followed by repeats of its first valid token. TRT's
    PV multiplication can still read masked values, so padding cannot reference
    potentially poisoned null storage. Empty rows point at reserved token zero
    and are explicitly zeroed after attention. Lengths and indices are runtime.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("sparse indices must be an int32 matrix")
    if sequence_lengths.shape != indices.shape[:1]:
        raise ValueError("one sequence length is required per sparse query")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("sequence lengths must be int32 or int64")
    if indices.device != sequence_lengths.device:
        raise ValueError("sparse metadata must share a device")
    if indices.shape[1] == 0:
        raise ValueError("sparse index capacity must be positive")
    if indices.is_cuda:
        from resin_models_cuda.glm5_3_flash.dsa_cuda import sparse_metadata_cuda

        return sparse_metadata_cuda(indices, sequence_lengths)
    width = (indices.shape[1] + 3) // 4 * 4
    packed = torch.zeros((indices.shape[0], width), dtype=torch.int32)
    counts = torch.empty(indices.shape[0], dtype=torch.int32)
    for row in range(indices.shape[0]):
        valid = indices[row][indices[row] >= 0]
        if valid.numel():
            packed[row].fill_(int(valid[0]))
        packed[row, : valid.numel()] = valid
        counts[row] = valid.numel()
    return packed, counts.clamp_min(1), sequence_lengths.int().clamp_min(1), counts == 0


def fp8_sparse_attention(
    query: Tensor,
    indices: Tensor,
    latent: Tensor,
    scale: float,
    *,
    workspace: DsaWorkspace | None,
    sequence_lengths: Tensor,
    max_seq_len: int,
) -> Tensor:
    """Quantize queries once, consume raw FP8 cache, and return BF16 latents."""
    if query.ndim != 3 or query.shape[1:] != (64, 512):
        raise ValueError("GLM FP8 MLA requires query [tokens,64,512]")
    if query.dtype != torch.bfloat16 or latent.dtype != torch.float8_e4m3fn:
        raise ValueError("GLM FP8 MLA requires BF16 queries and raw E4M3 cache")
    if latent.ndim != 3 or latent.shape[1:] != (64, 512):
        raise ValueError("GLM FP8 cache must have shape [pages,64,512]")
    if latent.stride(-1) != 1 or latent.stride(1) != 512 or latent.stride(0) % 512:
        raise ValueError("GLM FP8 cache requires compact, aligned latent rows")
    if indices.shape[0] != query.shape[0] or query.shape[0] > MAX_QUERY_ROWS:
        raise ValueError("GLM FP8 MLA accepts at most 128 query rows per call")
    if query.device != latent.device or query.device != indices.device:
        raise ValueError("GLM FP8 MLA inputs must share a device")
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive static capacity")
    if not query.shape[0]:
        return torch.empty_like(query)
    packed, topk_lengths, lengths, empty = sparse_metadata(indices, sequence_lengths)
    # Preserve the serving reference's raw E4M3 cast: no block scaling or clamp.
    quantized = query.to(torch.float8_e4m3fn)
    if not query.is_cuda:
        rows = ((latent.shape[0] - 1) * latent.stride(0) + 64 * 512) // 512
        envelope = latent.as_strided((rows, 512), (512, 1)).float()
        output = torch.zeros_like(query)
        for row in range(query.shape[0]):
            if empty[row]:
                continue
            selected = envelope[packed[row, : topk_lengths[row]].long()]
            scores = quantized[row].float() @ selected.T * scale
            output[row] = scores.softmax(-1) @ selected
        return output
    if workspace is None or workspace.buffer.device != query.device:
        raise ValueError("CUDA FP8 MLA requires its arena-owned DsaWorkspace")
    output = workspace.attention(
        query=quantized.unsqueeze(1),
        kv_cache=latent.unsqueeze(1),
        workspace_buffer=workspace.buffer,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        block_tables=packed.unsqueeze(1),
        seq_lens=lengths,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=packed.shape[-1],
        sparse_mla_top_k_lens=topk_lengths,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        enable_pdl=False,
        multi_ctas_kv_counter_buffer=workspace.counters,
    ).squeeze(1)
    return output.masked_fill(empty[:, None, None], 0)
