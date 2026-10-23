"""DeepInfer's paged FP4 MQA and CED selection on Engine-owned pages.

The kernel calls and candidate scheduling follow DeepInfer 774ce244's
``v4_modules/index_attn.py``. Both prefill and verification use its varlen paged
ABI, so neither path gathers history or materializes per-head score matrices.
"""

from typing import cast

import torch


def _weights_aligned(weights):
    # DeepGEMM loads weights through TMA: each BF16 row needs 16B alignment.
    if weights.stride(0) * weights.element_size() % 16 == 0:
        return weights
    heads = weights.shape[-1]
    storage = weights.new_empty(len(weights), (heads + 7) // 8 * 8)
    return storage[:, :heads].copy_(weights)


def index_metadata(cache, positions, ratio, runtime):
    """Share causal lengths and Engine page tables across index query sources."""
    metadata = runtime.setdefault("index_metadata", {})
    if ratio not in metadata:
        batch, queries = positions.shape
        lengths = ((positions + 1) // ratio).flatten().int()
        request_ids = torch.arange(batch, device=positions.device, dtype=torch.int32)
        request_ids = request_ids.repeat_interleave(queries)
        metadata[ratio] = (lengths, request_ids, cache.index_block_table(queries), {})
    return metadata[ratio]


def paged_index(
    indexer, hidden, query, latent, positions, capacity, cache, freqs, runtime
):
    import deep_gemm
    from deep_select.interface import topk

    from resin_models_cuda.deepseek_v41.attention import _cos_sin
    from resin_models_cuda.deepseek_v41.upstream.candidate_blocks import (
        candidate_block_max,
        get_candidate_compact_lens,
        restore_candidate_indices_inplace,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        NormArgs,
        RopeArgs,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        fused_norm_rope_hadamard_cast_store_kvcache as transform,
    )

    shape = positions.shape
    cos_sin = _cos_sin(freqs)
    cast_args = CastArgs("e2m1", 32, round_sf=True, sf_format="ue8m0")
    if indexer.owns_k:
        keys = indexer.wk(latent).reshape(-1, indexer.head_dim)
        slots = (
            torch.where(
                (positions + 1).remainder(indexer.ratio) == 0,
                cache.index_rows(indexer.layer, positions),
                -1,
            )
            .flatten()
            .int()
        )
        transform(
            keys,
            norm_args=NormArgs(indexer.k_norm.weight, indexer.k_norm.eps),
            rope_args=RopeArgs(
                (positions - positions.remainder(indexer.ratio)).flatten().int(),
                cos_sin,
                indexer.rope_dim,
                False,
                offset=indexer.head_dim - indexer.rope_dim,
            ),
            cast_args=cast_args,
            out_args=KVCacheOutArgs(
                cache.index_cache(indexer.layer),
                slots,
            ),
        )
    q, sf = cast(
        tuple[torch.Tensor, torch.Tensor],
        transform(
            query.reshape(-1, indexer.local_heads, indexer.head_dim),
            rope_args=RopeArgs(
                positions.flatten().int(),
                cos_sin,
                indexer.rope_dim,
                False,
                offset=indexer.head_dim - indexer.rope_dim,
            ),
            cast_args=cast_args,
        ),
    )
    weights = _weights_aligned(
        (
            indexer.weights_proj(hidden)
            * (indexer.head_dim**-0.5 * indexer.n_heads**-0.5)
        ).reshape(-1, indexer.local_heads)
    )
    # These tensors depend only on the current call's live page tables and
    # positions. Reuse them across the shared-K query sources of each ratio.
    lengths, request_ids, table, schedules = index_metadata(
        cache, positions, indexer.ratio, runtime
    )
    kv = cache.index_cache(indexer.layer)
    candidates = runtime.get("candidate_blocks")
    sparse = indexer.candidate_source >= 0 and indexer.layer > indexer.candidate_source
    paged_q = q[:, None], sf[:, None]
    if sparse:
        if candidates is None:
            raise RuntimeError("CED consumer executed before its candidate source")
        if "sparse" not in schedules:
            schedules["sparse"] = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                lengths,
                table,
                request_ids,
                kv.shape[1],
                candidates,
                q.dtype,
                indexer.block_size,
            )
            schedules["compact_lens"] = get_candidate_compact_lens(
                candidates, lengths, indexer.block_size
            )
        scores = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            paged_q,
            kv,
            weights,
            schedules["sparse"],
            candidates.shape[1],
            indexer.block_size,
        )
        valid_lens = schedules["compact_lens"]
    else:
        if "dense" not in schedules:
            schedules["dense"] = deep_gemm.get_paged_mqa_logits_metadata(
                lengths[:, None],
                kv.shape[1],
                deep_gemm.get_num_sms(),
                indices=request_ids,
            )
        # DeepInfer allocates against the fixed runtime context bound. Live
        # lengths above still mask unfinished/future keys in each invocation.
        width = max(indexer.max_kv_len, indexer.topk)
        if indexer.layer == indexer.candidate_source:
            width = max(width, indexer.candidate_topk * indexer.block_size)
        scores = deep_gemm.fp8_fp4_paged_mqa_logits(
            paged_q,
            kv,
            weights,
            lengths[:, None],
            table,
            schedules["dense"],
            width,
            indices=request_ids,
        )
        valid_lens = lengths
    if indexer.layer == indexer.candidate_source:
        blocks, valid_blocks = candidate_block_max(scores, lengths, indexer.block_size)
        runtime["candidate_blocks"] = topk(
            blocks,
            indexer.candidate_topk,
            sorted=False,
            end=valid_blocks,
            indices_type=torch.int32,
            sorted_index=True,
            idx_oob_fill_value=-1,
            return_value=False,
        )[1]
        # A new candidate source starts a fresh sparse schedule group.
        schedules.pop("sparse", None)
        schedules.pop("compact_lens", None)
    selected = topk(
        scores,
        indexer.topk,
        sorted=False,
        end=valid_lens,
        indices_type=torch.int32,
        sorted_index=True,
        idx_oob_fill_value=-1,
        return_value=False,
    )[1]
    if sparse:
        restore_candidate_indices_inplace(selected, candidates, indexer.block_size)
    return selected.reshape(*shape, indexer.topk)
