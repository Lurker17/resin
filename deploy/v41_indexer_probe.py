"""Independent CUDA checks for V4.1's paged FP4 indexer and candidate path."""

import math
from types import SimpleNamespace
from typing import cast

import torch


def _unpack(values, scales):
    """Decode packed E2M1/UE8M0 directly, independently of production kernels."""
    codes = values.view(torch.uint8)
    codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=values.device,
    )
    powers = scales.contiguous().view(torch.uint8).reshape(*values.shape[:-1], 4)
    scale = torch.pow(2.0, powers.int() - 127).repeat_interleave(32, -1)
    return lookup[codes] * scale


def _case(ratio, *, context=4096, batch=2, queries=5, heads=32):
    import deep_gemm

    from resin.backend.cuda.slab import DevicePageGroup
    from resin_models_cuda.deepseek_v41.cache import CacheArena, CacheGeometry
    from resin_models_cuda.deepseek_v41.indexer import index_metadata, paged_index
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        RopeArgs,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        fused_norm_rope_hadamard_cast_store_kvcache as transform,
    )

    args = SimpleNamespace(
        n_layers=3,
        n_mtp_layers=0,
        compress_ratios=(ratio, ratio, ratio),
        kv_source_layers=(0,),
        dspark_block_size=queries - 1,
        window_size=128,
    )
    geometry = CacheGeometry.build(args, prefill_chunk_size=128, max_batch_size=batch)
    pages = math.ceil(context / 128)
    arena = CacheArena(
        "cuda", geometry.layout(batch * pages + 8), geometry, dtype=torch.bfloat16
    )
    # Reverse physical pages and use different children per request. Source and
    # consumer layers must still resolve precisely the same shared index keys.
    physical = torch.arange(1, batch * pages + 1, device="cuda", dtype=torch.int32)
    physical = physical.flip(0).reshape(batch, pages)
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
    slots = slots.expand(batch, pages) % geometry.packings[1]
    group = DevicePageGroup(
        physical, slots, torch.zeros(batch, device="cuda", dtype=torch.int32)
    )
    cache = arena._make_graph_cache((group,) * 4)
    cos_sin = torch.cat(
        (
            torch.ones(context, 32, device="cuda"),
            torch.zeros(context, 32, device="cuda"),
        ),
        -1,
    )
    freqs = torch.complex(cos_sin[:, :32], cos_sin[:, 32:])
    cast_args = CastArgs("e2m1", 32, round_sf=True, sf_format="ue8m0")
    history = torch.arange(context, device="cuda")[None].expand(batch, -1)
    keys = torch.randn(batch, context, 128, device="cuda", dtype=torch.bfloat16)
    # C2's two source positions deliberately agree to avoid racy stores of two
    # different values into the same compressed slot in this focused probe.
    if ratio == 2:
        keys = keys[:, ::2].repeat_interleave(2, 1)
    transform(
        keys.reshape(-1, 128),
        rope_args=RopeArgs(history.flatten().int(), cos_sin, 64, False),
        cast_args=cast_args,
        out_args=KVCacheOutArgs(
            cache.index_cache(0), cache.index_rows(0, history).flatten()
        ),
    )
    query = torch.randn(batch, queries, heads, 128, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(batch, queries, heads, device="cuda", dtype=torch.bfloat16)
    # Include empty/incomplete compressed histories, a partial CED block, and
    # near-full lengths. Rejected future keys remain populated above.
    starts = torch.tensor([0, context - queries - 3], device="cuda")[:batch]
    positions = starts[:, None] + torch.arange(queries, device="cuda")
    indexer = SimpleNamespace(
        layer=0,
        ratio=ratio,
        owns_k=False,
        head_dim=128,
        local_heads=heads,
        n_heads=heads,
        rope_dim=64,
        topk=16,
        candidate_source=0,
        candidate_topk=16,
        block_size=8,
        weights_proj=lambda x: x,
    )
    runtime = {}
    selected = paged_index(
        indexer, hidden, query, None, positions, context, cache, freqs, runtime
    )
    candidates = runtime["candidate_blocks"]
    q, sf = cast(
        tuple[torch.Tensor, torch.Tensor],
        transform(
            query.reshape(-1, heads, 128),
            rope_args=RopeArgs(positions.flatten().int(), cos_sin, 64, False),
            cast_args=cast_args,
        ),
    )
    q_ref = _unpack(q, sf).reshape(batch, queries, heads, 128)
    logical = torch.arange(0, context, ratio, device="cuda")[None].expand(batch, -1)
    rows = cache.index_rows(0, logical).long()
    values, scales = cache.index_parts(0)
    page, offset = rows // values.shape[1], rows % values.shape[1]
    k_ref = _unpack(values[page, offset], scales[page, offset])
    weights = hidden * (128**-0.5 * heads**-0.5)
    reference = (
        torch.einsum("bshd,btd->bsht", q_ref, k_ref).relu() * weights[..., None]
    ).sum(-2)
    lengths, ids, table, schedules = index_metadata(cache, positions, ratio, runtime)
    scores = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q[:, None], sf[:, None]),
        cache.index_cache(0),
        weights.flatten(0, 1),
        lengths[:, None],
        table,
        schedules["dense"],
        context // ratio,
        indices=ids,
    )
    valid = torch.arange(context // ratio, device="cuda") < lengths[:, None]
    delta = scores[valid].float() - reference.flatten(0, 1)[valid]
    relative_l2 = (
        delta.norm() / reference.flatten(0, 1)[valid].norm().clamp_min(1e-8)
    ).item()
    if relative_l2 > 0.02:
        raise AssertionError(f"FP4 paged MQA relative L2: {relative_l2}")
    if not torch.equal(cache.index_rows(0, logical), cache.index_rows(2, logical)):
        raise AssertionError("shared index-K consumer changed page ownership")
    for row, valid_length in enumerate(lengths.tolist()):
        valid_selected = selected.flatten(0, 1)[row]
        valid_selected = valid_selected[valid_selected >= 0]
        if len(valid_selected) != min(indexer.topk, valid_length):
            raise AssertionError(
                "indexer returned the wrong number of valid selections"
            )
        if not (valid_selected < valid_length).all():
            raise AssertionError(
                "indexer selected a future or incomplete compressed key"
            )
        if valid_length and (valid_length - 1) // 8 not in candidates[row].tolist():
            raise AssertionError("CED omitted the newest partial block")
        if len(valid_selected):
            boundary = scores[row, :valid_length].topk(len(valid_selected)).values[-1]
            if (scores[row, valid_selected.long()] < boundary).any():
                raise AssertionError("DeepSelect omitted a higher-scoring valid key")
    indexer.layer = 2
    sparse = paged_index(
        indexer, hidden, query, None, positions, context, cache, freqs, runtime
    )
    for row in range(batch * queries):
        chosen = sparse.flatten(0, 1)[row]
        valid_chosen = chosen[chosen >= 0]
        if not (valid_chosen < lengths[row]).all():
            raise AssertionError("sparse indexer selected a future key")
        if not torch.isin(valid_chosen // 8, candidates[row]).all():
            raise AssertionError("sparse indexer escaped its CED candidates")

    # Recompute schedules inside the graph as serving does, then change causal
    # positions without recapturing. Compare exact selected IDs to eager.
    def call():
        state = {}
        indexer.layer = 0
        paged_index(
            indexer, hidden, query, None, positions, context, cache, freqs, state
        )
        indexer.layer = 2
        return paged_index(
            indexer, hidden, query, None, positions, context, cache, freqs, state
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    for shift in (0, 1, 2):
        if shift:
            positions.add_(1)
        expected = call().clone()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(output, expected):
            raise AssertionError("paged sparse indexer graph IDs differ from eager")
    return {
        "ratio": ratio,
        "batch": batch,
        "queries": queries,
        "context": context,
        "heads": heads,
        "score_relative_l2": relative_l2,
        "score_max_abs": delta.abs().max().item(),
        "graph_replays": 3,
        "cache_bytes_per_index_token": 68,
    }


def run_probe():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(47)
    return {"hardware": torch.cuda.get_device_name(), "cases": [_case(1), _case(2)]}
