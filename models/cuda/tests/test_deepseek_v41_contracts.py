"""Paged V4.1 state contracts against independent Torch equations.

The official 1017 inference/model.py is read as a specification; no checkpoint
Python is imported. These checks isolate state semantics from GPU numerics.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v41_model import implementation as implementation
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.slab import DevicePageGroup
from resin.bench import BenchmarkRunner
from resin_models_cuda.deepseek_v41.cache import CacheArena, CacheGeometry


@pytest.mark.parametrize("version", ["v4", "v41"])
def test_benchmark_router_permutation_uses_actual_ep_and_adp(
    implementation, monkeypatch, version
):
    import sys

    module = (
        implementation
        if version == "v41"
        else sys.modules["resin_models_cuda.deepseek_v4.model"]
    )
    args = module.ModelArgs(
        dim=8,
        n_routed_experts=384,
        n_activated_experts=6,
        n_hash_layers=0,
        score_func="sigmoid",
        route_scale=1.5,
    )
    gate = module.Gate(args) if version == "v41" else module.Gate(0, args)
    if version == "v4":
        monkeypatch.setattr(
            module,
            "bf16_linear_fp32",
            lambda hidden, weight: F.linear(hidden.float(), weight.float()),
        )
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.arange(384))
    hidden = torch.zeros(1, 8)
    _, ordinary = gate(hidden, torch.tensor([0]))
    assert (ordinary // 96 == 3).all()  # Ordinary top-k remains score-selected.
    BenchmarkRunner(
        SimpleNamespace(
            _model=nn.Sequential(gate),
            parallel_groups=SimpleNamespace(
                ep=SimpleNamespace(size=4), attn_dp=SimpleNamespace(local_rank=1)
            ),
        ),
        balanced_experts=True,
    )
    weights, indices = gate(hidden, torch.tensor([0]))
    assert indices.tolist() == [[193, 289, 2, 98, 194, 290]]
    assert gate.benchmark_expert_parallel_size == 4
    assert gate.benchmark_replica == 1
    torch.testing.assert_close(
        weights.sum(-1), torch.tensor([1.5], dtype=weights.dtype)
    )


def test_legacy_deepseek_tp_keeps_its_prepared_expert_owner_count(implementation):
    import sys

    from resin.backend.cuda.parallel import TensorParallelLayout

    module = sys.modules["resin_models_cuda.deepseek_v4.model"]
    weights = module.MoEWeights(
        0,
        module.ModelArgs(
            dim=8,
            moe_inter_dim=128,
            n_routed_experts=8,
            n_activated_experts=2,
            expert_dtype=None,
        ),
        expert_parallel=TensorParallelLayout(rank=0, size=2),
    )
    assert weights.gate.benchmark_expert_parallel_size == 2
    BenchmarkRunner(SimpleNamespace(_model=weights), balanced_experts=True)
    assert weights.gate.benchmark_expert_parallel_size == 2
    assert weights.gate.benchmark_replica == 0


def _args():
    return SimpleNamespace(
        n_layers=7,
        n_mtp_layers=1,
        dspark_block_size=5,
        compress_ratios=(0, 2, 2, 2, 2, 1, 1, 0),
        kv_source_layers=(1, 3, 5),
        window_size=128,
        head_dim=512,
        dim=8,
        norm_eps=1e-6,
    )


def _cache():
    geometry = CacheGeometry.build(_args(), prefill_chunk_size=130)
    groups, parent = [], 1
    for packing in geometry.packings:
        ordinal = torch.arange(6).reshape(2, 3)
        groups.append(
            DevicePageGroup(
                (parent + ordinal // packing).int(),
                (ordinal % packing).int(),
                torch.zeros(2, dtype=torch.int32),
            )
        )
        parent += math.ceil(6 / packing)
    arena = CacheArena("cpu", geometry.layout(parent), geometry, dtype=torch.bfloat16)
    arena.storage.zero_()
    return arena._make_graph_cache(tuple(groups))


def _compressor(implementation):
    module = implementation.Compressor(_args(), 1)
    with torch.no_grad():
        # Exact one-term projections remove CPU GEMM reduction differences
        # between chunk shapes; the test isolates pooling/state semantics.
        rows = torch.arange(512)
        module.wkv.weight.zero_()
        module.wgate.weight.zero_()
        module.wkv.weight[rows, rows % 8] = 0.25
        module.wgate.weight[rows, (rows + 3) % 8] = 0.125
    return module


def _pooled_reference(module, hidden):
    # The released ratio-2 compressor projects and gates in FP32, casts the
    # pooled latent to the model dtype, then applies its RMS normalization.
    values = F.linear(hidden.float(), module.wkv.weight)
    gates = F.linear(hidden.float(), module.wgate.weight)
    latent = (
        values.unflatten(1, (-1, 2)) * gates.unflatten(1, (-1, 2)).softmax(2)
    ).sum(2)
    latent = latent.to(hidden.dtype).float()
    latent = latent * torch.rsqrt(
        latent.square().mean(-1, keepdim=True) + module.norm.eps
    )
    return (latent * module.norm.weight.float()).to(hidden.dtype)


@pytest.mark.parametrize("chunk", [1, 3, 128, 130])
def test_ratio2_compressor_chunking_matches_completed_official_groups(
    implementation, chunk
):
    module, cache = _compressor(implementation), _cache()
    hidden = torch.randn(
        2, 130, 8, generator=torch.Generator().manual_seed(7)
    ).bfloat16()
    with torch.inference_mode():
        expected = _pooled_reference(module, hidden)
        for start in range(0, 130, chunk):
            end = min(start + chunk, 130)
            positions = torch.arange(start, end).expand(2, -1)
            actual = module(hidden[:, start:end], positions, cache)
            complete = (positions[0] + 1) % 2 == 0
            torch.testing.assert_close(
                actual[:, complete],
                expected[:, positions[0, complete] // 2],
                rtol=0,
                atol=0,
            )
            paired = complete & (positions[0] > start)
            # CUDA index_copy sees duplicate physical compressed rows. Those
            # writes are safe only because both completed-pair values agree.
            torch.testing.assert_close(
                actual[:, paired], actual[:, torch.where(paired)[0] - 1], rtol=0, atol=0
            )


def test_ratio2_rejected_tail_needs_no_relocation_or_accepted_state_mutation(
    implementation,
):
    module, cache = _compressor(implementation), _cache()
    generator = torch.Generator().manual_seed(71)
    original = torch.randn(2, 130, 8, generator=generator).bfloat16()
    corrected = original.clone()
    corrected[:, 125:] = torch.randn(2, 5, 8, generator=generator).bfloat16()
    with torch.inference_mode():
        module(original, torch.arange(130).expand(2, -1), cache)
        accepted_rows, _ = cache.compressor_rows(1, torch.arange(125).expand(2, -1))
        accepted = cache.compressor_values[accepted_rows].clone()
        expected = _pooled_reference(module, corrected)
        # Position 125 completes a pair whose first token was accepted. Future
        # rejected pairs must stay hidden until the corrected token completes it.
        for position in range(125, 130):
            actual = module(
                corrected[:, position : position + 1],
                torch.full((2, 1), position),
                cache,
            )
            if position % 2:
                torch.testing.assert_close(
                    actual[:, 0], expected[:, position // 2], rtol=0, atol=0
                )
        torch.testing.assert_close(
            cache.compressor_values[accepted_rows], accepted, rtol=0, atol=0
        )


def test_fp32_compressor_sources_and_packed_index_keys_never_alias():
    cache = _cache()
    positions = torch.arange(258).expand(2, -1)
    saved = []
    for source in (1, 3):
        values, scores = cache.compressor_rows(source, positions)
        cache.compressor_values[values] = source + 10
        cache.compressor_values[scores] = source + 20
        saved.extend([(values, source + 10), (scores, source + 20)])
    for source in (1, 3, 5):
        rows = cache.compressed_rows(
            source, cache.geometry.ratios[source], positions
        ).long()
        keys = cache.index_rows(source, positions).long()
        packed = cache.compressed_cache(source)
        packed[rows // packed.shape[1], rows % packed.shape[1], 0] = source + 30
        values, scales = cache.index_parts(source)
        pages, offsets = keys // packed.shape[1], keys % packed.shape[1]
        values[pages, offsets] = source + 40
        scales[pages, offsets] = source + 50
    for source in (1, 3, 5):
        rows = cache.compressed_rows(
            source, cache.geometry.ratios[source], positions
        ).long()
        keys = cache.index_rows(source, positions).long()
        packed = cache.compressed_cache(source)
        assert (
            packed[rows // packed.shape[1], rows % packed.shape[1], 0] == source + 30
        ).all()
        values, scales = cache.index_parts(source)
        pages, offsets = keys // packed.shape[1], keys % packed.shape[1]
        assert (values[pages, offsets] == source + 40).all()
        assert (scales[pages, offsets] == source + 50).all()
    for rows, value in saved:
        assert (cache.compressor_values[rows] == value).all()


def test_published_index_query_sources_share_keys_from_latest_kv_owner():
    path = Path(__file__).with_name("fixtures") / "deepseek_v41" / "inference.json"
    args = SimpleNamespace(**json.loads(path.read_text()))
    geometry = CacheGeometry.build(args, prefill_chunk_size=4096)
    assert geometry.compressor_sources == (2, 8, 14)
    assert geometry.index_offsets[2] == (3 * 64 + 128) * 288
    for layer in (20, 24, 28, 32, 36, 39):
        assert geometry.compressed_offsets[layer] == geometry.compressed_offsets[20]
        assert geometry.index_offsets[layer] == geometry.index_offsets[20]
    assert all(geometry.compressed_offsets[layer] is None for layer in (40, 41, 42))


def test_indexer_metadata_preserves_causal_lengths_and_request_grouping():
    from resin_models_cuda.deepseek_v41.indexer import index_metadata

    cache = _cache()
    positions = torch.arange(6).expand(2, -1)
    runtime = {}
    metadata = index_metadata(cache, positions, 2, runtime)
    lengths, requests, tables, schedules = metadata
    assert lengths.tolist() == [0, 1, 1, 2, 2, 3] * 2
    assert requests.tolist() == [0] * 6 + [1] * 6
    assert tables.shape[0] == positions.numel()
    physical = cache.groups[1].parent_ids * cache.geometry.packings[1]
    physical += cache.groups[1].child_slots
    assert torch.equal(tables[:2], physical)
    # Later query-source layers share exactly the same causal/page metadata.
    assert index_metadata(cache, positions, 2, runtime) is metadata
    assert schedules == {}
    c1 = index_metadata(cache, positions, 1, runtime)
    assert c1[0].tolist() == list(range(1, 7)) * 2
    assert c1 is not metadata


def test_indexer_accepts_the_shared_query_projection(implementation, monkeypatch):
    from resin_models_cuda.deepseek_v41 import indexer as kernels

    indexer = implementation.Indexer.__new__(implementation.Indexer)
    nn.Module.__init__(indexer)

    def duplicate_projection(_hidden):
        raise AssertionError("fused index Q was projected a second time")

    indexer.wq_b = duplicate_projection
    query = torch.ones(2, 5, 32 * 128)
    observed = []

    def execute(module, hidden, projected, *args):
        observed.append((module, projected))
        return projected[..., :1]

    monkeypatch.setattr(kernels, "paged_index", execute)
    result = indexer(None, None, None, None, None, None, None, None, query=query)
    assert observed == [(indexer, query)]
    assert result.shape == (2, 5, 1)


def test_final_hidden_collapses_with_last_ffn_pre_mix_instead_of_mean(implementation):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.engram_hash, model.hc_mult, model.target_layer_ids = None, 2, {0}
    model.embed, model.norm = nn.Embedding(1, 1), nn.Identity()
    model.embed.weight.data.fill_(1)

    class Layer(nn.Module):
        layer_id, engram = 0, None

        def forward(self, hidden, pre, ids, attention, *, prefill=False):
            return hidden + torch.tensor([1.0, 10.0])[
                None, None, :, None
            ], torch.tensor([[[2.0, 3.0]]])

    model.layers = nn.ModuleList([Layer()])
    hidden, collected = model._run(torch.tensor([[0]]), torch.tensor([[0]]), 1, None)
    torch.testing.assert_close(hidden, torch.tensor([[[37.0]]]))
    torch.testing.assert_close(collected, torch.tensor([[[1.0]]]))


def test_dspark_query_rope_starts_at_anchor_while_main_history_ends_before_it(
    implementation, monkeypatch
):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    hidden = torch.zeros(2, 3, 4)
    observed = []
    attention = SimpleNamespace(
        _project=lambda value: (value.clone(), value.clone()),
        wq_b=lambda value: value,
        kv_norm=nn.Identity(),
        n_local_heads=1,
        head_dim=4,
        rope_head_dim=2,
        window_size=128,
        softmax_scale=0.5,
        freqs_cis=torch.arange(200),
        layer_id=7,
        attn_sink=torch.zeros(1),
        attention_workspace=None,
        _finish=lambda output, positions: (
            observed.append(("finish", positions.clone())) or output
        ),
    )
    monkeypatch.setattr(
        implementation,
        "apply_rotary_emb",
        lambda value, freqs: observed.append(("rope", freqs.clone())),
    )
    monkeypatch.setattr(implementation, "pack_kv", lambda value, **kwargs: value)
    draft_cache, window_cache = object(), object()
    written = []
    monkeypatch.setattr(
        implementation,
        "store_kv",
        lambda target, indices, values: written.append((target, indices.clone())),
    )
    calls = []

    def fused(attention, query, positions, cache, indices, **kwargs):
        calls.append((positions.clone(), cache, indices.clone(), kwargs))
        return query

    monkeypatch.setattr(implementation, "fused_attention", fused)
    cache = SimpleNamespace(
        geometry=SimpleNamespace(draft_tokens=8),
        draft_cache=draft_cache,
        window_cache=lambda layer: window_cache,
        window_rows=lambda layer, positions: positions,
    )
    anchor = torch.tensor([7, 130])
    model._draft_attention(attention, hidden, anchor, cache)
    positions = torch.tensor([[7, 8, 9], [130, 131, 132]])
    assert [name for name, _ in observed] == ["rope", "finish"]
    for _, actual in observed:
        torch.testing.assert_close(actual, positions)
    query_positions, history_cache, history, extra = calls[0]
    assert history_cache is window_cache and extra["extra_cache"] is draft_cache
    torch.testing.assert_close(query_positions, positions)
    for row, start in enumerate((0, 2)):
        expected = torch.arange(start, int(anchor[row]))
        for query in history[row]:
            torch.testing.assert_close(query[query >= 0], expected)
    assert written[0][0] is draft_cache
    assert written[0][1].tolist() == [[0, 1, 2], [8, 9, 10]]
    assert extra["extra_indices"].tolist() == [[[0, 1, 2]] * 3, [[8, 9, 10]] * 3]
