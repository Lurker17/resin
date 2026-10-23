"""Sparse candidate padding must not change the attention reduction geometry."""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from test_deepseek_v4 import TestDeepSeekV4Weights as _ModelFixture

from resin.backend.cuda.parallel import TensorParallelContext


@pytest.fixture
def native(monkeypatch):
    model = _ModelFixture._native_model(monkeypatch)
    monkeypatch.setattr(model, "flashinfer_top_k", torch.topk)
    return model


@pytest.mark.parametrize("capacity", [4, 128])
def test_selected_rows_stay_sorted_and_fixed_width_across_score_padding(
    native, capacity
):
    selected = []
    for width in (64, 128):
        scores = torch.full((2, width), -torch.inf)
        scores[0, [1, 7, 15, 63]] = torch.tensor([2.0, 8.0, 1.0, 4.0])

        indices = native._compact_topk_indices(scores, capacity)

        assert indices.shape == (2, capacity)
        assert indices[0, :4].tolist() == [1, 7, 15, 63]
        assert (indices[0, 4:] == -1).all()
        assert (indices[1] == -1).all()
        selected.append(indices)
    torch.testing.assert_close(*selected)


def test_selection_retains_highest_scores_before_ordering_logical_positions(native):
    scores = torch.full((1, 128), -torch.inf)
    scores[0, [2, 3, 8, 19, 27, 62]] = torch.tensor([1.0, 6.0, 3.0, 5.0, 2.0, 4.0])

    selected = native._compact_topk_indices(scores, 4)

    assert selected.tolist() == [[3, 8, 19, 62]]


def test_unavailable_candidates_cannot_enter_the_active_sparse_prefix(native):
    scores = torch.ones(2, 64)
    complete = torch.tensor([[3], [0]])

    selected = native._compact_topk_indices(scores, 64, complete=complete)

    assert selected[0, :3].tolist() == [0, 1, 2]
    assert (selected[0, 3:] == -1).all()
    assert (selected[1] == -1).all()


@pytest.mark.parametrize(
    "maximum,topk,expected",
    [(1, 512, 64), (257, 512, 128), (4096, 512, 512), (4096, 64, 64)],
)
def test_model_capacity_depends_on_configured_context(native, maximum, topk, expected):
    args = native.ModelArgs(
        dim=4,
        n_heads=1,
        head_dim=4,
        rope_head_dim=4,
        q_lora_rank=4,
        o_groups=1,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        max_seq_len=maximum,
        compress_ratios=(4, 128),
    )
    indexed = native.Attention(0, args, tensor_parallel=TensorParallelContext())
    generated = native.Attention(1, args, tensor_parallel=TensorParallelContext())

    assert indexed.indexer.selected_capacity == expected
    assert indexed.compressed_capacity == expected
    assert generated.compressed_capacity == ((maximum + 127) // 128 + 3) // 4 * 4


@pytest.mark.parametrize("topk", [0, 1, 65])
def test_indexer_rejects_unaligned_selected_capacity_before_allocating(native, topk):
    with pytest.raises(ValueError, match="positive multiple of 64"):
        native.Indexer(0, native.ModelArgs(index_topk=topk))


def test_indexer_prefill_and_graph_map_the_same_fixed_sparse_rows(native, monkeypatch):
    capacity = 128
    indexer = SimpleNamespace(
        compress_ratio=4,
        rope_head_dim=2,
        head_dim=2,
        layer_id=2,
        n_local_heads=2,
        n_heads=2,
        softmax_scale=0.5,
        selected_capacity=capacity,
        tensor_parallel=SimpleNamespace(enabled=False),
        freqs_cis=torch.ones(512, 1, dtype=torch.complex64),
        compressor=SimpleNamespace(forward_prefill=lambda *args, **kwargs: None),
        wq_b=lambda qr: qr,
        weights_proj=lambda hidden: torch.ones(*hidden.shape[:2], 2),
    )
    indexer._select = MethodType(native.Indexer._select, indexer)
    positions = torch.tensor([[15, 19]])
    scores_widths = []

    def scores(q, *args, cache_width, **kwargs):
        scores_widths.append(cache_width)
        result = torch.full((*q.shape[:2], cache_width), -torch.inf)
        # Deliberately scramble score order. Position 15 can see four rows;
        # position 19 can see five. Unavailable padded scores remain -inf.
        result[0, 0, [0, 1, 2, 3]] = torch.tensor([2.0, 4.0, 1.0, 3.0])
        result[0, 1, [0, 1, 2, 3, 4]] = torch.tensor([2.0, 4.0, 1.0, 3.0, 5.0])
        return result

    def physical(layer, ratio, token_positions):
        return 100 + token_positions // ratio * 7

    def finish(indices, token_positions, ratio, offset, **kwargs):
        complete = (token_positions + 1) // ratio
        valid = (indices >= 0) & (indices < complete[..., None])
        return torch.where(valid, 100 + (indices + offset) * 7, -1).int()

    cache = SimpleNamespace(
        index_keys=torch.empty(32, 2),
        index_rows=lambda *args: positions,
        index_mapping=lambda layer: object(),
        compressed_rows=physical,
        compressed_mapping=lambda *args: object(),
    )
    for name in ("apply_rotary_emb", "rope_last_block_inplace", "fp4_act_quant"):
        monkeypatch.setattr(native, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(native, "rotate_activation", lambda tensor: tensor)
    monkeypatch.setattr(native, "index_scores", scores)
    monkeypatch.setattr(native, "finish_index_topk", finish)
    hidden = torch.arange(8, dtype=torch.float32).view(1, 2, 4)

    prefill = native.Indexer.forward_prefill(
        indexer, hidden, hidden, positions, 15, 20, cache=cache
    )
    graph = native.Indexer._select(
        indexer, hidden, hidden, positions, 0, 512, cache=cache
    )

    assert scores_widths == [64, 128]
    assert prefill.shape == graph.shape == (1, 2, capacity)
    torch.testing.assert_close(prefill, graph)
    assert prefill[0, 0, :4].tolist() == [100, 107, 114, 121]
    assert (prefill[0, 0, 4:] == -1).all()
    assert prefill[0, 1, :5].tolist() == [100, 107, 114, 121, 128]
    assert (prefill[0, 1, 5:] == -1).all()


def test_generated_compressed_geometry_is_shared_by_all_attention_phases(
    native, monkeypatch
):
    metadata_widths = []
    stream = SimpleNamespace(wait_stream=lambda other: None)

    def metadata(cache, positions, selected, generated_width):
        assert selected is None
        metadata_widths.append(generated_width)
        return (
            torch.empty(*positions.shape, 192, dtype=torch.int32),
            torch.ones_like(positions),
            positions + 1,
        )

    attention = SimpleNamespace(
        head_dim=4,
        n_local_heads=1,
        compress_ratio=128,
        compressed_capacity=8,
        layer_id=2,
        kv_norm=SimpleNamespace(weight=torch.ones(4)),
        freqs_cis=torch.ones(512, 2, dtype=torch.complex64),
        eps=1e-6,
        compressor=SimpleNamespace(
            forward_prefill=lambda *args, **kwargs: None,
            forward_verify=lambda *args, **kwargs: None,
        ),
        indexer=None,
        attn_sink=torch.zeros(1),
        attention_workspace=object(),
        decode_overlap_stream=stream,
        softmax_scale=0.5,
        _prepare_compression=lambda: None,
        _project=lambda hidden: (hidden, hidden),
        prefill_project=None,
        wq_b=lambda hidden: hidden,
        prefill_query=None,
        _paged_attention_metadata=metadata,
        _finish=lambda output, positions: output.squeeze(2),
        prefill_finish=None,
    )
    cache = SimpleNamespace(
        latents=torch.empty(16, 4),
        compressed_rows=lambda layer, ratio, positions: positions,
        compressed_mapping=lambda *args: object(),
        state_mapping=lambda *args: object(),
        window_mapping=lambda *args: object(),
    )
    for name in ("qk_norm_rope_block", "qk_norm_rope_store"):
        monkeypatch.setattr(native, name, lambda *args, **kwargs: None)
    for name in ("sparse_verify_attention", "sparse_decode_attention"):
        monkeypatch.setattr(native, name, lambda query, *args: query)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    hidden = torch.ones(1, 2, 4)
    native.Attention.forward_prefill(
        attention, hidden, torch.tensor([[23, 24]]), 23, 25, cache=cache
    )
    native.Attention.forward_verify(
        attention, hidden, torch.tensor([[319, 320]]), 512, cache=cache
    )
    native.Attention.forward_decode(
        attention, hidden[:, :1], torch.tensor([23]), 256, cache=cache
    )

    assert metadata_widths == [8, 8, 8]
