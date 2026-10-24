"""Context ranks preserve full recurrent state while partitioning query work."""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from test_deepseek_v4 import TestDeepSeekV4Weights as _ModelFixture

from resin.backend.cuda.context_parallel import (
    ContextParallelPlan,
    ContextParallelTokens,
)
from resin.backend.topology import RankGroup


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("captured_regions", [False, True])
@pytest.mark.parametrize(
    "length,size,rank", [(5, 2, 0), (5, 2, 1), (1, 4, 0), (1, 4, 3)]
)
def test_prefill_context_ranks_update_full_state_and_compute_only_local_queries(
    monkeypatch, ratio, captured_regions, length, size, rank
):
    native = _ModelFixture._native_model(monkeypatch)
    hidden = torch.arange(length * 4, dtype=torch.float32).view(1, length, 4)
    positions = 9 + torch.arange(length).view(1, -1)
    expected = hidden.squeeze(0) * 2
    group = RankGroup(tuple(range(size)), object(), rank)
    calls = []

    def gather(output, local, *, group):
        capacity = (length + size - 1) // size
        output.zero_()
        for peer in range(size):
            plan = ContextParallelPlan.build((length,), rank=peer, size=size)
            output[peer * capacity : peer * capacity + plan.local_tokens].copy_(
                expected[plan.start : plan.stop]
            )
        calls.append(("gather", local.clone()))

    tokens = ContextParallelTokens(
        (length,),
        group,
        device="cpu",
        distributed=SimpleNamespace(all_gather_into_tensor=gather),
    )
    local_positions = positions[:, tokens.plan.start : tokens.plan.stop]

    def compress(x, token_positions, **kwargs):
        calls.append(("compress", x.clone(), token_positions.clone()))

    def index(x, qr, token_positions, start, sequence_length, **kwargs):
        calls.append(
            ("index", x.clone(), token_positions.clone(), kwargs["query_indices"])
        )
        return torch.zeros((1, tokens.plan.local_tokens, 1), dtype=torch.int32)

    def kv_store(kv, norm, frequencies, cache, token_positions, epsilon, **kwargs):
        calls.append(("kv", kv.clone(), token_positions.clone()))

    def qk_store(q, kv, norm, frequencies, cache, token_positions, epsilon, **kwargs):
        calls.append(("query", q.shape[1], token_positions.clone()))

    def project(hidden):
        calls.append(("project", hidden.clone()))
        return hidden, hidden

    def project_query(qr):
        calls.append(("project_query", qr.clone()))
        return qr * 2

    def finish(output, token_positions):
        calls.append(("finish", output.clone(), token_positions.clone()))
        return output.squeeze(2)

    monkeypatch.setattr(native, "kv_norm_rope_store", kv_store)
    monkeypatch.setattr(native, "qk_norm_rope_block", qk_store)
    monkeypatch.setattr(native, "sparse_verify_attention", lambda q, *args: q)
    attention = SimpleNamespace(
        dim=4,
        head_dim=4,
        n_local_heads=1,
        compress_ratio=ratio,
        compressed_capacity=4 if ratio == 128 else 1 if ratio == 4 else 0,
        layer_id=2,
        kv_norm=SimpleNamespace(weight=torch.ones(4)),
        freqs_cis=torch.ones(32, 2, dtype=torch.complex64),
        eps=1e-6,
        compressor=SimpleNamespace(forward_prefill=compress),
        indexer=SimpleNamespace(forward_prefill=index) if ratio == 4 else None,
        attn_sink=torch.zeros(1),
        attention_workspace=object(),
        softmax_scale=0.5,
        _prepare_compression=lambda: None,
        _project=lambda x: (x, x),
        prefill_project=project if captured_regions else None,
        wq_b=lambda x: x * 2,
        prefill_query=project_query if captured_regions else None,
        _paged_attention_metadata=lambda *args: (None, None, None),
        _finish=lambda output, positions: output.squeeze(2),
        prefill_finish=finish if captured_regions else None,
    )
    cache = SimpleNamespace(
        latents=torch.empty(8, 4),
        compressed_rows=lambda *args: positions,
        window_mapping=lambda layer: object(),
    )

    output = native.Attention.forward_prefill(
        attention,
        hidden,
        positions,
        9,
        9 + length,
        cache=cache,
        context_parallel=tokens,
    )

    torch.testing.assert_close(output.squeeze(0), expected)
    assert [call[0] for call in calls].count("compress") == bool(ratio)
    assert [call[0] for call in calls].count("index") == (ratio == 4)
    for call in calls:
        if call[0] in {"compress", "index", "kv"}:
            torch.testing.assert_close(call[1], hidden)
            torch.testing.assert_close(call[2], positions)
        elif call[0] == "query":
            assert call[1] == tokens.plan.local_tokens
            torch.testing.assert_close(call[2], local_positions)
        elif call[0] == "gather":
            torch.testing.assert_close(
                call[1][: tokens.plan.local_tokens],
                expected[tokens.plan.start : tokens.plan.stop],
            )
    assert [call[0] for call in calls].count("kv") == 1
    assert [call[0] for call in calls].count("query") == bool(tokens.plan.local_tokens)
    assert [call[0] for call in calls].count("project") == captured_regions
    assert [call[0] for call in calls].count("finish") == (
        captured_regions and bool(tokens.plan.local_tokens)
    )
    assert [call[0] for call in calls].count("project_query") == (
        captured_regions and bool(tokens.plan.local_tokens)
    )
    if captured_regions:
        torch.testing.assert_close(calls[0][1], hidden)
        for call in calls:
            if call[0] == "finish":
                torch.testing.assert_close(call[2], local_positions)
            elif call[0] == "project_query":
                torch.testing.assert_close(
                    call[1], hidden[:, tokens.plan.start : tokens.plan.stop]
                )


@pytest.mark.parametrize("local_indices", [[1, 2], []])
def test_indexer_updates_complete_compressor_before_selecting_queries(
    monkeypatch, local_indices
):
    native = _ModelFixture._native_model(monkeypatch)
    x = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    positions = torch.tensor([[8, 9, 10, 11]])
    selected = torch.tensor(local_indices, dtype=torch.long)
    calls = []

    def compress(hidden, token_positions, **kwargs):
        calls.append(("compress", hidden.clone(), token_positions.clone()))

    def score(q, cache, weights, token_positions, *args, **kwargs):
        calls.append(("scores", q.shape[1], token_positions.clone()))
        return torch.zeros((1, len(local_indices), 64))

    monkeypatch.setattr(native, "rope_last_block_inplace", lambda *args: None)
    monkeypatch.setattr(native, "rotate_activation", lambda x: x)
    monkeypatch.setattr(native, "fp4_act_quant", lambda *args: None)
    monkeypatch.setattr(native, "index_scores", score)
    monkeypatch.setattr(native, "flashinfer_top_k", torch.topk)
    indexer = SimpleNamespace(
        compress_ratio=4,
        rope_head_dim=2,
        head_dim=2,
        layer_id=2,
        n_local_heads=2,
        n_heads=2,
        softmax_scale=0.5,
        index_topk=1,
        selected_capacity=1,
        tensor_parallel=SimpleNamespace(enabled=False),
        freqs_cis=torch.ones(32, 1, dtype=torch.complex64),
        compressor=SimpleNamespace(forward_prefill=compress),
        wq_b=lambda qr: qr,
        weights_proj=lambda hidden: torch.ones(*hidden.shape[:2], 2),
    )
    indexer._select = MethodType(native.Indexer._select, indexer)
    monkeypatch.setattr(
        native, "finish_index_topk", lambda indices, *args, **kwargs: indices
    )
    cache = SimpleNamespace(
        index_keys=torch.zeros(16, 2),
        index_rows=lambda *args: positions,
        index_mapping=lambda layer: object(),
        compressed_mapping=lambda *args: object(),
    )

    result = native.Indexer.forward_prefill(
        indexer, x, x, positions, 8, 12, cache=cache, query_indices=selected
    )

    assert result.shape == (1, len(local_indices), 1)
    assert calls[0][0] == "compress"
    torch.testing.assert_close(calls[0][1], x)
    torch.testing.assert_close(calls[0][2], positions)
    if local_indices:
        assert calls[1][0:2] == ("scores", len(local_indices))
        torch.testing.assert_close(calls[1][2], positions[:, selected])
    else:
        assert len(calls) == 1


@pytest.mark.parametrize("size", [1, 2])
def test_transformer_prefill_prepares_one_context_map_for_all_layer_queries(
    monkeypatch, size
):
    native = _ModelFixture._native_model(monkeypatch)
    group = RankGroup(tuple(range(size)), object(), 0)
    distributed = object()
    groups = SimpleNamespace(cp=group, distributed=distributed)
    context = object()
    prepared, attended = [], []
    hidden = torch.arange(20, dtype=torch.float32).view(1, 5, 4)

    def prepare(lengths, group_arg, **kwargs):
        prepared.append((lengths, group_arg, kwargs))
        return context

    def attention(x, positions, start, sequence_length, **kwargs):
        attended.append((x, positions, start, sequence_length, kwargs))
        return x * 2

    monkeypatch.setattr(native, "ContextParallelTokens", prepare)
    monkeypatch.setattr(native, "dynamic_gemm_rows", nullcontext)
    layer = SimpleNamespace(
        attn=SimpleNamespace(forward_prefill=attention),
        forward_attention=lambda x, ids, operation: operation(x),
    )
    model = SimpleNamespace(
        parallel_groups=groups,
        _forward_layers=lambda ids, operation: (operation(layer, hidden), None),
        _target_output=lambda output, main: (output, main),
    )

    output, main = native.Transformer.forward_prefill(
        model, torch.tensor([[3, 4, 5, 6, 7]]), 9, cache="cache"
    )

    torch.testing.assert_close(output, hidden * 2)
    assert main is None
    assert len(prepared) == (size > 1)
    if prepared:
        assert prepared == [
            ((5,), group, {"device": hidden.device, "distributed": distributed})
        ]
    assert attended[0][0] is hidden
    torch.testing.assert_close(attended[0][1], torch.arange(9, 14).view(1, -1))
    assert attended[0][2:4] == (9, 14)
    assert attended[0][4] == {
        "cache": "cache",
        "context_parallel": context if size > 1 else None,
    }
