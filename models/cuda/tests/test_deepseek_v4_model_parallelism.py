"""Model layers bind their own attention groups, separately from expert shards."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from test_deepseek_v4 import TestDeepSeekV4Weights as _ModelFixture

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout


def test_attention_layers_capture_their_own_tensor_parallel_context(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    handle = object()
    sharded = TensorParallelContext(TensorParallelLayout(rank=1, size=2), handle)
    local = TensorParallelContext()
    reductions, gathers = [], []

    def reduce(tensor, *, group):
        reductions.append(group)
        tensor.add_(5)

    def gather(outputs, tensor, *, group):
        gathers.append(group)
        for index, output in enumerate(outputs):
            output.copy_(tensor + index)

    monkeypatch.setattr(native.dist, "all_reduce", reduce)
    monkeypatch.setattr(native.dist, "all_gather", gather)
    embedding = native.ParallelEmbedding(8, 2, tensor_parallel=sharded)
    projection = native.RowParallelLinear(4, 2, tensor_parallel=sharded)
    head = native.ParallelHead(8, 2, tensor_parallel=sharded)
    column = native.ColumnParallelLinear(2, 4, tensor_parallel=sharded)
    # Constructing a second model's layers cannot change the first one's mesh.
    other_embedding = native.ParallelEmbedding(8, 2, tensor_parallel=local)
    other_projection = native.RowParallelLinear(4, 2, tensor_parallel=local)
    other_head = native.ParallelHead(8, 2, tensor_parallel=local)
    other_column = native.ColumnParallelLinear(2, 4, tensor_parallel=local)
    with torch.no_grad():
        embedding.weight.fill_(3)
        other_embedding.weight.fill_(2)
        projection.weight.fill_(1)
        other_projection.weight.fill_(2)
        head.weight.fill_(1)
        other_head.weight.fill_(2)

    assert embedding.vocab_start_idx == 4
    torch.testing.assert_close(
        embedding(torch.tensor([4, 0])), torch.tensor([[8.0, 8.0], [5.0, 5.0]])
    )
    torch.testing.assert_close(
        other_embedding(torch.tensor([4, 0])), torch.full((2, 2), 2.0)
    )
    torch.testing.assert_close(
        projection(torch.ones(1, 2, dtype=torch.bfloat16)),
        torch.full((1, 2), 7.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        other_projection(torch.ones(1, 4, dtype=torch.bfloat16)),
        torch.full((1, 2), 8.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        head(torch.ones(1, 1, 2)), torch.tensor([[2.0] * 4 + [3.0] * 4])
    )
    torch.testing.assert_close(other_head(torch.ones(1, 1, 2)), torch.full((1, 8), 4.0))
    assert column.weight.shape == (2, 2)
    assert other_column.weight.shape == (4, 2)
    assert reductions == [handle, handle]
    assert gathers == [handle]


def test_attention_and_expert_weight_meshes_are_independent(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    args = native.ModelArgs(
        dim=128,
        vocab_size=256,
        moe_inter_dim=128,
        n_layers=1,
        n_heads=4,
        index_n_heads=4,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=4,
        head_dim=128,
        compress_ratios=(0,),
        n_routed_experts=8,
        expert_dtype="fp4",
    )
    attention = native.Attention(0, args, tensor_parallel=TensorParallelContext())
    experts = native.MoEWeights(
        0, args, expert_parallel=TensorParallelLayout(rank=2, size=4)
    )
    sharded_attention = native.Attention(
        0,
        args,
        tensor_parallel=TensorParallelContext(TensorParallelLayout(size=2), object()),
    )
    assert attention.n_local_heads == 4
    assert attention.n_local_groups == 4
    assert sharded_attention.n_local_heads == 2
    assert sharded_attention.n_local_groups == 2
    assert experts.experts_start_idx == 4
    assert experts.experts_end_idx == 6
    assert [i for i, expert in enumerate(experts.experts) if expert is not None] == [
        4,
        5,
    ]


def test_empty_model_round_only_calls_prepared_routed_experts(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    calls = []

    def routed(hidden, indices, weights):
        assert hidden.shape == (0, 4) and hidden.dtype == torch.bfloat16
        assert indices.shape == weights.shape == (0, 3)
        assert indices.dtype == torch.int32 and weights.dtype == torch.bfloat16
        calls.append(True)

    model = SimpleNamespace(
        parallel_groups=object(),
        embed=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16), dim=4),
        layers=[
            SimpleNamespace(
                ffn=SimpleNamespace(gate=SimpleNamespace(topk=3), routed=routed)
            )
            for _ in range(2)
        ],
    )
    native.Transformer.forward_experts_empty(model)
    assert len(calls) == 2
