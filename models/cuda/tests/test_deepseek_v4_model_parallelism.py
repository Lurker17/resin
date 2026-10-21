"""Model layers bind their own attention groups, separately from expert shards."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_deepseek_v4 import TestDeepSeekV4Weights as _ModelFixture

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin_models_cuda.deepseek_v4.weights import load_deepseek_v4_weights


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("draft_tokens", [0, 1, 2, 5])
def test_compressor_ring_preserves_the_anchor_window_before_speculative_tail(
    monkeypatch, ratio, draft_tokens
):
    native = _ModelFixture._native_model(monkeypatch)
    args = native.ModelArgs(dim=64, head_dim=64, dspark_block_size=draft_tokens)
    compressor = native.Compressor(args, ratio, head_dim=64)
    logical_window = ratio * (2 if ratio == 4 else 1)
    required = logical_window + draft_tokens
    assert compressor.ring_size == 1 << (required - 1).bit_length()

    # The kernel preloads a ring ending at the final speculative position.
    # If the anchor itself completes a compression window, its oldest live
    # input must still occur in that preloaded chronology.
    anchor = 3 * ratio - 1
    oldest_loaded = anchor + draft_tokens - compressor.ring_size + 1
    assert oldest_loaded <= anchor - logical_window + 1


def test_attention_layers_capture_their_own_tensor_parallel_context(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    handle = Mock(spec=torch.distributed.ProcessGroup)
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
        moe_inter_dim=384,
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
        0,
        args,
        expert_parallel=TensorParallelLayout(rank=2, size=4),
        expert_tensor_parallel=TensorParallelLayout(rank=1, size=2),
    )
    sharded_attention = native.Attention(
        0,
        args,
        # The opaque group is never used: this test only checks shard geometry.
        tensor_parallel=TensorParallelContext(TensorParallelLayout(size=2), object()),  # ty: ignore[invalid-argument-type]
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
    assert experts.expert_intermediate_shard.start == 256
    assert experts.expert_intermediate_shard.stop == 384
    assert experts.experts[4].w1.weight.shape == (128, 64)
    assert experts.experts[4].w2.weight.shape == (128, 64)
    assert experts.experts[4].w3.weight.shape == (128, 64)
    assert experts.gate.weight.shape == (8, 128)
    assert experts.shared_experts.w1.weight.shape == (384, 128)
    assert experts.shared_experts.w2.weight.shape == (128, 384)


def test_streaming_loader_reconstructs_uneven_fp4_expert_tensor_shards(
    monkeypatch, tmp_path
):
    import resin_models_cuda.deepseek_v4.weights as weights

    native = _ModelFixture._native_model(monkeypatch)
    args = native.ModelArgs(
        dim=128,
        moe_inter_dim=384,
        n_routed_experts=4,
        expert_dtype="fp4",
    )
    full = native.MoEWeights(0, args, expert_parallel=TensorParallelLayout())

    def checkpoint_model(moe):
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.ffn = moe
        model.layers = torch.nn.ModuleList([layer])
        return model

    checkpoint = {}
    for name, tensor in checkpoint_model(full).state_dict().items():
        if tensor.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
            raw = torch.arange(tensor.numel()).remainder(251).to(torch.uint8)
            value = raw.reshape(tensor.shape).view(tensor.dtype)
            # Published FP4 weights are signed bytes; scales retain their dtype.
            if tensor.dtype == torch.float4_e2m1fn_x2:
                value = value.view(torch.int8)
        else:
            value = (
                torch.arange(tensor.numel()).remainder(31).reshape(tensor.shape) / 32
            ).to(tensor.dtype)
        checkpoint[name] = value

    reads = []

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def read(self, _path, name):
            reads.append(name)
            return checkpoint[name]

    path = tmp_path / "model.safetensors"
    monkeypatch.setattr(weights, "SafetensorReader", Reader)
    monkeypatch.setattr(
        weights,
        "indexed_tensor_files",
        lambda *_args, **_kwargs: {name: path for name in checkpoint},
    )
    for ep_rank in range(2):
        shards = []
        for expert_tp_rank in range(2):
            shard = native.MoEWeights(
                0,
                args,
                expert_parallel=TensorParallelLayout(rank=ep_rank, size=2),
                expert_tensor_parallel=TensorParallelLayout(
                    rank=expert_tp_rank, size=2
                ),
            )
            model = checkpoint_model(shard)
            reads.clear()
            report = load_deepseek_v4_weights(
                tmp_path,
                model,
                rank=0,
                tensor_parallel_size=1,
                device="cpu",
            )
            assert report["tensors"] == len(model.state_dict())
            assert set(reads) == set(model.state_dict())
            assert all(
                ".experts." not in name
                or int(name.split(".experts.")[1].split(".")[0]) // 2 == ep_rank
                for name in reads
            )
            for name, tensor in model.state_dict().items():
                if ".experts." not in name:
                    torch.testing.assert_close(tensor, checkpoint[name], rtol=0, atol=0)
            shards.append(model)

        for expert_id in range(2 * ep_rank, 2 * ep_rank + 2):
            for projection, axis in (("w1", 0), ("w2", 1), ("w3", 0)):
                for storage in ("weight", "scale"):
                    name = f"layers.0.ffn.experts.{expert_id}.{projection}.{storage}"
                    restored = torch.cat(
                        [
                            shard.state_dict()[name].view(torch.uint8)
                            for shard in shards
                        ],
                        dim=axis,
                    )
                    assert torch.equal(restored, checkpoint[name].view(torch.uint8))


@pytest.mark.parametrize(
    ("intermediate", "experts", "ep", "expert_tp", "message"),
    [
        (320, 4, 2, 2, "moe_inter_dim=320 must be divisible by 128"),
        (128, 4, 2, 2, "only 1 aligned units for TP=2"),
        (384, 3, 2, 2, "n_routed_experts=3 must be divisible by TP=2"),
    ],
)
def test_invalid_expert_mesh_is_rejected_before_weight_allocation(
    monkeypatch, intermediate, experts, ep, expert_tp, message
):
    native = _ModelFixture._native_model(monkeypatch)
    args = native.ModelArgs(
        dim=128,
        moe_inter_dim=intermediate,
        n_routed_experts=experts,
        expert_dtype="fp4",
    )

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("invalid expert geometry allocated model weights")

    monkeypatch.setattr(native.torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError, match=message):
        native.MoEWeights(
            0,
            args,
            expert_parallel=TensorParallelLayout(size=ep),
            expert_tensor_parallel=TensorParallelLayout(size=expert_tp),
        )


def test_bf16_expert_tensor_partials_sum_to_full_expert(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    monkeypatch.setattr(
        native,
        "swiglu",
        lambda gate, up, _limit: torch.nn.functional.silu(gate.float()) * up.float(),
    )
    args = native.ModelArgs(
        dim=8,
        moe_inter_dim=7,
        n_routed_experts=2,
        dtype="bf16",
    )
    full = native.MoEWeights(0, args, expert_parallel=TensorParallelLayout())
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in full.experts[1].parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) / 4)
    hidden = torch.randn(5, 8, generator=generator, dtype=torch.bfloat16)
    route_weights = torch.tensor([0.25, 0.5, 0.75, 1, 0.125]).view(-1, 1)
    expected = full.experts[1](hidden, route_weights).float()
    partials = []
    for rank in range(3):
        shard = native.MoEWeights(
            0,
            args,
            expert_parallel=TensorParallelLayout(rank=1, size=2),
            expert_tensor_parallel=TensorParallelLayout(rank=rank, size=3),
        )
        interval = shard.expert_intermediate_shard.as_slice()
        with torch.no_grad():
            shard.experts[1].w1.weight.copy_(full.experts[1].w1.weight[interval])
            shard.experts[1].w3.weight.copy_(full.experts[1].w3.weight[interval])
            shard.experts[1].w2.weight.copy_(full.experts[1].w2.weight[:, interval])
        partials.append(shard.experts[1](hidden, route_weights).float())

    # Each local GEMM rounds to BF16 before combining its partial output.
    torch.testing.assert_close(sum(partials), expected, rtol=0.02, atol=0.001)


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


def test_empty_speculative_stages_visit_their_own_expert_layers(monkeypatch):
    native = _ModelFixture._native_model(monkeypatch)
    calls = []

    def layer(name):
        def routed(hidden, indices, weights):
            assert hidden.shape == (0, 4)
            assert hidden.dtype == weights.dtype == torch.bfloat16
            assert indices.shape == weights.shape == (0, 3)
            calls.append(name)
            # Prepared distributed expert callbacks retain FP32 partials.
            return hidden.float()

        return SimpleNamespace(
            ffn=SimpleNamespace(gate=SimpleNamespace(topk=3), routed=routed)
        )

    model = SimpleNamespace(
        parallel_groups=object(),
        embed=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16), dim=4),
        layers=[layer("target-0"), layer("target-1")],
        mtp=[layer("draft-0"), layer("draft-1"), layer("draft-2")],
    )
    native.Transformer.forward_experts_empty(model, stage="draft")
    native.Transformer.forward_experts_empty(model)
    assert calls == ["draft-0", "draft-1", "draft-2", "target-0", "target-1"]

    with pytest.raises(ValueError, match="stage"):
        native.Transformer.forward_experts_empty(model, stage="invalid")
    assert len(calls) == 5
