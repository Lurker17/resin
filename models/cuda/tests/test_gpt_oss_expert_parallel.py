"""GPT-OSS keeps global routing separate from local expert storage/execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.topology import RankGroup
from resin_models_cuda.gpt_oss.mxfp4 import (
    EXPERT_CHECKPOINT_NAMES,
    kernel_expert_factory,
    moe_kernel_config,
    shard_checkpoint_tensor,
)


def _args():
    return SimpleNamespace(
        num_local_experts=8,
        num_experts_per_tok=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swiglu_limit=7.0,
    )


@pytest.mark.parametrize("name", EXPERT_CHECKPOINT_NAMES)
def test_all_expert_checkpoint_tensors_are_sharded_on_the_expert_axis(name):
    tensor = torch.arange(8 * 16).view(8, 4, 4)
    shard = shard_checkpoint_tensor(
        f"model.layers.0.mlp.experts.{name}",
        tensor,
        args=_args(),
        tensor_parallel=TensorParallelLayout(),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(shard, tensor[4:])
    assert shard.is_contiguous()


def test_expert_and_attention_shards_use_independent_axes():
    args = _args()
    tensor = torch.arange(8 * 128 * 32).view(8, 128, 2, 16)
    expert_shard = shard_checkpoint_tensor(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        tensor,
        args=args,
        tensor_parallel=TensorParallelLayout(rank=0, size=2),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(expert_shard, tensor[4:, :64])
    query = torch.arange(32 * 32).view(32, 32)
    query_shard = shard_checkpoint_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        query,
        args=args,
        tensor_parallel=TensorParallelLayout(rank=0, size=2),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(query_shard, query[:16])
    router = torch.arange(8 * 32).view(8, 32)
    assert (
        shard_checkpoint_tensor(
            "model.layers.0.mlp.router.weight",
            router,
            args=args,
            tensor_parallel=TensorParallelLayout(),
            expert_parallel=TensorParallelLayout(rank=1, size=2),
        )
        is router
    )


def test_ep_local_adapter_preserves_global_routing_weights_and_skips_empty():
    args = _args()
    local_args = SimpleNamespace(**vars(args))
    local_args.num_local_experts = 4
    config = moe_kernel_config(local_args, "sm90")
    groups = SimpleNamespace(
        ep=RankGroup((0, 1), object(), 1),
        attn_replication=RankGroup((1,), object(), 0),
    )
    calls = []
    storage = object()

    class Kernel:
        capabilities = SimpleNamespace(supports_fp32_partial=True)

        def forward_partial(self, hidden, routing, weights, actual):
            assert actual is config and weights is storage
            calls.append(routing)
            result = torch.zeros_like(hidden, dtype=torch.float32)
            for expert in range(config.num_experts):
                # CUTLASS's W4 expert map stops at the first matching slot.
                # A foreign zero-weight expert-0 placeholder would conceal
                # the real owned expert-0 route later in the same row.
                matches = routing.selected_experts == expert
                first = matches.int().argmax(1)
                scales = routing.expert_weights.gather(1, first[:, None])
                scales = scales * matches.any(1)[:, None]
                result += hidden * scales * (expert + 5)
            return result

    module = kernel_expert_factory(Kernel(), config, parallel_groups=groups)(args)
    module.install_prepared_weights(storage, device=torch.device("cpu"))
    hidden = torch.ones(3, 32, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 4], [7, 3], [5, 6]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    output = module.forward_local(hidden, indices, weights)
    torch.testing.assert_close(output[:, 0], torch.tensor([4.0, 3.2, 6.7]))
    assert calls[0].selected_experts.tolist() == [[4, 0], [3, 4], [1, 2]]
    torch.testing.assert_close(
        calls[0].expert_weights,
        torch.tensor([[0.0, 0.8], [0.4, 0.0], [0.3, 0.7]]),
    )
    assert calls[0].router_logits.shape == (3, 4)
    assert torch.equal(weights, torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]]))
    empty = module.forward_local(hidden[:0], indices[:0], weights[:0])
    assert empty.shape == (0, 32) and len(calls) == 1

    # Both the torch router (int64/BF16) and idle ranks use one wire format.
    routed = []

    def dispatch(hidden, indices, weights):
        routed.append((hidden, indices, weights))
        return hidden

    module._dispatch = dispatch
    module(hidden, indices, weights.bfloat16(), torch.empty(3, 8))
    module.forward_empty()
    for _, expert_ids, probabilities in routed:
        assert expert_ids.dtype == torch.int32
        assert probabilities.dtype == torch.float32
    assert routed[1][0].shape == (0, 32)
    assert routed[1][1].shape == routed[1][2].shape == (0, 2)


def test_ep_rejects_kernels_without_fp32_partial_execution():
    args = _args()
    config = moe_kernel_config(args, "sm90")
    with pytest.raises(ValueError, match="FP32 forward_partial"):
        kernel_expert_factory(object(), config, parallel_groups=object())
