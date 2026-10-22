"""Actual V4.1 prepared-FFN adapter against decoded Torch FFNs on an EP2 mesh."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn

from deploy.deepinfer_kernel_probe import (
    quantize_fp4_raw,
    quantize_fp8,
    reference_ffn,
)
from resin.backend.cuda.mega_moe import MegaMoe, MegaMoeCollective
from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import prepare_model
from resin.backend.topology import RankGroup
from resin.backend.torch import use_default_dtype


def _block_fp8(shape):
    """Generate checkpoint-style32x32 weights, retaining a decoded oracle."""
    source = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.02
    n, k = shape
    blocks = source.reshape(n // 32, 32, k // 32, 32)
    scale = (
        2 ** torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-4) / 448))
    ).to(torch.float8_e8m0fnu)
    full = scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    weight = (source / full).to(torch.float8_e4m3fn)
    return weight, scale, weight.float() * full


@torch.inference_mode()
def _block_hc():
    """Real Block wiring against its explicit checkpoint equations, including B>1."""
    from resin_models_cuda.deepseek_v41.hc import mega_ffn_transition
    from resin_models_cuda.deepseek_v41.model import Block, ModelArgs, RMSNorm

    class FFN(nn.Module):
        def forward(self, hidden, _ids):
            return (hidden.float() * 0.75).bfloat16()

    args = ModelArgs(dim=5120, norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20)
    block = Block.__new__(Block)
    nn.Module.__init__(block)
    block.args, block.prefill_ffn, block.ffn_transition = args, None, None
    block.ffn = FFN()
    with torch.device("cuda"):
        block.attn_norm, block.ffn_norm = (
            RMSNorm(args.dim, args.norm_eps),
            RMSNorm(args.dim, args.norm_eps),
        )
        for name in ("attn", "ffn"):
            for suffix, value in (
                ("fn", torch.randn(24, 4 * args.dim) * 0.01),
                ("scale", torch.randn(3) * 0.1),
                ("base", torch.randn(24) * 0.1),
            ):
                setattr(block, f"hc_{name}_{suffix}", nn.Parameter(value))
        trunk = torch.randn(4, 5, 4, args.dim).bfloat16()
        previous_pre = torch.rand(4, 5, 4)
        ids = torch.zeros(4, 5, dtype=torch.long)
    block.attn_norm.weight.fill_(1)
    block.ffn_norm.weight.fill_(1)
    output, next_pre = torch.empty_like(trunk), torch.empty_like(previous_pre)

    def call():
        values = block(
            trunk, previous_pre, ids, lambda value: (value.float() * 0.25).bfloat16()
        )
        output.copy_(values[0])
        next_pre.copy_(values[1])

    graph, results = torch.cuda.CUDAGraph(), []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            call()
            expected = output.clone(), next_pre.clone()
            block.ffn_transition = mega_ffn_transition
            for _ in range(3):
                call()
            eager = output.clone(), next_pre.clone()
            for name, actual, reference in zip(
                ("full_residual", "shifted_ffn_pre"), eager, expected, strict=True
            ):
                delta = actual.float() - reference.float()
                normalized = delta.square().sum() / (
                    actual.float().square().sum() + reference.float().square().sum()
                ).clamp_min(1e-20)
                if normalized >= 5e-5:
                    raise AssertionError(
                        f"Block mHC {name} normalized error={normalized.item()}"
                    )
                results.append(
                    dict(
                        output=name,
                        max_abs=delta.abs().max().item(),
                        normalized_squared_error=normalized.item(),
                        limit=5e-5,
                    )
                )
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for _ in range(3):
            graph.replay()
        stream.synchronize()
        if not torch.equal(output, eager[0]) or not torch.equal(next_pre, eager[1]):
            raise AssertionError("Block mHC graph differs from identical eager inputs")
        return dict(
            shape=list(trunk.shape),
            reference="unfused_checkpoint_equations",
            outputs=results,
            graph_bitwise_equal=True,
            replays=3,
        )
    finally:
        graph.reset()


@torch.inference_mode()
def run_probe(group):
    """Join an already initialized EP2 process group, leaving it alive on return."""
    from resin_models_cuda.deepseek_v41.experts import prepare_mega_moe
    from resin_models_cuda.deepseek_v41.linear import SharedExpert
    from resin_models_cuda.deepseek_v41.model import Gate, ModelArgs, MoEWeights

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world != 2:
        raise ValueError("the prepared mega-MoE probe requires EP2")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1731 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, hidden, intermediate, experts, topk = 8, 5120, 2304, 16, 6
    args = ModelArgs(
        dim=hidden,
        moe_inter_dim=intermediate,
        n_routed_experts=experts,
        n_activated_experts=topk,
        expert_dtype="fp4",
        swiglu_limit=10,
    )
    with use_default_dtype(torch.bfloat16), torch.device(device):
        source: Any = MoEWeights(
            0, args, expert_parallel=TensorParallelLayout(rank=rank, size=world)
        )
        gate = Gate(args)
        source.gate, source.shared_experts = gate, SharedExpert(args)
    gate.weight.normal_(std=hidden**-0.5)
    gate.bias.zero_()
    gate.benchmark_replica = rank
    gate.benchmark_expert_parallel_size = world
    first, first_sf, ref_first = quantize_fp4_raw(
        torch.randn(
            experts // world,
            2 * intermediate,
            hidden,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    second, second_sf, ref_second = quantize_fp4_raw(
        torch.randn(
            experts // world, hidden, intermediate, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    # Checkpoint container uses [up|gate]; the prepared adapter must invert it.
    source._fused_w13_weight.view(torch.int8).copy_(
        torch.cat((first[:, intermediate:], first[:, :intermediate]), 1)
    )
    source._fused_w13_scale.copy_(
        torch.cat((first_sf[:, intermediate:], first_sf[:, :intermediate]), 1).to(
            torch.float8_e8m0fnu
        )
    )
    source._fused_w2_weight.view(torch.int8).copy_(second)
    source._fused_w2_scale.copy_(second_sf.to(torch.float8_e8m0fnu))
    shared_first, sf_first, ref_shared_first = _block_fp8((2 * intermediate, hidden))
    shared_second, sf_second, ref_shared_second = _block_fp8((hidden, intermediate))
    for linear, weight, scale in (
        (
            source.shared_experts.w1,
            shared_first[:intermediate],
            sf_first[: intermediate // 32],
        ),
        (
            source.shared_experts.w3,
            shared_first[intermediate:],
            sf_first[intermediate // 32 :],
        ),
        (source.shared_experts.w2, shared_second, sf_second),
    ):
        linear.weight.copy_(weight)
        linear.scale.copy_(scale)
    raw_weight = weakref.ref(source._fused_w13_weight)
    model = nn.Sequential(source)
    del source
    collective = MegaMoeCollective(RankGroup(tuple(range(world)), group, rank), 16)
    prepare_model(
        model,
        lambda module: (
            prepare_mega_moe(module, collective)
            if isinstance(module, MoEWeights)
            else module
        ),
    )
    gc.collect()
    if raw_weight() is not None:
        raise AssertionError("prepared traversal retained consumed checkpoint W13")
    layer = cast(MegaMoe, model[0])
    inputs = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    original = inputs.clone()
    ids = torch.zeros(rows, dtype=torch.long, device=device)
    valid = torch.tensor(rows, dtype=torch.int32, device=device)
    output = torch.empty_like(inputs)

    def call():
        with collective.fixed_capacity(rows, valid_rows=valid):
            output.copy_(layer(inputs, ids))

    def reference(count):
        _, _, restored = quantize_fp8(inputs)
        weights, indices = gate(inputs, ids)
        indices[count:] = -1
        weights[count:] = 0
        gathered = []
        for value in (restored, indices, weights):
            parts = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(parts, value, group=group)
            gathered.append(torch.cat(parts))
        all_inputs, all_indices, all_weights = gathered
        result = torch.zeros(world * rows, hidden, device=device)
        for expert in range(experts // world):
            token, slot = (all_indices == rank * (experts // world) + expert).nonzero(
                as_tuple=True
            )
            if token.numel():
                result.index_add_(
                    0,
                    token,
                    reference_ffn(
                        all_inputs[token],
                        ref_first[expert],
                        ref_second[expert],
                        all_weights[token, slot],
                    ),
                )
        dist.all_reduce(result, group=group)
        result = result[rank * rows : rank * rows + count]
        if count:
            result += reference_ffn(
                restored[:count], ref_shared_first, ref_shared_second
            )
        return result.bfloat16()

    results, graph = [], None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for name in ("regular", "changed_routing", "idle_rank", "balanced"):
            count = 0 if name == "idle_rank" and rank == 1 else rows
            with torch.cuda.stream(stream):
                valid.fill_(count)
                inputs.copy_(original if name == "regular" else original.roll(17, -1))
                gate.bias.copy_(
                    torch.linspace(-0.5, 0.5, experts, device=device)
                    * (name != "regular")
                )
                expected = reference(count)
                call()
                eager = output.clone()
                # The balanced benchmark switch is a capture-time choice.
                if name == "balanced":
                    gate.benchmark_balanced_experts = True
                    expected = reference(count)
                    call()
                    eager = output.clone()
                    graph.reset()
                    with torch.cuda.graph(graph, stream=stream):
                        call()
                for _ in range(3):
                    graph.replay()
                stream.synchronize()
                if not torch.equal(output, eager):
                    raise AssertionError(
                        f"adapter {name} graph differs from identical eager inputs"
                    )
                if count:
                    delta = output[:count].float() - expected.float()
                    error = delta.norm() / expected.float().norm().clamp_min(1e-20)
                    if error >= 0.02:
                        raise AssertionError(
                            f"adapter {name} quantized oracle relative L2={error.item()}"
                        )
                    comparison = {
                        "relative_l2": error.item(),
                        "limit": 0.02,
                        "max_abs": delta.abs().max().item(),
                    }
                else:
                    comparison = {"empty_output": [0, hidden]}
                results.append(
                    {
                        "case": name,
                        "tokens": count,
                        "reference": comparison,
                        "graph_bitwise_equal": True,
                        "replays": 3,
                    }
                )
        # Exercise actual empty-source participation as well as masked graph rows.
        if rank == 1:
            layer.forward_empty()
        else:
            layer(inputs, ids)
        torch.cuda.synchronize()
        return {
            "rank": rank,
            "cases": results,
            "source_storage_released": True,
            "workspace_count": len(collective.buffers),
            "empty_collective": True,
            "block_hc": _block_hc(),
        }
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        collective.close()
        collective.close()
