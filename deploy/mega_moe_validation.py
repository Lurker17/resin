"""Actual V4.1 prepared-FFN adapter against decoded Torch FFNs on an EP2 mesh."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from deploy.deepinfer_kernel_probe import (
    _compare,
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


def _routing_comparison(gate, inputs, expected_indices, routes, count):
    """Separate near-tie selection changes from incorrect routing or FFN math."""
    weights, indices = routes
    if not indices[count:].eq(-1).all() or not weights[count:].eq(0).all():
        raise AssertionError("fused gate emitted routes for masked rows")
    result = dict(live_tokens=count, masked_tokens=len(inputs) - count)
    if not count:
        return result
    selected, expected = indices[:count], expected_indices[:count].long()
    if not ((selected >= 0) & (selected < len(gate.weight))).all():
        raise AssertionError("fused gate emitted an invalid expert ID")
    sorted_ids = selected.sort(-1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise AssertionError("fused gate emitted duplicate expert IDs")
    scores = F.softplus(inputs[:count].float() @ gate.weight.float().T).sqrt()
    ranking = scores + gate.bias
    selected_scores = scores.gather(1, selected)
    expected_weights = (
        selected_scores
        / (selected_scores.sum(-1, keepdim=True) + 1e-20)
        * gate.route_scale
    )
    # Selected weights are intentionally rounded to BF16 at the model boundary.
    result["weights_on_actual_ids"] = _compare(
        weights[:count], expected_weights, atol=2e-5, rtol=4e-3
    )
    result["rounded_weight_max_abs"] = (
        (weights[:count] - expected_weights.bfloat16().float()).abs().max().item()
    )
    mismatched = (sorted_ids != expected.sort(-1).values).any(-1)
    result.update(
        token_set_mismatches=int(mismatched.sum()),
        ordered_slot_mismatches=int((selected != expected).sum()),
        balanced=gate.benchmark_balanced_experts,
    )
    if gate.benchmark_balanced_experts:
        if not torch.equal(selected, expected):
            raise AssertionError("fused gate changed exact balanced benchmark routes")
        return result
    top = ranking.topk(gate.topk + 1, dim=-1).values
    gap = top[:, -2] - top[:, -1]
    crossing = (
        ranking.scatter(1, selected, -torch.inf).max(-1).values
        - ranking.gather(1, selected).min(-1).values
    )
    result.update(
        max_selection_crossing=crossing.max().item(),
        selection_crossing_limit=2e-5,
        min_reference_boundary_gap=gap.min().item(),
        mismatch_examples=[
            dict(
                token=token,
                reference_ids=expected[token].tolist(),
                actual_ids=selected[token].tolist(),
                reference_boundary_gap=gap[token].item(),
                selection_crossing=crossing[token].item(),
            )
            for token in mismatched.nonzero().flatten()[:8].tolist()
        ],
    )
    if (crossing > 2e-5).any():
        raise AssertionError(f"fused gate selected outside numerical ties: {result}")
    return result


def _output_comparison(actual, expected, *, limit, max_token_limit=None):
    if not len(expected):
        return {"empty_output": list(expected.shape)}
    delta = actual[: len(expected)].float() - expected.float()
    error = delta.norm() / expected.float().norm().clamp_min(1e-20)
    token_error = delta.norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-20)
    result = dict(
        relative_l2=error.item(),
        limit=limit,
        max_abs=delta.abs().max().item(),
        max_token_relative_l2=token_error.max().item(),
    )
    if max_token_limit is not None:
        result["max_token_relative_l2_limit"] = max_token_limit
    if error >= limit or (
        max_token_limit is not None and token_error.max() >= max_token_limit
    ):
        raise AssertionError(f"adapter quantized FFN reference failed: {result}")
    return result


@torch.inference_mode()
def _block_hc(layer):
    """Both lazy HC transitions, direct FP8 writes and the actual EP FFN."""
    from deploy.deepinfer_kernel_probe import _mhc_reference
    from resin.backend.cuda.mega_moe import shared_scale_rows
    from resin_models_cuda.deepseek_v41.hc import (
        HCLazyPost,
        materialize,
        mega_transition,
    )
    from resin_models_cuda.deepseek_v41.kernels import act_quant

    dim, tokens = layer.dim, min(1280, layer.collective.max_tokens)
    shape = (2, tokens // 2) if tokens % 2 == 0 else (1, tokens)
    block = SimpleNamespace(
        args=SimpleNamespace(
            dim=dim, norm_eps=1e-20, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20
        ),
        ffn=layer,
    )
    with torch.device("cuda"):
        for name in ("attn", "ffn"):
            setattr(block, f"hc_{name}_fn", torch.randn(24, 4 * dim) * 0.01)
            setattr(block, f"hc_{name}_scale", torch.randn(3) * 0.1)
            setattr(block, f"hc_{name}_base", torch.randn(24) * 0.1)
            setattr(
                block,
                f"{name}_norm",
                SimpleNamespace(weight=torch.ones(dim).bfloat16()),
            )
        source = torch.randn(*shape, dim).bfloat16()
        initial = HCLazyPost(
            torch.randn(*shape, 4, dim).bfloat16(),
            torch.rand(*shape, 4),
            torch.rand(*shape, 4),
            torch.rand(*shape, 4, 4),
        )
        ids = torch.zeros(*shape, dtype=torch.long)
        valid = torch.tensor(tokens, dtype=torch.int32)
        output = torch.empty(*shape, 4, dim, dtype=torch.bfloat16)
        norm = torch.empty_like(source)
        pre = torch.empty_like(initial.pre)

    def call():
        with layer.collective.fixed_capacity(tokens, valid_rows=valid):
            attention, state = mega_transition(block, source, initial, "attn")
            attention = (attention.float() * 0.25).bfloat16()
            normalized, state = mega_transition(
                block, attention, state, "ffn", prepare_ffn=True
            )
            output.copy_(materialize(layer.forward_prepared(normalized, ids), state))
            norm.copy_(normalized)
            pre.copy_(state.pre)

    def reference_transition(value, state, name):
        reference = _mhc_reference(
            dict(
                x=value.reshape(-1, dim),
                residual=state.residual.reshape(-1, 4, dim),
                shifted_prev_mix=state.pre.reshape(-1, 4, 1),
                post_mix=state.post.reshape(-1, 4, 1),
                comb_res_mix=state.comb.reshape(-1, 4, 4),
                fn=getattr(block, f"hc_{name}_fn"),
                mix_scales=getattr(block, f"hc_{name}_scale"),
                mix_bases=getattr(block, f"hc_{name}_base"),
                rmsnorm_weight=getattr(block, f"{name}_norm").weight,
                hc_norm_eps=1e-20,
                hc_pre_eps=1e-6,
                hc_post_scale=2,
                sinkhorn_eps=1e-6,
                num_sinkhorn_iters=20,
                rmsnorm_eps=1e-20,
                rmsnorm_scale=1,
            )
        )
        return reference["y_bf16"].reshape(*shape, dim), HCLazyPost(
            reference["new_residual"].reshape(*shape, 4, dim),
            reference["new_prev_mix"].reshape(*shape, 4),
            reference["new_post_mix"].reshape(*shape, 4),
            reference["new_comb_res_mix"].reshape(*shape, 4, 4),
        )

    graph, results = torch.cuda.CUDAGraph(), []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            attention, state = reference_transition(source, initial, "attn")
            reference_norm, reference_state = reference_transition(
                (attention.float() * 0.25).bfloat16(), state, "ffn"
            )
            for _ in range(3):
                call()
            eager = output.clone(), norm.clone(), pre.clone()
            for name, actual, expected in (
                ("shifted_ffn_pre", pre, reference_state.pre),
                ("ffn_normalization", norm, reference_norm),
            ):
                delta = actual.float() - expected.float()
                error = delta.square().sum() / (
                    actual.float().square().sum() + expected.float().square().sum()
                ).clamp_min(1e-20)
                if error >= 5e-5:
                    raise AssertionError(
                        f"lazy HC {name} normalized error={error.item()}"
                    )
                results.append(
                    dict(output=name, normalized_squared_error=error.item(), limit=5e-5)
                )
            values, scales = act_quant(
                norm.reshape(-1, dim), 32, "ue8m0", torch.float8_e8m0fnu
            )
            packed = scales.contiguous().view(torch.int32)
            if not torch.equal(
                values.view(torch.uint8), layer.workspace.x[:tokens].view(torch.uint8)
            ):
                raise AssertionError(
                    "fused HC FP8 differs from quantizing its BF16 normalization"
                )
            if not torch.equal(packed, layer.workspace.x_sf[:tokens]):
                raise AssertionError("fused HC routed scales differ")
            shared_rows = shared_scale_rows(
                tokens, layer._shared_block_m(tokens), device=norm.device
            )
            if not torch.equal(packed, layer.workspace.shared_l1_acts_sf[shared_rows]):
                raise AssertionError("fused HC shared scales differ")
            # Ordinary FFN preparation must consume exactly the same FP8 values.
            with layer.collective.fixed_capacity(tokens):
                ordinary = layer(norm, ids)
            attention, actual_state = mega_transition(block, source, initial, "attn")
            _, actual_state = mega_transition(
                block, (attention.float() * 0.25).bfloat16(), actual_state, "ffn"
            )
            if not torch.equal(materialize(ordinary, actual_state), eager[0]):
                raise AssertionError(
                    "prepared FFN changed the complete residual output"
                )
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            call()
        captures = []
        for name, count in (
            ("regular", tokens),
            ("changed_input", tokens),
            ("partial_rows", max(1, tokens // 2)),
            ("idle_rank", 0 if layer.collective.group.local_rank == 1 else tokens),
        ):
            with torch.cuda.stream(stream):
                if name == "changed_input":
                    source.copy_(source.roll(17, -1))
                    initial.pre.copy_(initial.pre.roll(1, -1))
                valid.fill_(count)
                call()
                eager = output.clone(), norm.clone(), pre.clone()
                for _ in range(3):
                    graph.replay()
            stream.synchronize()
            if any(
                not torch.equal(value, expected)
                for value, expected in zip((output, norm, pre), eager, strict=True)
            ):
                raise AssertionError(f"lazy HC {name} graph differs from eager inputs")
            if not layer.workspace.topk_idx[count:tokens].eq(-1).all():
                raise AssertionError("fused normalization routed masked graph rows")
            captures.append(dict(case=name, valid_tokens=count, replays=3))
        return dict(
            shape=list(initial.residual.shape),
            reference="unfused_checkpoint_equations",
            outputs=results,
            direct_fp8_and_scales_exact=True,
            prepared_ffn_bitwise_equal=True,
            graph_bitwise_equal=True,
            capture_cases=captures,
        )
    finally:
        graph.reset()


@torch.inference_mode()
def run_probe(group, *, rows: int = 8, max_tokens: int | None = None):
    """Join an already initialized EP2 process group, leaving it alive on return."""
    from resin_models_cuda.deepseek_v41.experts import prepare_mega_moe
    from resin_models_cuda.deepseek_v41.linear import (
        Linear,
        SharedExpert,
        prepare_linear,
    )
    from resin_models_cuda.deepseek_v41.model import Gate, ModelArgs, MoEWeights

    capacity = max(16, rows) if max_tokens is None else max_tokens
    if rows < 1 or capacity < rows:
        raise ValueError("probe rows must be positive and fit max_tokens")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world != 2:
        raise ValueError("the prepared mega-MoE probe requires EP2")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1731 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    hidden, intermediate, experts, topk = 5120, 2304, 16, 6
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
    collective = MegaMoeCollective(
        RankGroup(tuple(range(world)), group, rank), capacity
    )
    prepared_projections = []

    def lower(module):
        if isinstance(module, Linear):
            prepared_projections.append(module)
            return prepare_linear(module)
        return (
            prepare_mega_moe(module, collective)
            if isinstance(module, MoEWeights)
            else module
        )

    prepare_model(model, lower)
    if len(prepared_projections) != 3:
        raise AssertionError(
            "shared projections were not prepared before their parent FFN"
        )
    prepared_projections.clear()
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

    def reference(count, routes=None):
        _, _, restored = quantize_fp8(inputs)
        weights, indices = gate(inputs, ids) if routes is None else routes
        weights, indices = weights.clone(), indices.clone()
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
        for name in (
            "regular",
            "changed_routing",
            "partial_rows",
            "idle_rank",
            "balanced",
        ):
            count = 0 if name == "idle_rank" and rank == 1 else rows
            if name == "partial_rows":
                count = max(1, rows - 3) if rank == 0 else max(1, rows // 2)
            with torch.cuda.stream(stream):
                valid.fill_(count)
                inputs.copy_(original if name == "regular" else original.roll(17, -1))
                gate.bias.copy_(
                    torch.linspace(-0.5, 0.5, experts, device=device)
                    * (name != "regular")
                )
                if name == "balanced":
                    gate.benchmark_balanced_experts = True
                original_routes = gate(inputs, ids)
                expected = reference(count, original_routes)
                call()
                eager = output.clone()
                # The balanced benchmark switch is a capture-time choice.
                if name == "balanced":
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
                actual_routes = (
                    layer.workspace.topk_weights[:rows].clone(),
                    layer.workspace.topk_idx[:rows].clone(),
                )
                routing = _routing_comparison(
                    gate, inputs, original_routes[1], actual_routes, count
                )
                actual_expected = reference(count, actual_routes)
                results.append(
                    {
                        "case": name,
                        "tokens": count,
                        "reference": _output_comparison(output, expected, limit=0.02),
                        "actual_route_reference": _output_comparison(
                            output, actual_expected, limit=0.002, max_token_limit=0.02
                        ),
                        "router": routing,
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
            "tokens_per_rank": rows,
            "max_tokens": capacity,
            "shared_block_m": layer._shared_block_m(rows),
            "cases": results,
            "source_storage_released": True,
            "prepared_shared_projections": 3,
            "workspace_count": len(collective.buffers),
            "empty_collective": True,
            "block_hc": _block_hc(layer),
        }
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        collective.close()
        collective.close()
