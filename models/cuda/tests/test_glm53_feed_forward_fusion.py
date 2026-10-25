"""Compare fused FFN arithmetic to the original independent Torch operations.

Exact comparisons are intentional: routing near ties and BF16 midpoints must
not acquire a new approximation merely because several launches were fused.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash.feed_forward_fusion import clamped_swiglu, route
from resin_models_cuda.glm5_3_flash.model import DenseMLP, Router

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def legacy_swiglu(gate, up, limit):
    g = gate.clamp(max=limit).float()
    u = up.clamp(-limit, limit).float()
    return (F.silu(g) * u).to(gate.dtype)


def legacy_route(logits, bias, k=8, normalize=True, scaling=2.5):
    scores = logits.sigmoid()
    indices = torch.topk(scores + bias, k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if normalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return indices.int(), weights * scaling


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def capture(operation):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    return graph, output


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("top_k", [2, 8])
def test_cpu_router_bias_selects_but_never_weights(normalize, top_k):
    logits = torch.linspace(-4, 4, 16).repeat(3, 1)
    bias = torch.arange(16, 0, -1).float()
    actual = route(logits, bias, top_k, normalize, 2.5)
    expected = legacy_route(logits, bias, top_k, normalize, 2.5)
    exact(actual, expected)
    assert set(actual[0][0].tolist()) == set(range(top_k))
    selected = logits.sigmoid().gather(1, actual[0].long())
    if normalize:
        selected /= selected.sum(-1, keepdim=True) + 1e-20
    exact(actual[1], selected * 2.5)


def test_cpu_dense_rounding_and_fixed_projection_metadata():
    torch.manual_seed(16)
    weights = {
        f"{name}_proj.weight": torch.randn(16, 16).bfloat16()
        for name in ("gate", "up", "down")
    }
    hidden = torch.randn(3, 16).bfloat16()
    layer = DenseMLP(weights, 10.1)
    padded = F.pad(hidden, (0, 0, 0, 61))
    g, u = (
        F.linear(padded, weights["gate_proj.weight"]),
        F.linear(padded, weights["up_proj.weight"]),
    )
    expected = F.linear(legacy_swiglu(g, u, 10.1), weights["down_proj.weight"])[:3]
    exact(layer(hidden, projection_rows=64), expected)
    # The old BF16 SiLU intermediate is observably a different operation.
    torch.manual_seed(117)
    g, u = torch.randn(128, 16).bfloat16(), torch.randn(128, 16).bfloat16()
    actual = clamped_swiglu(g, u, 10.1)
    assert torch.any(actual != F.silu(g.clamp(max=10.1)) * u.clamp(-10.1, 10.1))
    exact(actual, legacy_swiglu(g, u, 10.1))


@CUDA
@torch.inference_mode()
@pytest.mark.parametrize("limit", [10.0, 10.1])
def test_cuda_swiglu_all_bf16_gate_values_match_original(limit):
    # Exhaust every BF16 gate bit pattern against several ordinary, saturated,
    # signed-zero and tiny up values. NaN payloads are not a routing contract.
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    gate = values.repeat(8).reshape(-1, 256)
    up_values = torch.tensor(
        [0.0, -0.0, 0.1, -0.75, 1.0, 9.9375, -10.25, 1e-30],
        device="cuda",
        dtype=torch.bfloat16,
    )
    up = up_values[:, None].expand(8, 65536).reshape_as(gate).clone()
    exact(clamped_swiglu(gate, up, limit), legacy_swiglu(gate, up, limit))


@CUDA
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_swiglu_strides_graph_and_unseen_rows(dtype):
    from resin_models_cuda.glm5_3_flash.feed_forward_fusion_cuda import _swiglu

    torch.manual_seed(711)
    width = 2048

    def inputs(rows):
        gate = (torch.randn(rows, width * 2 + 3, device="cuda") * 12).to(dtype)[
            :, 1 : 2 * width + 1 : 2
        ]
        up = (torch.randn(rows, width * 3 + 7, device="cuda") * 12).to(dtype)[
            :, 2 : 3 * width + 2 : 3
        ]
        return gate, up

    gate, up = inputs(7)
    graph, output = capture(lambda: clamped_swiglu(gate, up, 10))
    for _ in range(3):
        gate.normal_(std=11)
        up.normal_(std=13)
        graph.replay()
        exact(output, legacy_swiglu(gate, up, 10))
    cache = _swiglu.device_caches[torch.cuda.current_device()][0]
    before = set(cache)
    for rows in (0, 1, 2, 3, 4, 8, 31, 32, 64, 129, 513, 2049):
        gate, up = inputs(rows)
        exact(clamped_swiglu(gate, up, 10), legacy_swiglu(gate, up, 10))
    assert set(cache) == before, "SwiGLU token count created a new compiled variant"


@CUDA
@torch.inference_mode()
@pytest.mark.parametrize("normalize", [False, True])
def test_cuda_route_exact_ties_strides_graph_and_unseen_rows(normalize):
    from resin_models_cuda.glm5_3_flash.feed_forward_fusion_cuda import (
        _route_scores,
        _route_weights,
        route_scores,
    )

    torch.manual_seed(742)
    experts = 288
    bias = torch.randn(experts * 2, device="cuda")[::2]

    def inputs(rows):
        return torch.randn(rows, experts * 2 + 3, device="cuda")[
            :, 1 : experts * 2 + 1 : 2
        ]

    logits = inputs(7)
    graph, output = capture(lambda: route(logits, bias, 8, normalize, 2.5))
    for kind in ("ordinary", "ties", "near-ties", "saturated"):
        if kind == "ordinary":
            logits.normal_()
            bias.normal_()
        elif kind == "ties":
            logits.zero_()
            bias.zero_()
        elif kind == "near-ties":
            logits.copy_(torch.arange(experts, device="cuda").float() * 1e-7)
            bias.fill_(0.25)
        else:
            logits.copy_(torch.linspace(-100, 100, experts, device="cuda"))
            bias.copy_(-logits[0].sigmoid())
        graph.replay()
        exact(output, legacy_route(logits, bias, 8, normalize, 2.5))
        scores, corrected = route_scores(logits, bias)
        exact(scores, logits.sigmoid())
        exact(corrected, logits.sigmoid() + bias)
    kernels = (_route_scores, _route_weights)
    caches = [k.device_caches[torch.cuda.current_device()][0] for k in kernels]
    before = [set(c) for c in caches]
    for rows in (0, 1, 2, 3, 4, 8, 16, 32, 64, 129, 513, 2049):
        logits = inputs(rows)
        exact(
            route(logits, bias, 8, normalize, 2.5),
            legacy_route(logits, bias, 8, normalize, 2.5),
        )
    assert [set(c) for c in caches] == before, (
        "Router token count created a new compiled variant"
    )


@CUDA
@torch.inference_mode()
def test_cuda_route_normalization_matches_aten_sum_order():
    from resin_models_cuda.glm5_3_flash.feed_forward_fusion_cuda import route_weights

    torch.manual_seed(23)
    scores = torch.rand(73, 288, device="cuda").pow(12)
    scores[0].zero_()
    scores[1] = torch.logspace(-38, 0, 288, device="cuda")
    ids = torch.randint(288, (73, 8), device="cuda")
    for normalized in (False, True):
        selected = scores.gather(1, ids)
        if normalized:
            selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
        actual_ids, actual = route_weights(scores, ids, normalized, 2.5)
        exact(actual_ids, ids.int())
        exact(actual, selected * 2.5)


@CUDA
@torch.inference_mode()
def test_cuda_real_router_fixed_rows_preserves_original_projection_and_topk():
    torch.manual_seed(728)
    config = replace(Glm53Config(), hidden_size=64)
    weight = torch.randn(288, 64, device="cuda")
    bias = torch.randn(288, device="cuda")
    router = Router(config, weight, bias)
    hidden = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    graph, output = capture(lambda: router(hidden, projection_rows=64))
    for _ in range(3):
        hidden.normal_()
        router.bias.normal_()
        graph.replay()
        padded = F.pad(hidden, (0, 0, 0, 48))
        logits = F.linear(padded.float(), weight)
        expected = legacy_route(logits, router.bias)
        exact(output, tuple(v[:16] for v in expected))
