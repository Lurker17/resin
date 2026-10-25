"""Independent normalization equations, graph replay, and bounded JIT keys."""

from __future__ import annotations

import pytest
import torch

from resin_models_cuda.glm5_3_flash.normalization import gated_rms_norm, rms_norm


def _reference(value, weight, gate=None):
    floating = value.float()
    normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + 1e-5)
    if gate is None:
        return (normalized * weight.float()).to(value.dtype)
    return (normalized * weight.float() * torch.sigmoid(gate.float())).to(value.dtype)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("width", [128, 512, 1536])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_normalization_equations_graph_and_runtime_length(gated, width):
    from resin_models_cuda.glm5_3_flash.normalization_cuda import _normalize

    torch.manual_seed(137)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)

    def run(value, gate):
        if gated:
            return gated_rms_norm(value, weight, gate, 1e-5)
        return rms_norm(value, weight, 1e-5)

    value = torch.randn(7, width, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(value) * 5
    run(value, gate)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(value, gate)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(value, gate)
    value.normal_()
    gate.normal_(std=3)
    graph.replay()
    expected = _reference(value, weight, gate if gated else None)
    torch.testing.assert_close(output, expected, rtol=0.008, atol=0.004)
    cache = _normalize.device_caches[torch.cuda.current_device()][0]
    before = set(cache)
    for rows in (1, 8, 65, 513, 1025, 2049):
        value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(value) * 5
        actual = run(value, gate)
        expected = _reference(value, weight, gate if gated else None)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.004)
    assert set(cache) == before, "unseen token count created a JIT specialization"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_sparse_norm_does_not_round_before_weight_multiply(device):
    value = torch.tensor(
        [[0.5, 0.75, 1.0, 1.25] * 32], dtype=torch.bfloat16, device=device
    )
    weight = torch.tensor(
        [1.25, 0.75, 1.5, 0.625] * 32, dtype=torch.bfloat16, device=device
    )
    floating = value.float()
    normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + 1e-5)
    old_two_casts = normalized.to(value.dtype) * weight
    expected = (normalized * weight.float()).to(value.dtype)
    assert not torch.equal(old_two_casts, expected)
    torch.testing.assert_close(rms_norm(value, weight, 1e-5), expected, rtol=0, atol=0)
