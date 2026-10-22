"""Real SM90 MXFP4 partial execution survives CUDA graph route changes."""

from __future__ import annotations

import pytest
import torch

from resin.moe_kernels import MoEExpertWeights, MoEKernelConfig, MoERouting
from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires one Hopper SM90 GPU and the pinned FlashInfer build",
)


@pytest.mark.parametrize("num_tokens", [3, 2051])
def test_partial_cuda_graph_replays_active_and_idle_routes(num_tokens):
    device = torch.device("cuda")
    config = MoEKernelConfig(
        architecture="sm90",
        quantization="mxfp4",
        activation="swiglu",
        top_k=4,
        num_experts=4,
        hidden_size=128,
        intermediate_size=128,
        group_size=32,
        expert_bias=True,
        activation_alpha=1.702,
        activation_beta=1.0,
        activation_limit=7.0,
    )
    generator = torch.Generator(device=device).manual_seed(1729)

    def packed(shape):
        return torch.randint(
            0, 256, shape, device=device, dtype=torch.uint8, generator=generator
        )

    def bias(shape):
        return (
            torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            / 100
        )

    kernel = FlashInferSM90MXFP4Kernel()
    prepared = kernel.prepare_weights(
        MoEExpertWeights(
            gate_up=packed((4, 256, 64)),
            down=packed((4, 128, 64)),
            gate_up_scales=torch.full(
                (4, 256, 4), 120, device=device, dtype=torch.uint8
            ),
            down_scales=torch.full((4, 128, 4), 120, device=device, dtype=torch.uint8),
            gate_up_bias=bias((4, 256)),
            down_bias=bias((4, 128)),
        ),
        config,
    )
    hidden = torch.randn(
        (num_tokens, 128), device=device, dtype=torch.bfloat16, generator=generator
    )
    indices = torch.arange(4, device=device, dtype=torch.int32).repeat(num_tokens, 1)
    probabilities = torch.randn(
        num_tokens, 4, device=device, generator=generator
    ).softmax(-1)
    routing = MoERouting(
        torch.zeros(num_tokens, 4, device=device), indices, probabilities
    )
    output = torch.empty_like(hidden, dtype=torch.float32)

    def launch():
        output.copy_(kernel.forward_partial(hidden, routing, prepared, config))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()

    original_indices = indices.clone()
    original_probabilities = probabilities.clone()
    for phase in range(3):
        if phase == 1:
            # Fixed-capacity EP owners may receive no active assignments.
            indices.fill_(config.num_experts)
            probabilities.zero_()
        else:
            indices.copy_(original_indices.roll(1, 1) if phase else original_indices)
            probabilities.copy_(original_probabilities)
        reference = kernel.forward_partial(hidden, routing, prepared, config)
        graph.replay()
        torch.cuda.synchronize()
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()
        # Four nonzero FP32 contributions expose unordered atomic reductions;
        # 2051 tokens also cross the 8192-assignment kernel chunk boundary.
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        if phase == 1:
            assert not output.any()
