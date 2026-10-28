"""Real vision attention must remain eligible for fused CUDA SDPA.

The reference expands attention explicitly in FP32 instead of calling SDPA.
Projection, normalization and rotary rounding retain each checkpoint's contract.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from resin_models_cuda.deepseek_v41.vision import Attention
from resin_models_cuda.glm5_3_flash.vision import GlmVisionConfig, VisionAttention


@pytest.fixture(autouse=True)
def fp32_reference_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def inputs(family, rows, device, dtype):
    torch.manual_seed(325)
    # Both released towers use sixteen heads of sixty-four channels.
    if family == "deepseek":
        module = Attention(SimpleNamespace(vision_dim=1024, vision_n_heads=16))
    else:
        module = VisionAttention(GlmVisionConfig())
    module = module.to(device=device, dtype=dtype).eval().requires_grad_(False)
    x = torch.randn(rows, 1024, device=device, dtype=dtype) * 0.4
    # Vary the angles across tokens and head channels; no zero-RoPE shortcut.
    angles = (
        torch.arange(rows, device=device, dtype=torch.float32)[:, None, None]
        * torch.linspace(0.001, 0.1, 32, device=device)[None, None]
    )
    if family == "glm":
        angles = torch.cat((angles, angles), -1)
    return module, x, angles.cos(), angles.sin()


def explicit_attention(module, x, cos, sin, family):
    """Independent full, bidirectional FP32 softmax attention, no SDPA call."""
    projection = module.wqkv if family == "deepseek" else module.qkv
    q, k, v = F.linear(x, projection.weight, projection.bias).reshape(
        x.shape[0], 3, 16, 64
    ).unbind(1)
    if family == "glm":
        def norm(value, layer):
            result = value.float() * torch.rsqrt(
                value.float().square().mean(-1, keepdim=True) + layer.eps
            )
            return layer.weight * result.to(value.dtype)

        q, k = norm(q, module.q_norm), norm(k, module.k_norm)

    def rotate(value):
        left, right = value.float().chunk(2, -1)
        if family == "deepseek":
            rotated = torch.cat(
                (left * cos - right * sin, right * cos + left * sin), -1
            )
        else:
            rotated = value.float() * cos + torch.cat((-right, left), -1) * sin
        return rotated.to(value.dtype)

    q, k = rotate(q).transpose(0, 1).float(), rotate(k).transpose(0, 1).float()
    values = v.transpose(0, 1).float()
    scores = (q @ k.transpose(-2, -1)) * (64**-0.5)
    context = (scores.softmax(-1) @ values).to(x.dtype)
    flattened = context.transpose(0, 1).reshape(x.shape[0], -1)
    output = module.wo if family == "deepseek" else module.proj
    return F.linear(flattened, output.weight, output.bias)


@pytest.mark.parametrize("family", ["deepseek", "glm"])
@pytest.mark.parametrize("rows", [19, 64])
@torch.inference_mode()
def test_vision_attention_cpu_matches_explicit_fp32_oracle(family, rows):
    module, x, cos, sin = inputs(family, rows, "cpu", torch.float32)
    expected = explicit_attention(module, x, cos, sin, family)
    torch.testing.assert_close(module(x, cos, sin), expected, atol=1e-6, rtol=2e-5)


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="requires CUDA fused BF16 SDPA",
)


@CUDA
@pytest.mark.parametrize("family", ["deepseek", "glm"])
@torch.inference_mode()
def test_vision_attention_production_selects_flash_for_unseen_lengths(family):
    # No test-side backend override: the production path must choose Flash
    # even when all SDPA alternatives are enabled by the caller.
    for rows in (73, 129):
        module, x, cos, sin = inputs(family, rows, "cuda", torch.bfloat16)
        with sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        ):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profiler:
                module(x, cos, sin)
                torch.cuda.synchronize()
        operators = {event.key for event in profiler.key_averages()}
        assert "aten::_scaled_dot_product_flash_attention" in operators
        assert "aten::_scaled_dot_product_cudnn_attention" not in operators
        assert "aten::_scaled_dot_product_attention_math" not in operators


@CUDA
@pytest.mark.parametrize("family", ["deepseek", "glm"])
@pytest.mark.parametrize("rows", [64, 255, 1521])
@pytest.mark.parametrize("backend", ["default", "flash_only"])
@torch.inference_mode()
def test_vision_attention_fused_backend_and_fp32_oracle(family, rows, backend):
    module, x, cos, sin = inputs(family, rows, "cuda", torch.bfloat16)
    expected = explicit_attention(module, x, cos, sin, family)
    selection = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if backend == "flash_only"
        else nullcontext()
    )
    # Disabling every fallback makes the former three-dimensional layout fail.
    # The default path is checked independently against the same FP32 oracle.
    with selection:
        actual = module(x, cos, sin)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@CUDA
@pytest.mark.parametrize("family", ["deepseek", "glm"])
@pytest.mark.parametrize("backend", ["default", "flash_only"])
@torch.inference_mode()
def test_vision_attention_graph_observes_changed_inputs(family, backend):
    module, x, cos, sin = inputs(family, 255, "cuda", torch.bfloat16)
    selection = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if backend == "flash_only"
        else nullcontext()
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with selection:
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(x, cos, sin)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(x, cos, sin)
        graph.replay()
        first = output.clone()
        # Keep addresses/geometry fixed, but alter both content and rotary data.
        x.mul_(0.7).add_(0.13)
        cos.mul_(0.9)
        sin.mul_(0.8)
        expected = explicit_attention(module, x, cos, sin, family)
        eager = module(x, cos, sin)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-2)
        assert not torch.equal(output, first), "attention graph ignored changed image"
        graph.reset()
