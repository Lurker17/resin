"""Bitwise old-layout oracle for strided/planar KDA convolution."""

from functools import cache
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from resin_models_cuda.glm5_3_flash import kernels


@cache
def _legacy_kernel() -> Any:
    # Exact convolution from qualified hnvkb1uk before the layout-only change.
    # Keep this independent oracle frozen: it reads compact token-major QKV.
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["state_stride", "final_stride"])
    def legacy(
        QKV,
        Weight,
        Initial,
        Final,
        Offsets,
        Sequences,
        Out,
        state_stride,
        final_stride,
        CHANNELS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        channel = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        seq = tl.load(Sequences + token).to(tl.int64)
        start = tl.load(Offsets + seq).to(tl.int64)
        end = tl.load(Offsets + seq + 1).to(tl.int64)
        valid = channel < CHANNELS
        acc = tl.full((BLOCK,), 0, tl.float32)
        for j in tl.static_range(4):
            source = token - 3 + j
            current = tl.load(
                QKV + (source * 3 + group) * CHANNELS + channel,
                valid & (source >= start),
                other=0,
            ).to(tl.float32)
            previous = tl.load(
                Initial
                + seq * state_stride
                + (group * CHANNELS + channel) * 3
                + source
                - start
                + 3,
                valid & (source < start),
                other=0,
            ).to(tl.float32)
            weight = tl.load(
                Weight + (group * CHANNELS + channel) * 4 + j, valid, other=0
            )
            acc += (current + previous) * weight
        value = acc * tl.sigmoid(acc)
        tl.store(Out + (token * 3 + group) * CHANNELS + channel, value, valid)
        if token == end - 1:
            for j in tl.static_range(3):
                source = end - 3 + j
                final_current = tl.load(
                    QKV + (source * 3 + group) * CHANNELS + channel,
                    valid & (source >= start),
                    other=0,
                )
                final_previous = tl.load(
                    Initial
                    + seq * state_stride
                    + (group * CHANNELS + channel) * 3
                    + source
                    - start
                    + 3,
                    valid & (source < start),
                    other=0,
                )
                tl.store(
                    Final + seq * final_stride + (group * CHANNELS + channel) * 3 + j,
                    final_current + final_previous,
                    valid,
                )

    return legacy


def _source(tokens, channels, *, padding, channel_stride, device):
    stride = 3 * channels * channel_stride + padding
    owner = torch.full(
        (tokens, stride), float("nan"), dtype=torch.bfloat16, device=device
    )
    value = owner.as_strided(
        (tokens, 3, channels), (stride, channels * channel_stride, channel_stride)
    )
    value.copy_(torch.randn_like(value) * 0.1)
    return owner, value


def _cache(lengths, channels, *, device):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    # A layer view in the old batch-major state buffer has a larger batch stride.
    owner = torch.randn(
        len(lengths), 2, 3, channels, 3, dtype=torch.bfloat16, device=device
    )
    history = owner[:, 1]
    return SimpleNamespace(
        convolution=history,
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
        sequence_indices=torch.tensor(
            [row for row, length in enumerate(lengths) for _ in range(length)],
            dtype=torch.long,
            device=device,
        ),
    )


def _legacy(qkv, weight, state):
    packed = qkv.contiguous()
    output, final = torch.empty_like(packed), state.convolution.clone()
    tokens, _, channels = packed.shape
    if tokens:
        _legacy_kernel()[(tokens, 3, (channels + 255) // 256)](
            packed,
            weight,
            state.convolution,
            final,
            state.cu_seqlens,
            state.sequence_indices,
            output,
            state.convolution.stride(0),
            final.stride(0),
            channels,
            256,
        )
    return output, final


@pytest.mark.parametrize("lengths", [[0], [1, 0, 1], [4, 0, 7]])
@pytest.mark.parametrize("channel_stride", [1, 2])
def test_cpu_planar_convolution_keeps_default_api_and_empty_history(
    lengths, channel_stride
):
    _, source = _source(
        sum(lengths), 8, padding=9, channel_stride=channel_stride, device="cpu"
    )
    state = _cache(lengths, 8, device="cpu")
    weight = torch.randn(3, 8, 4)
    before = state.convolution.clone()
    expected, expected_final = kernels.convolution(source.contiguous(), weight, state)
    actual, final = kernels.convolution(source, weight, state, planar=True)
    assert actual.shape == (3, sum(lengths), 8)
    assert all(value.is_contiguous() for value in actual)
    torch.testing.assert_close(actual.permute(1, 0, 2), expected, atol=0, rtol=0)
    torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
    torch.testing.assert_close(state.convolution, before, atol=0, rtol=0)
    for row, length in enumerate(lengths):
        if not length:
            torch.testing.assert_close(final[row], before[row], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("channels", [256, 8192])
@torch.inference_mode()
def test_cuda_planar_strides_match_old_kernel_without_row_specialization(channels):
    from resin_models_cuda.glm5_3_flash.kernels_cuda import _conv

    torch.manual_seed(815)
    weight = torch.randn(3, channels, 4, device="cuda")
    _, source = _source(2, channels, padding=320, channel_stride=1, device="cuda")
    state = _cache([1, 0, 1], channels, device="cuda")
    kernels.convolution(source, weight, state, planar=True)
    jit_cache = _conv.device_caches[torch.cuda.current_device()][0]
    warmed = set(jit_cache)
    # Includes K0/K3 row counts, packed-sequence boundaries and new planar
    # strides T*C. One startup invocation, never a context-length warmup grid.
    for lengths, padding, channel_stride in (
        ([0, 0], 320, 1),
        ([1], 0, 1),
        ([4, 0, 4], 320, 1),
        ([3, 0, 4], 17, 2),
        ([63, 0, 2], 320, 1),
        ([1, 64, 0], 1, 2),
        ([510, 0, 3], 320, 1),
        ([1, 1024], 320, 1),
    ):
        _, source = _source(
            sum(lengths),
            channels,
            padding=padding,
            channel_stride=channel_stride,
            device="cuda",
        )
        state = _cache(lengths, channels, device="cuda")
        before = state.convolution.clone()
        expected, expected_final = _legacy(source, weight, state)
        for planar in (False, True):
            actual, final = kernels.convolution(source, weight, state, planar=planar)
            torch.testing.assert_close(
                actual.permute(1, 0, 2) if planar else actual, expected, atol=0, rtol=0
            )
            torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
        torch.testing.assert_close(state.convolution, before, atol=0, rtol=0)
        assert set(jit_cache) == warmed, "source/plane strides specialized the kernel"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cuda_planar_convolution_graph_changes_values_lengths_and_history():
    _, source = _source(12, 256, padding=320, channel_stride=1, device="cuda")
    state = _cache([4, 4, 4, 0], 256, device="cuda")
    weight = torch.randn(3, 256, 4, device="cuda")
    for _ in range(3):
        kernels.convolution(source, weight, state, planar=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final = kernels.convolution(source, weight, state, planar=True)
    for lengths in ([1, 3, 8, 0], [0, 4, 3, 5]):
        source.copy_(torch.randn_like(source) * 0.2)
        replacement = _cache(lengths, 256, device="cuda")
        state.convolution.copy_(replacement.convolution)
        state.cu_seqlens.copy_(replacement.cu_seqlens)
        state.sequence_indices.copy_(replacement.sequence_indices)
        weight.mul_(0.75)
        expected, expected_final = _legacy(source, weight, state)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.permute(1, 0, 2), expected, atol=0, rtol=0)
        torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
