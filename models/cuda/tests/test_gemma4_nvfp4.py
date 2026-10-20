from __future__ import annotations

import pytest
import torch

from resin_models_cuda.nvfp4 import (
    Nvfp4Linear,
    Nvfp4LinearWeights,
    prepare_nvfp4_moe_weights,
    quantize_nvfp4,
    swizzle_block_scales,
)


def test_block_scale_swizzle_matches_128x4_byte_layout() -> None:
    values = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    scales = values.to(torch.float8_e4m3fn)

    swizzled = swizzle_block_scales(scales).view(torch.uint8).reshape(-1)
    linear = scales.view(torch.uint8)

    def offset(row: int, column: int) -> int:
        return (
            (column // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4
        )

    for row, column in [(0, 0), (31, 3), (32, 0), (127, 7)]:
        assert swizzled[offset(row, column)] == linear[row, column]


def test_nvfp4_lowering_preserves_checkpoint_and_uses_w4a16_contract() -> None:
    source = Nvfp4LinearWeights(16, 8)
    source.weight.zero_()
    source.weight_scale.fill_(1.0)
    source.input_scale.fill_(123.0)
    source.weight_scale_2.fill_(0.25)
    checkpoint_shapes = {
        name: value.shape for name, value in source.state_dict().items()
    }
    calls = []

    def prepare(weight, scales, alpha, **kwargs):
        calls.append((weight.shape, scales.shape, alpha.item(), kwargs))
        return weight, scales, alpha

    def mm(hidden, weight, scales, alpha, **kwargs):
        del weight, scales
        assert alpha.item() == 0.25
        assert kwargs["backend"] == "cudnn"
        assert kwargs["out_dtype"] == torch.bfloat16
        return torch.full(
            (hidden.shape[0], 8), 7.0, dtype=torch.bfloat16, device=hidden.device
        )

    with pytest.raises(RuntimeError, match="lowered"):
        source(torch.ones(3, 16, dtype=torch.bfloat16))
    linear = Nvfp4Linear(source, "cudnn", prepare=prepare, mm=mm)
    result = linear(torch.ones(3, 16, dtype=torch.bfloat16))

    assert calls == [
        (
            torch.Size([8, 8]),
            torch.Size([128, 4]),
            0.25,
            {"backend": "cudnn", "block_size": 16},
        )
    ]
    assert result.shape == (3, 8)
    assert result.tolist() == [[7.0] * 8] * 3
    assert not hasattr(linear, "input_scale")
    assert {
        name: value.shape for name, value in source.state_dict().items()
    } == checkpoint_shapes
    assert linear.weight is source.weight


def test_nvfp4_moe_activation_quantization_uses_group16_linear_scales() -> None:
    calls = []

    def quantize(hidden, scale, group_size, ue8m0, swizzled):
        calls.append((hidden.shape, scale.item(), group_size, ue8m0, swizzled))
        return (
            torch.zeros(hidden.shape[0], hidden.shape[1] // 2, dtype=torch.uint8),
            torch.zeros(hidden.shape[0], hidden.shape[1] // 16, dtype=torch.uint8),
        )

    packed, scales = quantize_nvfp4(
        torch.ones(3, 32, dtype=torch.bfloat16),
        torch.tensor(0.25),
        quantize=quantize,
    )

    assert calls == [(torch.Size([3, 32]), 0.25, 16, False, False)]
    assert packed.shape == (3, 16)
    assert scales.shape == (3, 2)
    assert scales.dtype == torch.float8_e4m3fn


def test_nvfp4_moe_weight_preparation_is_in_place_and_idempotent() -> None:
    calls = []

    def reorder(tensor):
        calls.append(("reorder", tuple(tensor.shape)))
        return tensor.flip(0)

    def shuffle(tensor, tile):
        calls.append(("shuffle", tuple(tensor.shape), tile))
        return tensor.clone()

    def interleave(tensor):
        calls.append(("interleave", tuple(tensor.shape)))
        return tensor + 1

    w13 = torch.zeros(2, 64, 16, dtype=torch.uint8)
    w13_scales = torch.zeros(2, 64, 2, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(2, 32, 16, dtype=torch.uint8)
    w2_scales = torch.zeros(2, 32, 2, dtype=torch.float8_e4m3fn)

    prepare_nvfp4_moe_weights(
        w13,
        w13_scales,
        w2,
        w2_scales,
        reorder=reorder,
        shuffle=shuffle,
        interleave=interleave,
    )
    first_calls = list(calls)
    prepare_nvfp4_moe_weights(
        w13,
        w13_scales,
        w2,
        w2_scales,
        reorder=reorder,
        shuffle=shuffle,
        interleave=interleave,
    )

    assert calls == first_calls
    assert torch.all(w13_scales.view(torch.uint8) == 1)
    assert torch.all(w2_scales.view(torch.uint8) == 1)
