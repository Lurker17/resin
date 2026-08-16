from __future__ import annotations

import torch

from resin_models_cuda.gemma4.nvfp4 import NvFp4Linear, swizzle_block_scales


def test_block_scale_swizzle_matches_128x4_byte_layout() -> None:
    values = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    scales = values.to(torch.float8_e4m3fn)

    swizzled = swizzle_block_scales(scales).view(torch.uint8).reshape(-1)
    linear = scales.view(torch.uint8)

    def offset(row: int, column: int) -> int:
        return (
            (column // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + column % 4
        )

    for row, column in [(0, 0), (31, 3), (32, 0), (127, 7)]:
        assert swizzled[offset(row, column)] == linear[row, column]


def test_nvfp4_linear_prepares_once_and_uses_w4a16_contract() -> None:
    linear = NvFp4Linear(16, 8)
    linear.weight.zero_()
    linear.weight_scale.fill_(1.0)
    linear.input_scale.fill_(123.0)
    linear.weight_scale_2.fill_(0.25)
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

    linear.prepare("cudnn", prepare=prepare, mm=mm)
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
    assert linear.input_scale is None
