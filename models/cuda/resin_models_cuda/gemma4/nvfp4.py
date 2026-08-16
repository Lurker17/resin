"""FlashInfer W4A16 adapter for ModelOpt NVFP4 dense linears."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def swizzle_block_scales(scales: Tensor) -> Tensor:
    """Convert linear ModelOpt scales to the canonical 128x4 layout."""

    if scales.ndim != 2:
        raise ValueError("NVFP4 block scales must have shape [out, in / 16]")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError("NVFP4 block scales must use float8_e4m3fn")
    rows, columns = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.zeros(
        (1, padded_rows, padded_columns), dtype=scales.dtype, device=scales.device
    )
    padded[:, :rows, :columns] = scales
    return (
        padded.reshape(1, padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(padded_rows, padded_columns)
    )


class NvFp4Linear(nn.Module):
    """One checkpoint-shaped NVFP4 linear prepared once for BF16 activations."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 16):
        super().__init__()
        if in_features % group_size:
            raise ValueError("NVFP4 input width must be divisible by group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.backend: str | None = None
        self._mm: Callable[..., Tensor] | None = None
        self.register_buffer(
            "weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features, in_features // group_size, dtype=torch.float8_e4m3fn
            ),
        )
        self.register_buffer("input_scale", torch.empty(1, dtype=torch.float32))
        self.register_buffer("weight_scale_2", torch.empty(1, dtype=torch.float32))

    def prepare(
        self,
        backend: str,
        *,
        prepare: Callable[..., tuple[Tensor, Tensor, Tensor | None]] | None = None,
        mm: Callable[..., Tensor] | None = None,
    ) -> None:
        if self.backend is not None:
            raise RuntimeError("NVFP4 weights were already prepared")
        if prepare is None or mm is None:
            from flashinfer.gemm import mm_bf16_fp4, prepare_bf16_fp4_weights

            prepare = prepare_bf16_fp4_weights
            mm = mm_bf16_fp4
        scales = swizzle_block_scales(self.weight_scale)
        weight, scales, alpha = prepare(
            self.weight,
            scales,
            self.weight_scale_2.float().reshape(1),
            backend=backend,
            block_size=self.group_size,
        )
        self.weight = weight
        self.weight_scale = scales
        if alpha is None:
            self.weight_scale_2 = None
        else:
            self.weight_scale_2 = alpha
        self.input_scale = None
        self.backend = backend
        self._mm = mm

    def forward(self, hidden: Tensor, *, out: Tensor | None = None) -> Tensor:
        if self._mm is None or self.backend is None:
            raise RuntimeError("NVFP4 weights must be prepared before inference")
        if hidden.dtype != torch.bfloat16:
            raise TypeError("NVFP4 W4A16 linears require bfloat16 activations")
        shape = (*hidden.shape[:-1], self.out_features)
        flat = hidden.reshape(-1, self.in_features)
        flat_out = None if out is None else out.reshape(-1, self.out_features)
        result = self._mm(
            flat,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            backend=self.backend,
            out_dtype=torch.bfloat16,
            out=flat_out,
            block_size=self.group_size,
        )
        return result.reshape(shape)


def prepare_nvfp4_modules(model: nn.Module, backend: str) -> None:
    for module in model.modules():
        if isinstance(module, NvFp4Linear):
            module.prepare(backend)


__all__ = ["NvFp4Linear", "prepare_nvfp4_modules", "swizzle_block_scales"]
