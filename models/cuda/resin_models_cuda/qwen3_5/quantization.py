"""Qwen3.5 ModelOpt FP8 projection modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModelOptFp8Linear(nn.Module):
    """Tensor-scaled W8A8 linear matching unified ModelOpt tensor names."""

    weight: Tensor
    input_scale: Tensor
    weight_scale: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))

    def forward(self, hidden: Tensor) -> Tensor:
        if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
            raise TypeError("ModelOpt FP8 linears require CUDA BF16 activations")
        matrix = hidden.reshape(-1, self.in_features)
        quantized = torch.clamp(
            matrix.float() / self.input_scale,
            min=-448.0,
            max=448.0,
        ).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return output.view(*hidden.shape[:-1], self.out_features)


__all__ = ["ModelOptFp8Linear"]
