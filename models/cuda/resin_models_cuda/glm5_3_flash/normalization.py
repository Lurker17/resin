"""GLM normalization with the serving reference's FP32 accumulation boundaries."""

from __future__ import annotations

import torch
from torch import Tensor


def rms_norm(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    if value.is_cuda:
        from resin_models_cuda.glm5_3_flash.normalization_cuda import normalize

        return normalize(value, weight, eps)
    floating = value.float()
    normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + eps)
    # SGLang's RMSNorm keeps the weighted product in FP32 until the final cast.
    return (normalized * weight.float()).to(value.dtype)


def gated_rms_norm(value: Tensor, weight: Tensor, gate: Tensor, eps: float) -> Tensor:
    if value.is_cuda:
        from resin_models_cuda.glm5_3_flash.normalization_cuda import normalize

        return normalize(value, weight, eps, gate=gate)
    floating = value.float()
    normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + eps)
    # KDA keeps normalization, weight multiplication and gating in FP32.
    return (normalized * weight.float() * torch.sigmoid(gate.float())).to(value.dtype)
