"""GLM feed-forward elementwise boundaries with unchanged PyTorch top-k.

The CUDA implementation fuses only arithmetic around the existing projection
and top-k operators. CPU execution remains the literal serving equations.
"""

import torch
from torch.nn import functional as F


def clamped_swiglu(gate, up, limit):
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError(
            "SwiGLU gate and up must have the same shape, dtype and device"
        )
    if (
        gate.is_cuda
        and gate.ndim == 2
        and gate.dtype
        in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )
    ):
        from resin_models_cuda.glm5_3_flash.feed_forward_fusion_cuda import swiglu

        return swiglu(gate, up, limit)
    # Clamp in the input dtype, evaluate SiLU/product in FP32, round only once.
    gate32 = gate.clamp(max=limit).float()
    up32 = up.clamp(-limit, limit).float()
    return (F.silu(gate32) * up32).to(gate.dtype)


def route(logits, bias, top_k, normalize, scaling):
    if logits.ndim != 2 or bias.shape != (logits.shape[1],):
        raise ValueError("Router logits and correction bias must match expert count")
    if logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("Router scoring requires FP32 logits and correction bias")
    if bias.device != logits.device:
        raise ValueError("Router logits and correction bias must share a device")
    if not 1 <= top_k <= logits.shape[1]:
        raise ValueError("Router top-k must fit the expert count")
    if logits.is_cuda and top_k == 8:
        from resin_models_cuda.glm5_3_flash.feed_forward_fusion_cuda import (
            route_scores,
            route_weights,
        )

        scores, corrected = route_scores(logits, bias)
        # Keep the exact same top-k implementation, including tie order. Bias
        # affects expert selection only, never the returned combination scores.
        indices = torch.topk(corrected, top_k, dim=-1, sorted=False).indices
        return route_weights(scores, indices, normalize, scaling)
    scores = logits.sigmoid()
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if normalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return indices.int(), weights * scaling
