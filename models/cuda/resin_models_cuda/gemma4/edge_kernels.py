"""Ampere BF16 kernels and attention adapters for Gemma Edge.

Triton is imported lazily; CPU model and checkpoint imports remain supported.
cuBLASLt GELU uses accumulator precision before its final BF16 cast. The
norm/rotary/residual fusions preserve the reference BF16 intermediate rounds.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sparse_gather_dot(hidden: Tensor, weights: Tensor, token_indices: Tensor) -> Tensor:
    """Score selected canonical vocabulary rows without materializing embeddings.

    This is an optional assistant-head candidate. Accumulation is FP32 and the
    output retains the hidden dtype, matching BF16 GEMV's numerical contract.
    The independently qualified reference remains available in edge_assistant.
    """
    if not hidden.is_cuda or not weights.is_cuda or not token_indices.is_cuda:
        raise ValueError("sparse gather-dot requires CUDA tensors")
    if hidden.ndim != 2 or weights.ndim != 2 or token_indices.ndim != 2:
        raise ValueError("sparse gather-dot expects matrices")
    rows, width = hidden.shape
    if rows != token_indices.shape[0] or width != weights.shape[1]:
        raise ValueError("sparse gather-dot tensor geometry does not match")
    if hidden.dtype != weights.dtype or hidden.dtype not in {
        torch.bfloat16,
        torch.float16,
        torch.float32,
    }:
        raise ValueError("sparse gather-dot requires matching floating dtypes")
    if token_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("sparse gather-dot token indices must be integers")
    if hidden.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("sparse gather-dot needs contiguous projection channels")
    if not width or width > 2048:
        raise ValueError("sparse gather-dot supports channel widths 1 through 2048")
    from resin_models_cuda.gemma4.edge_kernels_triton import launch_sparse_gather_dot

    output = torch.empty(token_indices.shape, dtype=hidden.dtype, device=hidden.device)
    launch_sparse_gather_dot(hidden, weights, token_indices, output)
    return output


def linear_gelu(
    hidden: Tensor,
    weight: Tensor,
    *,
    zero_bias: Tensor,
) -> Tensor:
    """Compute a bias-free BF16 projection with cuBLASLt's GELU epilogue.

    GELU consumes the FP32 accumulator before the final BF16 output cast. The
    caller prepares an all-zero bias once, outside graph capture.
    """
    if (
        not hidden.is_cuda
        or hidden.dtype != torch.bfloat16
        or weight.dtype != hidden.dtype
        or weight.device != hidden.device
        or hidden.ndim != 2
        or weight.ndim != 2
        or hidden.shape[1] != weight.shape[1]
        or hidden.stride(-1) != 1
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "linear/GELU requires CUDA BF16 [rows, K] and contiguous [N, K]"
        )
    if zero_bias is None or (
        zero_bias.shape != (weight.shape[0],)
        or zero_bias.device != hidden.device
        or zero_bias.dtype != hidden.dtype
        or not zero_bias.is_contiguous()
    ):
        raise ValueError("cuBLASLt requires a prepared contiguous zero bias")
    # beta=1 and a one-dimensional bias select PyTorch's Lt activation path.
    return torch._addmm_activation(zero_bias, hidden, weight.t(), use_gelu=True)


def gelu_tanh_mul(gate: Tensor, up: Tensor) -> Tensor:
    """Fuse GELU and multiply while retaining the BF16 GELU output round."""
    if (
        not gate.is_cuda
        or gate.dtype != torch.bfloat16
        or up.dtype != gate.dtype
        or up.device != gate.device
        or gate.shape != up.shape
        or not gate.is_contiguous()
        or not up.is_contiguous()
    ):
        raise ValueError("GELU/multiply requires matching contiguous CUDA BF16 tensors")
    from resin_models_cuda.gemma4.edge_kernels_triton import launch_gelu_tanh_mul

    output = torch.empty_like(gate)
    launch_gelu_tanh_mul(gate, up, output)
    return output


def strided_qk_rmsnorm_rope(
    query: Tensor,
    key: Tensor | None,
    query_weight: Tensor,
    key_weight: Tensor | None,
    cos: Tensor,
    sin: Tensor,
    *,
    eps: float,
) -> tuple[Tensor, Tensor | None]:
    """Normalize Q/K heads and rotate, preserving separate BF16 rounding steps.

    Q and K may be views of a packed projection with padded row/head strides.
    Existing BF16 cos/sin values also encode partial rotary dimensions. Value
    normalization remains separate, as in the reference model.
    """
    tensors = [query, query_weight, cos, sin]
    if key is not None:
        if key_weight is None:
            raise ValueError("key normalization requires a weight")
        tensors.extend((key, key_weight))
    elif key_weight is not None:
        raise ValueError("key_weight requires a key tensor")
    if any(t.device != query.device for t in tensors):
        raise ValueError("norm/RoPE tensors must share a device")
    if not query.is_cuda or any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("strided norm/RoPE requires CUDA BF16 tensors")
    if query.ndim != 3 or query.shape[-1] not in (256, 512):
        raise ValueError("query must have shape [tokens, heads, 256 or 512]")
    tokens, heads, width = query.shape
    if not 0 < heads <= 32 or query.stride(-1) != 1:
        raise ValueError("query needs 1..32 heads and contiguous channels")
    if query_weight.shape != (width,) or query_weight.stride(0) != 1:
        raise ValueError("query weight must have contiguous head channels")
    if cos.shape != (tokens, width) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have shape [tokens, head_dim]")
    if key is not None and (
        key_weight is None
        or key.ndim != 3
        or key.shape[0] != tokens
        or key.shape[-1] != width
        or not 0 < key.shape[1] <= 32
        or key.stride(-1) != 1
        or key_weight.shape != (width,)
        or key_weight.stride(0) != 1
    ):
        raise ValueError("key tensor or weight has incompatible geometry")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    query_output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    key_output = (
        None
        if key is None
        else torch.empty_like(key, memory_format=torch.contiguous_format)
    )
    if tokens:
        from resin_models_cuda.gemma4.edge_kernels_triton import launch_norm_rope

        launch_norm_rope(
            query,
            key,
            query_weight,
            key_weight,
            cos,
            sin,
            query_output,
            key_output,
            eps,
        )
    return query_output, key_output


def norm_residual(
    hidden: Tensor,
    weight: Tensor,
    residual: Tensor,
    *,
    eps: float,
    scale: Tensor | None = None,
) -> Tensor:
    """Return BF16(residual + BF16(RMSNorm(hidden))), optionally scaled.

    Unlike add-before-normalization kernels, this matches Gemma's post-norm
    residual order. A final layer scalar is applied only after the BF16 add.
    The active implementation does not call this optional candidate.
    """
    tensors = [hidden, weight, residual] + ([] if scale is None else [scale])
    if not hidden.is_cuda or any(t.device != hidden.device for t in tensors):
        raise ValueError("post-norm fusion requires tensors on one CUDA device")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("post-norm fusion requires BF16 inputs")
    if hidden.ndim != 2 or residual.shape != hidden.shape:
        raise ValueError("hidden and residual must be matching matrices")
    if not 0 < hidden.shape[1] <= 8192:
        raise ValueError("post-norm fusion supports widths 1 through 8192")
    if hidden.stride(-1) != 1 or residual.stride(-1) != 1:
        raise ValueError("post-norm channels must be contiguous")
    if weight.shape != (hidden.shape[1],) or weight.stride(0) != 1:
        raise ValueError("post-norm weight has incompatible geometry")
    if scale is not None and scale.numel() != 1:
        raise ValueError("layer scale must contain one BF16 value")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    if len(hidden):
        from resin_models_cuda.gemma4.edge_kernels_triton import launch_norm_residual

        launch_norm_residual(hidden, weight, residual, scale, output, eps)
    return output


class UniformCausalPagedVerifyWrapper:
    """Use FA2's fixed query-length schedule for speculative verification.

    The public surface matches BatchPrefillWithPagedKVCacheWrapper for the
    uniform, bottom-right causal query blocks used by MTP. This is an optional
    experiment; the active runner continues to use the prefill wrapper.
    """

    def __init__(
        self,
        workspace,
        layout,
        *,
        query_width,
        split_kv=None,
        use_cuda_graph=False,
        qo_indptr_buf=None,
        paged_kv_indptr_buf=None,
        paged_kv_indices_buf=None,
        paged_kv_last_page_len_buf=None,
        **kwargs,
    ):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        if type(query_width) is not int or query_width <= 1:
            raise ValueError("a verifier requires at least two queries per request")
        if kwargs.get("backend", "fa2") not in {"auto", "fa2"}:
            raise ValueError("uniform causal verification requires the FA2 backend")
        self.query_width = query_width
        self.split_kv = split_kv
        self._wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            layout,
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=True,
            backend="fa2",
            paged_kv_indptr_buffer=paged_kv_indptr_buf,
            paged_kv_indices_buffer=paged_kv_indices_buf,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buf,
        )

    def __getattr__(self, name):
        return getattr(self._wrapper, name)

    def plan(self, qo_indptr, *args, **kwargs):
        if not kwargs.pop("causal", True):
            raise ValueError("uniform verification must be causal")
        if self.split_kv is not None:
            kwargs["disable_split_kv"] = not self.split_kv
        self._wrapper.plan(
            *args,
            q_len_per_req=self.query_width,
            is_causal=True,
            **kwargs,
        )

    def run(self, *args, **kwargs):
        return self._wrapper.run(*args, **kwargs)


__all__ = [
    "sparse_gather_dot",
    "linear_gelu",
    "gelu_tanh_mul",
    "strided_qk_rmsnorm_rope",
    "norm_residual",
    "UniformCausalPagedVerifyWrapper",
    "update_reserved_paged_metadata",
]


def update_reserved_paged_metadata(inputs, *, query_width: int, sliding_window: int):
    """Derive exact readable CSR and physical writes from GPU accepted positions.

    The immutable dense tables contain the native reservation envelope. Future
    pages are never exposed as readable KV. This requires fixed no-split FA2
    schedules; callers must establish schedule invariance before enabling it.
    """
    from resin.backend.sampling import SAMPLING_OFFSET_STRIDE
    from resin_models_cuda.gemma4.edge_kernels_triton import (
        launch_reserved_paged_metadata,
    )

    if not inputs["positions"].is_cuda or query_width <= 1 or sliding_window <= 0:
        raise ValueError(
            "reserved metadata requires CUDA positions and valid query/window widths"
        )
    launch_reserved_paged_metadata(
        inputs, query_width, sliding_window, SAMPLING_OFFSET_STRIDE
    )
