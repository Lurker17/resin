# Adapted from DeepSeek-V4-Flash-0731 (MIT).
"""Resin-owned DeepSeek-V4 target-model hierarchy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from deep_gemm import tf32_hc_prenorm_gemm
from flashinfer.mhc import mhc_post
from flashinfer.topk import top_k as flashinfer_top_k
from torch import nn

from resin.backend.cuda.context_parallel import ContextParallelTokens
from resin.backend.cuda.parallel import (
    TensorParallelContext,
    TensorParallelLayout,
    TensorShard,
)
from resin.backend.loading import prepare_model
from resin.backend.torch import benchmark_expert_indices, use_default_dtype
from resin.backend.transformer import RMSNorm as SharedRMSNorm
from resin_models_cuda.deepseek_v4.attention import (
    sparse_decode_attention,
    sparse_verify_attention,
)
from resin_models_cuda.deepseek_v4.cache import (
    C4_STATE_GROUP,
    C128_STATE_GROUP,
    INDEX_STATE_GROUP,
    CacheArena,
    DeviceCache,
    DeviceMapping,
)
from resin_models_cuda.deepseek_v4.cutile import (
    act_quant,
    clear_rejected_compress_state,
    compress_verify,
    dspark_attention_metadata,
    finish_index_topk,
    fp4_act_quant,
    hadamard_transform,
    hash_sqrt_softplus,
    index_scores,
    kv_norm_rope_store,
    qk_norm_rope_block,
    qk_norm_rope_store,
    rope_last_block_inplace,
    swiglu,
    top6_sqrt_softplus,
    verify_attention_metadata,
)
from resin_models_cuda.deepseek_v4.experts import RoutedExperts
from resin_models_cuda.deepseek_v4.gemm import (
    dynamic_gemm_rows,
    fp4_gemm,
    fp8_gemm,
    fp8_gemm_uses_packed_scales,
)
from resin_models_cuda.deepseek_v4.tilelang_kernels import mhc_pre_norm

if TYPE_CHECKING:
    from resin.backend.topology import ParallelGroups

block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
_MHC_PRE_NUM_SPLITS = 64


def _default_tensor_parallel() -> TensorParallelContext:
    """Compatibility for callers using the default process group as TP."""

    size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=size),
        dist.group.WORLD if size > 1 else None,
    )


@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""

    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 0
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # dspark
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256

    def expert_intermediate_shard(self, layout: TensorParallelLayout) -> TensorShard:
        """Partition complete quantization blocks across expert tensor ranks."""

        alignment = 128 if self.expert_dtype == "fp4" or self.dtype == "fp8" else 1
        return layout.aligned_shard(
            self.moe_inter_dim, alignment=alignment, name="moe_inter_dim"
        )


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each attention-TP rank holds one contiguous vocabulary shard.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        size, rank = self.tensor_parallel.size, self.tensor_parallel.rank
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % size == 0, (
            f"Vocabulary size must be divisible by attention TP (size={size})"
        )
        self.part_vocab_size = vocab_size // size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = (
            self.tensor_parallel.sum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel.enabled:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if self.tensor_parallel.enabled:
            y[mask] = 0
        return self.allreduce(y)


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(
            x,
            block_size,
            scale_fmt,
            scale_dtype,
            packed_scales=True,
        )
        return fp4_gemm(
            x, s, weight, cast(torch.Tensor, getattr(weight, "scale")), scale_dtype
        )
    elif weight.dtype == torch.float8_e4m3fn:
        rows = x.numel() // x.shape[-1]
        x, s = act_quant(
            x,
            block_size,
            scale_fmt,
            scale_dtype,
            packed_scales=fp8_gemm_uses_packed_scales(rows),
        )
        return fp8_gemm(
            x, s, weight, cast(torch.Tensor, getattr(weight, "scale")), scale_dtype
        )
    else:
        return F.linear(x, weight)


def grouped_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run independent per-group output projections."""

    return torch.einsum("bsgd,grd->bsgr", x, weight)


def bf16_linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply CUDA BF16 operands while accumulating into FP32."""

    output_shape = (*x.shape[:-1], weight.shape[0])
    return torch.mm(
        x.reshape(-1, x.shape[-1]),
        weight.t(),
        out_dtype=torch.float32,
    ).view(output_shape)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        *,
        weight_storage: torch.Tensor | None = None,
        scale_storage: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            weight_shape = (out_features, in_features // 2)
            if weight_storage is None:
                weight_storage = torch.empty(weight_shape, dtype=torch.float4_e2m1fn_x2)
            if weight_storage.shape != weight_shape or weight_storage.dtype != dtype:
                raise ValueError("invalid shared FP4 weight storage")
            self.weight = nn.Parameter(weight_storage)
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            scale_shape = (scale_out_features, scale_in_features)
            if scale_storage is None:
                scale_storage = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu)
            if (
                scale_storage.shape != scale_shape
                or scale_storage.dtype != torch.float8_e8m0fnu
            ):
                raise ValueError("invalid shared FP4 scale storage")
            self.scale = nn.Parameter(scale_storage)
            setattr(self.weight, "scale", self.scale)
        elif dtype == torch.float8_e4m3fn:
            weight_shape = (out_features, in_features)
            if weight_storage is None:
                weight_storage = torch.empty(weight_shape, dtype=dtype)
            if weight_storage.shape != weight_shape or weight_storage.dtype != dtype:
                raise ValueError("invalid shared FP8 weight storage")
            self.weight = nn.Parameter(weight_storage)
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            scale_shape = (scale_out_features, scale_in_features)
            if scale_storage is None:
                scale_storage = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu)
            if (
                scale_storage.shape != scale_shape
                or scale_storage.dtype != torch.float8_e8m0fnu
            ):
                raise ValueError("invalid shared FP8 scale storage")
            self.scale = nn.Parameter(scale_storage)
            setattr(self.weight, "scale", self.scale)
        else:
            if scale_storage is not None:
                raise ValueError("unquantized weights do not have scale storage")
            weight_shape = (out_features, in_features)
            if weight_storage is None:
                weight_storage = torch.empty(weight_shape, dtype=dtype)
            if weight_storage.shape != weight_shape or weight_storage.dtype != dtype:
                raise ValueError("invalid shared unquantized weight storage")
            self.weight = nn.Parameter(weight_storage)
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        size = self.tensor_parallel.size
        assert out_features % size == 0, (
            f"Output features must be divisible by attention TP (size={size})"
        )
        self.part_out_features = out_features // size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        size = self.tensor_parallel.size
        assert in_features % size == 0, (
            f"Input features must be divisible by attention TP (size={size})"
        )
        self.part_in_features = in_features // size
        super().__init__(self.part_in_features, out_features, bias, dtype)
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = (
            self.tensor_parallel.sum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if self.tensor_parallel.enabled:
            y = self.allreduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(SharedRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps, weight_dtype=torch.bfloat16)


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    row_aligned = x.size(1) == 1 and freqs_cis.size(0) == x.size(0)
    if freqs_cis.ndim == 3:
        if x.ndim == 4:
            freqs_cis = freqs_cis.unsqueeze(2)
    elif x.ndim == 3:
        freqs_cis = freqs_cis.view(
            x.size(0) if row_aligned else 1, x.size(1), x.size(-1)
        )
    else:
        freqs_cis = freqs_cis.view(
            x.size(0) if row_aligned else 1, x.size(1), 1, x.size(-1)
        )
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    assert x.dtype == torch.bfloat16
    return hadamard_transform(x)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    _fused_projection_weight: torch.Tensor

    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # The checkpoint stores both projections in BF16. Keep their public
        # parameter views in one allocation so the shared input uses one GEMM,
        # with FP32 accumulation selected explicitly at the call site.
        projection_width = coff * self.head_dim
        self.register_buffer(
            "_fused_projection_weight",
            torch.empty(2 * projection_width, self.dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.wkv = Linear(
            self.dim,
            projection_width,
            dtype=torch.bfloat16,
            weight_storage=self._fused_projection_weight[:projection_width],
        )
        self.wgate = Linear(
            self.dim,
            projection_width,
            dtype=torch.bfloat16,
            weight_storage=self._fused_projection_weight[projection_width:],
        )
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        logical_window = compress_ratio * coff
        retained_ring_size = (
            (
                math.ceil((logical_window + args.dspark_block_size) / compress_ratio)
                * compress_ratio
            )
            if args.dspark_block_size
            else logical_window
        )
        # cuTile requires static tile extents to be powers of two. Paged state
        # maps entries outside the Engine-retained window to the null parent.
        self.ring_size = 1 << (retained_ring_size - 1).bit_length()
        # Speculative decode appends raw projected rows into a padded ring.
        # The extra capacity prevents an unaccepted tail from overwriting the
        # committed compressor window.
        self.kv_state: torch.Tensor | None
        self.register_buffer("kv_state", None, persistent=False)
        self.score_state: torch.Tensor | None
        self.register_buffer("score_state", None, persistent=False)
        self.freqs_cis: torch.Tensor | None = None

    def _state_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_state is None or self.score_state is None:
            raise RuntimeError("compressor state is not bound to the cache arena")
        return self.kv_state, self.score_state

    def forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        start_position: int,
        cache_pool: torch.Tensor,
        cache_rows: torch.Tensor,
        cache: DeviceCache,
        state_group: int,
    ) -> None:
        """Append an arbitrary prompt suffix with one vectorized cache update."""

        batch_size, block_length, _ = x.shape
        ratio = self.compress_ratio
        if positions.shape != (batch_size, block_length):
            raise ValueError("prefill compressor positions must match the token block")
        if cache_rows.shape != positions.shape:
            raise ValueError("prefill compressor rows must match the token block")

        projected = bf16_linear_fp32(x, self._fused_projection_weight)
        raw_values, raw_scores = projected.chunk(2, dim=-1)
        adjusted_scores = raw_scores + self.ape[positions.remainder(ratio)]
        state_pages, state_offsets = cache.state_locations(state_group, positions)
        value_state, score_state = self._state_tensors()
        flat_pages = state_pages.flatten().long()
        flat_offsets = state_offsets.flatten().long()
        valid = flat_pages >= 0
        flat_pages = flat_pages[valid]
        flat_offsets = flat_offsets[valid]
        value_state[flat_pages, flat_offsets] = raw_values.flatten(0, 1)[valid]
        score_state[flat_pages, flat_offsets] = adjusted_scores.flatten(0, 1)[valid]

        first_complete = (ratio - 1 - start_position) % ratio
        if first_complete >= block_length:
            return
        complete_offsets = torch.arange(
            first_complete,
            block_length,
            ratio,
            dtype=torch.long,
            device=x.device,
        )
        if complete_offsets.numel() == 0:
            return
        complete_positions = positions.index_select(1, complete_offsets)
        window = ratio * (2 if self.overlap else 1)
        history_positions = complete_positions.unsqueeze(-1) - torch.arange(
            window - 1,
            -1,
            -1,
            dtype=positions.dtype,
            device=positions.device,
        )
        history_pages, history_offsets = cache.state_locations(
            state_group, history_positions
        )
        cached_values = value_state[
            history_pages.clamp_min(0).long(), history_offsets.long()
        ]
        cached_scores = score_state[
            history_pages.clamp_min(0).long(), history_offsets.long()
        ]
        current_indices = history_positions - start_position
        from_current = (current_indices >= 0) & (current_indices < block_length)
        current_indices = current_indices.clamp(0, block_length - 1).long()
        gather_indices = current_indices.unsqueeze(-1).expand(
            *current_indices.shape, raw_values.shape[-1]
        )
        current_values = torch.gather(
            raw_values.unsqueeze(1).expand(-1, complete_offsets.numel(), -1, -1),
            2,
            gather_indices,
        )
        current_scores = torch.gather(
            adjusted_scores.unsqueeze(1).expand(-1, complete_offsets.numel(), -1, -1),
            2,
            gather_indices,
        )
        history_values = torch.where(
            from_current.unsqueeze(-1), current_values, cached_values
        ).float()
        history_scores = torch.where(
            from_current.unsqueeze(-1), current_scores, cached_scores
        ).float()
        if self.overlap:
            dimension = self.head_dim
            history_values = torch.cat(
                (
                    history_values[:, :, :ratio, :dimension],
                    history_values[:, :, ratio:, dimension:],
                ),
                dim=2,
            )
            history_scores = torch.cat(
                (
                    history_scores[:, :, :ratio, :dimension],
                    history_scores[:, :, ratio:, dimension:],
                ),
                dim=2,
            )
        pooled = (history_values * history_scores.softmax(dim=2)).sum(dim=2)
        pooled = self.norm(pooled.to(x.dtype))
        rope_positions = complete_positions + 1 - ratio
        if self.freqs_cis is None:
            raise RuntimeError("compressor frequencies are not initialized")
        apply_rotary_emb(
            pooled[..., -self.rope_head_dim :],
            self.freqs_cis[rope_positions],
        )
        if self.rotate:
            pooled = rotate_activation(pooled)
            fp4_act_quant(pooled, fp4_block_size, True)
        else:
            act_quant(
                pooled[..., : -self.rope_head_dim],
                64,
                scale_fmt,
                scale_dtype,
                True,
            )
        complete_rows = cache_rows.index_select(1, complete_offsets)
        cache_pool.index_copy_(
            0,
            complete_rows.flatten().long(),
            pooled.flatten(0, 1).to(cache_pool.dtype),
        )

    def forward_verify(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        cache_pool: torch.Tensor,
        cache_mapping: DeviceMapping,
        state_mapping: DeviceMapping,
    ) -> torch.Tensor:
        """Update a fixed verification block with one projection and scan."""

        value_state, score_state = self._state_tensors()
        assert self.freqs_cis is not None
        kv, score = bf16_linear_fp32(x, self._fused_projection_weight).chunk(2, dim=-1)
        compress_verify(
            kv,
            score,
            self.ape,
            value_state,
            score_state,
            self.norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache_pool,
            positions,
            self.compress_ratio,
            self.rotate,
            self.norm.eps,
            cache_mapping=cache_mapping,
            state_mapping=state_mapping,
            state_ring_size=self.ring_size,
        )
        return kv[..., : self.head_dim]

    def clear_rejected(
        self,
        positions: torch.Tensor,
        processed_counts: torch.Tensor,
        block_length: int,
        *,
        state_mapping: DeviceMapping,
    ) -> None:
        """Clear only rollback-sensitive c128 rows after target acceptance."""

        if self.compress_ratio == 128:
            value_state, score_state = self._state_tensors()
            clear_rejected_compress_state(
                value_state,
                score_state,
                positions,
                processed_counts,
                block_length,
                state_mapping=state_mapping,
                state_ring_size=self.ring_size,
            )


def _compact_topk_indices(
    scores: torch.Tensor, capacity: int, *, complete: torch.Tensor | None = None
) -> torch.Tensor:
    """Select a fixed-capacity logical KV set in canonical ascending order.

    Scoring remains context-sized. Invalid future candidates sort behind all
    selected causal rows, and padding never enters the active sparse prefix.
    """

    width = min(capacity, scores.shape[-1])
    if width == 0:
        return torch.full(
            (*scores.shape[:-1], capacity),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
    values, indices = flashinfer_top_k(scores, width, sorted=True)
    valid = torch.isfinite(values)
    if complete is not None:
        valid &= indices < complete
    sentinel = torch.iinfo(indices.dtype).max
    logical = indices.masked_fill(~valid, sentinel).sort(dim=-1).values
    logical.masked_fill_(logical == sentinel, -1)
    return F.pad(logical, (0, capacity - width), value=-1)


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        compress_ratio: int = 4,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        if (
            type(args.index_topk) is not int
            or args.index_topk <= 0
            or args.index_topk % 64
        ):
            raise ValueError("DeepSeek-V4 index_topk must be a positive multiple of 64")
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // self.tensor_parallel.size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.selected_capacity = min(
            self.index_topk, math.ceil(args.max_seq_len / compress_ratio / 64) * 64
        )
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            tensor_parallel=self.tensor_parallel,
        )
        self.weights_proj = ColumnParallelLinear(
            self.dim,
            self.n_heads,
            dtype=torch.bfloat16,
            tensor_parallel=self.tensor_parallel,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = (
            self.tensor_parallel.sum
        )

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.freqs_cis: torch.Tensor | None = None

    def forward_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        position: torch.Tensor,
        offset: int,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Select compressed positions with graph-static score geometry."""

        self.compressor.forward_verify(
            x,
            position[:, None],
            cache_pool=cache.index_keys.view(-1, self.head_dim),
            cache_mapping=cache.index_mapping(self.layer_id),
            state_mapping=cache.state_mapping(INDEX_STATE_GROUP),
        )
        return self._forward_graph(
            x, qr, position[:, None], offset, graph_seq_len, cache=cache
        )

    def forward_prefill(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        start_position: int,
        sequence_length: int,
        *,
        cache: DeviceCache,
        query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select compressed history for an arbitrary cached prompt suffix."""

        batch_size, block_length, _ = x.shape
        ratio, rope_dim = self.compress_ratio, self.rope_head_dim
        if positions.shape != (batch_size, block_length):
            raise ValueError("index prefill positions must match the token block")
        if self.freqs_cis is None:
            raise RuntimeError("index frequencies are not initialized")

        self.compressor.forward_prefill(
            x,
            positions,
            start_position=start_position,
            cache_pool=cache.index_keys.view(-1, self.head_dim),
            cache_rows=cache.index_rows(self.layer_id, positions),
            cache=cache,
            state_group=INDEX_STATE_GROUP,
        )
        if query_indices is not None:
            x = x.index_select(1, query_indices)
            qr = qr.index_select(1, query_indices)
            positions = positions.index_select(1, query_indices)
            block_length = query_indices.numel()
        if block_length == 0:
            return torch.empty(
                (batch_size, 0, self.selected_capacity),
                dtype=torch.int32,
                device=x.device,
            )
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rope_dim:], self.freqs_cis[positions])
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        weights = self.weights_proj(x)
        width = math.ceil(sequence_length / ratio / 64) * 64
        if width == 0:
            return torch.full(
                (batch_size, block_length, self.selected_capacity),
                -1,
                dtype=torch.int32,
                device=x.device,
            )
        index_score = index_scores(
            q,
            cache.index_keys.view(-1, self.head_dim),
            weights,
            positions,
            ratio,
            self.softmax_scale * self.n_heads**-0.5,
            cache_mapping=cache.index_mapping(self.layer_id),
            cache_width=width,
        )
        if self.tensor_parallel.enabled:
            index_score = self.allreduce(index_score)
        complete = (positions + 1).div(ratio, rounding_mode="floor")
        logical = _compact_topk_indices(
            index_score.flatten(0, 1),
            self.selected_capacity,
            complete=complete.reshape(-1, 1),
        ).view(batch_size, block_length, self.selected_capacity)
        physical = cache.compressed_rows(
            self.layer_id,
            ratio,
            logical.clamp_min(0) * ratio,
        )
        return torch.where(logical >= 0, physical, -1).to(torch.int32)

    def _forward_graph(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        offset: int,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Score one fixed graph block with one projection and collective."""

        bsz, block_length, _ = x.shape
        ratio = self.compress_ratio
        width = math.ceil(graph_seq_len / ratio / 64) * 64
        if width == 0:
            return torch.full(
                (bsz, block_length, self.selected_capacity),
                -1,
                dtype=torch.int32,
                device=x.device,
            )

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        assert self.freqs_cis is not None
        rope_last_block_inplace(
            q,
            torch.view_as_real(self.freqs_cis),
            positions,
        )
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        weights = self.weights_proj(x)
        index_cache = cache.index_keys.view(-1, self.head_dim)
        index_score = index_scores(
            q,
            index_cache,
            weights,
            positions,
            ratio,
            self.softmax_scale * self.n_heads**-0.5,
            cache_mapping=cache.index_mapping(self.layer_id),
            cache_width=width,
        )
        if self.tensor_parallel.enabled:
            index_score = self.allreduce(index_score)
        complete = (positions + 1).div(ratio, rounding_mode="floor")
        topk_idxs = _compact_topk_indices(
            index_score.flatten(0, 1),
            self.selected_capacity,
            complete=complete.reshape(-1, 1),
        )
        topk_idxs = topk_idxs.view(*index_score.shape[:2], self.selected_capacity)
        selected = finish_index_topk(
            topk_idxs,
            positions,
            ratio,
            offset,
            cache_mapping=cache.compressed_mapping(self.layer_id, ratio),
        )
        return selected


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""

    freqs_cis: torch.Tensor
    _fused_wqkv_weight: torch.Tensor
    _fused_wqkv_scale: torch.Tensor

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // self.tensor_parallel.size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // self.tensor_parallel.size
        self.window_size = args.window_size
        # Verification writes a whole speculative block before acceptance is
        # known. A small physical guard band keeps those future writes from
        # overwriting the still-live logical sliding window after rejection.
        self.window_capacity = args.window_size + (
            args.dspark_block_size + 1 if args.dspark_block_size else 0
        )
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self._fused_wqkv = default_dtype == torch.float8_e4m3fn
        if self._fused_wqkv:
            fused_width = self.q_lora_rank + self.head_dim
            self.register_buffer(
                "_fused_wqkv_weight",
                torch.empty(fused_width, self.dim, dtype=default_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_fused_wqkv_scale",
                torch.empty(
                    fused_width // block_size,
                    self.dim // block_size,
                    dtype=torch.float8_e8m0fnu,
                ),
                persistent=False,
            )
            setattr(self._fused_wqkv_weight, "scale", self._fused_wqkv_scale)
            self.wq_a = Linear(
                self.dim,
                self.q_lora_rank,
                weight_storage=self._fused_wqkv_weight[: self.q_lora_rank],
                scale_storage=self._fused_wqkv_scale[: self.q_lora_rank // block_size],
            )
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            tensor_parallel=self.tensor_parallel,
        )
        if self._fused_wqkv:
            self.wkv = Linear(
                self.dim,
                self.head_dim,
                weight_storage=self._fused_wqkv_weight[self.q_lora_rank :],
                scale_storage=self._fused_wqkv_scale[self.q_lora_rank // block_size :],
            )
        else:
            self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
            tensor_parallel=self.tensor_parallel,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * args.o_lora_rank,
            self.dim,
            tensor_parallel=self.tensor_parallel,
        )
        self.softmax_scale = self.head_dim**-0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(
                    layer_id,
                    args,
                    self.compress_ratio,
                    tensor_parallel=self.tensor_parallel,
                )
            else:
                self.indexer = None
            # Attention sees one prepared sparse-row geometry in every phase,
            # including when a peer's longer context selects a larger graph.
            self.compressed_capacity = (
                self.indexer.selected_capacity
                if self.indexer is not None
                else math.ceil(args.max_seq_len / self.compress_ratio / 4) * 4
            )
        else:
            self.compressed_capacity = 0

        self.attention_workspace: torch.Tensor | None = None
        self.decode_overlap_stream: torch.cuda.Stream | None = None
        self.prefill_project: (
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None
        self.prefill_query: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.prefill_finish: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _paged_attention_metadata(
        self,
        cache: DeviceCache,
        token_positions: torch.Tensor,
        compressed_indices: torch.Tensor | None = None,
        generated_compressed_width: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build physical sparse rows inside the existing metadata launch."""

        compressed_mapping = None
        if self.compress_ratio:
            compressed_mapping = cache.compressed_mapping(
                self.layer_id, self.compress_ratio
            )
        return verify_attention_metadata(
            token_positions[:, 0],
            token_positions.shape[1],
            self.window_size,
            self.window_capacity,
            compressed_indices,
            0,
            self.compress_ratio,
            generated_compressed_width,
            window_mapping=cache.window_mapping(self.layer_id),
            compressed_mapping=compressed_mapping,
        )

    def _prepare_compression(self) -> None:
        if not self.compress_ratio or self.compressor.freqs_cis is not None:
            return
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
            self.indexer.compressor.freqs_cis = self.freqs_cis

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fused_wqkv:
            q_input, kv = linear(x, self._fused_wqkv_weight).split(
                (self.q_lora_rank, self.head_dim), dim=-1
            )
        else:
            q_input, kv = self.wq_a(x), self.wkv(x)
        return self.q_norm(q_input), kv

    def _finish(self, output: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 1:
            apply_rotary_emb(
                output[..., -self.rope_head_dim :],
                self.freqs_cis.index_select(0, positions),
                True,
            )
        else:
            rope_last_block_inplace(
                output,
                torch.view_as_real(self.freqs_cis),
                positions,
                inverse=True,
            )
        output = output.view(*output.shape[:2], self.n_local_groups, -1)
        weight = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        return self.wo_b(grouped_bf16_linear(output, weight).flatten(2))

    def forward_decode(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Run one reusable fixed-geometry decode attention step."""

        ratio = self.compress_ratio
        self._prepare_compression()
        qr, kv = self._project(x)

        compressed_width = 0
        compress_topk_idxs = None
        generated_compressed_width = 0
        main_stream = None
        if ratio:
            padded_width = math.ceil(graph_seq_len / ratio / 4) * 4
            compressed_width = padded_width
            assert self.decode_overlap_stream is not None
            # Compression/indexing depends on normalized Q but is otherwise
            # independent of the dense Q projection and window-cache update.
            main_stream = torch.cuda.current_stream(x.device)
            self.decode_overlap_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.decode_overlap_stream):
                self.compressor.forward_verify(
                    x,
                    position[:, None],
                    cache_pool=cache.latents.view(-1, self.head_dim),
                    cache_mapping=cache.compressed_mapping(self.layer_id, ratio),
                    state_mapping=cache.state_mapping(
                        C4_STATE_GROUP if ratio == 4 else C128_STATE_GROUP
                    ),
                )
                if self.indexer is not None:
                    compress_topk_idxs = self.indexer.forward_decode(
                        x,
                        qr,
                        position,
                        0,
                        compressed_width * ratio,
                        cache=cache,
                    )
                else:
                    generated_compressed_width = self.compressed_capacity

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        qk_norm_rope_store(
            q,
            kv,
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache.latents.view(-1, self.head_dim),
            position,
            self.eps,
            cache_mapping=cache.window_mapping(self.layer_id),
        )

        if ratio:
            assert main_stream is not None
            assert self.decode_overlap_stream is not None
            main_stream.wait_stream(self.decode_overlap_stream)
        topk_idxs, sparse_lengths, sequence_lengths = self._paged_attention_metadata(
            cache,
            position[:, None],
            compress_topk_idxs if ratio else None,
            generated_compressed_width,
        )
        sparse_lengths = sparse_lengths.flatten()
        sequence_lengths = sequence_lengths.flatten()

        assert self.attention_workspace is not None
        o = sparse_decode_attention(
            q,
            cache.latents,
            cache.latents,
            self.attn_sink,
            topk_idxs,
            sparse_lengths,
            sequence_lengths,
            self.attention_workspace,
            self.softmax_scale,
        )
        return self._finish(o, position)

    def forward_prefill(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        start_position: int,
        sequence_length: int,
        *,
        cache: DeviceCache,
        context_parallel: ContextParallelTokens | None = None,
    ) -> torch.Tensor:
        """Prefill a cached suffix, optionally splitting queries across CP ranks.

        MLA cache and compressor tails stay replicated; local query results
        are gathered before the transformer block consumes them.
        """

        batch_size, block_length, _ = x.shape
        ratio = self.compress_ratio
        if token_positions.shape != (batch_size, block_length):
            raise ValueError("prefill positions must match the token block")
        if context_parallel is not None and batch_size != 1:
            raise ValueError("DeepSeek context prefill requires one sequence")
        self._prepare_compression()
        qr, kv = (self.prefill_project or self._project)(x)

        if ratio:
            state_group = C4_STATE_GROUP if ratio == 4 else C128_STATE_GROUP
            self.compressor.forward_prefill(
                x,
                token_positions,
                start_position=start_position,
                cache_pool=cache.latents.view(-1, self.head_dim),
                cache_rows=cache.compressed_rows(self.layer_id, ratio, token_positions),
                cache=cache,
                state_group=state_group,
            )

        query_indices = (
            None if context_parallel is None else context_parallel.token_indices
        )
        if context_parallel is not None:
            # Compressor tails and MLA KV remain replicated. Preserve their
            # complete ordered updates before partitioning any query work.
            kv_norm_rope_store(
                kv,
                self.kv_norm.weight,
                torch.view_as_real(self.freqs_cis),
                cache.latents.view(-1, self.head_dim),
                token_positions,
                self.eps,
                cache_mapping=cache.window_mapping(self.layer_id),
            )
        compressed_indices = None
        generated_compressed_width = 0
        if ratio:
            if self.indexer is not None:
                compressed_indices = self.indexer.forward_prefill(
                    x,
                    qr,
                    token_positions,
                    start_position,
                    sequence_length,
                    cache=cache,
                    query_indices=query_indices,
                )
            else:
                generated_compressed_width = self.compressed_capacity

        if query_indices is not None:
            qr = qr.index_select(1, query_indices)
            kv = kv.index_select(1, query_indices)
            query_positions = token_positions.index_select(1, query_indices)
            if query_indices.numel() == 0:
                local = x.new_empty((0, self.dim))
                assert context_parallel is not None
                return context_parallel.gather(local).unsqueeze(0)
        else:
            query_positions = token_positions
        q = (self.prefill_query or self.wq_b)(qr).unflatten(
            -1, (self.n_local_heads, self.head_dim)
        )

        # This fused normalization also stores local KV again. With CP these
        # writes match the complete store above, while keeping Q normalization
        # identical to the unpartitioned kernel.
        qk_norm_rope_block(
            q,
            kv,
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache.latents.view(-1, self.head_dim),
            query_positions,
            self.eps,
            cache_mapping=cache.window_mapping(self.layer_id),
        )
        topk_indices, sparse_lengths, sequence_lengths = self._paged_attention_metadata(
            cache,
            query_positions,
            compressed_indices,
            generated_compressed_width,
        )
        if self.attention_workspace is None:
            raise RuntimeError("attention workspace is not initialized")
        o = sparse_verify_attention(
            q,
            cache.latents,
            cache.latents,
            self.attn_sink,
            topk_indices,
            sparse_lengths,
            sequence_lengths,
            self.attention_workspace,
            self.softmax_scale,
        )
        output = (self.prefill_finish or self._finish)(o, query_positions)
        if context_parallel is not None:
            output = context_parallel.gather(output.squeeze(0)).unsqueeze(0)
        return output

    def forward_verify(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Verify a short causal token block through padded recurrent rings."""

        batch_size, block_length, _ = x.shape
        ratio = self.compress_ratio
        self._prepare_compression()
        qr, kv = self._project(x)

        if token_positions.shape != (batch_size, block_length):
            raise ValueError("target layers must share one verification position block")
        if ratio:
            self.compressor.forward_verify(
                x,
                token_positions,
                cache_pool=cache.latents.view(-1, self.head_dim),
                cache_mapping=cache.compressed_mapping(self.layer_id, ratio),
                state_mapping=cache.state_mapping(
                    C4_STATE_GROUP if ratio == 4 else C128_STATE_GROUP
                ),
            )
            if self.indexer is not None:
                self.indexer.compressor.forward_verify(
                    x,
                    token_positions,
                    cache_pool=cache.index_keys.view(-1, self.indexer.head_dim),
                    cache_mapping=cache.index_mapping(self.layer_id),
                    state_mapping=cache.state_mapping(INDEX_STATE_GROUP),
                )

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))

        block_compressed_indices = None
        generated_compressed_width = 0
        if ratio:
            if self.indexer is not None:
                block_compressed_indices = self.indexer._forward_graph(
                    x,
                    qr,
                    token_positions,
                    0,
                    graph_seq_len,
                    cache=cache,
                )
            else:
                generated_compressed_width = self.compressed_capacity

        qk_norm_rope_block(
            q,
            kv,
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache.latents.view(-1, self.head_dim),
            token_positions,
            self.eps,
            cache_mapping=cache.window_mapping(self.layer_id),
        )
        topk_indices, sparse_lengths, sequence_lengths = self._paged_attention_metadata(
            cache,
            token_positions,
            block_compressed_indices,
            generated_compressed_width,
        )

        assert self.attention_workspace is not None
        o = sparse_verify_attention(
            q,
            cache.latents,
            cache.latents,
            self.attn_sink,
            topk_indices,
            sparse_lengths,
            sequence_lengths,
            self.attention_workspace,
            self.softmax_scale,
        )
        return self._finish(o, token_positions)


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.benchmark_balanced_experts = False
        self.benchmark_replica = 0
        self.benchmark_expert_parallel_size = 1
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32)
            )

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = bf16_linear_fp32(x, self.weight)
        if (
            not self.benchmark_balanced_experts
            and self.hash
            and self.score_func == "sqrtsoftplus"
            and self.topk == 6
        ):
            return hash_sqrt_softplus(
                scores,
                input_ids,
                self.tid2eid,
                self.route_scale,
            )
        if (
            not self.benchmark_balanced_experts
            and (scores.shape[-1] & (scores.shape[-1] - 1)) == 0
            and self.topk == 6
            and self.score_func == "sqrtsoftplus"
            and self.bias is not None
        ):
            return top6_sqrt_softplus(scores, self.bias, self.route_scale)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.benchmark_balanced_experts:
            indices = benchmark_expert_indices(
                scores,
                self.topk,
                self.benchmark_replica,
                self.benchmark_expert_parallel_size,
            )
        elif self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.to(torch.bfloat16), indices.to(torch.int32)


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""

    _fused_w13_weight: torch.Tensor
    _fused_w13_scale: torch.Tensor

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype=None,
        swiglu_limit=0,
        storage: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
    ):
        super().__init__()
        if storage is None:
            resolved_dtype = dtype or default_dtype
            self._fused_w13 = resolved_dtype == torch.float8_e4m3fn
            if self._fused_w13:
                self.register_buffer(
                    "_fused_w13_weight",
                    torch.empty(2 * inter_dim, dim, dtype=resolved_dtype),
                    persistent=False,
                )
                self.register_buffer(
                    "_fused_w13_scale",
                    torch.empty(
                        2 * inter_dim // block_size,
                        dim // block_size,
                        dtype=torch.float8_e8m0fnu,
                    ),
                    persistent=False,
                )
                setattr(self._fused_w13_weight, "scale", self._fused_w13_scale)
                self.w1 = Linear(
                    dim,
                    inter_dim,
                    dtype=resolved_dtype,
                    weight_storage=self._fused_w13_weight[:inter_dim],
                    scale_storage=self._fused_w13_scale[: inter_dim // block_size],
                )
                self.w3 = Linear(
                    dim,
                    inter_dim,
                    dtype=resolved_dtype,
                    weight_storage=self._fused_w13_weight[inter_dim:],
                    scale_storage=self._fused_w13_scale[inter_dim // block_size :],
                )
            else:
                self.w1 = Linear(dim, inter_dim, dtype=resolved_dtype)
                self.w3 = Linear(dim, inter_dim, dtype=resolved_dtype)
            self.w2 = Linear(inter_dim, dim, dtype=dtype)
        else:
            self._fused_w13 = False
            self.w1 = Linear(
                dim,
                inter_dim,
                dtype=dtype,
                weight_storage=storage[0][0],
                scale_storage=storage[0][1],
            )
            self.w2 = Linear(
                inter_dim,
                dim,
                dtype=dtype,
                weight_storage=storage[1][0],
                scale_storage=storage[1][1],
            )
            self.w3 = Linear(
                dim,
                inter_dim,
                dtype=dtype,
                weight_storage=storage[2][0],
                scale_storage=storage[2][1],
            )
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        if self._fused_w13:
            gate, up = linear(x, self._fused_w13_weight).chunk(2, dim=-1)
        else:
            gate = self.w1(x)
            up = self.w3(x)
        x = swiglu(gate, up, self.swiglu_limit)
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoEWeights(nn.Module):
    """Checkpoint-shaped gate, shared expert and rank-local routed storage."""

    _fused_w13_weight: torch.Tensor
    _fused_w13_scale: torch.Tensor
    _fused_w2_weight: torch.Tensor
    _fused_w2_scale: torch.Tensor
    _fused_output_scales: torch.Tensor
    _fused_clamp_limits: torch.Tensor

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        *,
        expert_parallel: TensorParallelLayout | None = None,
        expert_tensor_parallel: TensorParallelLayout | None = None,
    ):
        super().__init__()
        expert_parallel = expert_parallel or _default_tensor_parallel().layout
        expert_tensor_parallel = expert_tensor_parallel or TensorParallelLayout()
        self.layer_id = layer_id
        self.dim = args.dim
        owned = expert_parallel.even_shard(
            args.n_routed_experts, name="n_routed_experts"
        )
        self.expert_tensor_parallel = expert_tensor_parallel
        self.expert_intermediate_shard = args.expert_intermediate_shard(
            expert_tensor_parallel
        )
        self.routed_intermediate_size = args.moe_inter_dim
        intermediate = self.expert_intermediate_shard.size
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = owned.size
        self.n_activated_experts = args.n_activated_experts
        self.swiglu_limit = args.swiglu_limit
        self.decode_overlap_stream: torch.cuda.Stream | None = None
        self.experts_start_idx = owned.start
        self.experts_end_idx = owned.stop
        self.gate = Gate(layer_id, args)
        self.gate.benchmark_expert_parallel_size = expert_parallel.size
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self._fused_fp4 = expert_dtype == torch.float4_e2m1fn_x2
        if self._fused_fp4:
            self.register_buffer(
                "_fused_w13_weight",
                torch.empty(
                    self.n_local_experts,
                    2 * intermediate,
                    args.dim // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w13_scale",
                torch.empty(
                    self.n_local_experts,
                    2 * intermediate,
                    args.dim // fp4_block_size,
                    dtype=torch.float8_e8m0fnu,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w2_weight",
                torch.empty(
                    self.n_local_experts,
                    args.dim,
                    intermediate // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w2_scale",
                torch.empty(
                    self.n_local_experts,
                    args.dim,
                    intermediate // fp4_block_size,
                    dtype=torch.float8_e8m0fnu,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_output_scales",
                torch.ones(self.n_local_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_fused_clamp_limits",
                torch.full(
                    (self.n_local_experts,),
                    args.swiglu_limit,
                    dtype=torch.float32,
                ),
                persistent=False,
            )

        def expert_storage(expert_id: int):
            if not self._fused_fp4:
                return None
            local = expert_id - self.experts_start_idx
            split = intermediate
            return (
                (
                    self._fused_w13_weight[local, split:],
                    self._fused_w13_scale[local, split:],
                ),
                (
                    self._fused_w2_weight[local],
                    self._fused_w2_scale[local],
                ),
                (
                    self._fused_w13_weight[local, :split],
                    self._fused_w13_scale[local, :split],
                ),
            )

        # ModuleList supports None placeholders at runtime for unowned experts.
        self.experts = cast(
            Sequence[Expert | None],
            nn.ModuleList(
                cast(
                    Iterable[nn.Module],
                    [
                        Expert(
                            args.dim,
                            intermediate,
                            dtype=expert_dtype,
                            swiglu_limit=args.swiglu_limit,
                            storage=expert_storage(i),
                        )
                        if self.experts_start_idx <= i < self.experts_end_idx
                        else None
                        for i in range(self.n_routed_experts)
                    ],
                )
            ),
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("routed expert weights must be prepared before inference")


class MoE(nn.Module):
    """Architectural gating, routed execution and shared-expert composition."""

    def __init__(self, source: MoEWeights, routed: nn.Module):
        super().__init__()
        self.dim = source.dim
        self.gate = source.gate
        self.shared_experts = source.shared_experts
        self.routed = routed
        self.decode_overlap_stream = source.decode_overlap_stream

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        main_stream = None
        shared = None
        if self.decode_overlap_stream is not None:
            # The shared expert does not depend on routing, so decode can hide
            # it behind the routed experts and their tensor-parallel reduction.
            main_stream = torch.cuda.current_stream(x.device)
            self.decode_overlap_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.decode_overlap_stream):
                shared = self.shared_experts(x)

        weights, indices = self.gate(x, input_ids.flatten())
        y = self.routed(x, indices, weights)
        if main_stream is None:
            shared = self.shared_experts(x)
        else:
            assert self.decode_overlap_stream is not None
            main_stream.wait_stream(self.decode_overlap_stream)
        assert shared is not None
        y += shared
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""

    attention_cls = Attention

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        *,
        tensor_parallel: TensorParallelContext | None = None,
        expert_parallel: TensorParallelLayout | None = None,
        expert_tensor_parallel: TensorParallelLayout | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args, tensor_parallel=tensor_parallel)
        self.ffn = MoEWeights(
            layer_id,
            args,
            expert_parallel=expert_parallel,
            expert_tensor_parallel=expert_tensor_parallel,
        )
        self.prefill_ffn: Callable | None = None
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with use_default_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre_graph(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
    ):
        """Run the small-token HC projection and fused pre-normalization."""

        outer_shape = x.shape[:-2]
        num_tokens = math.prod(outer_shape)
        hc_hidden_size = x.shape[-2] * x.shape[-1]
        residual = x.reshape(num_tokens, x.shape[-2], x.shape[-1]).contiguous()
        flat = residual.reshape(num_tokens, hc_hidden_size)
        dot_mix = torch.empty(
            (_MHC_PRE_NUM_SPLITS, num_tokens, hc_fn.shape[0]),
            dtype=torch.float32,
            device=x.device,
        )
        sqrsum = torch.empty(
            (_MHC_PRE_NUM_SPLITS, num_tokens),
            dtype=torch.float32,
            device=x.device,
        )
        tf32_hc_prenorm_gemm(
            flat,
            hc_fn,
            dot_mix,
            sqrsum,
            num_splits=_MHC_PRE_NUM_SPLITS,
        )
        post, comb, layer_input = mhc_pre_norm(
            dot_mix,
            sqrsum,
            residual,
            hc_scale,
            hc_base,
            norm_weight,
            rms_epsilon=self.norm_eps,
            norm_epsilon=self.norm_eps,
            pre_epsilon=self.hc_eps,
            sinkhorn_epsilon=self.hc_eps,
            sinkhorn_iterations=self.hc_sinkhorn_iters,
        )
        return (
            layer_input.reshape(*outer_shape, x.shape[-1]),
            post.reshape(*outer_shape, x.shape[-2]),
            comb.reshape(*outer_shape, x.shape[-2], x.shape[-2]),
        )

    def forward_attention(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run the shared small-token HC/attention/FFN block skeleton."""

        residual = x
        x, post, comb = self.hc_pre_graph(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
        )
        x = attention(x)
        x = mhc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre_graph(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight,
        )
        x = (self.prefill_ffn or self.ffn)(x, input_ids)
        return mhc_post(x, residual, post, comb)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class ParallelHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        part_vocab_size = vocab_size // self.tensor_parallel.size
        # The checkpoint stores lm_head in BF16; FP32 accumulation keeps the
        # final sampling logits stable across tensor-parallel ranks.
        self.weight = nn.Parameter(
            torch.empty(part_vocab_size, dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, *, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        if self.tensor_parallel.enabled:
            all_logits = [
                torch.empty_like(logits) for _ in range(self.tensor_parallel.size)
            ]
            dist.all_gather(
                all_logits, logits, group=self.tensor_parallel.process_group
            )
            logits = torch.cat(all_logits, dim=-1)
        return logits


class DSparkAttention(Attention):
    """Dense-window attention for one parallel DSpark draft block."""

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__(layer_id, args, tensor_parallel=tensor_parallel)
        if self.compress_ratio:
            raise ValueError("DSpark layers must use uncompressed attention")

    def prefill_main(self, main_x: torch.Tensor, *, cache: DeviceCache) -> None:
        """Populate the draft stage's rolling target-hidden cache."""

        batch_size, sequence_length, _ = main_x.shape
        start = max(0, sequence_length - self.window_size)
        positions = torch.arange(start, sequence_length, device=main_x.device)
        kv_norm_rope_store(
            self.wkv(main_x[:, start:]),
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache.latents.view(-1, self.head_dim),
            positions.view(1, -1).expand(batch_size, -1),
            self.kv_norm.eps,
            cache_mapping=cache.window_mapping(self.layer_id),
        )

    def store_main_block(
        self,
        main_x: torch.Tensor,
        position: torch.Tensor,
        *,
        cache: DeviceCache,
    ) -> None:
        """Commit a block; later unaccepted slots remain logically invisible."""

        block_length = main_x.shape[1]
        positions = position[:, None] + torch.arange(block_length, device=main_x.device)
        kv_norm_rope_store(
            self.wkv(main_x),
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            cache.latents.view(-1, self.head_dim),
            positions,
            self.kv_norm.eps,
            cache_mapping=cache.window_mapping(self.layer_id),
        )

    def forward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        batch_size, draft_length, _ = x.shape
        draft_positions = position[:, None] + torch.arange(
            draft_length, device=x.device
        )
        frequencies = self.freqs_cis[draft_positions]
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -self.rope_head_dim :], frequencies)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], frequencies)
        act_quant(
            kv[..., : -self.rope_head_dim],
            64,
            scale_fmt,
            scale_dtype,
            True,
        )

        indices, sparse_lengths, sequence_lengths = dspark_attention_metadata(
            position,
            draft_length,
            self.window_size,
            self.window_capacity,
            window_mapping=cache.window_mapping(self.layer_id),
        )
        if self.attention_workspace is None:
            raise RuntimeError("DSpark attention workspace is not initialized")
        o = sparse_verify_attention(
            q,
            cache.latents,
            kv.contiguous(),
            self.attn_sink,
            indices,
            sparse_lengths,
            sequence_lengths,
            self.attention_workspace,
            self.softmax_scale,
            enable_pdl=False,
        )
        apply_rotary_emb(o[..., -self.rope_head_dim :], frequencies, True)
        o = o.view(batch_size, draft_length, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = grouped_bf16_linear(o, wo_a)
        return self.wo_b(o.flatten(2))


class DSparkMarkovHead(nn.Module):
    """Low-rank autoregressive correction over parallel draft logits."""

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.markov_w1 = ParallelEmbedding(
            vocab_size, markov_rank, tensor_parallel=tensor_parallel
        )
        self.markov_w2 = ParallelHead(
            vocab_size, markov_rank, tensor_parallel=tensor_parallel
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids), full_logits=True)


class DSparkBlock(Block):
    """One checkpoint DSpark stage under the ``mtp.*`` namespace."""

    attn: DSparkAttention
    attention_cls = DSparkAttention

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        *,
        tensor_parallel: TensorParallelContext | None = None,
        expert_parallel: TensorParallelLayout | None = None,
        expert_tensor_parallel: TensorParallelLayout | None = None,
    ):
        super().__init__(
            layer_id,
            args,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            expert_tensor_parallel=expert_tensor_parallel,
        )
        self.stage_id = layer_id - args.n_layers
        if self.stage_id == 0:
            if not args.dspark_target_layer_ids:
                raise ValueError("DSpark requires target hidden layers")
            self.main_proj = Linear(
                args.dim * len(args.dspark_target_layer_ids), args.dim
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if self.stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size,
                args.dspark_markov_rank,
                tensor_parallel=tensor_parallel,
            )
            hc_dim = self.hc_mult * args.dim
            with use_default_dtype(torch.float32):
                self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        return self.forward_attention(
            x,
            input_ids,
            lambda layer_input: self.attn(layer_input, position, cache=cache),
        )


class Transformer(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Attention TP and expert ownership are bound independently at construction."""

    _attention_workspace: torch.Tensor

    def __init__(
        self,
        args: ModelArgs,
        *,
        tensor_parallel: TensorParallelContext | None = None,
        parallel_groups: ParallelGroups | None = None,
    ):
        global default_dtype, scale_fmt, scale_dtype
        if tensor_parallel is None and parallel_groups is not None:
            group = parallel_groups.attn_tp
            tensor_parallel = TensorParallelContext(
                TensorParallelLayout(rank=group.local_rank, size=group.size),
                group.handle if group.size > 1 else None,
            )
        tensor_parallel = tensor_parallel or _default_tensor_parallel()
        expert_parallel = (
            TensorParallelLayout(
                rank=parallel_groups.ep.local_rank, size=parallel_groups.ep.size
            )
            if parallel_groups is not None
            else tensor_parallel.layout
        )
        expert_tensor_parallel = (
            TensorParallelLayout(
                rank=parallel_groups.expert_tp.local_rank,
                size=parallel_groups.expert_tp.size,
            )
            if parallel_groups is not None
            else TensorParallelLayout()
        )
        # Validate quantization boundaries before any model storage is allocated.
        args.expert_intermediate_shard(expert_tensor_parallel)
        expert_parallel.even_shard(args.n_routed_experts, name="n_routed_experts")
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = (
            torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        )
        super().__init__()
        self.parallel_groups = parallel_groups
        self.embed = ParallelEmbedding(
            args.vocab_size, args.dim, tensor_parallel=tensor_parallel
        )
        self.layers = cast(
            Sequence[Block],
            torch.nn.ModuleList(
                Block(
                    layer_id,
                    args,
                    tensor_parallel=tensor_parallel,
                    expert_parallel=expert_parallel,
                    expert_tensor_parallel=expert_tensor_parallel,
                )
                for layer_id in range(args.n_layers)
            ),
        )
        self.mtp = cast(
            Sequence[DSparkBlock],
            torch.nn.ModuleList(
                DSparkBlock(
                    args.n_layers + stage_id,
                    args,
                    tensor_parallel=tensor_parallel,
                    expert_parallel=expert_parallel,
                    expert_tensor_parallel=expert_tensor_parallel,
                )
                for stage_id in range(args.n_mtp_layers)
            ),
        )
        self.dspark_block_size = args.dspark_block_size
        self.dspark_noise_token_id = args.dspark_noise_token_id
        self.target_layer_ids = frozenset(
            args.dspark_target_layer_ids if self.mtp else ()
        )
        self.register_buffer(
            "_attention_workspace",
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8),
            persistent=False,
        )
        self._decode_overlap_stream = torch.cuda.Stream()
        for layer in (*self.layers, *self.mtp):
            layer.attn.attention_workspace = self._attention_workspace
            layer.attn.decode_overlap_stream = self._decode_overlap_stream
            layer.ffn.decode_overlap_stream = self._decode_overlap_stream
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelHead(
            args.vocab_size, args.dim, tensor_parallel=tensor_parallel
        )
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with use_default_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def prepare_weights(self, reducer: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Lower loaded expert containers into local kernels and communication."""

        groups = getattr(self, "parallel_groups", None)
        prepare_model(
            self,
            lambda source: (
                MoE(
                    source,
                    RoutedExperts(
                        source,
                        reduce=reducer,
                        block_size=block_size,
                        scale_fmt=scale_fmt,
                        scale_dtype=scale_dtype,
                        groups=groups,
                    ),
                )
                if isinstance(source, MoEWeights)
                else source
            ),
        )

    def forward_experts_empty(
        self, *, stage: Literal["target", "draft"] = "target"
    ) -> None:
        """Join one EP model round without attention, sampling, or cache writes.

        Other attention-DP replicas may have work while this replica is idle.
        Its experts still receive and execute their tokens in every layer.
        Speculative decode calls the draft stages first, then the target
        verification layers. Prefill only enters the target layers.
        """

        if self.parallel_groups is None:
            raise RuntimeError("empty expert rounds require explicit parallel groups")
        if stage not in ("target", "draft"):
            raise ValueError("expert round stage must be target or draft")
        layers = self.layers if stage == "target" else self.mtp
        hidden = self.embed.weight.new_empty((0, self.embed.dim))
        for layer in layers:
            ffn = cast(MoE, layer.ffn)
            indices = torch.empty(
                (0, ffn.gate.topk), device=hidden.device, dtype=torch.int32
            )
            weights = hidden.new_empty(indices.shape)
            ffn.routed(hidden, indices, weights)

    def set_allreduce(
        self,
        reducer: Callable[[torch.Tensor], torch.Tensor],
        *,
        indexer_reducer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Install TP reduction on embeddings, projections and indexers.

        Routed experts receive their reduction policy during weight preparation.
        """

        collective_types = (ParallelEmbedding, RowParallelLinear, Indexer)
        for module in self.modules():
            if isinstance(module, collective_types):
                module.allreduce = (
                    indexer_reducer
                    if isinstance(module, Indexer) and indexer_reducer is not None
                    else reducer
                )

    def bind_cache_arena(self, arena: CacheArena) -> None:
        """Attach compressor fields to their packed physical parent views."""

        for layer in self.layers:
            attention = layer.attn
            compressor = getattr(attention, "compressor", None)
            if compressor is None:
                continue
            compressor.kv_state, compressor.score_state = arena.state_tensors(
                layer.layer_id
            )
            compressor.freqs_cis = attention.freqs_cis
            indexer = getattr(attention, "indexer", None)
            if indexer is not None:
                (
                    indexer.compressor.kv_state,
                    indexer.compressor.score_state,
                ) = arena.state_tensors(layer.layer_id, indexer=True)
                indexer.compressor.freqs_cis = attention.freqs_cis
                indexer.freqs_cis = attention.freqs_cis

    def _forward_layers(
        self,
        input_ids: torch.Tensor,
        forward_layer: Callable[[Block, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the shared embedding/layer/hidden-state collection skeleton."""

        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        for layer_id, layer in enumerate(self.layers):
            h = forward_layer(layer, h)
            if layer_id in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
        h = self.layers[-1].hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return h, main_hidden

    def _target_output(
        self,
        hidden: torch.Tensor,
        main_hidden: torch.Tensor | None,
        *,
        full_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.head(self.norm(hidden), full_logits=full_logits), main_hidden

    @torch.inference_mode()
    def forward_decode(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Decode one token using only graph-staged position data."""

        logits, _ = self.forward_decode_target(
            input_ids, position, graph_seq_len, cache=cache
        )
        return logits

    @torch.inference_mode()
    def forward_decode_target(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode target tokens and expose DSpark's selected hidden states."""

        h, main_hidden = self._forward_layers(
            input_ids,
            lambda layer, hidden: layer.forward_attention(
                hidden,
                input_ids,
                lambda layer_input: layer.attn.forward_decode(
                    layer_input, position, graph_seq_len, cache=cache
                ),
            ),
        )
        return self._target_output(h, main_hidden)

    @torch.inference_mode()
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        start_position: int,
        *,
        cache: DeviceCache,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prefill one cached suffix and return only its final-token logits."""

        token_positions = start_position + torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).view(1, -1)
        sequence_length = start_position + input_ids.shape[1]
        groups = self.parallel_groups
        context_parallel = (
            ContextParallelTokens(
                (input_ids.shape[1],),
                groups.cp,
                device=input_ids.device,
                distributed=groups.distributed,
            )
            if groups is not None and groups.cp.size > 1
            else None
        )
        with dynamic_gemm_rows():
            h, main_hidden = self._forward_layers(
                input_ids,
                lambda layer, hidden: layer.forward_attention(
                    hidden,
                    input_ids,
                    lambda layer_input: layer.attn.forward_prefill(
                        layer_input,
                        token_positions,
                        start_position,
                        sequence_length,
                        cache=cache,
                        context_parallel=context_parallel,
                    ),
                ),
            )
            if return_hidden:
                return self.norm(h), main_hidden
            return self._target_output(h, main_hidden)

    @torch.inference_mode()
    def forward_verify(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
        *,
        cache: DeviceCache,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one fixed target verification block."""

        token_positions = position[:, None] + torch.arange(
            input_ids.shape[1], device=input_ids.device
        )
        h, main_hidden = self._forward_layers(
            input_ids,
            lambda layer, hidden: layer.forward_attention(
                hidden,
                input_ids,
                lambda layer_input: layer.attn.forward_verify(
                    layer_input, token_positions, graph_seq_len, cache=cache
                ),
            ),
        )
        return self._target_output(h, main_hidden, full_logits=True)

    @torch.inference_mode()
    def clear_rejected_speculative_state(
        self,
        positions: torch.Tensor,
        processed_counts: torch.Tensor,
        block_length: int,
        *,
        cache: DeviceCache,
    ) -> None:
        """Discard rejected recurrent rows after engine-level acceptance."""

        for layer in self.layers:
            compressor = getattr(layer.attn, "compressor", None)
            if compressor is not None and compressor.compress_ratio == 128:
                compressor.clear_rejected(
                    positions,
                    processed_counts,
                    block_length,
                    state_mapping=cache.state_mapping(C128_STATE_GROUP),
                )

    def _project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        if not self.mtp:
            raise RuntimeError("this checkpoint runner has no DSpark stages")
        first = self.mtp[0]
        return first.main_norm(first.main_proj(main_hidden))

    @torch.inference_mode()
    def prefill_dspark(self, main_hidden: torch.Tensor, *, cache: DeviceCache) -> None:
        with dynamic_gemm_rows():
            main_x = self._project_main(main_hidden)
            for layer in self.mtp:
                layer.attn.prefill_main(main_x, cache=cache)

    @torch.inference_mode()
    def commit_dspark_main_block(
        self,
        main_hidden: torch.Tensor,
        position: torch.Tensor,
        *,
        cache: DeviceCache,
    ) -> None:
        main_x = self._project_main(main_hidden)
        for layer in self.mtp:
            layer.attn.store_main_block(main_x, position, cache=cache)

    @torch.inference_mode()
    def draft(
        self,
        anchor_ids: torch.Tensor,
        position: torch.Tensor,
        *,
        cache: DeviceCache,
    ) -> torch.Tensor:
        """Draft a fixed greedy block from one target anchor per batch row."""

        if not self.mtp or self.dspark_block_size <= 0:
            raise RuntimeError("DSpark is not configured")
        input_ids = anchor_ids.new_full(
            (anchor_ids.shape[0], self.dspark_block_size),
            self.dspark_noise_token_id,
        )
        input_ids[:, 0] = anchor_ids.flatten()
        h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.mtp:
            h = layer(h, position, input_ids, cache=cache)

        last = self.mtp[-1]
        hidden = last.hc_head(h, last.hc_head_fn, last.hc_head_scale, last.hc_head_base)
        logits = self.head(last.norm(hidden), full_logits=True)
        previous = anchor_ids.flatten()
        output = []
        for index in range(self.dspark_block_size):
            biased = logits[:, index] + last.markov_head(previous)
            previous = biased.argmax(dim=-1)
            output.append(previous)
        return torch.stack(output, dim=1)
