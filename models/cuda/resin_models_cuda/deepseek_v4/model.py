# Adapted from DeepSeek-V4-Flash-0731 (MIT).
"""Resin-owned DeepSeek-V4 target-model hierarchy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from deep_gemm import tf32_hc_prenorm_gemm
from flashinfer.mhc import mhc_post
from flashinfer.topk import top_k as flashinfer_top_k
from torch import nn

from resin.backend.torch import use_default_dtype
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
    moe_act_quant,
    qk_norm_rope_block,
    qk_norm_rope_store,
    rope_last_block_inplace,
    swiglu,
    top6_sqrt_softplus,
    verify_attention_metadata,
)
from resin_models_cuda.deepseek_v4.gemm import (
    dynamic_gemm_rows,
    fp4_fused_moe,
    fp4_gemm,
    fp8_gemm,
    fp8_gemm_uses_packed_scales,
    prepare_fp4_fused_moe_weights,
)
from resin_models_cuda.deepseek_v4.tilelang_kernels import mhc_pre_norm

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
_MHC_PRE_NUM_SPLITS = 64


def _distributed_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Default in-place tensor-parallel reduction."""

    if world_size > 1:
        dist.all_reduce(tensor)
    return tensor


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
    compress_ratios: tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
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


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, (
            f"Vocabulary size must be divisible by world size (world_size={world_size})"
        )
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = _distributed_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
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
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        rows = x.numel() // x.shape[-1]
        x, s = act_quant(
            x,
            block_size,
            scale_fmt,
            scale_dtype,
            packed_scales=fp8_gemm_uses_packed_scales(rows),
        )
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
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
            self.weight.scale = self.scale = nn.Parameter(scale_storage)
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
            self.weight.scale = self.scale = nn.Parameter(scale_storage)
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
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert out_features % world_size == 0, (
            f"Output features must be divisible by world size (world_size={world_size})"
        )
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert in_features % world_size == 0, (
            f"Input features must be divisible by world size (world_size={world_size})"
        )
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = _distributed_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
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
                math.ceil(
                    max(logical_window, compress_ratio + args.dspark_block_size + 1)
                    / compress_ratio
                )
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


def _compact_topk_indices(scores: torch.Tensor, width: int) -> torch.Tensor:
    """Keep finite cache candidates ahead of graph-padding entries."""

    _, indices = flashinfer_top_k(scores, width, sorted=True)
    return indices


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, layer_id: int, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(
            self.dim, self.n_heads, dtype=torch.bfloat16
        )
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = _distributed_sum

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
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rope_dim:], self.freqs_cis[positions])
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        weights = self.weights_proj(x)
        width = math.ceil(sequence_length / ratio / 64) * 64
        if width == 0:
            return torch.empty(
                (batch_size, block_length, 0),
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
        if world_size > 1:
            index_score = self.allreduce(index_score)
        topk_width = min(self.index_topk, width)
        topk_indices = torch.topk(index_score, topk_width, dim=-1, sorted=True).indices
        complete = (positions + 1).div(ratio, rounding_mode="floor")
        logical = torch.where(topk_indices < complete.unsqueeze(-1), topk_indices, -1)
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
            return torch.empty(
                (bsz, block_length, 0), dtype=torch.int32, device=x.device
            )

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
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
        if world_size > 1:
            index_score = self.allreduce(index_score)
        topk_width = min(self.index_topk, width)
        topk_idxs = _compact_topk_indices(index_score.flatten(0, 1), topk_width)
        topk_idxs = topk_idxs.view(*index_score.shape[:2], topk_width)
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

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
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
            self._fused_wqkv_weight.scale = self._fused_wqkv_scale
            self.wq_a = Linear(
                self.dim,
                self.q_lora_rank,
                weight_storage=self._fused_wqkv_weight[: self.q_lora_rank],
                scale_storage=self._fused_wqkv_scale[: self.q_lora_rank // block_size],
            )
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
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
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(layer_id, args, self.compress_ratio)
            else:
                self.indexer = None

        self.attention_workspace: torch.Tensor | None = None
        self.decode_overlap_stream: torch.cuda.Stream | None = None
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
                    generated_compressed_width = compressed_width

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
    ) -> torch.Tensor:
        """Prefill one cached suffix with paged sparse block attention."""

        batch_size, block_length, _ = x.shape
        ratio = self.compress_ratio
        if token_positions.shape != (batch_size, block_length):
            raise ValueError("prefill positions must match the token block")
        self._prepare_compression()
        qr, kv = self._project(x)

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

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
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
                )
            else:
                generated_compressed_width = math.ceil(sequence_length / ratio / 4) * 4

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
        return self._finish(o, token_positions)

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

        compressed_width = 0
        if ratio:
            padded_width = math.ceil(graph_seq_len / ratio / 4) * 4
            compressed_width = padded_width

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
                generated_compressed_width = compressed_width

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
        if self.hash and self.score_func == "sqrtsoftplus" and self.topk == 6:
            return hash_sqrt_softplus(
                scores,
                input_ids,
                self.tid2eid,
                self.route_scale,
            )
        if (
            (scores.shape[-1] & (scores.shape[-1] - 1)) == 0
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
        if self.hash:
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
                self._fused_w13_weight.scale = self._fused_w13_scale
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


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // world_size experts."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, (
            f"Number of experts must be divisible by world size (world_size={world_size})"
        )
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.swiglu_limit = args.swiglu_limit
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = _distributed_sum
        self.decode_overlap_stream: torch.cuda.Stream | None = None
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self._fused_fp4 = expert_dtype == torch.float4_e2m1fn_x2
        if self._fused_fp4:
            self.register_buffer(
                "_fused_w13_weight",
                torch.empty(
                    self.n_local_experts,
                    2 * args.moe_inter_dim,
                    args.dim // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w13_scale",
                torch.empty(
                    self.n_local_experts,
                    2 * args.moe_inter_dim,
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
                    args.moe_inter_dim // 2,
                    dtype=expert_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_fused_w2_scale",
                torch.empty(
                    self.n_local_experts,
                    args.dim,
                    args.moe_inter_dim // fp4_block_size,
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
            split = args.moe_inter_dim
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

        self.experts = nn.ModuleList(
            [
                Expert(
                    args.dim,
                    args.moe_inter_dim,
                    dtype=expert_dtype,
                    swiglu_limit=args.swiglu_limit,
                    storage=expert_storage(i),
                )
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    def _fused_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        quantized, scales = moe_act_quant(x, block_size, scale_fmt, scale_dtype)
        output = fp4_fused_moe(
            quantized,
            scales,
            indices,
            weights,
            self._fused_w13_weight,
            self._fused_w13_scale,
            self._fused_w2_weight,
            self._fused_w2_scale,
            self._fused_output_scales,
            self._fused_clamp_limits,
            num_experts=self.n_routed_experts,
            local_expert_offset=self.experts_start_idx,
            scale_dtype=scale_dtype,
        )
        return output

    def prepare_fused_weights(self) -> None:
        """Put loaded FP4 checkpoint weights in FlashInfer's runtime layout."""

        if self._fused_fp4:
            prepare_fp4_fused_moe_weights(
                self._fused_w13_weight,
                self._fused_w13_scale,
                self._fused_w2_weight,
                self._fused_w2_scale,
            )

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
        if self._fused_fp4:
            y = self._fused_experts(x, weights, indices)
        else:
            y = torch.zeros_like(x, dtype=torch.bfloat16)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
        if world_size > 1:
            y = self.allreduce(y)
        if main_stream is None:
            shared = self.shared_experts(x)
        else:
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

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
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
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        part_vocab_size = vocab_size // world_size
        # The checkpoint stores lm_head in BF16; FP32 accumulation keeps the
        # final sampling logits stable across tensor-parallel ranks.
        self.weight = nn.Parameter(
            torch.empty(part_vocab_size, dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, *, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


class DSparkAttention(Attention):
    """Dense-window attention for one parallel DSpark draft block."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
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

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = ParallelEmbedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelHead(vocab_size, markov_rank)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids), full_logits=True)


class DSparkBlock(Block):
    """One checkpoint DSpark stage under the ``mtp.*`` namespace."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
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
                args.vocab_size, args.dspark_markov_rank
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
    Sets global state (world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__."""

    def __init__(self, args: ModelArgs):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = (
            torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        )
        super().__init__()
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.mtp = torch.nn.ModuleList(
            DSparkBlock(args.n_layers + stage_id, args)
            for stage_id in range(args.n_mtp_layers)
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
        self.head = ParallelHead(args.vocab_size, args.dim)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with use_default_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def set_allreduce(
        self,
        reducer: Callable[[torch.Tensor], torch.Tensor],
        *,
        indexer_reducer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Install one runner-owned TP reduction policy on every collective."""

        collective_types = (ParallelEmbedding, RowParallelLinear, Indexer, MoE)
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prefill one cached suffix and return only its final-token logits."""

        token_positions = start_position + torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).view(1, -1)
        sequence_length = start_position + input_ids.shape[1]
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
                    ),
                ),
            )
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
