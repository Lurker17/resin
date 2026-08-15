# Adapted from DeepSeek-V4-Flash-0731 (MIT).
"""Resin-owned DeepSeek-V4 target-model hierarchy."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flashinfer.mhc import mhc_post, mhc_pre_big_fuse_with_prenorm
from quack import rmsnorm as quack_rmsnorm
from torch import nn

from resin.models.deepseek_v4.attention import (
    sparse_decode_attention,
    sparse_topk_lengths,
)
from resin.models.deepseek_v4.cutile import (
    act_quant,
    compress_decode,
    fp4_act_quant,
    hadamard_transform,
    qk_norm_rope_store,
    sparse_attn,
    top6_sqrt_softplus,
)
from resin.models.deepseek_v4.gemm import (
    fp4_fused_moe,
    fp4_gemm,
    fp8_gemm,
    prepare_fp4_fused_moe_weights,
)

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


def _distributed_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Default in-place tensor-parallel reduction."""

    if world_size > 1:
        dist.all_reduce(tensor)
    return tensor


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


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
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
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
    dspark_target_layer_ids: Tuple[int, ...] = ()
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
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)


def grouped_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run independent per-group output projections."""

    return torch.einsum("bsgd,grd->bsgr", x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        *,
        weight_storage: Optional[torch.Tensor] = None,
        scale_storage: Optional[torch.Tensor] = None,
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
            y = y.float()
            y = self.allreduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        if x.is_cuda:
            return quack_rmsnorm(x, self.weight, eps=self.eps)
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


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


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


@lru_cache(2)
def get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_decode_window_topk_idxs(
    window_size: int, position: torch.Tensor
) -> torch.Tensor:
    """Build one fixed-width ring-cache row from a staged device position."""

    logical = torch.arange(
        window_size, dtype=torch.long, device=position.device
    ).unsqueeze(0)
    start = torch.where(
        position >= window_size - 1,
        (position + 1).remainder(window_size),
        torch.zeros_like(position),
    ).unsqueeze(1)
    physical = (logical + start).remainder(window_size)
    # The fixed-width row must not expose cache slots beyond a short prefix.
    # FlashInfer also receives the logical length, but DSpark's parallel draft
    # attention consumes these indices directly.
    physical = torch.where(logical <= position.unsqueeze(1), physical, -1)
    return physical.to(torch.int32).unsqueeze(1)


def get_decode_compress_topk_idxs(
    ratio: int,
    position: torch.Tensor,
    width: int,
    offset: int,
) -> torch.Tensor:
    """Build a graph-static compressed-cache row and mask unfinished groups."""

    logical = torch.arange(width, dtype=torch.long, device=position.device).unsqueeze(0)
    complete = (position + 1).div(ratio, rounding_mode="floor").unsqueeze(1)
    return (
        torch.where(logical < complete, logical + offset, -1)
        .to(torch.int32)
        .unsqueeze(1)
    )


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
        # The checkpoint stores both projections in BF16; the released model computes
        # them in FP32. Keep their public parameter views in one allocation so a
        # shared-input decode can use one GEMM.
        projection_width = coff * self.head_dim
        self.register_buffer(
            "_fused_projection_weight",
            torch.empty(2 * projection_width, self.dim, dtype=torch.float32),
            persistent=False,
        )
        self.wkv = Linear(
            self.dim,
            projection_width,
            dtype=torch.float32,
            weight_storage=self._fused_projection_weight[:projection_width],
        )
        self.wgate = Linear(
            self.dim,
            projection_width,
            dtype=torch.float32,
            weight_storage=self._fused_projection_weight[projection_width:],
        )
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        # compression need fp32
        x = x.float()
        kv, score = F.linear(x, self._fused_projection_weight).chunk(2, dim=-1)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

    def forward_decode(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Update recurrent compression state from a graph-staged position."""

        assert self.kv_cache is not None
        values = x.float()
        kv, score = F.linear(values, self._fused_projection_weight).chunk(2, dim=-1)
        compress_decode(
            kv,
            score,
            self.ape,
            self.kv_state[: x.shape[0]],
            self.score_state[: x.shape[0]],
            self.norm.weight,
            torch.view_as_real(self.freqs_cis),
            self.kv_cache[: x.shape[0]],
            position,
            self.compress_ratio,
            self.rotate,
            self.norm.eps,
        )
        return kv[..., : self.head_dim]


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
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
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim
            ),
            persistent=False,
        )
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4 simulation for q and kv in indexer
        fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            index_score = self.allreduce(index_score)
        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs

    def forward_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        position: torch.Tensor,
        offset: int,
        graph_seq_len: int,
    ) -> torch.Tensor:
        """Select compressed positions with graph-static score geometry."""

        bsz, _, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        self.compressor.forward_decode(x, position)

        width = min(self.kv_cache.shape[1], graph_seq_len // ratio)
        if width == 0:
            return torch.empty((bsz, 1, 0), dtype=torch.int32, device=x.device)

        freqs_cis = self.freqs_cis.index_select(0, position)
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        cache = self.kv_cache[:bsz, :width]
        index_score = torch.einsum("bshd,btd->bsht", q, cache)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        complete = (position + 1).div(ratio, rounding_mode="floor")
        valid = torch.arange(width, device=x.device).unsqueeze(0) < complete.unsqueeze(
            1
        )
        index_score = index_score.masked_fill(~valid.unsqueeze(1), float("-inf"))
        if world_size > 1:
            index_score = self.allreduce(index_score)
        topk_width = min(self.index_topk, width)
        topk_idxs = index_score.topk(topk_width, dim=-1)[1]
        return torch.where(
            topk_idxs < complete.view(bsz, 1, 1),
            topk_idxs + offset,
            -1,
        ).to(torch.int32)


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
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )
        if self.compress_ratio:
            self.register_buffer(
                "compressed_kv_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.head_dim,
                ),
                persistent=False,
            )
        else:
            self.compressed_kv_cache = None
        self.attention_workspace: torch.Tensor = None
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

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.compressed_kv_cache
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        if self._fused_wqkv:
            qkv = linear(x, self._fused_wqkv_weight)
            q_input, kv = qkv.split((self.q_lora_rank, self.head_dim), dim=-1)
        else:
            q_input, kv = self.wq_a(x), self.wkv(x)
        qr = q = self.q_norm(q_input)
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset).int()
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
                attention_cache = torch.cat(
                    (self.kv_cache[:bsz], self.compressed_kv_cache[:bsz]), dim=1
                )
            else:
                attention_cache = self.kv_cache[:bsz]
            o = sparse_attn(
                q, attention_cache, self.attn_sink, topk_idxs, self.softmax_scale
            )
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # BF16 einsum is faster than the available tiny-M grouped FP8 paths.
        o = grouped_bf16_linear(o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x

    def forward_decode(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
    ) -> torch.Tensor:
        """Run one reusable fixed-geometry decode attention step."""

        bsz, _, _ = x.size()
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.compressed_kv_cache
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        if self._fused_wqkv:
            qkv = linear(x, self._fused_wqkv_weight)
            q_input, kv = qkv.split((self.q_lora_rank, self.head_dim), dim=-1)
        else:
            q_input, kv = self.wq_a(x), self.wkv(x)
        qr = q = self.q_norm(q_input)

        compressed_width = 0
        compress_topk_idxs = None
        main_stream = None
        if ratio:
            assert self.compressed_kv_cache is not None
            compressed_width = min(
                self.compressed_kv_cache.shape[1],
                math.ceil(graph_seq_len / ratio / 4) * 4,
            )
            assert self.decode_overlap_stream is not None
            # Compression/indexing depends on normalized Q but is otherwise
            # independent of the dense Q projection and window-cache update.
            main_stream = torch.cuda.current_stream(x.device)
            self.decode_overlap_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.decode_overlap_stream):
                if self.indexer is not None:
                    self.compressor.forward_decode(x, position)
                    compress_topk_idxs = self.indexer.forward_decode(
                        x, qr, position, 0, compressed_width * ratio
                    )
                else:
                    self.compressor.forward_decode(x, position)
                    compress_topk_idxs = get_decode_compress_topk_idxs(
                        ratio,
                        position,
                        compressed_width,
                        0,
                    )

        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        qk_norm_rope_store(
            q,
            kv,
            self.kv_norm.weight,
            torch.view_as_real(self.freqs_cis),
            self.kv_cache[:bsz, :win],
            position,
            self.eps,
        )
        topk_idxs = get_decode_window_topk_idxs(win, position)
        cache_rows = None
        if bsz > 1:
            cache_rows = torch.arange(bsz, dtype=torch.int32, device=x.device).view(
                bsz, 1, 1
            )
            # FlashInfer sees each dense sequence row as one page in a shared pool,
            # so sparse indices are physical pool offsets rather than row-local IDs.
            topk_idxs = torch.where(topk_idxs >= 0, topk_idxs + cache_rows * win, -1)

        compressed_complete = torch.zeros_like(position)
        compressed_index_width = 0
        if ratio:
            compressed_complete = torch.minimum(
                (position + 1).div(ratio, rounding_mode="floor"),
                torch.full_like(position, compressed_width),
            )
            assert main_stream is not None and compress_topk_idxs is not None
            main_stream.wait_stream(self.decode_overlap_stream)
            if cache_rows is not None:
                compress_topk_idxs = torch.where(
                    compress_topk_idxs >= 0,
                    compress_topk_idxs + cache_rows * self.compressed_kv_cache.shape[1],
                    -1,
                )
            compressed_index_width = compress_topk_idxs.shape[-1]
            topk_idxs = torch.cat((topk_idxs, compress_topk_idxs), dim=-1)

        assert self.attention_workspace is not None
        window_cache = self.kv_cache[:bsz, :win]
        compressed_cache = self.compressed_kv_cache[:bsz] if ratio else window_cache
        o = sparse_decode_attention(
            q,
            window_cache,
            compressed_cache,
            self.attn_sink,
            topk_idxs,
            sparse_topk_lengths(compressed_complete, compressed_index_width, win),
            (position + 1).to(torch.int32),
            self.attention_workspace,
            self.softmax_scale,
        )
        freqs_cis = self.freqs_cis.index_select(0, position)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, 1, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = grouped_bf16_linear(o, wo_a)
        return self.wo_b(o.flatten(2))

    def _recurrent_buffers(self) -> tuple[torch.Tensor, ...]:
        if not self.compress_ratio:
            return ()
        buffers = [self.compressor.kv_state, self.compressor.score_state]
        if self.indexer is not None:
            buffers.extend(
                (
                    self.indexer.compressor.kv_state,
                    self.indexer.compressor.score_state,
                )
            )
        return tuple(buffers)

    def forward_verify(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Verify a short causal token block while checkpointing recurrence."""

        batch_size, block_length, _ = x.shape
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.compressed_kv_cache
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        if self._fused_wqkv:
            qkv = linear(x, self._fused_wqkv_weight)
            q_input, kv = qkv.split((self.q_lora_rank, self.head_dim), dim=-1)
        else:
            q_input, kv = self.wq_a(x), self.wkv(x)
        qr = q = self.q_norm(q_input)
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))

        recurrent = self._recurrent_buffers()
        checkpoints = [(buffer, [buffer[:batch_size].clone()]) for buffer in recurrent]
        compressed_width = 0
        if ratio:
            assert self.compressed_kv_cache is not None
            compressed_width = min(
                self.compressed_kv_cache.shape[1],
                math.ceil(graph_seq_len / ratio / 4) * 4,
            )

        outputs = []
        for index in range(block_length):
            token_position = position + index
            compressed_indices = None
            compressed_index_width = 0
            if ratio:
                self.compressor.forward_decode(x[:, index : index + 1], token_position)
                if self.indexer is not None:
                    compressed_indices = self.indexer.forward_decode(
                        x[:, index : index + 1],
                        qr[:, index : index + 1],
                        token_position,
                        0,
                        graph_seq_len,
                    )
                else:
                    compressed_indices = get_decode_compress_topk_idxs(
                        ratio,
                        token_position,
                        compressed_width,
                        0,
                    )
            qk_norm_rope_store(
                q[:, index : index + 1],
                kv[:, index : index + 1],
                self.kv_norm.weight,
                torch.view_as_real(self.freqs_cis),
                self.kv_cache[:batch_size, :win],
                token_position,
                self.eps,
            )
            topk_indices = get_decode_window_topk_idxs(win, token_position)
            cache_rows = None
            if batch_size > 1:
                cache_rows = torch.arange(
                    batch_size, dtype=torch.int32, device=x.device
                ).view(batch_size, 1, 1)
                topk_indices = torch.where(
                    topk_indices >= 0, topk_indices + cache_rows * win, -1
                )

            compressed_complete = torch.zeros_like(token_position)
            if ratio:
                assert compressed_indices is not None
                compressed_complete = torch.minimum(
                    (token_position + 1).div(ratio, rounding_mode="floor"),
                    torch.full_like(token_position, compressed_width),
                )
                if cache_rows is not None:
                    compressed_indices = torch.where(
                        compressed_indices >= 0,
                        compressed_indices
                        + cache_rows * self.compressed_kv_cache.shape[1],
                        -1,
                    )
                compressed_index_width = compressed_indices.shape[-1]
                topk_indices = torch.cat((topk_indices, compressed_indices), dim=-1)

            assert self.attention_workspace is not None
            window_cache = self.kv_cache[:batch_size, :win]
            compressed_cache = (
                self.compressed_kv_cache[:batch_size] if ratio else window_cache
            )
            output = sparse_decode_attention(
                q[:, index : index + 1],
                window_cache,
                compressed_cache,
                self.attn_sink,
                topk_indices,
                sparse_topk_lengths(compressed_complete, compressed_index_width, win),
                (token_position + 1).to(torch.int32),
                self.attention_workspace,
                self.softmax_scale,
            )
            frequencies = self.freqs_cis.index_select(0, token_position)
            apply_rotary_emb(output[..., -rd:], frequencies, True)
            outputs.append(output)
            for buffer, snapshots in checkpoints:
                snapshots.append(buffer[:batch_size].clone())

        o = torch.cat(outputs, dim=1)
        o = o.view(batch_size, block_length, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = grouped_bf16_linear(o, wo_a)
        frozen = [
            (buffer, torch.stack(snapshots, dim=0)) for buffer, snapshots in checkpoints
        ]
        return self.wo_b(o.flatten(2)), frozen


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
        input_ids: Optional[torch.Tensor] = None,
        *,
        optimized: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if (
            optimized
            and scores.is_cuda
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
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype=None,
        swiglu_limit=0,
        storage: Optional[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ]
        ] = None,
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
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        if self._fused_w13:
            gate, up = linear(x, self._fused_w13_weight).chunk(2, dim=-1)
            gate, up = gate.float(), up.float()
        else:
            gate = self.w1(x).float()
            up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
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
        quantized, scales = act_quant(x, block_size, scale_fmt, scale_dtype)
        output = fp4_fused_moe(
            quantized,
            scales,
            indices.to(torch.int32),
            weights.to(torch.bfloat16),
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
        return output.float()

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
        if shape[1] == 1 and self.decode_overlap_stream is not None:
            # The shared expert does not depend on routing, so decode can hide
            # it behind the routed experts and their tensor-parallel reduction.
            main_stream = torch.cuda.current_stream(x.device)
            self.decode_overlap_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.decode_overlap_stream):
                shared = self.shared_experts(x)

        weights, indices = self.gate(x, input_ids.flatten(), optimized=shape[1] == 1)
        if self._fused_fp4:
            y = self._fused_experts(x, weights, indices)
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
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
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        mixes = F.linear(x.flatten(2).float(), hc_fn)
        post, comb, layer_input = mhc_pre_big_fuse_with_prenorm(
            mixes,
            x,
            hc_scale,
            hc_base,
            rms_eps=self.norm_eps,
            mhc_pre_eps=self.hc_eps,
            mhc_sinkhorn_eps=self.hc_eps,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post.squeeze(-1), comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        return mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor],
        *attn_args,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, *attn_args)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def forward_decode(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        input_ids: torch.Tensor,
        graph_seq_len: int,
    ) -> torch.Tensor:
        """Run a target block without host-side position specialization."""

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn.forward_decode(x, position, graph_seq_len)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self.hc_post(x, residual, post, comb)

    def forward_verify(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        input_ids: torch.Tensor,
        graph_seq_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one target layer over a short causal verification block."""

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x, checkpoints = self.attn.forward_verify(x, position, graph_seq_len)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self.hc_post(x, residual, post, comb), checkpoints

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

    def _main_kv(self, main_x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        frequencies = self.freqs_cis[position]
        kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], frequencies)
        act_quant(
            kv[..., : -self.rope_head_dim],
            64,
            scale_fmt,
            scale_dtype,
            True,
        )
        return kv

    def prefill_main(self, main_x: torch.Tensor) -> None:
        """Populate the draft stage's rolling target-hidden cache."""

        batch_size, sequence_length, _ = main_x.shape
        positions = torch.arange(sequence_length, device=main_x.device)
        kv = self._main_kv(main_x, positions)
        if sequence_length <= self.window_size:
            self.kv_cache[:batch_size, :sequence_length] = kv
            return
        cutoff = sequence_length % self.window_size
        tail = kv[:, -self.window_size :]
        self.kv_cache[:batch_size, cutoff:], self.kv_cache[:batch_size, :cutoff] = (
            tail.split([self.window_size - cutoff, cutoff], dim=1)
        )

    def store_main(self, main_x: torch.Tensor, position: torch.Tensor) -> None:
        """Commit one target hidden state at each row's logical position."""

        rows = torch.arange(main_x.shape[0], device=main_x.device)
        self.kv_cache[rows, position.remainder(self.window_size)] = self._main_kv(
            main_x, position
        ).squeeze(1)

    def store_main_block(self, main_x: torch.Tensor, position: torch.Tensor) -> None:
        """Commit a block; later unaccepted slots remain logically invisible."""

        block_length = main_x.shape[1]
        positions = position[:, None] + torch.arange(block_length, device=main_x.device)
        kv = self._main_kv(main_x, positions)
        rows = torch.arange(main_x.shape[0], device=main_x.device)[:, None]
        self.kv_cache[rows, positions.remainder(self.window_size)] = kv

    def forward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, draft_length, _ = x.shape
        self.store_main(main_x, position)

        draft_positions = (
            position[:, None] + 1 + torch.arange(draft_length, device=x.device)
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

        history = get_decode_window_topk_idxs(self.window_size, position).expand(
            -1, draft_length, -1
        )
        draft = (
            (
                self.window_size
                + torch.arange(draft_length, dtype=torch.int32, device=x.device)
            )
            .view(1, 1, -1)
            .expand(batch_size, draft_length, -1)
        )
        indices = torch.cat((history, draft), dim=-1).contiguous()
        o = sparse_attn(
            q,
            torch.cat((self.kv_cache[:batch_size], kv), dim=1),
            self.attn_sink,
            indices,
            self.softmax_scale,
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
            with set_dtype(torch.float32):
                self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, position, main_x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self.hc_post(x, residual, post, comb)


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
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8),
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
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def set_allreduce(self, reducer: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Install one runner-owned TP reduction policy on every collective."""

        collective_types = (ParallelEmbedding, RowParallelLinear, Indexer, MoE)
        for module in self.modules():
            if isinstance(module, collective_types):
                module.allreduce = reducer

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        logits, _ = self.forward_target(input_ids, start_pos)
        return logits

    @torch.inference_mode()
    def forward_target(self, input_ids: torch.Tensor, start_pos: int = 0):
        """Run target prefill and return the hidden states consumed by DSpark."""

        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        for layer_id, layer in enumerate(self.layers):
            h = layer(h, start_pos, input_ids)
            if layer_id in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
        h = self.layers[-1].hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return self.head(self.norm(h)), main_hidden

    @torch.inference_mode()
    def forward_decode(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
    ) -> torch.Tensor:
        """Decode one token using only graph-staged position data."""

        logits, _ = self.forward_decode_target(input_ids, position, graph_seq_len)
        return logits

    @torch.inference_mode()
    def forward_decode_target(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
    ):
        """Decode target tokens and expose DSpark's selected hidden states."""

        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        for layer_id, layer in enumerate(self.layers):
            h = layer.forward_decode(h, position, input_ids, graph_seq_len)
            if layer_id in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
        h = self.layers[-1].hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return self.head(self.norm(h)), main_hidden

    @torch.inference_mode()
    def forward_verify(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        graph_seq_len: int,
    ):
        """Run a fixed target verification block and retain recurrent checkpoints."""

        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_id, layer in enumerate(self.layers):
            h, layer_checkpoints = layer.forward_verify(
                h, position, input_ids, graph_seq_len
            )
            checkpoints.extend(layer_checkpoints)
            if layer_id in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
        h = self.layers[-1].hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return self.head(self.norm(h), full_logits=True), main_hidden, checkpoints

    def _project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        if not self.mtp:
            raise RuntimeError("this checkpoint runner has no DSpark stages")
        first = self.mtp[0]
        return first.main_norm(first.main_proj(main_hidden))

    @torch.inference_mode()
    def prefill_dspark(self, main_hidden: torch.Tensor) -> None:
        main_x = self._project_main(main_hidden)
        for layer in self.mtp:
            layer.attn.prefill_main(main_x)

    @torch.inference_mode()
    def commit_dspark_main_block(
        self, main_hidden: torch.Tensor, position: torch.Tensor
    ) -> None:
        main_x = self._project_main(main_hidden)
        for layer in self.mtp:
            layer.attn.store_main_block(main_x, position)

    @torch.inference_mode()
    def draft(
        self,
        anchor_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Draft a fixed greedy block from one target anchor per batch row."""

        if main_hidden is None or not self.mtp or self.dspark_block_size <= 0:
            raise RuntimeError("DSpark is not configured")
        main_x = self._project_main(main_hidden)
        input_ids = anchor_ids.new_full(
            (anchor_ids.shape[0], self.dspark_block_size),
            self.dspark_noise_token_id,
        )
        input_ids[:, 0] = anchor_ids.flatten()
        h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.mtp:
            h = layer(h, position, input_ids, main_x)

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
