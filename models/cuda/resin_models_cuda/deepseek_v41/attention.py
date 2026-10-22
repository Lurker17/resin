"""Pinned FlashMLA packed KV and prepared fused attention projections."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.mega_moe import prepare_weights
from resin_models_cuda.deepseek_v41.kernels import act_quant, fp4_act_quant, unpack_kv


def pack_kv(value, *, fp4=False):
    data, scales = (
        fp4_act_quant(value, 16, scale_dtype=torch.float8_e4m3fn)
        if fp4
        else act_quant(value, 32, "ue8m0", torch.float8_e8m0fnu)
    )
    return torch.cat((data.view(torch.uint8), scales.view(torch.uint8)), -1)


def store_kv(cache, indices, values):
    indices = indices.flatten().long()
    cache[indices // cache.shape[1], indices % cache.shape[1], 0] = values.reshape(
        -1, values.shape[-1]
    )


def gather_kv(cache, indices):
    if not indices.numel():
        return cache.new_empty(0, 512, dtype=torch.bfloat16)
    return unpack_kv(cache, indices.flatten().int())


def padded_indices(indices):
    return F.pad(indices.int(), (0, -indices.shape[-1] % 8), value=-1).contiguous()


class QueryProjection(nn.Module):
    weight: torch.Tensor
    scale: torch.Tensor

    def __init__(self, source, heads, dim):
        super().__init__()
        from flash_mla import fused_norm_rope_attn_rope_cast as fused

        weight, scales = prepare_weights(source.weight, source.scale, block_rows=32)
        weight, scales = fused.permute_q_b_proj((weight, scales), heads, dim)
        self.register_buffer("weight", weight.detach())
        self.register_buffer("scale", scales.detach())

    def forward(self, hidden):
        import deep_gemm

        values, scales = act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu)
        output = hidden.new_empty(
            hidden.numel() // hidden.shape[-1], self.weight.shape[0]
        )
        deep_gemm.fp8_gemm_nt(
            (
                values.reshape(-1, values.shape[-1]),
                scales.reshape(-1, scales.shape[-1]).float(),
            ),
            (self.weight, self.scale),
            output,
            recipe=(1, 1, 32),
        )
        return output.reshape(*hidden.shape[:-1], -1)


class OutputProjection(nn.Module):
    weight: torch.Tensor
    scale: torch.Tensor

    def __init__(self, source, groups, rank):
        super().__init__()
        from flash_mla import fused_norm_rope_attn_rope_cast as fused

        width = source.weight.shape[-1]
        weight = source.weight.reshape(groups, rank, width)
        scales = (
            source.scale.float()
            .repeat_interleave(32, 0)
            .reshape(groups, rank, width // 32)
        )
        weight, scales = prepare_weights(weight, scales)
        weight, scales = fused.permute_wv_proj((weight, scales), width // 512, 512)
        self.register_buffer("weight", weight.detach())
        self.register_buffer("scale", scales.detach())

    def forward(self, values, scales):
        import deep_gemm

        # This layout helper copies Float32 words; keep the packed UE8M0 bits.
        scales = (
            deep_gemm.get_mn_major_tma_aligned_tensor(
                scales.transpose(0, 1).view(torch.float32)
            )
            .view(torch.int32)
            .transpose(0, 1)
        )
        output = values.new_empty(
            values.shape[0],
            self.weight.shape[0],
            self.weight.shape[1],
            dtype=torch.bfloat16,
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (values, scales),
            (self.weight, self.scale),
            output,
            recipe=(1, 1, 32),
        )
        return output


@lru_cache(2)
def _cos_sin(freqs):
    return torch.cat((freqs.real, freqs.imag), -1).contiguous()


def prepare_attention(attention):
    attention.wq_b = QueryProjection(
        attention.wq_b, attention.n_local_heads, attention.head_dim
    )
    attention.wo_a = OutputProjection(
        attention.wo_a, attention.n_local_groups, attention.o_lora_rank
    )
    attention.register_buffer(
        "cos_sin",
        _cos_sin(attention.freqs_cis),
        persistent=False,
    )


def fused_attention(
    attention,
    query,
    positions,
    cache,
    indices,
    *,
    extra_cache=None,
    extra_indices=None,
    prefill=False,
):
    from flash_mla import fused_norm_rope_attn_rope_cast as fused

    common = (
        False,
        0.0,
        positions.flatten().int(),
        False,
        attention.rope_head_dim,
        attention.cos_sin,
        attention.n_local_groups,
        32,
        True,
        True,
        True,
    )
    query = query.reshape(-1, attention.n_local_heads, attention.head_dim)
    indices = padded_indices(indices.reshape(-1, indices.shape[-1]))
    if prefill:
        values, scales, _, _ = fused.prefill(
            *common,
            q=query,
            kv=cache,
            indices=indices[:, None],
            sm_scale=attention.softmax_scale,
            attn_sink=attention.attn_sink,
        )
    else:
        values, scales, _ = fused.decode(
            *common,
            q=query,
            k_cache=cache,
            indices_in_kvcache=indices,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None
            if extra_indices is None
            else padded_indices(extra_indices.reshape(indices.shape[0], -1)),
            sm_scale=attention.softmax_scale,
            attn_sink=attention.attn_sink,
        )
    return values, scales
