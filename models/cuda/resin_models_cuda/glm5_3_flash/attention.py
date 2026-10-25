"""GLM-5.3-Flash KDA and NoPE sparse MLA over Engine-owned cache views.

Architecture semantics: Transformers 50e35cc0ca18ec19b80653690de553e6decf74f5,
models/glm5_next (Apache-2.0). This module owns no request-ID cache or allocator.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.normalization import gated_rms_norm, rms_norm
from resin_models_cuda.glm5_3_flash.projections import bmm, linear


class KdaAttention(nn.Module):
    """Packed-variable-length KDA with separate read and final-state ownership."""

    in_weight: Tensor
    fg_weight: Tensor
    conv_weight: Tensor
    a_log: Tensor
    dt_bias: Tensor
    out_norm: Tensor
    out_weight: Tensor

    def __init__(self, config, layer_id: int, weights: dict[str, Tensor]):
        super().__init__()
        self.layer_id = layer_id
        self.heads = config.linear_num_heads
        self.dim = config.linear_head_dim
        self.width = self.heads * self.dim
        self.eps = config.rms_norm_eps
        self.lower_bound = config.linear_lower_bound
        if config.linear_conv_kernel_dim != 4:
            raise ValueError("GLM KDA adapter currently requires conv4")
        self.register_buffer(
            "in_weight",
            torch.cat(
                [
                    weights[name + ".weight"]
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "b_proj",
                        "f_a_proj",
                        "g_a_proj",
                    )
                ]
            ),
        )
        self.register_buffer(
            "fg_weight",
            torch.stack((weights["f_b_proj.weight"], weights["g_b_proj.weight"])),
        )
        self.register_buffer(
            "conv_weight",
            torch.stack(
                [
                    weights[name + "_conv1d.weight"].squeeze(1)
                    for name in ("q", "k", "v")
                ]
            ).float(),
        )
        self.register_buffer("a_log", weights["A_log"].reshape(-1).float())
        self.register_buffer("dt_bias", weights["dt_bias"].reshape(-1).float())
        self.register_buffer("out_norm", weights["o_norm.weight"])
        self.register_buffer("out_weight", weights["o_proj.weight"])

    def forward(
        self, hidden: Tensor, positions: Tensor, cache, *, projection_rows=0
    ) -> Tensor:
        del positions  # Packed offsets and recurrence contain all positional state.
        if hidden.shape[0] == 0:
            return hidden
        projected = linear(hidden, self.in_weight, projection_rows=projection_rows)
        qkv, beta, fg_a = projected.split(
            (self.width * 3, self.heads, self.dim * 2), -1
        )
        # Preserve the packed projection's row stride. The convolution consumes
        # it directly; this view also retains its owner for speculative rollback.
        qkv = qkv.reshape(-1, 3, self.width)
        forget, output_gate = bmm(
            fg_a.reshape(-1, 2, self.dim).transpose(0, 1),
            self.fg_weight.transpose(1, 2),
            projection_rows=projection_rows,
            input_strides=(self.dim, self.in_weight.shape[0], 1),
            input_storage_offset=self.in_weight.shape[0] - 2 * self.dim,
        ).unbind(0)
        convolved, convolution_state = kernels.convolution(
            qkv, self.conv_weight, cache, planar=True
        )
        query, key, value = convolved.reshape(3, -1, self.heads, self.dim).unbind(0)
        tape = getattr(cache, "speculative_tape", None)
        if tape is not None:
            # FLA requires a contiguous [B,H,K,V] initial state. Share this one
            # packed copy between target verification and recurrence-only replay.
            cache.recurrent = cache.recurrent.contiguous()
            tape.record(
                cache,
                qkv,
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                forget.reshape(-1, self.heads, self.dim).contiguous(),
                beta,
                self.a_log,
                self.dt_bias,
                self.lower_bound,
            )
        core, recurrent_state = kernels.kda(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            forget.reshape(-1, self.heads, self.dim).contiguous(),
            beta,
            self.a_log,
            self.dt_bias,
            self.lower_bound,
            cache,
        )
        # The caller commits these states only after the owning forward succeeds.
        cache.recurrent = recurrent_state
        cache.convolution = convolution_state
        normalized = gated_rms_norm(
            core, self.out_norm, output_gate.reshape_as(core), self.eps
        )
        return linear(
            normalized.to(hidden.dtype).reshape(-1, self.width),
            self.out_weight,
            projection_rows=projection_rows,
        )


class SparseAttention(nn.Module):
    """DSA index-pool4 selection and absorbed NoPE MLA, including native MTP."""

    q_a: Tensor
    q_norm: Tensor
    q_b: Tensor
    kv_a: Tensor
    kv_norm: Tensor
    o: Tensor
    index_q: Tensor
    index_k: Tensor
    index_norm: Tensor
    index_bias: Tensor
    index_weight: Tensor
    index_gate: Tensor
    index_ape: Tensor
    key_absorb: Tensor
    value_expand: Tensor

    def __init__(self, config, layer_id: int, weights: dict[str, Tensor]):
        super().__init__()
        self.layer_id = layer_id
        self.heads = config.num_attention_heads
        self.query_dim = config.qk_nope_head_dim
        self.value_dim = config.v_head_dim
        self.latent_dim = config.kv_lora_rank
        self.index_heads = config.index_n_heads
        self.index_dim = config.index_head_dim
        self.topk = config.index_topk
        self.eps = config.rms_norm_eps
        if config.qk_rope_head_dim != 0 or config.index_kpool != 4:
            raise ValueError("GLM sparse adapter requires NoPE MLA and pool4")
        if self.topk % 4:
            raise ValueError("GLM sparse token budget must contain whole pools")
        names = {
            "q_a": "q_a_proj.weight",
            "q_norm": "q_a_layernorm.weight",
            "q_b": "q_b_proj.weight",
            "kv_a": "kv_a_proj_with_mqa.weight",
            "kv_norm": "kv_a_layernorm.weight",
            "o": "o_proj.weight",
            "index_q": "indexer.wq_b.weight",
            "index_k": "indexer.wk.weight",
            "index_norm": "indexer.k_norm.weight",
            "index_bias": "indexer.k_norm.bias",
            "index_weight": "indexer.weights_proj.weight",
            "index_gate": "indexer.index_kpool_compress_gate",
            "index_ape": "indexer.index_kpool_compress_ape",
        }
        for attr, name in names.items():
            # SGLang's pool indexer explicitly promotes the head-gate projection
            # to FP32 on load and projects FP32 inputs before weighting scores.
            weight = weights[name].float() if attr == "index_weight" else weights[name]
            self.register_buffer(attr, weight)
        kv_b = weights["kv_b_proj.weight"].reshape(
            self.heads, self.query_dim + self.value_dim, self.latent_dim
        )
        self.register_buffer("key_absorb", kv_b[:, : self.query_dim].contiguous())
        self.register_buffer("value_expand", kv_b[:, self.query_dim :].contiguous())

    def forward(
        self, hidden: Tensor, positions: Tensor, cache, *, projection_rows=0
    ) -> Tensor:
        if hidden.shape[0] == 0:
            return hidden

        def project(value, weight):
            return linear(value, weight, projection_rows=projection_rows)

        residual = rms_norm(project(hidden, self.q_a), self.q_norm, self.eps)
        query = project(residual, self.q_b).reshape(-1, self.heads, self.query_dim)
        latent = rms_norm(project(hidden, self.kv_a), self.kv_norm, self.eps)
        index_query = project(residual, self.index_q).reshape(
            -1, self.index_heads, self.index_dim
        )
        index_keys = F.layer_norm(
            project(hidden, self.index_k),
            (self.index_dim,),
            self.index_norm,
            self.index_bias,
            1e-6,
        )
        index_gates = project(hidden, self.index_gate)
        index_weights = project(hidden.float(), self.index_weight)
        kernels.store_sparse(latent, index_keys, index_gates, self.index_ape, cache)
        absorbed = (
            bmm(
                query.transpose(0, 1),
                self.key_absorb,
                projection_rows=projection_rows,
                input_strides=(self.query_dim, self.heads * self.query_dim, 1),
            )
            .transpose(0, 1)
            .contiguous()
        )
        # Bound score workspace even when prefill spans a large context. Context
        # width remains runtime data in each scoring kernel, not a JIT key.
        pieces = []
        for start in range(0, hidden.shape[0], 128):
            end = min(start + 128, hidden.shape[0])
            selected = kernels.select_indices(
                index_query[start:end],
                index_weights[start:end],
                positions[start:end],
                cache.sequence_indices[start:end],
                cache,
                self.topk,
            )
            pieces.append(
                kernels.sparse_attention(
                    absorbed[start:end], selected, cache.latent, self.query_dim**-0.5
                )
            )
        attended = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        expanded = bmm(
            attended.transpose(0, 1),
            self.value_expand.transpose(1, 2),
            projection_rows=projection_rows,
            input_strides=(self.latent_dim, self.heads * self.latent_dim, 1),
        ).transpose(0, 1)
        return project(expanded.reshape(-1, self.heads * self.value_dim), self.o)


def Attention(config, layer_id: int, weights: dict[str, Tensor]):
    """Native MTP is an additional sparse-attention layer without KDA state."""
    kind = (
        config.layer_types[layer_id]
        if layer_id < len(config.layer_types)
        else "deepseek_sparse_attention"
    )
    cls = KdaAttention if kind == "linear_attention" else SparseAttention
    return cls(config, layer_id, weights)
