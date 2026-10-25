"""GLM-5.3-Flash mHC blocks, routed NVFP4 experts and native MTP.

Model equations follow Z.ai's Transformers reference and SGLang's GLM5 Next
implementation (Apache-2.0). NVFP4 clamp preparation follows SGLang's ModelOpt
adapter at 5971f47bc8eacd8989cb6dd67e46b4fc6dbec2ad.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.expert_parallel import ExpertParallelDispatch
from resin_models_cuda.glm5_3_flash.projections import fixed_rows
from resin_models_cuda.glm5_3_flash.weights import nvfp4_expert_scales
from resin_models_cuda.nvfp4 import (
    Nvfp4Linear,
    Nvfp4LinearWeights,
    prepare_nvfp4_moe_weights,
    quantize_nvfp4,
)

if TYPE_CHECKING:
    from resin_models_cuda.glm5_3_flash.vision import GlmVisionModel


def rms_norm(hidden, weight, eps):
    if hidden.is_cuda:
        from quack import rmsnorm

        return rmsnorm(hidden, weight, eps=eps)
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().square().mean(-1, keepdim=True) + eps
    )
    return normalized.to(hidden.dtype) * weight


def swiglu(gate, up, limit):
    from resin_models_cuda.glm5_3_flash.feed_forward_fusion import clamped_swiglu

    return clamped_swiglu(gate, up, limit)


class Linear(nn.Module):
    weight: Tensor

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


def linear(weights, name, *, backend="cute-dsl"):
    weight = weights[f"{name}.weight"]
    if weight.dtype != torch.uint8:
        return Linear(weight)
    out_features, packed = weight.shape
    with torch.device("meta"):
        source = Nvfp4LinearWeights(packed * 2, out_features)
    for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        setattr(source, suffix, weights[f"{name}.{suffix}"])
    return Nvfp4Linear(source, backend)


class DenseMLP(nn.Module):
    def __init__(self, weights, limit, *, backend="cute-dsl"):
        super().__init__()
        self.gate = linear(weights, "gate_proj", backend=backend)
        self.up = linear(weights, "up_proj", backend=backend)
        self.down = linear(weights, "down_proj", backend=backend)
        self.limit = limit

    def forward(self, hidden, *, projection_rows=0):
        return fixed_rows(hidden, self._forward, projection_rows=projection_rows)

    def _forward(self, hidden):
        if hidden.shape[0] == 0:
            # An idle attention replica still joins routed expert collectives;
            # its local dense/shared GEMMs must not launch a zero-M CuTe kernel.
            return torch.empty_like(hidden)

        # Preserve both the prepared CuTe W4A16 dense operator and ordinary
        # BF16 shared F.linear. Target decode and verification share a fixed physical row capacity.
        return self.down(swiglu(self.gate(hidden), self.up(hidden), self.limit))


class Router(nn.Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, config, weight, bias):
        super().__init__()
        self.config = config
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())

    def forward(self, hidden, *, projection_rows=0):
        return fixed_rows(hidden, self._forward, projection_rows=projection_rows)

    def _forward(self, hidden):
        from resin_models_cuda.glm5_3_flash.feed_forward_fusion import route

        return route(
            F.linear(hidden.float(), self.weight),
            self.bias,
            self.config.num_experts_per_tok,
            self.config.norm_topk_prob,
            self.config.routed_scaling_factor,
        )


class Nvfp4Experts(nn.Module):
    """Rank-local TRT-LLM W4A4 experts with global checkpoint calibration."""

    w13: Tensor
    w13_scales: Tensor
    w2: Tensor
    w2_scales: Tensor
    input_scale_quant: Tensor
    g1_alphas: Tensor
    g1_scale_c: Tensor
    g2_alphas: Tensor
    clamp_limits: Tensor

    def __init__(
        self, config, reader, prefix, *, ep_size=1, ep_rank=0, max_tokens=32768
    ):
        super().__init__()
        count = config.n_routed_experts // ep_size
        self.total, self.start, self.count = (
            config.n_routed_experts,
            ep_rank * count,
            count,
        )
        self.intermediate, self.hidden = (
            config.moe_intermediate_size,
            config.hidden_size,
        )
        self.max_tokens = max_tokens
        h, m = self.hidden, self.intermediate
        for name, shape, dtype in (
            ("w13", (count, 2 * m, h // 2), torch.uint8),
            ("w13_scales", (count, 2 * m, h // 16), torch.float8_e4m3fn),
            ("w2", (count, h, m // 2), torch.uint8),
            ("w2_scales", (count, h, m // 16), torch.float8_e4m3fn),
        ):
            self.register_buffer(
                name, torch.empty(shape, dtype=dtype, device=reader.device)
            )
        scales = torch.empty(count, 3, dtype=torch.float32, device="cpu")
        input_scales, output_scales = [], []
        # SGLang uses the global maximum, including experts on other EP ranks.
        # These are only scalar tensors; unowned packed weight bodies stay on disk.
        for expert in range(self.total):
            base = f"{prefix}{expert}."
            for projection in ("gate_proj", "up_proj"):
                input_scales.append(
                    float(
                        reader.read(base + projection + ".input_scale", cpu=True).max()
                    )
                )
            output_scales.append(
                float(reader.read(base + "down_proj.input_scale", cpu=True).max())
            )
        for local in range(count):
            base = f"{prefix}{self.start + local}."
            for position, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                output_width, input_width = (h, m) if name == "down_proj" else (m, h)
                weight = reader.read(
                    base + name + ".weight",
                    (output_width, input_width // 2),
                    torch.uint8,
                )
                block_scale = reader.read(
                    base + name + ".weight_scale",
                    (output_width, input_width // 16),
                    torch.float8_e4m3fn,
                )
                if name == "down_proj":
                    self.w2[local].copy_(weight)
                    self.w2_scales[local].copy_(block_scale)
                else:
                    # TRT-LLM consumes up followed by gate, before its row shuffle.
                    rows = slice(m, 2 * m) if name == "gate_proj" else slice(0, m)
                    self.w13[local, rows].copy_(weight)
                    self.w13_scales[local, rows].copy_(block_scale)
                scale = reader.read(base + name + ".weight_scale_2", cpu=True)
                if scale.numel() != 1:
                    raise ValueError("expected a scalar NVFP4 expert weight_scale_2")
                scales[local, position] = scale.reshape(())
        if any(
            not isfinite(value) or value <= 0
            for value in (*input_scales, *output_scales)
        ) or not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError("NVFP4 calibration scales must be finite and positive")
        a, b = max(input_scales), max(output_scales)
        scales = scales.to(reader.device)
        prepared = nvfp4_expert_scales(
            scales[:, 0], scales[:, 1], scales[:, 2], a, b, config.swiglu_limit
        )
        for name, value in prepared.items():
            self.register_buffer(name, value.to(reader.device))
        prepare_nvfp4_moe_weights(self.w13, self.w13_scales, self.w2, self.w2_scales)

    def forward(self, hidden, indices, weights):
        if not hidden.shape[0]:
            return torch.empty_like(hidden)
        if hidden.shape[0] > self.max_tokens:
            raise ValueError(
                "NVFP4 expert receive capacity exceeds its prepared token bound"
            )
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType

        packed, scales = quantize_nvfp4(hidden, self.input_scale_quant)
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(indices.int(), weights.to(torch.bfloat16)),
            routing_bias=None,
            hidden_states=packed,
            hidden_states_scale=scales,
            gemm1_weights=self.w13,
            gemm1_weights_scale=self.w13_scales,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=self.clamp_limits,
            gemm2_weights=self.w2,
            gemm2_weights_scale=self.w2_scales,
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.g1_alphas,
            output2_scale_scalar=self.g2_alphas,
            num_experts=self.total,
            top_k=indices.shape[1],
            n_group=1,
            topk_group=1,
            intermediate_size=self.intermediate,
            local_expert_offset=self.start,
            local_num_experts=self.count,
            routed_scaling_factor=None,
            routing_method_type=2,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=self.max_tokens,
            enable_pdl=False,
        )
        return result[0] if isinstance(result, (tuple, list)) else result


class Bf16Experts(nn.Module):
    """BF16 native-MTP experts; capture-safe local CUTLASS execution."""

    w13: Tensor
    w2: Tensor
    alpha: Tensor
    beta: Tensor
    clamp: Tensor

    def __init__(
        self, config, reader, prefix, *, ep_size=1, ep_rank=0, max_tokens=32768
    ):
        super().__init__()
        self.count = config.n_routed_experts // ep_size
        self.start, self.limit = ep_rank * self.count, config.swiglu_limit
        self.max_tokens = max_tokens
        h, m = config.hidden_size, config.moe_intermediate_size
        self.register_buffer(
            "w13",
            torch.empty(
                self.count, 2 * m, h, dtype=torch.bfloat16, device=reader.device
            ),
        )
        self.register_buffer(
            "w2",
            torch.empty(self.count, h, m, dtype=torch.bfloat16, device=reader.device),
        )
        for local in range(self.count):
            base = f"{prefix}{self.start + local}."
            self.w13[local, :m].copy_(
                reader.read(base + "up_proj.weight", (m, h), torch.bfloat16)
            )
            self.w13[local, m:].copy_(
                reader.read(base + "gate_proj.weight", (m, h), torch.bfloat16)
            )
            self.w2[local].copy_(
                reader.read(base + "down_proj.weight", (h, m), torch.bfloat16)
            )
        for name, value in (("alpha", 1.0), ("beta", 0.0), ("clamp", self.limit)):
            self.register_buffer(
                name,
                torch.full(
                    (self.count,), value, dtype=torch.float32, device=reader.device
                ),
            )

    def forward(self, hidden, indices, weights):
        if not hidden.shape[0]:
            return torch.empty_like(hidden)
        if hidden.shape[0] > self.max_tokens:
            raise ValueError(
                "BF16 expert receive capacity exceeds its prepared token bound"
            )
        from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

        selected = indices - self.start
        selected = torch.where(
            (selected >= 0) & (selected < self.count), selected, self.count
        ).int()
        output = torch.empty_like(hidden)
        cutlass_fused_moe(
            input=hidden,
            token_selected_experts=selected,
            token_final_scales=weights.float(),
            fc1_expert_weights=self.w13,
            fc2_expert_weights=self.w2,
            output_dtype=torch.bfloat16,
            quant_scales=[],
            swiglu_alpha=self.alpha,
            swiglu_beta=self.beta,
            swiglu_limit=self.clamp,
            output=output,
            tune_max_num_tokens=self.max_tokens,
            enable_pdl=False,
            activation_type=ActivationType.Swiglu,
            use_fused_finalize=False,
        )
        return output


class SparseMLP(nn.Module):
    def __init__(self, config, weights, experts, groups=None):
        super().__init__()
        self.router = Router(
            config, weights["gate.weight"], weights["gate.e_score_correction_bias"]
        )
        self.shared = DenseMLP(
            {
                k.removeprefix("shared_experts."): v
                for k, v in weights.items()
                if k.startswith("shared_experts.")
            },
            config.swiglu_limit,
        )
        self.experts = experts
        self.dispatch = (
            None
            if groups is None or groups.device_domain.size == 1
            else ExpertParallelDispatch(
                experts, num_experts=config.n_routed_experts, groups=groups
            )
        )

    def forward(self, hidden, *, projection_rows=0):
        ids, weights = self.router(hidden, projection_rows=projection_rows)
        routed = (
            self.experts(hidden, ids, weights)
            if self.dispatch is None
            else self.dispatch(hidden, ids, weights)
        )
        return routed + self.shared(hidden, projection_rows=projection_rows)


class HyperConnection(nn.Module):
    fn: Tensor
    scale: Tensor
    base: Tensor

    def __init__(self, config, weights, name):
        super().__init__()
        self.config = config
        for suffix in ("fn", "scale", "base"):
            self.register_buffer(suffix, weights[f"hc_{name}_{suffix}"].float())

    def forward(self, hidden, norm_weight=None):
        c, hc = self.config, self.config.hc_mult
        if hidden.is_cuda:
            # This model uses the same mHC primitive ABI as DeepInfer, without
            # its shifted pre-mix or weighted final head semantics.
            import deep_gemm

            import_module("resin_models_cuda.deepseek_v41.tilelang_kernels")
            kernel = import_module(
                "tile_kernels.mhc.pre_big_fuse_kernel"
            )._mhc_pre_big_fuse
            rows, width = hidden.shape[0], c.hidden_size
            # Keep one reduction regime across contexts and sampling settings.
            # This matches the serving reference's small-batch reduction order.
            splits = 64
            partials = hidden.new_empty(
                (splits, rows, hc * (hc + 2)), dtype=torch.float32
            )
            squares = hidden.new_empty((splits, rows), dtype=torch.float32)
            deep_gemm.tf32_hc_prenorm_gemm(
                hidden.flatten(1), self.fn, partials, squares, splits
            )
            post = hidden.new_empty((rows, hc), dtype=torch.float32)
            comb = hidden.new_empty((rows, hc, hc), dtype=torch.float32)
            value = hidden.new_empty((rows, width))
            if norm_weight is not None:
                from resin_models_cuda.glm5_3_flash.mhc_cuda import pre_with_norm

                pre_with_norm(
                    partials,
                    squares,
                    self.scale,
                    self.base,
                    hidden,
                    post,
                    comb.flatten(1),
                    value,
                    norm_weight,
                    width,
                    c.rms_norm_eps,
                    c.hc_eps,
                    c.hc_eps,
                    2.0,
                    c.hc_sinkhorn_iters,
                    c.rms_norm_eps,
                    splits,
                    hc,
                )
                return value, post, comb
            kernel(
                width,
                c.rms_norm_eps,
                c.hc_eps,
                c.hc_eps,
                2.0,
                c.hc_sinkhorn_iters,
                splits,
                hc,
            )(
                partials,
                squares,
                self.scale,
                self.base,
                hidden,
                post,
                comb.flatten(1),
                value,
            )
            return value, post, comb
        flat = hidden.flatten(1).float()
        mixed = F.linear(
            flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + c.rms_norm_eps),
            self.fn,
        )
        pre, post, comb = mixed.split((hc, hc, hc * hc), -1)
        pre_b, post_b, comb_b = self.base.split((hc, hc, hc * hc))
        pre = torch.sigmoid(pre * self.scale[0] + pre_b) + c.hc_eps
        post = 2 * torch.sigmoid(post * self.scale[1] + post_b)
        comb = (comb * self.scale[2] + comb_b).reshape(-1, hc, hc).softmax(
            -1
        ) + c.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        for _ in range(c.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        value = (pre[..., None] * hidden).sum(1).to(hidden.dtype)
        if norm_weight is not None:
            value = rms_norm(value, norm_weight, c.rms_norm_eps)
        return value, post, comb

    @staticmethod
    def post(output, residual, post, comb):
        if output.is_cuda:
            kernel = import_module("tile_kernels.mhc.post_kernel").mhc_post_fwd
            return kernel(
                output[None], residual[None], post[None, :, :, None], comb[None]
            )[0]
        # Match the reference's BF16 multiplication boundaries.
        return post.to(output.dtype)[..., None] * output[:, None] + torch.matmul(
            comb.to(output.dtype).transpose(-1, -2), residual
        )


class DecoderLayer(nn.Module):
    attn_norm: Tensor
    ffn_norm: Tensor
    prefill_ffn: Callable[..., Tensor] | None

    def __init__(self, config, layer_id, weights, mlp, attention_factory):
        super().__init__()
        self.layer_id, self.config = layer_id, config
        self.attention = attention_factory(
            config,
            layer_id,
            {
                k.removeprefix("self_attn."): v
                for k, v in weights.items()
                if k.startswith("self_attn.")
            },
        )
        self.mlp = mlp
        self.prefill_ffn = None
        self.attn_hc = HyperConnection(config, weights, "attn")
        self.ffn_hc = HyperConnection(config, weights, "ffn")
        self.register_buffer("attn_norm", weights["input_layernorm.weight"])
        self.register_buffer("ffn_norm", weights["post_attention_layernorm.weight"])

    def forward(self, hidden, positions, cache):
        projection_rows = getattr(cache, "projection_rows", 0)
        residual = hidden
        value, post, comb = self.attn_hc(hidden, self.attn_norm)
        value = self.attention(
            value,
            positions,
            cache.layer(self.layer_id),
            projection_rows=projection_rows,
        )
        hidden = HyperConnection.post(value, residual, post, comb)
        residual = hidden
        value, post, comb = self.ffn_hc(hidden, self.ffn_norm)
        if getattr(cache, "is_decode", False):
            # Outside prefill scope the callback calls this same operation.
            # Keep explicit target metadata out of prefill/draft callbacks.
            value = self.feed_forward(value, projection_rows=projection_rows)
        else:
            value = (
                self.feed_forward(value)
                if self.prefill_ffn is None
                else self.prefill_ffn(value)
            )
        return HyperConnection.post(value, residual, post, comb)

    def feed_forward(self, hidden, token_ids=None, *, projection_rows=0):
        return (
            self.mlp(hidden)
            if projection_rows == 0
            else self.mlp(hidden, projection_rows=projection_rows)
        )


class MtpLayer(nn.Module):
    enorm: Tensor
    hnorm: Tensor
    projection: Tensor
    attn_norm: Tensor
    ffn_norm: Tensor
    head_norm: Tensor
    prefill_ffn: Callable[..., Tensor] | None

    def __init__(self, config, weights, mlp, attention_factory):
        super().__init__()
        self.config, self.layer_id = config, config.num_hidden_layers
        self.attention = attention_factory(
            config,
            self.layer_id,
            {
                k.removeprefix("self_attn."): v
                for k, v in weights.items()
                if k.startswith("self_attn.")
            },
        )
        self.mlp = mlp
        self.prefill_ffn = None
        for name, key in (
            ("enorm", "enorm.weight"),
            ("hnorm", "hnorm.weight"),
            ("projection", "eh_proj.weight"),
            ("attn_norm", "input_layernorm.weight"),
            ("ffn_norm", "post_attention_layernorm.weight"),
            ("head_norm", "shared_head.norm.weight"),
        ):
            self.register_buffer(name, weights[key])

    def forward(self, embedded, previous_hidden, positions, cache):
        eps = self.config.rms_norm_eps
        hidden = F.linear(
            torch.cat(
                (
                    rms_norm(embedded, self.enorm, eps),
                    rms_norm(previous_hidden, self.hnorm, eps),
                ),
                -1,
            ),
            self.projection,
        )
        hidden = hidden + self.attention(
            rms_norm(hidden, self.attn_norm, eps), positions, cache.layer(self.layer_id)
        )
        value = rms_norm(hidden, self.ffn_norm, eps)
        hidden = hidden + (
            self.feed_forward(value)
            if self.prefill_ffn is None
            else self.prefill_ffn(value)
        )
        return hidden

    def feed_forward(self, hidden, token_ids=None):
        return self.mlp(hidden)


class Glm53Model(nn.Module):
    embedding: Tensor
    norm: Tensor
    head: Tensor
    weight_load: dict[str, Any]
    mtp: MtpLayer | None
    vision: GlmVisionModel | None

    def __init__(self, config, embedding, norm, head, layers, mtp=None):
        super().__init__()
        self.config = config
        self.register_buffer("embedding", embedding)
        self.register_buffer("norm", norm)
        self.register_buffer("head", head)
        self.layers = nn.ModuleList(layers)
        self.mtp = mtp
        self.vision = None
        self.weight_load = {}

    def forward_hidden(self, token_ids, positions, cache, *, hidden=None):
        hidden = F.embedding(token_ids, self.embedding) if hidden is None else hidden
        hidden = hidden[:, None].expand(-1, self.config.hc_mult, -1).contiguous()
        for layer in self.layers:
            hidden = layer(hidden, positions, cache)
        return hidden.mean(1)

    def logits(self, hidden, *, projection_rows=0):
        return fixed_rows(
            hidden,
            lambda value: F.linear(
                rms_norm(value, self.norm, self.config.rms_norm_eps), self.head
            ),
            projection_rows=projection_rows,
        )

    def forward_experts_empty(self):
        hidden = self.embedding.new_empty((0, self.config.hidden_size))
        for module in self.layers:
            layer = cast(DecoderLayer, module)
            if layer.prefill_ffn is None:
                layer.feed_forward(hidden)
            else:
                layer.prefill_ffn(hidden)

    def forward_mtp_experts_empty(self):
        if self.mtp is None:
            raise RuntimeError("native MTP weights were not loaded")
        hidden = self.embedding.new_empty((0, self.config.hidden_size))
        if self.mtp.prefill_ffn is None:
            return self.mtp.feed_forward(hidden)
        return self.mtp.prefill_ffn(hidden)

    def forward(self, token_ids, positions, cache, *, hidden=None, return_hidden=False):
        hidden = self.forward_hidden(token_ids, positions, cache, hidden=hidden)
        logits = self.logits(
            hidden, projection_rows=getattr(cache, "projection_rows", 0)
        )
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self, token_ids, previous_hidden, positions, cache, *, input_embeddings=None
    ):
        if self.mtp is None:
            raise RuntimeError("native MTP weights were not loaded")
        embedded = (
            F.embedding(token_ids, self.embedding)
            if input_embeddings is None
            else input_embeddings
        )
        hidden = self.mtp(embedded, previous_hidden, positions, cache)
        # NextN consumes the prior step after its final shared-head norm, just
        # as the official SGLang draft worker captures its recursive state.
        hidden = rms_norm(hidden, self.mtp.head_norm, self.config.rms_norm_eps)
        return F.linear(hidden, self.head), hidden
