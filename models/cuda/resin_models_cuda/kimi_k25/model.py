"""Native Kimi K2.5-family text target and one-layer EAGLE3 MLA draft."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.cuda.parallel import TensorParallelContext
from resin_models_cuda.kimi_k25.cache import PAGE_TOKENS, DeviceCache
from resin_models_cuda.kimi_k25.config import (
    KimiK25Config,
    KimiK25DraftConfig,
    RopeConfig,
)
from resin_models_cuda.nvfp4 import prepare_nvfp4_moe_weights, quantize_nvfp4


class RMSNorm(nn.Module):
    """BF16 RMSNorm dispatched to Quack on CUDA."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.is_cuda:
            from quack import rmsnorm

            return rmsnorm(hidden, self.weight, eps=self.eps)
        normalized = hidden.float()
        normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden.dtype)


class ParallelEmbedding(nn.Module):
    """Vocabulary-sharded embedding with one TP reduction."""

    def __init__(self, vocab_size: int, hidden_size: int, tp: TensorParallelContext):
        super().__init__()
        shard = tp.layout.even_shard(vocab_size, name="vocab_size")
        self.vocab_start = shard.start
        self.vocab_stop = shard.stop
        self.tp = tp
        self.weight = nn.Parameter(
            torch.empty(shard.size, hidden_size, dtype=torch.bfloat16)
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        local = token_ids - self.vocab_start
        masked = (token_ids < self.vocab_start) | (token_ids >= self.vocab_stop)
        local = local.masked_fill(masked, 0)
        hidden = F.embedding(local, self.weight)
        hidden.masked_fill_(masked[..., None], 0)
        return self.tp.sum(hidden)


class ParallelHead(nn.Module):
    """Vocabulary-sharded BF16 head returning globally ordered FP32 logits."""

    def __init__(self, vocab_size: int, hidden_size: int, tp: TensorParallelContext):
        super().__init__()
        shard = tp.layout.even_shard(vocab_size, name="vocab_size")
        self.tp = tp
        self.weight = nn.Parameter(
            torch.empty(shard.size, hidden_size, dtype=torch.float32)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        local = F.linear(hidden.float(), self.weight)
        if not self.tp.enabled:
            return local
        gathered = [torch.empty_like(local) for _ in range(self.tp.size)]
        dist.all_gather(gathered, local, group=self.tp.process_group)
        return torch.cat(gathered, dim=-1)


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp: TensorParallelContext):
        super().__init__()
        shard = tp.layout.even_shard(out_features, name="linear output")
        self.weight = nn.Parameter(
            torch.empty(shard.size, in_features, dtype=torch.bfloat16)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        shard = tp.layout.even_shard(in_features, name="linear input")
        self.weight = nn.Parameter(
            torch.empty(out_features, shard.size, dtype=torch.bfloat16)
        )
        self.reduce = reduce

    def partial(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.reduce(self.partial(hidden))


@lru_cache(8)
def precompute_rope(
    head_dim: int,
    max_sequence_length: int,
    rope: RopeConfig,
) -> Tensor:
    """Precompute DeepSeek YaRN frequencies without sequence specialization."""

    frequencies = 1.0 / (
        rope.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    if rope.factor <= 1:
        angles = torch.outer(
            torch.arange(max_sequence_length, dtype=torch.float32), frequencies
        )
        return torch.polar(torch.ones_like(angles), angles)

    def correction(rotations: float) -> float:
        return (
            head_dim
            * math.log(rope.original_context_length / (rotations * 2 * math.pi))
            / (2 * math.log(rope.theta))
        )

    low = max(math.floor(correction(rope.beta_fast)), 0)
    high = min(math.ceil(correction(rope.beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001
    ramp = (
        (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)
    ).clamp(0, 1)
    smooth = 1 - ramp
    frequencies = frequencies / rope.factor * (1 - smooth) + frequencies * smooth
    angles = torch.outer(
        torch.arange(max_sequence_length, dtype=torch.float32), frequencies
    )
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(hidden: Tensor, positions: Tensor, frequencies: Tensor) -> Tensor:
    """Apply rotary embeddings in place to [batch, tokens, heads, dim]."""

    selected = frequencies[positions].unsqueeze(2)
    complex_hidden = torch.view_as_complex(
        hidden.float().reshape(*hidden.shape[:-1], hidden.shape[-1] // 2, 2)
    )
    hidden.copy_(torch.view_as_real(complex_hidden * selected).flatten(-2))
    return hidden


class MlaPrefill:
    """Sequence-generic paged MLA plan for one eager prefill batch."""

    def __init__(
        self,
        workspace: Tensor,
        *,
        num_heads: int,
        kv_lora_rank: int,
        rope_head_dim: int,
        scale: float,
    ) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
        self.device = workspace.device
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.scale = scale
        self.page_count = 0

    def plan(self, query_length: int, sequence_length: int) -> None:
        """Plan runtime lengths without compiling a sequence-specific kernel."""

        if not 0 < query_length <= sequence_length:
            raise ValueError("MLA prefill lengths must satisfy 0 < query <= sequence")
        self.page_count = math.ceil(sequence_length / PAGE_TOKENS)
        self.wrapper.plan(
            torch.tensor([0, query_length], dtype=torch.int32, device=self.device),
            torch.tensor([0, self.page_count], dtype=torch.int32, device=self.device),
            torch.arange(self.page_count, dtype=torch.int32, device=self.device),
            torch.tensor([sequence_length], dtype=torch.int32, device=self.device),
            self.num_heads,
            self.kv_lora_rank,
            self.rope_head_dim,
            PAGE_TOKENS,
            True,
            self.scale,
            torch.bfloat16,
            torch.bfloat16,
        )

    def run(
        self,
        q_nope: Tensor,
        q_rope: Tensor,
        kv_cache: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        """Run causal prefill over the live pages selected by the Engine."""

        if q_nope.shape[0] != 1 or block_tables.shape[0] != 1:
            raise ValueError(
                "Kimi eager MLA prefill currently runs one sequence at a time"
            )
        page_ids = block_tables[0, : self.page_count].long()
        ckv_cache = kv_cache[page_ids, :, : self.kv_lora_rank].to(torch.bfloat16)
        kpe_cache = kv_cache[
            page_ids,
            :,
            self.kv_lora_rank : self.kv_lora_rank + self.rope_head_dim,
        ].to(torch.bfloat16)
        output = self.wrapper.run(
            q_nope.flatten(0, 1),
            q_rope.flatten(0, 1),
            ckv_cache,
            kpe_cache,
            return_lse=False,
        )
        return output.unsqueeze(0)


class MlaAttention(nn.Module):
    """DeepSeek-V3 MLA with matrix-absorbed paged decode."""

    def __init__(
        self,
        config: KimiK25Config | KimiK25DraftConfig,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
        workspace: Tensor,
        frequencies: Tensor,
        *,
        input_size: int | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        heads = tp.layout.even_shard(
            config.num_attention_heads, name="attention heads"
        ).size
        self.hidden_size = hidden
        self.local_heads = heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        projection_input = hidden if input_size is None else input_size
        self.q_a_proj = nn.Linear(
            projection_input, config.q_lora_rank, bias=False, dtype=torch.bfloat16
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            bias=False,
            dtype=torch.bfloat16,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            projection_input,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            dtype=torch.bfloat16,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.v_head_dim,
            hidden,
            tp,
            reduce,
        )
        self.scale = (
            config.qk_nope_head_dim + config.qk_rope_head_dim
        ) ** -0.5 * config.rope.attention_scale_multiplier
        self.__dict__["workspace"] = workspace
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _run_decode_attention(
        self,
        query: Tensor,
        kv_cache: Tensor,
        block_tables: Tensor,
        sequence_lengths: Tensor,
        max_sequence_length: int,
    ) -> Tensor:
        from tokenspeed_mla import tokenspeed_mla_decode

        if kv_cache.dtype == torch.float8_e4m3fn:
            limit = torch.finfo(kv_cache.dtype).max
            query = query.clamp(-limit, limit)
        query = query.to(kv_cache.dtype)
        output = tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=sequence_lengths.to(torch.int32),
            max_seq_len=max_sequence_length,
            softmax_scale=self.scale,
            output_scale=1.0,
            is_var_seq=True,
            causal_mask=True,
            enable_pdl=True,
        )
        return output.to(torch.bfloat16)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        *,
        kv_cache: Tensor,
        block_tables: Tensor,
        write_rows: Tensor,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        prefill: MlaPrefill | None = None,
    ) -> Tensor:
        batch, tokens = positions.shape
        q_a = self.q_a_proj(hidden)
        q = self.q_b_proj(self.q_a_layernorm(q_a)).view(
            batch,
            tokens,
            self.local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        latent = self.kv_a_proj_with_mqa(hidden)
        ckv, k_rope = latent.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        ckv = self.kv_a_layernorm(ckv)
        apply_rope(q_rope, positions, self.frequencies)
        apply_rope(k_rope.unsqueeze(2), positions, self.frequencies)
        cache_rows = torch.cat((ckv, k_rope), dim=-1)
        if kv_cache.dtype == torch.float8_e4m3fn:
            limit = torch.finfo(kv_cache.dtype).max
            cache_rows = cache_rows.clamp(-limit, limit)
        encoded_rows = cache_rows.flatten(0, 1).to(kv_cache.dtype).view(torch.uint8)
        kv_cache.view(torch.uint8).reshape(-1, encoded_rows.shape[-1]).index_copy_(
            0, write_rows.flatten().long(), encoded_rows
        )

        kv_b = self.kv_b_proj.weight.view(
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        key_weight, value_weight = kv_b.split(
            (self.qk_nope_head_dim, self.v_head_dim), dim=1
        )
        absorbed_q = torch.einsum("bthd,hdc->bthc", q_nope, key_weight)
        expanded_q_rope = q_rope.expand(-1, -1, self.local_heads, -1)
        if prefill is None:
            query = torch.cat((absorbed_q, expanded_q_rope), dim=-1)
            attended = self._run_decode_attention(
                query,
                kv_cache,
                block_tables,
                sequence_lengths,
                max_sequence_length,
            )
        else:
            attended = prefill.run(
                absorbed_q,
                expanded_q_rope,
                kv_cache,
                block_tables,
            )
        values = torch.einsum("bthc,hvc->bthv", attended, value_weight)
        return self.o_proj(values.flatten(-2))


class DenseMlp(nn.Module):
    """Tensor-parallel SwiGLU used by dense and shared experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, tp)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, tp)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, tp, reduce)

    def partial(self, hidden: Tensor) -> Tensor:
        activated = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj.partial(activated)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj.reduce(self.partial(hidden))


class Nvfp4Experts(nn.Module):
    """Local ModelOpt NVFP4 experts executed by one routed MoE kernel."""

    def __init__(self, config: KimiK25Config, tp: TensorParallelContext) -> None:
        super().__init__()
        expert_shard = tp.layout.even_shard(
            config.num_routed_experts, name="routed experts"
        )
        local_experts = expert_shard.size
        hidden = config.hidden_size
        intermediate = config.moe_intermediate_size
        self.global_experts = config.num_routed_experts
        self.local_expert_offset = expert_shard.start
        self.local_experts = local_experts
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.top_k = config.experts_per_token
        self.route_scale = config.route_scale
        self.normalize_topk = config.normalize_topk
        self.tp = tp
        self.gate = nn.Linear(
            hidden, config.num_routed_experts, bias=False, dtype=torch.float32
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_routed_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "w13",
            torch.empty(
                local_experts, 2 * intermediate, hidden // 2, dtype=torch.uint8
            ),
        )
        self.register_buffer(
            "w13_scales",
            torch.empty(
                local_experts,
                2 * intermediate,
                hidden // 16,
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "w2",
            torch.empty(local_experts, hidden, intermediate // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "w2_scales",
            torch.empty(
                local_experts,
                hidden,
                intermediate // 16,
                dtype=torch.float8_e4m3fn,
            ),
        )
        for name, shape in (
            ("w13_input_scales", (local_experts, 2)),
            ("w13_weight_scales", (local_experts, 2)),
            ("w2_input_scales", (local_experts,)),
            ("w2_weight_scales", (local_experts,)),
        ):
            self.register_buffer(name, torch.empty(shape, dtype=torch.float32))
        self.register_buffer("input_scale_quant", torch.empty((), dtype=torch.float32))
        self.register_buffer(
            "g1_scale_c", torch.empty(local_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "g1_alphas", torch.empty(local_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "g2_alphas", torch.empty(local_experts, dtype=torch.float32)
        )
        self.prepared = False

    def prepare(self) -> None:
        if self.prepared:
            return
        w13_input = self.w13_input_scales.max()
        w2_input = self.w2_input_scales.max()
        if self.tp.enabled:
            dist.all_reduce(
                w13_input, op=dist.ReduceOp.MAX, group=self.tp.process_group
            )
            dist.all_reduce(w2_input, op=dist.ReduceOp.MAX, group=self.tp.process_group)
        # FlashInfer's fused gated GEMM consumes W13 in W3/W1 order: up, gate.
        up_alpha = w13_input * self.w13_weight_scales[:, 0]
        gate_alpha = w13_input * self.w13_weight_scales[:, 1]
        self.input_scale_quant.copy_(w13_input.reciprocal())
        self.g1_alphas.copy_(gate_alpha)
        self.g1_scale_c.copy_(w2_input.reciprocal() * up_alpha)
        self.g2_alphas.copy_(w2_input * self.w2_weight_scales)
        prepare_nvfp4_moe_weights(self.w13, self.w13_scales, self.w2, self.w2_scales)
        self.w13_input_scales = None
        self.w13_weight_scales = None
        self.w2_input_scales = None
        self.w2_weight_scales = None
        self.prepared = True

    def forward(self, hidden: Tensor) -> Tensor:
        if not self.prepared:
            raise RuntimeError("Kimi NVFP4 experts must be prepared after loading")
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType

        shape = hidden.shape
        flat = hidden.reshape(-1, self.hidden_size)
        scores = torch.sigmoid(F.linear(flat.float(), self.gate.weight))
        _, topk_ids = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        )
        topk_weights = scores.gather(1, topk_ids)
        if self.normalize_topk:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = (topk_weights * self.route_scale).to(torch.bfloat16)
        packed, scales = quantize_nvfp4(flat, self.input_scale_quant)
        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids.to(torch.int32), topk_weights),
            routing_bias=None,
            hidden_states=packed,
            hidden_states_scale=scales,
            gemm1_weights=self.w13,
            gemm1_weights_scale=self.w13_scales,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.w2,
            gemm2_weights_scale=self.w2_scales,
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.g1_alphas,
            output2_scale_scalar=self.g2_alphas,
            num_experts=self.global_experts,
            top_k=self.top_k,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=8192,
            enable_pdl=True,
        )[0]
        return output.view(shape)


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiK25Config,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.experts = Nvfp4Experts(config, tp)
        self.shared_experts = DenseMlp(
            config.hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            tp,
            reduce,
        )
        self.reduce = reduce

    def forward(self, hidden: Tensor) -> Tensor:
        routed = self.experts(hidden)
        shared = self.shared_experts.partial(hidden)
        return self.reduce(routed + shared)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: KimiK25Config,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
        workspace: Tensor,
        frequencies: Tensor,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MlaAttention(config, tp, reduce, workspace, frequencies)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_id < config.first_dense_layers:
            self.mlp = DenseMlp(
                config.hidden_size, config.intermediate_size, tp, reduce
            )
        else:
            self.mlp = KimiMoE(config, tp, reduce)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        prefill: MlaPrefill | None = None,
    ) -> Tensor:
        residual = hidden
        attention = self.self_attn(
            self.input_layernorm(hidden),
            positions,
            kv_cache=cache.target,
            block_tables=cache.target_pages(self.layer_id),
            write_rows=cache.target_rows(self.layer_id, positions),
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            prefill=prefill,
        )
        hidden = residual + attention
        mlp = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp


class KimiTarget(nn.Module):
    """61-layer DeepSeek-V3 target with selected EAGLE auxiliary states."""

    def __init__(
        self,
        config: KimiK25Config,
        aux_layer_ids: tuple[int, ...],
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
        workspace: Tensor,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.aux_layer_ids = aux_layer_ids
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size, tp)
        frequencies = precompute_rope(
            config.qk_rope_head_dim, max_sequence_length, config.rope
        ).to(workspace.device)
        self.layers = nn.ModuleList(
            DecoderLayer(
                layer,
                config,
                tp,
                reduce,
                workspace,
                frequencies,
            )
            for layer in range(config.num_hidden_layers)
        )
        first_attention = self.layers[0].self_attn
        self.prefill = MlaPrefill(
            workspace,
            num_heads=first_attention.local_heads,
            kv_lora_rank=first_attention.kv_lora_rank,
            rope_head_dim=first_attention.qk_rope_head_dim,
            scale=first_attention.scale,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelHead(config.vocab_size, config.hidden_size, tp)

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        *,
        full_logits: bool,
        prefill_length: int | None = None,
        input_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = (
            self.embed_tokens(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if hidden.shape != (*token_ids.shape, self.config.hidden_size):
            raise ValueError("Kimi input embeddings do not match token IDs")
        auxiliary = []
        prefill = None
        if prefill_length is not None:
            self.prefill.plan(token_ids.shape[1], prefill_length)
            prefill = self.prefill
        for layer_id, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                positions,
                cache,
                sequence_lengths,
                max_sequence_length,
                prefill,
            )
            if layer_id in self.aux_layer_ids:
                auxiliary.append(hidden)
        if len(auxiliary) != len(self.aux_layer_ids):
            raise RuntimeError("Kimi target did not collect every EAGLE layer")
        normalized = self.norm(hidden)
        logits = self.lm_head(normalized if full_logits else normalized[:, -1])
        return logits, torch.cat(auxiliary, dim=-1)


class EagleDraft(nn.Module):
    """One BF16 EAGLE3/3.1 layer with its own paged MLA cache."""

    def __init__(
        self,
        config: KimiK25DraftConfig,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
        workspace: Tensor,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp = tp
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size, tp)
        self.fc_norm = (
            nn.ModuleList(
                RMSNorm(config.hidden_size, config.rms_norm_eps)
                for _ in config.aux_layer_ids
            )
            if config.fc_norm
            else None
        )
        self.fc = ColumnParallelLinear(
            config.hidden_size * len(config.aux_layer_ids), config.hidden_size, tp
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        frequencies = precompute_rope(
            config.qk_rope_head_dim, max_sequence_length, config.rope
        ).to(workspace.device)
        self.self_attn = MlaAttention(
            config,
            tp,
            reduce,
            workspace,
            frequencies,
            input_size=2 * config.hidden_size,
        )
        self.prefill = MlaPrefill(
            workspace,
            num_heads=self.self_attn.local_heads,
            kv_lora_rank=self.self_attn.kv_lora_rank,
            rope_head_dim=self.self_attn.qk_rope_head_dim,
            scale=self.self_attn.scale,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DenseMlp(config.hidden_size, config.intermediate_size, tp, reduce)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelHead(config.vocab_size, config.hidden_size, tp)

    def project_target(self, auxiliary: Tensor) -> Tensor:
        if self.fc_norm is not None:
            chunks = auxiliary.chunk(len(self.fc_norm), dim=-1)
            auxiliary = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)],
                dim=-1,
            )
        local = self.fc(auxiliary)
        if not self.tp.enabled:
            return local
        gathered = [torch.empty_like(local) for _ in range(self.tp.size)]
        dist.all_gather(gathered, local, group=self.tp.process_group)
        return torch.cat(gathered, dim=-1)

    def forward(
        self,
        token_ids: Tensor,
        target_hidden: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        *,
        return_logits: bool = True,
        prefill_length: int | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        embeds = self.input_layernorm(self.embed_tokens(token_ids))
        normalized_hidden = self.hidden_norm(target_hidden)
        attention_input = torch.cat((embeds, normalized_hidden), dim=-1)
        prefill = None
        if prefill_length is not None:
            self.prefill.plan(token_ids.shape[1], prefill_length)
            prefill = self.prefill
        hidden = target_hidden + self.self_attn(
            attention_input,
            positions,
            kv_cache=cache.draft,
            block_tables=cache.draft_pages(),
            write_rows=cache.draft_rows(positions),
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            prefill=prefill,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        normalized = self.norm(hidden)
        auxiliary = normalized if self.config.norm_output else hidden
        logits = self.lm_head(normalized) if return_logits else None
        return logits, auxiliary


class KimiForCausalLM(nn.Module):
    """Target plus EAGLE3 orchestration over one Engine-owned cache."""

    def __init__(
        self,
        target_config: KimiK25Config,
        draft_config: KimiK25DraftConfig,
        tp: TensorParallelContext,
        reduce: Callable[[Tensor], Tensor],
        workspace: Tensor,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.target = KimiTarget(
            target_config,
            draft_config.target_capture_layer_ids,
            tp,
            reduce,
            workspace,
            max_sequence_length,
        )
        self.draft = EagleDraft(
            draft_config,
            tp,
            reduce,
            workspace,
            max_sequence_length,
        )

    def prepare_nvfp4(self) -> None:
        for layer in self.target.layers:
            mlp = layer.mlp
            if isinstance(mlp, KimiMoE):
                mlp.experts.prepare()

    @torch.inference_mode()
    def target_forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        cache: DeviceCache,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        *,
        full_logits: bool,
        prefill_length: int | None = None,
        input_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits, auxiliary = self.target(
            token_ids,
            positions,
            cache,
            sequence_lengths,
            max_sequence_length,
            full_logits=full_logits,
            prefill_length=prefill_length,
            input_embeddings=input_embeddings,
        )
        projected = self.draft.project_target(auxiliary)
        cache.store_aux(projected, positions)
        return logits, projected


__all__ = [
    "DenseMlp",
    "EagleDraft",
    "KimiForCausalLM",
    "KimiMoE",
    "KimiTarget",
    "MlaAttention",
    "Nvfp4Experts",
    "ParallelEmbedding",
    "ParallelHead",
    "RMSNorm",
    "apply_rope",
    "precompute_rope",
]
