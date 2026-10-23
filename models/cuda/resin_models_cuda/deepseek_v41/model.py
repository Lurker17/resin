"""DeepSeek V4.1's shared-KV/CED and shifted-HC text architecture.

The runner, sampling, graph lifetime, EP transport and prepared FP4 expert
storage are shared with V4. Version-specific attention and residual semantics
are explicit here; checkpoint repository Python is never imported.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.context_parallel import ContextParallelTokens
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.loading import prepare_model
from resin.backend.torch import benchmark_expert_indices
from resin_models_cuda.deepseek_v4.experts import RoutedExperts as V4RoutedExperts
from resin_models_cuda.deepseek_v4.gemm import fp4_fused_moe
from resin_models_cuda.deepseek_v4.model import (
    ModelArgs as V4Args,
)
from resin_models_cuda.deepseek_v4.model import (
    MoE,
    MoEWeights,
    ParallelEmbedding,
    ParallelHead,
    RMSNorm,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from resin_models_cuda.deepseek_v41.attention import (
    fused_attention,
    gather_kv,
    pack_kv,
    store_attention_kv,
    store_kv,
)
from resin_models_cuda.deepseek_v41.hc import HCLazyPost, materialize, mega_transition
from resin_models_cuda.deepseek_v41.kernels import (
    act_quant,
    hc_split_sinkhorn,
)
from resin_models_cuda.deepseek_v41.linear import Linear, SharedExpert


@dataclass
class ModelArgs(V4Args):
    kv_source_layers: tuple[int, ...] = ()
    index_source_layers: tuple[int, ...] = ()
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 8
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 0
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    image_token_id: int = 129264


class RoutedExperts(V4RoutedExperts):
    def local(self, hidden, indices, weights):
        if not hidden.shape[0]:
            return torch.zeros_like(
                hidden, dtype=torch.float32 if self.dispatch else hidden.dtype
            )
        quantized, scales = act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu)
        partial = fp4_fused_moe(
            quantized,
            scales.view(torch.float8_e4m3fn),
            indices,
            weights,
            self.w13_weight,
            self.w13_scale,
            self.w2_weight,
            self.w2_scale,
            self.output_scales,
            self.clamp_limits,
            num_experts=self.num_experts,
            local_expert_offset=self.start,
            scale_dtype=torch.float8_e8m0fnu,
        )
        return partial.float() if self.dispatch else partial


class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.topk, self.dim = args.n_activated_experts, args.dim
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(
            torch.empty(args.n_routed_experts, dtype=torch.float32)
        )
        self.temperature, self.route_scale = args.gate_temp, args.route_scale
        self.normalized, self.score_func = args.norm_topk_prob, args.score_func
        self.benchmark_balanced_experts = False
        self.benchmark_replica = 0
        self.benchmark_expert_parallel_size = 1

    def forward(self, hidden, _input_ids):
        scores = F.linear(hidden.float(), self.weight.float()) / self.temperature
        scores = (
            scores.softmax(-1)
            if self.score_func == "softmax"
            else scores.sigmoid()
            if self.score_func == "sigmoid"
            else F.softplus(scores).sqrt()
        )
        if self.benchmark_balanced_experts:
            indices = benchmark_expert_indices(
                scores,
                self.topk,
                self.benchmark_replica,
                self.benchmark_expert_parallel_size,
            )
        else:
            indices = (scores + self.bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.normalized and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return (weights * self.route_scale).bfloat16(), indices.int()


class Compressor(nn.Module):
    """Checkpoint layout and reference pooling; lowered to PreparedCompressor."""

    def __init__(self, args, layer):
        super().__init__()
        self.layer, self.ratio = layer, args.compress_ratios[layer]
        self.wkv = Linear(
            args.dim,
            args.head_dim,
            dtype=torch.float32 if self.ratio > 1 else torch.bfloat16,
        )
        if self.ratio > 1:
            self.wgate = Linear(args.dim, args.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(args.head_dim, args.norm_eps)

    def forward(self, hidden, positions, cache):
        if self.ratio == 1:
            return self.norm(self.wkv(hidden))
        values, scores = self.wkv(hidden.float()), self.wgate(hidden.float())
        vr, sr = cache.compressor_rows(self.layer, positions)
        pool = cache.compressor_values
        pool.index_copy_(0, vr.flatten(), values.reshape(-1, values.shape[-1]))
        pool.index_copy_(0, sr.flatten(), scores.reshape(-1, scores.shape[-1]))
        group = (
            positions[..., None]
            - positions[..., None].remainder(self.ratio)
            + torch.arange(self.ratio, device=hidden.device)
        )
        vr, sr = cache.compressor_rows(self.layer, group)
        latent = (pool[vr] * pool[sr].softmax(-2)).sum(-2).to(hidden.dtype)
        return self.norm(latent)


class Indexer(nn.Module):
    def __init__(self, args, layer, tp):
        super().__init__()
        self.layer, self.ratio = layer, args.compress_ratios[layer]
        self.owns_k = layer in args.kv_source_layers
        self.n_heads, self.local_heads = (
            args.index_n_heads,
            args.index_n_heads // tp.size,
        )
        self.head_dim, self.rope_dim = args.index_head_dim, args.rope_head_dim
        self.topk = args.index_topk
        self.candidate_source = args.candidate_source_layer
        self.candidate_topk, self.block_size = (
            args.candidate_topk_blocks,
            args.candidate_block_size,
        )
        self.wq_b = Linear(
            args.q_lora_rank,
            args.index_n_heads * args.index_head_dim,
            tensor_parallel=tp,
            axis=0,
        )
        self.weights_proj = Linear(
            args.dim,
            args.index_n_heads,
            dtype=torch.bfloat16,
            tensor_parallel=tp,
            axis=0,
        )
        if self.owns_k:
            self.wk = Linear(args.head_dim, args.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(args.index_head_dim, args.norm_eps)
        self.allreduce = tp.sum

    def forward(
        self,
        hidden,
        qr,
        latent,
        positions,
        capacity,
        cache,
        freqs,
        runtime,
        *,
        query=None,
    ):
        from resin_models_cuda.deepseek_v41.indexer import paged_index

        query = self.wq_b(qr) if query is None else query
        return paged_index(
            self, hidden, query, latent, positions, capacity, cache, freqs, runtime
        )


class Attention(nn.Module):
    freqs_cis: torch.Tensor
    project_input: Callable
    project_query: Callable

    def __init__(self, layer, args, tp):
        super().__init__()
        if tp.size != 1:
            raise ValueError("DeepSeek-V4.1 MLA has one KV head and requires attn_tp=1")
        self.layer_id, self.dim, self.head_dim = layer, args.dim, args.head_dim
        self.q_lora_rank, self.rope_head_dim = args.q_lora_rank, args.rope_head_dim
        self.n_local_heads, self.n_local_groups = (
            args.n_heads // tp.size,
            args.o_groups // tp.size,
        )
        self.window_size, self.ratio = args.window_size, args.compress_ratios[layer]
        self.o_lora_rank, self.softmax_scale = args.o_lora_rank, args.head_dim**-0.5
        self.wq_a, self.wkv = (
            Linear(args.dim, args.q_lora_rank),
            Linear(args.dim, args.head_dim),
        )
        self.q_norm, self.kv_norm = (
            RMSNorm(args.q_lora_rank, args.norm_eps),
            RMSNorm(args.head_dim, args.norm_eps),
        )
        self.wq_b = Linear(
            args.q_lora_rank, args.n_heads * args.head_dim, tensor_parallel=tp, axis=0
        )
        self.wo_a = Linear(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            tensor_parallel=tp,
            axis=0,
        )
        self.wo_b = Linear(
            args.o_groups * args.o_lora_rank, args.dim, tensor_parallel=tp, axis=1
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        if layer in args.kv_source_layers:
            self.compressor = Compressor(args, layer)
        if layer in args.index_source_layers:
            self.indexer = Indexer(args, layer, tp)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.rope_head_dim,
                args.max_seq_len,
                args.original_seq_len if self.ratio else 0,
                args.compress_rope_theta if self.ratio else args.rope_theta,
                args.rope_factor,
                args.beta_fast,
                args.beta_slow,
            ),
            persistent=False,
        )
        self.prefill_project = self.prefill_query = self.prefill_finish = None

    def _project(self, hidden):
        if hasattr(self, "project_input"):
            return self.project_input(hidden)
        return self.q_norm(self.wq_a(hidden)), self.wkv(hidden)

    def _finish(self, output, positions):
        output = self.wo_a(*output)
        return self.wo_b(output.reshape(*positions.shape, -1))

    def store(self, hidden, positions, cache):
        if hasattr(self, "cos_sin"):
            store_attention_kv(
                self,
                self.wkv(hidden),
                positions,
                cache.window_cache(self.layer_id),
                cache.window_rows(self.layer_id, positions),
            )
            return
        kv = self.kv_norm(self.wkv(hidden))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], self.freqs_cis[positions])
        store_kv(
            cache.window_cache(self.layer_id),
            cache.window_rows(self.layer_id, positions),
            pack_kv(kv),
        )

    def forward(
        self,
        hidden,
        positions,
        capacity,
        cache,
        runtime,
        context_parallel=None,
        *,
        prefill=False,
    ):
        # Prefill projection callbacks adapt one sequence to flat graph rows;
        # decode and verification retain their independent batch dimension.
        qr, kv = (
            self.prefill_project if prefill and self.prefill_project else self._project
        )(hidden)
        if hasattr(self, "cos_sin"):
            store_attention_kv(
                self,
                kv,
                positions,
                cache.window_cache(self.layer_id),
                cache.window_rows(self.layer_id, positions),
            )
        else:
            kv = self.kv_norm(kv)
            apply_rotary_emb(kv[..., -self.rope_head_dim :], self.freqs_cis[positions])
            store_kv(
                cache.window_cache(self.layer_id),
                cache.window_rows(self.layer_id, positions),
                pack_kv(kv),
            )
        project_query = (
            self.prefill_query
            if prefill and self.prefill_query
            else getattr(self, "project_query", None)
        )
        projected = project_query(qr) if project_query else self.wq_b(qr)
        query, index_query = (
            projected if isinstance(projected, tuple) else (projected, None)
        )
        latent = (
            self.compressor(hidden, positions, cache)
            if hasattr(self, "compressor")
            else None
        )
        if hasattr(self, "indexer"):
            runtime["indices"] = self.indexer(
                hidden,
                qr,
                latent,
                positions,
                capacity,
                cache,
                self.freqs_cis,
                runtime,
                query=index_query,
            )
        if latent is not None:
            compressed_positions = positions - positions.remainder(self.ratio)
            if hasattr(self, "cos_sin"):
                store_attention_kv(
                    self,
                    latent,
                    compressed_positions,
                    cache.compressed_cache(self.layer_id),
                    torch.where(
                        (positions + 1).remainder(self.ratio) == 0,
                        cache.compressed_rows(self.layer_id, self.ratio, positions),
                        -1,
                    ),
                    compressed=True,
                )
            else:
                apply_rotary_emb(
                    latent[..., -self.rope_head_dim :],
                    self.freqs_cis[compressed_positions],
                )
                store_kv(
                    cache.compressed_cache(self.layer_id),
                    cache.compressed_rows(self.layer_id, self.ratio, positions),
                    pack_kv(latent, fp4=True),
                )
        if context_parallel is not None:
            query, positions = (
                query.index_select(1, context_parallel.token_indices),
                positions.index_select(1, context_parallel.token_indices),
            )
            if not positions.shape[1]:
                return context_parallel.gather(hidden.new_empty(0, self.dim))[None]
        query = query.unflatten(-1, (self.n_local_heads, self.head_dim))
        columns = torch.arange(self.window_size, device=hidden.device)
        counts = (positions + 1).clamp_max(self.window_size)
        windows = positions[..., None] + 1 - counts[..., None] + columns
        indices = torch.where(
            columns < counts[..., None], cache.window_rows(self.layer_id, windows), -1
        )
        extra_cache = extra_indices = None
        if self.ratio:
            selected = runtime["indices"]
            if context_parallel is not None:
                selected = selected.index_select(1, context_parallel.token_indices)
            physical = cache.compressed_rows(
                self.layer_id, self.ratio, selected.clamp_min(0) * self.ratio
            )
            extra_cache = cache.compressed_cache(self.layer_id)
            extra_indices = torch.where(selected >= 0, physical, -1)
        if prefill:
            # Gather compact BF16 inputs for the upstream prefill kernel;
            # retained history stays packed in the shared slab pool.
            start = max(0, capacity - hidden.shape[1] - self.window_size + 1)
            history = torch.arange(start, capacity, device=hidden.device)[None]
            values = gather_kv(
                cache.window_cache(self.layer_id),
                cache.window_rows(self.layer_id, history),
            )
            indices = torch.where(columns < counts[..., None], windows - start, -1)
            if self.ratio:
                compressed = (
                    torch.arange(capacity // self.ratio, device=hidden.device)[None]
                    * self.ratio
                )
                extra = gather_kv(
                    extra_cache,
                    cache.compressed_rows(self.layer_id, self.ratio, compressed),
                )
                indices = torch.cat(
                    (
                        indices,
                        torch.where(selected >= 0, selected + values.shape[0], -1),
                    ),
                    -1,
                )
                values = torch.cat((values, extra))
            output = fused_attention(
                self, query, positions, values[:, None], indices, prefill=True
            )
        else:
            output = fused_attention(
                self,
                query,
                positions,
                cache.window_cache(self.layer_id),
                indices,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
            )
        output = (
            self.prefill_finish if prefill and self.prefill_finish else self._finish
        )(output, positions)
        return (
            context_parallel.gather(output.squeeze(0))[None]
            if context_parallel is not None
            else output
        )


class Block(nn.Module):
    main_norm: RMSNorm
    main_proj: Linear
    norm: RMSNorm
    markov_head: Any
    confidence_head: Any
    engram: Any
    """Shifted HC: attention consumes the previous FFN's pre-mix."""

    def __init__(self, layer, args, tp, ep, etp, engram=None):
        super().__init__()
        self.layer_id, self.args = layer, args
        self.attn = Attention(layer, args, tp)
        moe_args = (
            replace(
                args,
                n_routed_experts=args.dspark_n_routed_experts or args.n_routed_experts,
                n_activated_experts=args.dspark_n_activated_experts
                or args.n_activated_experts,
            )
            if layer >= args.n_layers
            else args
        )
        self.ffn: Any = MoEWeights(
            layer, moe_args, expert_parallel=ep, expert_tensor_parallel=etp
        )
        self.ffn.gate = Gate(moe_args)
        self.ffn.gate.benchmark_expert_parallel_size = ep.size
        self.ffn.shared_experts = SharedExpert(args)
        self.engram = engram
        self.attn_norm, self.ffn_norm = (
            RMSNorm(args.dim, args.norm_eps),
            RMSNorm(args.dim, args.norm_eps),
        )
        self.prefill_ffn = None
        self.lazy_hc = False
        for name in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{name}_fn",
                nn.Parameter(
                    torch.empty(
                        (2 + args.hc_mult) * args.hc_mult,
                        args.hc_mult * args.dim,
                        dtype=torch.float32,
                    )
                ),
            )
            setattr(
                self,
                f"hc_{name}_base",
                nn.Parameter(
                    torch.empty((2 + args.hc_mult) * args.hc_mult, dtype=torch.float32)
                ),
            )
            setattr(
                self,
                f"hc_{name}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32)),
            )

    def mixes(self, hidden, name):
        flat = hidden.flatten(2).float()
        mixed = F.linear(flat, getattr(self, f"hc_{name}_fn")) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.args.norm_eps
        )
        return hc_split_sinkhorn(
            mixed,
            getattr(self, f"hc_{name}_scale"),
            getattr(self, f"hc_{name}_base"),
            self.args.hc_mult,
            self.args.hc_sinkhorn_iters,
            self.args.hc_eps,
        )

    @staticmethod
    def collapse(hidden, pre):
        return (hidden.float() * pre[..., None]).sum(-2).to(hidden.dtype)

    @staticmethod
    def expand(output, residual, post, comb):
        return (
            post[..., None] * output[..., None, :]
            + (comb[..., None] * residual.unsqueeze(-2)).sum(-3)
        ).to(output.dtype)

    def forward(self, hidden, pre, ids, attention, *, prefill=False):
        if self.lazy_hc:
            if isinstance(pre, HCLazyPost):
                value, state = mega_transition(self, hidden, pre, "attn")
            else:
                current_pre, post, comb = self.mixes(hidden, "attn")
                value = self.attn_norm(self.collapse(hidden, pre))
                state = HCLazyPost(hidden, current_pre, post, comb)
            output = attention(value)
            prepared = not prefill
            value, state = mega_transition(
                self, output, state, "ffn", prepare_ffn=prepared
            )
            output = (
                self.ffn.forward_prepared(value, ids)
                if prepared
                else (self.prefill_ffn or self.ffn)(value, ids)
            )
            return output, state
        residual = hidden
        attn_pre, post, comb = self.mixes(hidden, "attn")
        output = attention(self.attn_norm(self.collapse(hidden, pre)))
        residual = self.expand(output, residual, post, comb)
        ffn_pre, post, comb = self.mixes(residual, "ffn")
        hidden = self.ffn_norm(self.collapse(residual, attn_pre))
        output = (self.prefill_ffn or self.ffn)(hidden, ids)
        return self.expand(output, residual, post, comb), ffn_pre


class MarkovHead(nn.Module):
    def __init__(self, args, tp):
        super().__init__()
        self.embed = ParallelEmbedding(
            args.vocab_size, args.dspark_markov_rank, tensor_parallel=tp
        )
        self.head = ParallelHead(
            args.vocab_size, args.dspark_markov_rank, tensor_parallel=tp
        )

    def forward(self, ids):
        embedded = self.embed(ids)
        return self.head(embedded, full_logits=True), embedded


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden, markov_embed):
        return self.proj(torch.cat((hidden, markov_embed), -1).float()).squeeze(-1)


class Transformer(nn.Module):
    layers: Sequence[Block]
    mtp: Sequence[Block]
    """The V4.1 model on Engine-owned pages; no per-request Python state."""

    def __init__(
        self, args, *, tensor_parallel=None, parallel_groups=None, tokenizer=None
    ):
        super().__init__()
        tp = tensor_parallel or TensorParallelContext()
        if tp.size != 1 or (
            parallel_groups is not None and parallel_groups.attn_tp.size != 1
        ):
            raise ValueError("DeepSeek-V4.1 MLA has one KV head and requires attn_tp=1")
        ep = (
            TensorParallelLayout(
                rank=parallel_groups.ep.local_rank, size=parallel_groups.ep.size
            )
            if parallel_groups
            else tp.layout
        )
        etp = (
            TensorParallelLayout(
                rank=parallel_groups.expert_tp.local_rank,
                size=parallel_groups.expert_tp.size,
            )
            if parallel_groups
            else TensorParallelLayout()
        )
        args.expert_intermediate_shard(etp)
        ep.even_shard(args.n_routed_experts, name="n_routed_experts")
        if args.n_mtp_layers:
            ep.even_shard(args.dspark_n_routed_experts, name="dspark_n_routed_experts")
        self.args, self.parallel_groups, self.tp = args, parallel_groups, tp
        self.embed = ParallelEmbedding(args.vocab_size, args.dim, tensor_parallel=tp)
        self.layers = cast(
            Sequence[Block],
            nn.ModuleList(Block(i, args, tp, ep, etp) for i in range(args.n_layers)),
        )
        self.mtp = cast(
            Sequence[Block],
            nn.ModuleList(
                Block(args.n_layers + i, args, tp, ep, etp)
                for i in range(args.n_mtp_layers)
            ),
        )
        self.norm, self.head = (
            RMSNorm(args.dim, args.norm_eps),
            ParallelHead(args.vocab_size, args.dim, tensor_parallel=tp),
        )
        self.hc_mult = args.hc_mult
        self.dspark_block_size, self.dspark_noise_token_id = (
            args.dspark_block_size,
            args.dspark_noise_token_id,
        )
        self.target_layer_ids = frozenset(
            args.dspark_target_layer_ids if self.mtp else ()
        )
        if args.engram_layer_ids:
            from resin.backend.embedding import ShardedEmbeddingLookup
            from resin_models_cuda.deepseek_v41.engram import (
                Engram,
                EngramRuntime,
                NgramHash,
            )

            self.engram_hash = NgramHash(args, tokenizer)
            lookup = (
                ShardedEmbeddingLookup(
                    parallel_groups.device_domain, args.engram_head_dim
                )
                if parallel_groups is not None
                and parallel_groups.device_domain.size > 1
                else None
            )
            for layer in self.layers:
                if layer.layer_id in args.engram_layer_ids:
                    layer.engram = Engram(args, layer.layer_id, lookup=lookup)
            self.engram_runtime = EngramRuntime(
                {
                    layer.layer_id: layer.engram
                    for layer in self.layers
                    if layer.engram is not None
                }
            )
        else:
            self.engram_hash = None
        if self.mtp:
            first, last = self.mtp[0], self.mtp[-1]
            first.main_proj = Linear(
                args.dim * len(args.dspark_target_layer_ids), args.dim
            )
            first.main_norm = RMSNorm(args.dim, args.norm_eps)
            last.norm = RMSNorm(args.dim, args.norm_eps)
            last.markov_head = MarkovHead(args, tp)
            last.confidence_head = ConfidenceHead(args.dim + args.dspark_markov_rank)

    def prepare_weights(self, reducer, *, expert_backend="flashinfer", max_tokens=1):
        collective = None
        if expert_backend == "deepgemm":
            from resin.backend.cuda.mega_moe import MegaMoeCollective

            if self.parallel_groups is None:
                raise ValueError("DeepGEMM requires explicit parallel groups")
            if self.args.hc_mult != 4 or self.args.dim % 1024:
                raise ValueError(
                    "DeepGEMM shifted HC requires hc_mult=4 and dim divisible by 1024"
                )
            collective = MegaMoeCollective(self.parallel_groups.ep, max_tokens)
        elif expert_backend != "flashinfer":
            raise ValueError(f"unknown expert backend: {expert_backend}")

        def lower(source):
            if isinstance(source, Attention):
                from resin_models_cuda.deepseek_v41.attention import prepare_attention

                prepare_attention(source)
            if isinstance(source, Block) and collective is not None:
                source.lazy_hc = True
            if isinstance(source, ParallelHead):
                from resin_models_cuda.deepseek_v41.linear import PreparedVocabHead

                return PreparedVocabHead(source)
            if isinstance(source, Linear):
                from resin_models_cuda.deepseek_v41.linear import prepare_linear

                return prepare_linear(source)
            if not isinstance(source, MoEWeights):
                return source
            if collective is not None:
                from resin_models_cuda.deepseek_v41.experts import prepare_mega_moe

                return prepare_mega_moe(source, collective)
            return MoE(
                source,
                RoutedExperts(
                    source,
                    reduce=reducer,
                    block_size=32,
                    scale_fmt="ue8m0",
                    scale_dtype=torch.float8_e8m0fnu,
                    groups=self.parallel_groups,
                ),
            )

        prepare_model(
            self,
            lower,
        )

    def set_allreduce(self, reducer, *, indexer_reducer=None):
        for module in self.modules():
            if (
                isinstance(module, ParallelEmbedding)
                or isinstance(module, Linear)
                and module.axis == 1
            ):
                module.allreduce = reducer
            elif isinstance(module, Indexer):
                module.allreduce = indexer_reducer or reducer

    def bind_cache_arena(self, _arena):
        pass  # Every state access receives the current immutable page tables.

    def forward_experts_empty(self, *, stage="target", prefill=False):
        hooks = (
            self.engram_runtime.fetch(None)
            if stage == "target" and self.engram_hash is not None
            else {}
        )
        for layer in self.layers if stage == "target" else self.mtp:
            if layer.engram is not None:
                hooks[layer.layer_id]()
            if prefill:
                weight = self.embed.weight
                operation = layer.prefill_ffn
                if operation is None:
                    raise RuntimeError("empty prefill requires prepared FFN callbacks")
                operation(
                    weight.new_empty((0, self.args.dim)),
                    weight.new_empty((0,), dtype=torch.int64),
                )
            else:
                layer.ffn.forward_empty()

    def _run(self, ids, positions, capacity, cache, cp=None, *, prefill=False):
        hashes = (
            self.engram_hash(ids, positions, cache)
            if self.engram_hash is not None
            else None
        )
        hooks = self.engram_runtime.fetch(hashes) if hashes is not None else {}
        hidden = self.embed(ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre = hidden.new_zeros(*ids.shape, self.hc_mult, dtype=torch.float32)
        pre[..., 0] = 1
        runtime, collected = {}, []
        for layer in self.layers:
            # Engram and feature taps observe all residual streams. Everywhere
            # else, keep the previous post-mix lazy so the next HC can fuse it.
            if isinstance(pre, HCLazyPost) and (
                layer.engram is not None or layer.layer_id in self.target_layer_ids
            ):
                hidden, pre = materialize(hidden, pre), pre.pre
            if layer.engram is not None:
                assert hashes is not None
                hidden = layer.engram(hidden, embed_hook=hooks[layer.layer_id])
            if layer.layer_id in self.target_layer_ids:
                collected.append(hidden.mean(-2))
            hidden, pre = layer(
                hidden,
                pre,
                ids,
                lambda value, layer=layer: layer.attn(
                    value, positions, capacity, cache, runtime, cp, prefill=prefill
                ),
                prefill=prefill,
            )
        if isinstance(pre, HCLazyPost):
            hidden, pre = materialize(hidden, pre), pre.pre
        return self.norm(Block.collapse(hidden, pre)), torch.cat(
            collected, -1
        ) if collected else None

    @torch.inference_mode()
    def forward_prefill(self, input_ids, start_position, *, cache, return_hidden=False):
        positions = (
            start_position
            + torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        )
        cp = (
            ContextParallelTokens(
                (input_ids.shape[1],),
                self.parallel_groups.cp,
                device=input_ids.device,
                distributed=self.parallel_groups.distributed,
            )
            if self.parallel_groups is not None and self.parallel_groups.cp.size > 1
            else None
        )
        hidden, main = self._run(
            input_ids,
            positions,
            start_position + input_ids.shape[1],
            cache,
            cp,
            prefill=True,
        )
        return (hidden if return_hidden else self.head(hidden)), main

    @torch.inference_mode()
    def forward_decode(self, input_ids, position, graph_seq_len, *, cache):
        hidden, _ = self._run(input_ids, position[:, None], graph_seq_len, cache)
        return self.head(hidden)

    @torch.inference_mode()
    def forward_verify(self, input_ids, position, graph_seq_len, *, cache):
        positions = position[:, None] + torch.arange(
            input_ids.shape[1], device=input_ids.device
        )
        hidden, main = self._run(input_ids, positions, graph_seq_len, cache)
        return self.head(hidden, full_logits=True), main

    def clear_rejected_speculative_state(
        self, positions, processed_counts, block_length, *, cache
    ):
        # Future rows can remain physically present: causal lengths hide them,
        # and corrected token, compressor and n-gram rows overwrite before use.
        pass

    def _project_main(self, hidden):
        return self.mtp[0].main_norm(self.mtp[0].main_proj(hidden))

    def prefill_dspark(self, main_hidden, *, cache):
        hidden = self._project_main(main_hidden)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
        for layer in self.mtp:
            layer.attn.store(hidden, positions, cache)

    def commit_dspark_main_block(self, main_hidden, position, *, cache):
        hidden = self._project_main(main_hidden)
        positions = position[:, None] + torch.arange(
            hidden.shape[1], device=hidden.device
        )
        for layer in self.mtp:
            layer.attn.store(hidden, positions, cache)

    def _draft_attention(self, attention, hidden, position, cache):
        batch, block = hidden.shape[:2]
        positions = position[:, None] + torch.arange(block, device=hidden.device)
        qr, kv = attention._project(hidden)
        query = attention.wq_b(qr).unflatten(
            -1, (attention.n_local_heads, attention.head_dim)
        )
        drafts = torch.arange(batch, device=hidden.device)[
            :, None
        ] * cache.geometry.draft_tokens + torch.arange(block, device=hidden.device)
        # Draft keys are transient and bidirectional within this block. They
        # must never overwrite the committed main-hidden history at the anchor.
        if hasattr(attention, "cos_sin"):
            store_attention_kv(attention, kv, positions, cache.draft_cache, drafts)
        else:
            kv = attention.kv_norm(kv)
            apply_rotary_emb(
                kv[..., -attention.rope_head_dim :], attention.freqs_cis[positions]
            )
            store_kv(cache.draft_cache, drafts, pack_kv(kv))
        columns = torch.arange(attention.window_size, device=hidden.device)
        counts = position.clamp_max(attention.window_size)
        history = position[:, None] - counts[:, None] + columns
        indices = torch.where(
            columns < counts[:, None],
            cache.window_rows(attention.layer_id, history),
            -1,
        )
        indices = indices[:, None].expand(batch, block, -1)
        output = fused_attention(
            attention,
            query,
            positions,
            cache.window_cache(attention.layer_id),
            indices,
            extra_cache=cache.draft_cache,
            extra_indices=drafts[:, None].expand(batch, block, -1),
        )
        return attention._finish(output, positions)

    @torch.inference_mode()
    def draft(self, anchor_ids, position, *, cache):
        ids = anchor_ids.new_full(
            (anchor_ids.shape[0], self.dspark_block_size), self.dspark_noise_token_id
        )
        ids[:, 0] = anchor_ids.flatten()
        hidden = self.embed(ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre = hidden.new_zeros(*ids.shape, self.hc_mult, dtype=torch.float32)
        pre[..., 0] = 1
        for layer in self.mtp:
            hidden, pre = layer(
                hidden,
                pre,
                ids,
                lambda value, layer=layer: self._draft_attention(
                    layer.attn, value, position, cache
                ),
            )
        last = self.mtp[-1]
        if isinstance(pre, HCLazyPost):
            hidden, pre = materialize(hidden, pre), pre.pre
        hidden = Block.collapse(hidden, pre)
        logits = self.head(last.norm(hidden), full_logits=True)
        previous, outputs = anchor_ids.flatten(), []
        for i in range(self.dspark_block_size):
            bias, _ = last.markov_head(previous)
            previous = (logits[:, i] + bias).argmax(-1)
            outputs.append(previous)
        return torch.stack(outputs, -1)
