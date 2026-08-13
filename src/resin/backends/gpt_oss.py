# Portions of the model implementation are adapted from OpenAI gpt-oss and
# Hugging Face Transformers.
# Copyright 2025 OpenAI and The HuggingFace Team. Licensed under Apache-2.0.

"""Tokenizer-free GPT-OSS execution on PyTorch.

The implementation follows OpenAI's reference model and the public Hugging
Face checkpoint layout. Resin's Rust control plane owns request lifecycle and
batch construction; this module owns weights and physical per-request KV state.
PyTorch, Hugging Face Hub, and safetensors are imported only when this backend
is actually used.

The dense expert implementation is deliberately isolated behind the
``experts`` module boundary. The released 20B and 120B checkpoints store those
matrices as MXFP4 blocks, so a Hopper MXFP4 expert kernel can replace that one
module without changing attention, routing, caching, or the runner contract.
"""

from __future__ import annotations

import glob
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resin import BatchPlan


DEFAULT_MODEL = "openai/gpt-oss-20b"
MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"
MODEL_REVISIONS = {
    MODEL_20B: "6cee5e81ee83917806bbde320786a8fb61efebee",
    MODEL_120B: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
}
_MODEL_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
)


@dataclass(frozen=True, slots=True)
class _GptOssArgs:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    head_dim: int
    sliding_window: int
    rms_norm_eps: float
    rope_theta: float
    initial_context_length: int
    rope_scaling_factor: float
    rope_beta_fast: float
    rope_beta_slow: float
    max_position_embeddings: int
    swiglu_limit: float
    attention_bias: bool
    layer_types: tuple[str, ...]
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_local_experts": self.num_local_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "head_dim": self.head_dim,
            "sliding_window": self.sliding_window,
            "initial_context_length": self.initial_context_length,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"GPT-OSS dimensions must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"unsupported GPT-OSS layer types: {sorted(unsupported)}")
        if self.rope_scaling_factor < 1.0:
            raise ValueError("rope scaling factor must be at least one")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.attention_dropout != 0.0:
            raise ValueError("the GPT-OSS inference runner requires zero dropout")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _GptOssArgs:
        if config.get("model_type") != "gpt_oss":
            raise ValueError(
                f"expected a GPT-OSS checkpoint, got {config.get('model_type')!r}"
            )

        hf_experts = config.get("num_local_experts")
        reference_experts = config.get("num_experts")
        if (
            hf_experts is not None
            and reference_experts is not None
            and hf_experts != reference_experts
        ):
            raise ValueError("conflicting GPT-OSS expert counts in config")
        num_experts = hf_experts if hf_experts is not None else reference_experts
        if num_experts is None:
            raise ValueError("GPT-OSS config is missing num_local_experts")

        hf_top_k = config.get("num_experts_per_tok")
        reference_top_k = config.get("experts_per_token")
        if (
            hf_top_k is not None
            and reference_top_k is not None
            and hf_top_k != reference_top_k
        ):
            raise ValueError("conflicting GPT-OSS top-k values in config")
        top_k = hf_top_k if hf_top_k is not None else reference_top_k
        if top_k is None:
            raise ValueError("GPT-OSS config is missing num_experts_per_tok")

        rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
        factor = rope.get("factor", config.get("rope_scaling_factor", 1.0))
        initial_context = rope.get(
            "original_max_position_embeddings",
            config.get("initial_context_length", 4096),
        )
        beta_fast = rope.get("beta_fast", config.get("rope_ntk_beta", 32.0))
        beta_slow = rope.get("beta_slow", config.get("rope_ntk_alpha", 1.0))

        layer_types = config.get("layer_types")
        if layer_types is None:
            layer_types = [
                "sliding_attention" if index % 2 == 0 else "full_attention"
                for index in range(config["num_hidden_layers"])
            ]

        if config.get("tie_word_embeddings", False):
            raise ValueError("GPT-OSS requires untied input and output embeddings")

        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            head_dim=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            sliding_window=config["sliding_window"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", config.get("rope_theta", 150_000.0)),
            initial_context_length=initial_context,
            rope_scaling_factor=factor,
            rope_beta_fast=beta_fast,
            rope_beta_slow=beta_slow,
            max_position_embeddings=config.get("max_position_embeddings", 131_072),
            swiglu_limit=config.get("swiglu_limit", 7.0),
            attention_bias=config.get("attention_bias", True),
            layer_types=tuple(layer_types),
            attention_dropout=config.get("attention_dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
            pad_token_id=config.get("pad_token_id"),
            eos_token_id=config.get("eos_token_id"),
        )


class _KVCache:
    """Physical KV state for one request and one attention layer.

    Full-attention layers retain all keys. Sliding layers retain exactly the
    preceding ``window - 1`` keys needed by the next token. A multi-token chunk
    still attends over its complete local chunk before retained state is cut.
    """

    def __init__(self, torch, sliding_window: int | None) -> None:
        self._torch = torch
        self.sliding_window = sliding_window
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def stored_tokens(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]

    def update_and_fetch(self, keys, values):
        previous = self.stored_tokens
        key_start = self.offset - previous
        if key_start < 0:
            raise ValueError("KV cache contains more tokens than the request has seen")

        if self.keys is None:
            combined_keys = keys
            combined_values = values
        else:
            combined_keys = self._torch.cat((self.keys, keys), dim=-2)
            combined_values = self._torch.cat((self.values, values), dim=-2)

        if self.sliding_window is None:
            self.keys = combined_keys
            self.values = combined_values
        else:
            keep = self.sliding_window - 1
            if keep:
                self.keys = combined_keys[..., -keep:, :]
                self.values = combined_values[..., -keep:, :]
            else:
                self.keys = combined_keys[..., :0, :]
                self.values = combined_values[..., :0, :]
        self.offset += keys.shape[-2]
        return combined_keys, combined_values, key_start


class _SequenceCache:
    def __init__(self, torch, args: _GptOssArgs) -> None:
        self.layers = [
            _KVCache(
                torch,
                args.sliding_window if layer_type == "sliding_attention" else None,
            )
            for layer_type in args.layer_types
        ]
        self.offset = 0


class _BatchedKVCache:
    """A transient tensor view over equally sized per-request layer caches."""

    def __init__(self, torch, caches: tuple[_KVCache, ...]) -> None:
        if not caches:
            raise ValueError("a physical cache batch must contain at least one request")
        self._torch = torch
        self._caches = caches
        self.sliding_window = caches[0].sliding_window
        if any(cache.sliding_window != self.sliding_window for cache in caches[1:]):
            raise ValueError("batched KV caches must use the same attention window")
        stored_tokens = caches[0].stored_tokens
        if any(cache.stored_tokens != stored_tokens for cache in caches[1:]):
            raise ValueError(
                "physical batching requires equal per-layer KV cache lengths"
            )

    def update_and_fetch(self, keys, values):
        if keys.shape[0] != len(self._caches) or values.shape[0] != len(self._caches):
            raise ValueError("the tensor batch does not match the physical cache batch")

        combined_keys = []
        combined_values = []
        key_starts = []
        for index, cache in enumerate(self._caches):
            # Keep request ownership physical as well as logical. A mere batch
            # slice would retain the entire projection allocation after sibling
            # requests are released.
            request_keys, request_values, key_start = cache.update_and_fetch(
                keys[index : index + 1].clone(),
                values[index : index + 1].clone(),
            )
            combined_keys.append(request_keys)
            combined_values.append(request_values)
            key_starts.append(key_start)
        return (
            self._torch.cat(combined_keys, dim=0),
            self._torch.cat(combined_values, dim=0),
            self._torch.tensor(key_starts, dtype=self._torch.long, device=keys.device),
        )


class _BatchedSequenceCache:
    """One model-call view that preserves independent request cache ownership."""

    def __init__(self, torch, caches: tuple[_SequenceCache, ...]) -> None:
        if len(caches) < 2:
            raise ValueError("a physical cache batch requires at least two requests")
        layer_count = len(caches[0].layers)
        if any(len(cache.layers) != layer_count for cache in caches[1:]):
            raise ValueError("batched sequence caches must have identical layer counts")
        for cache in caches:
            if any(layer.offset != cache.offset for layer in cache.layers):
                raise ValueError("sequence and layer cache offsets are inconsistent")
        self._caches = caches
        self.layers = [
            _BatchedKVCache(
                torch,
                tuple(cache.layers[layer_index] for cache in caches),
            )
            for layer_index in range(layer_count)
        ]

    @property
    def batch_size(self) -> int:
        return len(self._caches)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(cache.offset for cache in self._caches)

    def advance(self, token_count: int) -> None:
        for cache in self._caches:
            cache.offset += token_count


def _make_model_class(torch, *, expert_factory: Callable | None = None):
    """Build the module hierarchy lazily, without a module-level torch import.

    ``expert_factory`` is the intentional kernel seam. It receives parsed model
    arguments and must return a module implementing
    ``forward(tokens, topk_indices, topk_weights, router_logits)``. Packed
    kernels use the full logits to build gather/scatter metadata; the default
    stores dense expert matrices for correctness tests and unquantized
    checkpoints.
    """

    nn = torch.nn

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
            self.variance_epsilon = eps

        def forward(self, hidden):
            input_dtype = hidden.dtype
            normalized = hidden.float()
            variance = normalized.square().mean(dim=-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * normalized).to(input_dtype)

    class RotaryEmbedding(nn.Module):
        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.head_dim = args.head_dim
            self.base = args.rope_theta
            self.initial_context_length = args.initial_context_length
            self.scaling_factor = args.rope_scaling_factor
            # OpenAI's reference calls these NTK beta and alpha respectively.
            self.ntk_beta = args.rope_beta_fast
            self.ntk_alpha = args.rope_beta_slow
            self.register_buffer(
                "inv_freq",
                self._make_inv_freq(torch.device("cpu")),
                persistent=False,
            )
            self.concentration = self._concentration()

        def _concentration(self) -> float:
            if self.scaling_factor > 1.0:
                return 0.1 * math.log(self.scaling_factor) + 1.0
            return 1.0

        def _make_inv_freq(self, device):
            freq = self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
            if self.scaling_factor <= 1.0:
                return 1.0 / freq

            half_dim = self.head_dim / 2
            low = (
                half_dim
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                half_dim
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            if not 0 < low < high < half_dim - 1:
                raise ValueError(
                    "invalid YaRN parameters for the configured GPT-OSS head size"
                )
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (
                torch.arange(half_dim, dtype=torch.float32, device=device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            return interpolation * (1 - mask) + extrapolation * mask

        def materialize(self, device) -> None:
            self.inv_freq = self._make_inv_freq(device)

        def forward(self, positions, dtype):
            frequencies = torch.einsum(
                "bl,d->bld", positions.float(), self.inv_freq.float()
            )
            return (
                (frequencies.cos() * self.concentration).to(dtype),
                (frequencies.sin() * self.concentration).to(dtype),
            )

    def apply_rotary(hidden, cos, sin):
        first, second = hidden.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )

    class Attention(nn.Module):
        def __init__(self, args: _GptOssArgs, layer_idx: int) -> None:
            super().__init__()
            self.layer_type = args.layer_types[layer_idx]
            self.head_dim = args.head_dim
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_key_value_heads
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            self.scale = self.head_dim**-0.5
            self.sliding_window = (
                args.sliding_window if self.layer_type == "sliding_attention" else None
            )
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.head_dim,
                bias=args.attention_bias,
            )
            self.k_proj = nn.Linear(
                args.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=args.attention_bias,
            )
            self.v_proj = nn.Linear(
                args.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=args.attention_bias,
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                args.hidden_size,
                bias=args.attention_bias,
            )
            self.sinks = nn.Parameter(torch.empty(self.num_heads))

        def forward(self, hidden, positions, position_embeddings, cache):
            batch, length, _ = hidden.shape
            query = self.q_proj(hidden).view(
                batch, length, self.num_heads, self.head_dim
            )
            key = self.k_proj(hidden).view(
                batch, length, self.num_kv_heads, self.head_dim
            )
            value = self.v_proj(hidden).view(
                batch, length, self.num_kv_heads, self.head_dim
            )
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            cos, sin = position_embeddings
            query = apply_rotary(query, cos, sin)
            key = apply_rotary(key, cos, sin)

            key, value, key_start = cache.update_and_fetch(key, value)
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

            if isinstance(key_start, int):
                key_start = torch.full(
                    (batch,),
                    key_start,
                    dtype=positions.dtype,
                    device=positions.device,
                )
            else:
                key_start = key_start.to(dtype=positions.dtype)
            key_positions = key_start[:, None] + torch.arange(
                key.shape[-2],
                dtype=positions.dtype,
                device=positions.device,
            )
            allowed = key_positions[:, None, :] <= positions[:, :, None]
            if self.sliding_window is not None:
                allowed &= key_positions[:, None, :] >= (
                    positions[:, :, None] - self.sliding_window + 1
                )

            logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
            logits = logits * self.scale
            logits = logits.masked_fill(~allowed[:, None, :, :], -float("inf"))
            sinks = self.sinks.float().view(1, self.num_heads, 1, 1)
            sinks = sinks.expand(batch, -1, length, -1)
            combined_logits = torch.cat((logits, sinks), dim=-1)
            probabilities = torch.softmax(combined_logits, dim=-1)[..., :-1]
            output = torch.matmul(probabilities, value.float()).to(hidden.dtype)
            output = output.transpose(1, 2).contiguous().view(batch, length, -1)
            return self.o_proj(output)

    class TopKRouter(nn.Module):
        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.top_k = args.num_experts_per_tok
            self.weight = nn.Parameter(
                torch.empty(args.num_local_experts, args.hidden_size)
            )
            self.bias = nn.Parameter(torch.empty(args.num_local_experts))

        def forward(self, hidden):
            logits = nn.functional.linear(hidden, self.weight, self.bias)
            top_values, indices = torch.topk(logits, self.top_k, dim=-1, sorted=True)
            weights = torch.softmax(top_values, dim=-1, dtype=top_values.dtype)
            return logits, weights, indices

    class DenseExperts(nn.Module):
        """Dense reference expert linears; replace this module for MXFP4."""

        storage_format = "dense"

        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.intermediate_size = args.intermediate_size
            self.swiglu_limit = args.swiglu_limit
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    args.num_local_experts,
                    args.hidden_size,
                    2 * args.intermediate_size,
                )
            )
            self.gate_up_proj_bias = nn.Parameter(
                torch.empty(args.num_local_experts, 2 * args.intermediate_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    args.num_local_experts,
                    args.intermediate_size,
                    args.hidden_size,
                )
            )
            self.down_proj_bias = nn.Parameter(
                torch.empty(args.num_local_experts, args.hidden_size)
            )

        def _apply_gate(self, gate_up):
            gate = gate_up[..., ::2].clamp(max=self.swiglu_limit)
            up = gate_up[..., 1::2].clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            glu = gate * torch.sigmoid(1.702 * gate)
            return glu * (up + 1)

        def forward(self, hidden, router_indices, routing_weights, router_logits=None):
            del router_logits
            gate_up_weights = self.gate_up_proj[router_indices]
            gate_up_bias = self.gate_up_proj_bias[router_indices]
            gate_up = (
                torch.einsum("tkhi,th->tki", gate_up_weights, hidden) + gate_up_bias
            )
            activated = self._apply_gate(gate_up)
            down_weights = self.down_proj[router_indices]
            down_bias = self.down_proj_bias[router_indices]
            output = torch.einsum("tkih,tki->tkh", down_weights, activated) + down_bias
            return torch.einsum("tkh,tk->th", output, routing_weights)

    class MLP(nn.Module):
        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.router = TopKRouter(args)
            self.experts = (
                DenseExperts(args) if expert_factory is None else expert_factory(args)
            )

        def forward(self, hidden):
            batch, length, width = hidden.shape
            flat = hidden.reshape(-1, width)
            router_logits, routing_weights, router_indices = self.router(flat)
            output = self.experts(flat, router_indices, routing_weights, router_logits)
            return output.reshape(batch, length, width)

    class DecoderLayer(nn.Module):
        def __init__(self, args: _GptOssArgs, layer_idx: int) -> None:
            super().__init__()
            self.self_attn = Attention(args, layer_idx)
            self.mlp = MLP(args)
            self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

        def forward(self, hidden, positions, position_embeddings, cache):
            residual = hidden
            hidden = self.input_layernorm(hidden)
            hidden = self.self_attn(hidden, positions, position_embeddings, cache)
            hidden = residual + hidden
            residual = hidden
            hidden = self.post_attention_layernorm(hidden)
            hidden = self.mlp(hidden)
            return residual + hidden

    class Backbone(nn.Module):
        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(
                args.vocab_size, args.hidden_size, padding_idx=args.pad_token_id
            )
            self.layers = nn.ModuleList(
                [DecoderLayer(args, index) for index in range(args.num_hidden_layers)]
            )
            self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
            self.rotary_emb = RotaryEmbedding(args)

        def forward(self, inputs, positions, cache: _SequenceCache):
            hidden = self.embed_tokens(inputs)
            position_embeddings = self.rotary_emb(positions, hidden.dtype)
            for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
                hidden = layer(hidden, positions, position_embeddings, layer_cache)
            return self.norm(hidden)

    class GptOssForCausalLM(nn.Module):
        def __init__(self, args: _GptOssArgs) -> None:
            super().__init__()
            self.args = args
            self.model = Backbone(args)
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            self.apply(self._initialize)

        def _initialize(self, module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.args.initializer_range
                )
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, Attention):
                nn.init.normal_(module.sinks, mean=0.0, std=self.args.initializer_range)
            elif isinstance(module, TopKRouter):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.args.initializer_range
                )
                nn.init.normal_(module.bias, mean=0.0, std=self.args.initializer_range)
            elif isinstance(module, DenseExperts):
                nn.init.normal_(
                    module.gate_up_proj,
                    mean=0.0,
                    std=self.args.initializer_range,
                )
                nn.init.zeros_(module.gate_up_proj_bias)
                nn.init.normal_(
                    module.down_proj,
                    mean=0.0,
                    std=self.args.initializer_range,
                )
                nn.init.zeros_(module.down_proj_bias)

        def make_cache(self) -> _SequenceCache:
            return _SequenceCache(torch, self.args)

        def make_batch_cache(
            self, caches: list[_SequenceCache] | tuple[_SequenceCache, ...]
        ) -> _BatchedSequenceCache:
            return _BatchedSequenceCache(torch, tuple(caches))

        def materialize_buffers(self, device) -> None:
            self.model.rotary_emb.materialize(device)

        def forward(
            self,
            inputs,
            cache: _SequenceCache | None = None,
            positions=None,
            *,
            logits_to_keep: int | None = None,
        ):
            if inputs.ndim != 2:
                raise ValueError("inputs must have shape [batch, sequence]")
            if inputs.shape[1] == 0:
                raise ValueError("inputs must contain at least one token")
            if cache is None:
                if inputs.shape[0] != 1:
                    raise ValueError(
                        "batched inputs require explicit per-request physical caches"
                    )
                cache = self.make_cache()
            if isinstance(cache, _SequenceCache):
                if inputs.shape[0] != 1:
                    raise ValueError(
                        "one physical request cache cannot serve multiple sequences"
                    )
                offsets = (cache.offset,)
            else:
                if inputs.shape[0] != cache.batch_size:
                    raise ValueError(
                        "the input batch does not match the physical cache batch"
                    )
                offsets = cache.offsets
            if positions is None:
                positions = torch.stack(
                    [
                        torch.arange(
                            offset,
                            offset + inputs.shape[1],
                            dtype=torch.long,
                            device=inputs.device,
                        )
                        for offset in offsets
                    ]
                )
            if positions.shape != inputs.shape:
                raise ValueError("positions must have the same shape as inputs")
            # The runner validates GPU-bound metadata while it is still a
            # Python list. Keep this defensive check for direct CPU use without
            # introducing a host synchronization in every CUDA decode step.
            if positions.device.type == "cpu":
                for index, offset in enumerate(offsets):
                    expected = torch.arange(
                        offset,
                        offset + inputs.shape[1],
                        dtype=positions.dtype,
                        device=positions.device,
                    )
                    if not bool(torch.equal(positions[index], expected)):
                        raise ValueError(
                            "positions must be contiguous with the request cache"
                        )
            if any(
                offset + inputs.shape[1] > self.args.max_position_embeddings
                for offset in offsets
            ):
                raise ValueError("position exceeds max_position_embeddings")

            hidden = self.model(inputs, positions, cache)
            if isinstance(cache, _SequenceCache):
                cache.offset += inputs.shape[1]
            else:
                cache.advance(inputs.shape[1])
            if logits_to_keep is not None:
                if logits_to_keep <= 0:
                    raise ValueError("logits_to_keep must be positive")
                hidden = hidden[:, -logits_to_keep:, :]
            return self.lm_head(hidden)

    return GptOssForCausalLM


def _download_model(
    repo_id: str, revision: str | None, *, allow_mxfp4: bool = False
) -> Path:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise ImportError(
            "loading GPT-OSS by repository id requires huggingface-hub; "
            "install it or pass a local model directory"
        ) from error

    # Inspect the tiny config before asking the Hub for tens of gigabytes of
    # packed weights. Without an explicit kernel, fail before shard download.
    config_file = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    _check_quantization(json.loads(config_file.read_text()), allow_mxfp4=allow_mxfp4)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(_MODEL_FILES),
        )
    )


def _resolve_model(
    model: str | Path, revision: str | None, *, allow_mxfp4: bool = False
) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        raise ValueError(f"model path is not a directory: {candidate}")
    repo_id = str(model)
    selected_revision = MODEL_REVISIONS.get(repo_id) if revision is None else revision
    return _download_model(repo_id, selected_revision, allow_mxfp4=allow_mxfp4)


def _resolve_dtype(torch, dtype):
    if dtype is None or not isinstance(dtype, str):
        return dtype
    name = dtype.removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype!r}")
    return resolved


def _check_quantization(config: dict[str, Any], *, allow_mxfp4: bool = False) -> None:
    quantization = config.get("quantization_config") or config.get("quantization")
    if not quantization:
        return
    method = quantization.get("quant_method")
    if method == "mxfp4":
        if allow_mxfp4:
            return
        raise NotImplementedError(
            "the released GPT-OSS checkpoint uses MXFP4 expert weights; "
            "Resin's dense reference expert module cannot load packed blocks yet. "
            "An H100 MXFP4 expert kernel must be selected at this boundary"
        )
    raise ValueError(f"unsupported GPT-OSS quantization method: {method!r}")


def _load_model(model_path: Path, *, device, dtype=None, moe_kernel=None):
    try:
        import torch
    except ImportError as error:
        raise ImportError("GptOssRunner requires PyTorch") from error

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    args = _GptOssArgs.from_config(config)
    _check_quantization(config, allow_mxfp4=moe_kernel is not None)
    quantization = config.get("quantization_config") or config.get("quantization")
    if quantization and quantization.get("quant_method") == "mxfp4":
        from .gpt_oss_packed import load_mxfp4_model

        return load_mxfp4_model(
            model_path,
            device=device,
            kernel=moe_kernel,
            dtype=dtype,
        )

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError(
            "loading GPT-OSS weights requires safetensors; install safetensors"
        ) from error

    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    model_class = _make_model_class(torch)
    with torch.device("meta"):
        model = model_class(args)

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"no model safetensors found in {model_path}")
    weights = {}
    for weight_file in weight_files:
        shard = load_file(weight_file, device=str(load_target))
        duplicates = weights.keys() & shard.keys()
        if duplicates:
            raise ValueError(f"duplicate tensors across weight shards: {duplicates}")
        weights.update(shard)
    model.load_state_dict(weights, strict=True, assign=True)
    model.materialize_buffers(load_target)
    if target != load_target:
        model.to(target)

    resolved_dtype = _resolve_dtype(torch, dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    # GPT-OSS keeps normalization scales and arithmetic in float32 even though
    # the released non-expert checkpoint tensors are BF16.
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm":
            module.float()
    model.materialize_buffers(target)
    model.eval()
    return torch, model, config


def _default_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "Resin's GPT-OSS backend requires CUDA; pass device='cpu' only for "
        "explicit test or reference execution"
    )


class GptOssRunner:
    """Greedy GPT-OSS runner consuming Rust-built integer-token batches.

    Equal-length sequences with equally sized KV caches execute in one physical
    batch. Prefill is split into bounded, contiguous chunks without serializing
    requests; decode remains one physical model call. Heterogeneous chunks are
    rejected instead of being mislabeled as a batched execution.
    """

    supports_continuation_prefill = True
    prefill_chunk_size = 8192

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype=None,
        moe_kernel=None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise ImportError("GptOssRunner requires PyTorch") from error
        self.prefill_chunk_size = type(self).prefill_chunk_size
        self.device = torch.device(device or _default_device(torch))
        self.model_path = _resolve_model(
            model,
            revision,
            allow_mxfp4=moe_kernel is not None,
        )
        self._torch, self._model, self.config = _load_model(
            self.model_path,
            device=self.device,
            dtype=dtype,
            moe_kernel=moe_kernel,
        )
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = self.config["vocab_size"]
        self._cache: dict[int, _SequenceCache] = {}

    def forward(self, plan: BatchPlan) -> list[int]:
        torch = self._torch
        token_ids = plan.token_ids
        positions = plan.positions
        indptr = plan.sequence_indptr
        work = []
        for index, request_id in enumerate(plan.sequence_ids):
            start, end = indptr[index : index + 2]
            cache = self._cache.get(request_id)
            offset = 0 if cache is None else cache.offset
            sequence_positions = positions[start:end]
            expected_positions = list(range(offset, offset + end - start))
            if sequence_positions != expected_positions:
                raise ValueError(
                    f"non-contiguous positions for request {request_id}: "
                    f"expected {expected_positions}, got {sequence_positions}"
                )
            sequence_tokens = token_ids[start:end]
            if any(token_id >= self.vocab_size for token_id in sequence_tokens):
                raise ValueError(f"token id is out of range for request {request_id}")
            work.append((request_id, sequence_tokens, sequence_positions, cache))

        lengths = {len(sequence_tokens) for _, sequence_tokens, _, _ in work}
        if len(lengths) != 1:
            raise ValueError(
                "physical GPT-OSS batching requires equal token counts per sequence"
            )

        request_caches = [
            cache if cache is not None else self._model.make_cache()
            for _, _, _, cache in work
        ]
        model_cache = (
            request_caches[0]
            if len(request_caches) == 1
            else self._model.make_batch_cache(request_caches)
        )
        inputs = torch.tensor(
            [sequence_tokens for _, sequence_tokens, _, _ in work],
            dtype=torch.long,
            device=self.device,
        )
        position_tensor = torch.tensor(
            [sequence_positions for _, _, sequence_positions, _ in work],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self._model(
                inputs,
                model_cache,
                position_tensor,
                logits_to_keep=1,
            )
        output = logits[:, -1, :].argmax(dim=-1).tolist()

        for (request_id, _, _, old_cache), cache in zip(
            work, request_caches, strict=True
        ):
            if old_cache is None:
                self._cache[request_id] = cache
        return output

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)
