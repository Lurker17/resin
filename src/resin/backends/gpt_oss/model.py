# Portions of the model implementation are adapted from OpenAI gpt-oss and
# Hugging Face Transformers.
# Copyright 2025 OpenAI and The HuggingFace Team. Licensed under Apache-2.0.

"""Tokenizer-free GPT-OSS model and request-owned attention state.

The expert factory is the sole seam between the readable dense implementation
and packed MXFP4 kernels. Attention, routing, and state ownership stay the same.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .config import GptOssConfig


class AttentionState:
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


class GptOssState:
    """Per-request state with explicit full and sliding attention layers."""

    def __init__(self, torch, args: GptOssConfig) -> None:
        self.layers = [
            AttentionState(
                torch,
                args.sliding_window if layer_type == "sliding_attention" else None,
            )
            for layer_type in args.layer_types
        ]
        self.offset = 0


class BatchedAttentionState:
    """A transient tensor view over equally sized per-request layer caches."""

    def __init__(self, torch, caches: tuple[AttentionState, ...]) -> None:
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


class BatchedGptOssState:
    """One model-call view that preserves independent request cache ownership."""

    def __init__(self, torch, caches: tuple[GptOssState, ...]) -> None:
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
            BatchedAttentionState(
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


def make_model_class(torch, *, expert_factory: Callable | None = None):
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
        def __init__(self, args: GptOssConfig) -> None:
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
        def __init__(self, args: GptOssConfig, layer_idx: int) -> None:
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
        def __init__(self, args: GptOssConfig) -> None:
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

        def __init__(self, args: GptOssConfig) -> None:
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
        def __init__(self, args: GptOssConfig) -> None:
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
        def __init__(self, args: GptOssConfig, layer_idx: int) -> None:
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
        def __init__(self, args: GptOssConfig) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(
                args.vocab_size, args.hidden_size, padding_idx=args.pad_token_id
            )
            self.layers = nn.ModuleList(
                [DecoderLayer(args, index) for index in range(args.num_hidden_layers)]
            )
            self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
            self.rotary_emb = RotaryEmbedding(args)

        def forward(self, inputs, positions, cache: GptOssState):
            hidden = self.embed_tokens(inputs)
            position_embeddings = self.rotary_emb(positions, hidden.dtype)
            for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
                hidden = layer(hidden, positions, position_embeddings, layer_cache)
            return self.norm(hidden)

    class GptOssForCausalLM(nn.Module):
        def __init__(self, args: GptOssConfig) -> None:
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

        def make_cache(self) -> GptOssState:
            return GptOssState(torch, self.args)

        def make_batch_cache(
            self, caches: list[GptOssState] | tuple[GptOssState, ...]
        ) -> BatchedGptOssState:
            return BatchedGptOssState(torch, tuple(caches))

        def materialize_buffers(self, device) -> None:
            self.model.rotary_emb.materialize(device)

        def forward(
            self,
            inputs,
            cache: GptOssState | None = None,
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
            if isinstance(cache, GptOssState):
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
            if isinstance(cache, GptOssState):
                cache.offset += inputs.shape[1]
            else:
                cache.advance(inputs.shape[1])
            if logits_to_keep is not None:
                if logits_to_keep <= 0:
                    raise ValueError("logits_to_keep must be positive")
                hidden = hidden[:, -logits_to_keep:, :]
            return self.lm_head(hidden)

    return GptOssForCausalLM
