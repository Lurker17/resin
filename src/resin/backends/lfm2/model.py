# Portions of the model implementation are adapted from Hugging Face Transformers.
# Copyright 2025 The HuggingFace Team. Licensed under Apache-2.0.

"""Tokenizer-free LFM2 model and request-owned recurrent state."""

from __future__ import annotations

from .config import Lfm2Config


class FullAttentionState:
    """Growing full-attention state for one request and one layer."""

    def __init__(self, torch) -> None:
        self._torch = torch
        self.keys = None
        self.values = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._torch.cat((self.keys, keys), dim=-2)
            self.values = self._torch.cat((self.values, values), dim=-2)
        return self.keys, self.values


class ShortConvState:
    """Fixed-size recurrent state for one short-convolution layer."""

    def __init__(self) -> None:
        self.state = None


class Lfm2State:
    """Per-request state with an explicit representation for each layer kind."""

    def __init__(self, torch, layer_types: tuple[str, ...]) -> None:
        self.layers = [
            FullAttentionState(torch)
            if layer_type == "full_attention"
            else ShortConvState()
            for layer_type in layer_types
        ]
        self.offset = 0


def make_model_class(torch):
    """Create the module hierarchy lazily, without importing torch at module import."""

    nn = torch.nn
    functional = torch.nn.functional

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden):
            input_dtype = hidden.dtype
            normalized = hidden.float()
            variance = normalized.square().mean(dim=-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * normalized.to(input_dtype)

    def rotate_half(hidden):
        first, second = hidden.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    class Attention(nn.Module):
        def __init__(self, args: Lfm2Config) -> None:
            super().__init__()
            dim = args.hidden_size
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_key_value_heads
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            self.head_dim = dim // self.num_heads
            self.scale = self.head_dim**-0.5
            self.rope_theta = args.rope_theta
            self.q_layernorm = RMSNorm(self.head_dim, args.norm_eps)
            self.k_layernorm = RMSNorm(self.head_dim, args.norm_eps)
            self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
            self.register_buffer(
                "inv_freq",
                self._make_inv_freq(self.q_proj.weight.device),
                persistent=False,
            )

        def _make_inv_freq(self, device):
            exponents = torch.arange(
                0, self.head_dim, 2, device=device, dtype=torch.float32
            )
            return 1.0 / (self.rope_theta ** (exponents / self.head_dim))

        def materialize_rope(self, device) -> None:
            self.inv_freq = self._make_inv_freq(device)

        def _apply_rope(self, queries, keys, positions):
            frequencies = torch.einsum(
                "bl,d->bld", positions.float(), self.inv_freq.float()
            )
            embeddings = torch.cat((frequencies, frequencies), dim=-1)
            cos = embeddings.cos().to(queries.dtype).unsqueeze(1)
            sin = embeddings.sin().to(queries.dtype).unsqueeze(1)
            queries = (queries * cos) + (rotate_half(queries) * sin)
            keys = (keys * cos) + (rotate_half(keys) * sin)
            return queries, keys

        def forward(self, hidden, positions, cache: FullAttentionState):
            batch, length, _ = hidden.shape
            queries = self.q_proj(hidden).view(
                batch, length, self.num_heads, self.head_dim
            )
            keys = self.k_proj(hidden).view(
                batch, length, self.num_kv_heads, self.head_dim
            )
            values = self.v_proj(hidden).view(
                batch, length, self.num_kv_heads, self.head_dim
            )
            queries = self.q_layernorm(queries).transpose(1, 2)
            keys = self.k_layernorm(keys).transpose(1, 2)
            values = values.transpose(1, 2)
            queries, keys = self._apply_rope(queries, keys, positions)

            previous = cache.offset
            keys, values = cache.update_and_fetch(keys, values)
            keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
            values = values.repeat_interleave(self.num_kv_groups, dim=1)

            attention_mask = None
            if length > 1:
                query_positions = torch.arange(
                    previous, previous + length, device=hidden.device
                )
                key_positions = torch.arange(keys.shape[-2], device=hidden.device)
                attention_mask = key_positions[None, :] <= query_positions[:, None]
            output = functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scale,
            )
            output = output.transpose(1, 2).contiguous().view(batch, length, -1)
            return self.out_proj(output)

    class ShortConv(nn.Module):
        def __init__(self, args: Lfm2Config) -> None:
            super().__init__()
            self.hidden_size = args.hidden_size
            self.cache_length = args.conv_cache_length
            self.conv = nn.Conv1d(
                args.hidden_size,
                args.hidden_size,
                kernel_size=self.cache_length,
                groups=args.hidden_size,
                bias=args.conv_bias,
                padding=0,
            )
            self.in_proj = nn.Linear(
                args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias
            )
            self.out_proj = nn.Linear(
                args.hidden_size, args.hidden_size, bias=args.conv_bias
            )

        def forward(self, hidden, cache: ShortConvState):
            gate_b, gate_c, value = self.in_proj(hidden).chunk(3, dim=-1)
            gated = (gate_b * value).transpose(1, 2)
            if cache.state is None:
                state = gated.new_zeros(
                    gated.shape[0], gated.shape[1], self.cache_length - 1
                )
            else:
                state = cache.state
            convolution_input = torch.cat((state, gated), dim=-1)
            if self.cache_length > 1:
                cache.state = convolution_input[..., -(self.cache_length - 1) :]
            else:
                cache.state = convolution_input[..., :0]
            output = functional.conv1d(
                convolution_input,
                self.conv.weight,
                self.conv.bias,
                groups=self.hidden_size,
            )
            return self.out_proj(gate_c * output.transpose(1, 2))

    class MLP(nn.Module):
        def __init__(self, args: Lfm2Config) -> None:
            super().__init__()
            size = args.intermediate_size
            if args.auto_adjust_ff_dim:
                size = int(2 * size / 3)
                size = int(args.ffn_dim_multiplier * size)
                size = args.multiple_of * (
                    (size + args.multiple_of - 1) // args.multiple_of
                )
            self.w1 = nn.Linear(args.hidden_size, size, bias=False)
            self.w3 = nn.Linear(args.hidden_size, size, bias=False)
            self.w2 = nn.Linear(size, args.hidden_size, bias=False)

        def forward(self, hidden):
            return self.w2(functional.silu(self.w1(hidden)) * self.w3(hidden))

    class DecoderLayer(nn.Module):
        def __init__(self, args: Lfm2Config, layer_type: str) -> None:
            super().__init__()
            self.is_attention_layer = layer_type == "full_attention"
            if self.is_attention_layer:
                self.self_attn = Attention(args)
            else:
                self.conv = ShortConv(args)
            self.feed_forward = MLP(args)
            self.operator_norm = RMSNorm(args.hidden_size, args.norm_eps)
            self.ffn_norm = RMSNorm(args.hidden_size, args.norm_eps)

        def forward(self, hidden, positions, cache):
            normalized = self.operator_norm(hidden)
            if self.is_attention_layer:
                residual = self.self_attn(normalized, positions, cache)
            else:
                residual = self.conv(normalized, cache)
            hidden = hidden + residual
            return hidden + self.feed_forward(self.ffn_norm(hidden))

    class Backbone(nn.Module):
        def __init__(self, args: Lfm2Config) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(
                args.vocab_size, args.hidden_size, padding_idx=args.pad_token_id
            )
            self.layers = nn.ModuleList(
                [DecoderLayer(args, layer_type) for layer_type in args.layer_types]
            )
            self.embedding_norm = RMSNorm(args.hidden_size, args.norm_eps)

        def forward(self, inputs, positions, cache: Lfm2State):
            hidden = self.embed_tokens(inputs)
            for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
                hidden = layer(hidden, positions, layer_cache)
            return self.embedding_norm(hidden)

    class Lfm2ForCausalLM(nn.Module):
        def __init__(self, args: Lfm2Config) -> None:
            super().__init__()
            self.args = args
            self.model = Backbone(args)

        def make_cache(self) -> Lfm2State:
            return Lfm2State(torch, self.args.layer_types)

        def materialize_buffers(self, device) -> None:
            for module in self.modules():
                if isinstance(module, Attention):
                    module.materialize_rope(device)

        def forward(
            self,
            inputs,
            cache: Lfm2State | None = None,
            positions=None,
            *,
            logits_to_keep: int | None = None,
        ):
            if inputs.ndim != 2:
                raise ValueError("inputs must have shape [batch, sequence]")
            if inputs.shape[0] != 1:
                raise ValueError("a physical request cache holds exactly one sequence")
            if cache is None:
                cache = self.make_cache()
            if positions is None:
                positions = torch.arange(
                    cache.offset,
                    cache.offset + inputs.shape[1],
                    dtype=torch.long,
                    device=inputs.device,
                ).unsqueeze(0)
            if positions.shape != inputs.shape:
                raise ValueError("positions must have the same shape as inputs")
            hidden = self.model(inputs, positions, cache)
            cache.offset += inputs.shape[1]
            if logits_to_keep is not None:
                if logits_to_keep <= 0:
                    raise ValueError("logits_to_keep must be positive")
                hidden = hidden[:, -logits_to_keep:, :]
            return functional.linear(hidden, self.model.embed_tokens.weight)

    return Lfm2ForCausalLM
