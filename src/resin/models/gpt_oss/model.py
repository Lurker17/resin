# Portions are adapted from OpenAI gpt-oss and Hugging Face Transformers
# (Apache-2.0).

"""GPT-OSS modules and explicit full/sliding attention state."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.models.gpt_oss.config import GptOssConfig


class AttentionState:
    """Keys and values retained by one full or sliding attention layer."""

    def __init__(self, window: int | None) -> None:
        self.window = window
        self.keys: Tensor | None = None
        self.values: Tensor | None = None
        self.length = 0

    @property
    def stored(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, int]:
        key_start = self.length - self.stored
        if key_start < 0:
            raise ValueError("attention state is longer than its request")
        all_keys = keys if self.keys is None else torch.cat((self.keys, keys), dim=-2)
        all_values = (
            values if self.values is None else torch.cat((self.values, values), dim=-2)
        )
        if self.window is None:
            self.keys, self.values = all_keys, all_values
        else:
            keep = self.window - 1
            self.keys = all_keys[..., -keep:, :] if keep else all_keys[..., :0, :]
            self.values = all_values[..., -keep:, :] if keep else all_values[..., :0, :]
        self.length += keys.shape[-2]
        return all_keys, all_values, key_start


class GptOssState:
    """One request's ordered full and sliding attention state."""

    def __init__(self, config: GptOssConfig) -> None:
        self.layers = [
            AttentionState(
                config.sliding_window if kind == "sliding_attention" else None
            )
            for kind in config.layer_types
        ]
        self.length = 0


class BatchedAttentionState:
    """A transient batched view that preserves request-owned allocations."""

    def __init__(self, states: tuple[AttentionState, ...]) -> None:
        if not states:
            raise ValueError("an attention-state batch cannot be empty")
        self.states = states
        self.window = states[0].window
        if any(state.window != self.window for state in states[1:]):
            raise ValueError("batched attention states must use one window")
        if any(state.stored != states[0].stored for state in states[1:]):
            raise ValueError("batched attention states must have equal lengths")

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if keys.shape[0] != len(self.states) or values.shape[0] != len(self.states):
            raise ValueError("tensor and physical state batch sizes differ")
        rows = [
            state.append(
                keys[index : index + 1].clone(), values[index : index + 1].clone()
            )
            for index, state in enumerate(self.states)
        ]
        return (
            torch.cat([row[0] for row in rows]),
            torch.cat([row[1] for row in rows]),
            torch.tensor(
                [row[2] for row in rows], dtype=torch.long, device=keys.device
            ),
        )


class BatchedGptOssState:
    """One physical model-call view over independent request states."""

    def __init__(self, states: tuple[GptOssState, ...]) -> None:
        if len(states) < 2:
            raise ValueError("a state batch requires at least two requests")
        layer_count = len(states[0].layers)
        if any(len(state.layers) != layer_count for state in states[1:]):
            raise ValueError("batched requests must have identical layer counts")
        if any(
            layer.length != state.length for state in states for layer in state.layers
        ):
            raise ValueError("request and layer state lengths differ")
        self.states = states
        self.layers = [
            BatchedAttentionState(tuple(state.layers[index] for state in states))
            for index in range(layer_count)
        ]

    @property
    def batch_size(self) -> int:
        return len(self.states)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(state.length for state in self.states)

    def advance(self, count: int) -> None:
        for state in self.states:
            state.length += count


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        dtype = hidden.dtype
        normalized = hidden.float()
        normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * normalized).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        self.initial_context = config.initial_context_length
        self.factor = config.rope_scaling_factor
        self.beta_fast = config.rope_beta_fast
        self.beta_slow = config.rope_beta_slow
        self.concentration = (
            0.1 * math.log(self.factor) + 1.0 if self.factor > 1 else 1.0
        )
        self.register_buffer(
            "inv_freq", self._frequencies(torch.device("cpu")), persistent=False
        )

    def _frequencies(self, device) -> Tensor:
        frequencies = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
            / self.head_dim
        )
        if self.factor <= 1:
            return 1.0 / frequencies
        half_dim = self.head_dim / 2
        low = (
            half_dim
            * math.log(self.initial_context / (self.beta_fast * 2 * math.pi))
            / math.log(self.base)
        )
        high = (
            half_dim
            * math.log(self.initial_context / (self.beta_slow * 2 * math.pi))
            / math.log(self.base)
        )
        if not 0 < low < high < half_dim - 1:
            raise ValueError("invalid YaRN parameters for the GPT-OSS head size")
        interpolation = 1.0 / (self.factor * frequencies)
        extrapolation = 1.0 / frequencies
        ramp = (torch.arange(half_dim, device=device) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        return interpolation * (1 - mask) + extrapolation * mask

    def materialize(self, device) -> None:
        self.inv_freq = self._frequencies(device)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        angles = torch.einsum("bl,d->bld", positions.float(), self.inv_freq.float())
        return (
            (angles.cos() * self.concentration).to(dtype),
            (angles.sin() * self.concentration).to(dtype),
        )


def apply_rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class Attention(nn.Module):
    qkv_proj: nn.Linear

    def __init__(self, config: GptOssConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        width = config.hidden_size
        self.q_proj = nn.Linear(
            width, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            width, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            width, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, width, bias=config.attention_bias
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

    def pack_qkv_projection(self) -> None:
        """Replace the loaded local Q/K/V shards with one packed projection."""

        projections = (self.q_proj, self.k_proj, self.v_proj)
        weight = torch.cat(tuple(projection.weight for projection in projections))
        packed = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=self.q_proj.bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
        packed.weight = nn.Parameter(
            weight,
            requires_grad=self.q_proj.weight.requires_grad,
        )
        if self.q_proj.bias is not None:
            packed.bias = nn.Parameter(
                torch.cat(
                    tuple(
                        projection.bias
                        for projection in projections
                        if projection.bias is not None
                    )
                ),
                requires_grad=self.q_proj.bias.requires_grad,
            )
        self.qkv_proj = packed
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        state: AttentionState | BatchedAttentionState,
    ) -> Tensor:
        batch, length, _ = hidden.shape
        query = (
            self.q_proj(hidden)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden)
            .view(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden)
            .view(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        query = apply_rotary(query, *rotary)
        key = apply_rotary(key, *rotary)
        key, value, key_start = state.append(key, value)
        key = key.repeat_interleave(self.kv_groups, dim=1)
        value = value.repeat_interleave(self.kv_groups, dim=1)

        starts = (
            torch.full((batch,), key_start, device=positions.device)
            if isinstance(key_start, int)
            else key_start
        )
        starts = starts.to(dtype=positions.dtype)
        key_positions = starts[:, None] + torch.arange(
            key.shape[-2], device=positions.device
        )
        allowed = key_positions[:, None] <= positions[:, :, None]
        if self.sliding_window is not None:
            allowed &= (
                key_positions[:, None]
                >= positions[:, :, None] - self.sliding_window + 1
            )
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~allowed[:, None], -float("inf"))
        sinks = (
            self.sinks.float()
            .view(1, self.num_heads, 1, 1)
            .expand(batch, -1, length, -1)
        )
        probabilities = torch.softmax(torch.cat((logits, sinks), -1), -1)[..., :-1]
        output = torch.matmul(probabilities, value.float()).to(hidden.dtype)
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, -1))


class TopKRouter(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size)
        )
        self.bias = nn.Parameter(torch.empty(config.num_local_experts))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = F.linear(hidden, self.weight, self.bias)
        values, indices = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        return logits, torch.softmax(values, dim=-1), indices


class DenseExperts(nn.Module):
    """Readable dense experts used by tests and unquantized checkpoints."""

    storage_format = "dense"

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        experts, hidden, intermediate = (
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_size = intermediate
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(experts, hidden, 2 * intermediate))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(experts, 2 * intermediate))
        self.down_proj = nn.Parameter(torch.empty(experts, intermediate, hidden))
        self.down_proj_bias = nn.Parameter(torch.empty(experts, hidden))

    def forward(
        self,
        hidden: Tensor,
        expert_indices: Tensor,
        routing_weights: Tensor,
        router_logits: Tensor | None = None,
    ) -> Tensor:
        del router_logits
        gate_up = torch.einsum(
            "tkhi,th->tki", self.gate_up_proj[expert_indices], hidden
        )
        gate_up += self.gate_up_proj_bias[expert_indices]
        gate = gate_up[..., ::2].clamp(max=self.swiglu_limit)
        up = gate_up[..., 1::2].clamp(-self.swiglu_limit, self.swiglu_limit)
        activated = gate * torch.sigmoid(1.702 * gate) * (up + 1)
        output = torch.einsum(
            "tkih,tki->tkh", self.down_proj[expert_indices], activated
        )
        output += self.down_proj_bias[expert_indices]
        return torch.einsum("tkh,tk->th", output, routing_weights)


ExpertFactory = Callable[[GptOssConfig], nn.Module]


class MLP(nn.Module):
    def __init__(
        self, config: GptOssConfig, expert_factory: ExpertFactory | None
    ) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = (
            DenseExperts(config) if expert_factory is None else expert_factory(config)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, weights, indices = self.router(flat)
        return self.experts(flat, indices, weights, logits).reshape(shape)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        layer_index: int,
        expert_factory: ExpertFactory | None,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_index)
        self.mlp = MLP(config, expert_factory)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden: Tensor, positions: Tensor, rotary, state) -> Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), positions, rotary, state
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class GptOssModel(nn.Module):
    def __init__(
        self, config: GptOssConfig, expert_factory: ExpertFactory | None
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, index, expert_factory)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, tokens: Tensor, positions: Tensor, state) -> Tensor:
        hidden = self.embed_tokens(tokens)
        rotary = self.rotary_emb(positions, hidden.dtype)
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            hidden = layer(hidden, positions, rotary, layer_state)
        return self.norm(hidden)


class GptOssForCausalLM(nn.Module):
    def __init__(
        self, config: GptOssConfig, expert_factory: ExpertFactory | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.args = config
        self.model = GptOssModel(config, expert_factory)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Attention):
            nn.init.normal_(module.sinks, std=std)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, std=std)
            nn.init.normal_(module.bias, std=std)
        elif isinstance(module, DenseExperts):
            nn.init.normal_(module.gate_up_proj, std=std)
            nn.init.zeros_(module.gate_up_proj_bias)
            nn.init.normal_(module.down_proj, std=std)
            nn.init.zeros_(module.down_proj_bias)

    def make_cache(self) -> GptOssState:
        return GptOssState(self.config)

    def make_batch_cache(self, states) -> BatchedGptOssState:
        return BatchedGptOssState(tuple(states))

    def materialize_buffers(self, device) -> None:
        self.model.rotary_emb.materialize(device)

    def forward(
        self,
        tokens: Tensor,
        state: GptOssState | BatchedGptOssState | None = None,
        positions: Tensor | None = None,
        *,
        logits_to_keep: int | None = None,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty sequence]")
        if state is None:
            if tokens.shape[0] != 1:
                raise ValueError("batched tokens require explicit request states")
            state = self.make_cache()
        if isinstance(state, GptOssState):
            if tokens.shape[0] != 1:
                raise ValueError("one request state cannot serve a tensor batch")
            lengths = (state.length,)
        else:
            if tokens.shape[0] != state.batch_size:
                raise ValueError("tensor and physical state batch sizes differ")
            lengths = state.lengths
        if positions is None:
            positions = torch.stack(
                [
                    torch.arange(
                        length,
                        length + tokens.shape[1],
                        dtype=torch.long,
                        device=tokens.device,
                    )
                    for length in lengths
                ]
            )
        if positions.shape != tokens.shape:
            raise ValueError("positions must have the same shape as tokens")
        if positions.device.type == "cpu":
            for row, length in zip(positions, lengths, strict=True):
                expected = torch.arange(
                    length, length + tokens.shape[1], dtype=row.dtype
                )
                if not torch.equal(row, expected):
                    raise ValueError("positions must continue the request state")
        if any(
            length + tokens.shape[1] > self.config.max_position_embeddings
            for length in lengths
        ):
            raise ValueError("position exceeds max_position_embeddings")

        hidden = self.model(tokens, positions, state)
        if isinstance(state, GptOssState):
            state.length += tokens.shape[1]
        else:
            state.advance(tokens.shape[1])
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            hidden = hidden[:, -logits_to_keep:]
        return self.lm_head(hidden)
