# Portions of the model implementation are adapted from Hugging Face Transformers.
# Copyright 2025 The HuggingFace Team. Licensed under Apache-2.0.

"""Tokenizer-free LFM2 execution on PyTorch.

Rust owns request lifecycle and batch construction. This module owns the model
weights and the physical per-request attention/short-convolution caches. PyTorch,
Hugging Face Hub, and safetensors are imported only when the backend is used.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resin import BatchPlan


DEFAULT_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
_MODEL_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
)


@dataclass(frozen=True, slots=True)
class _Lfm2TorchArgs:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    conv_bias: bool
    conv_cache_length: int
    intermediate_size: int
    multiple_of: int
    auto_adjust_ff_dim: bool
    ffn_dim_multiplier: float
    rope_theta: float
    layer_types: tuple[str, ...]
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {"conv", "full_attention"}
        if unsupported:
            raise ValueError(f"unsupported LFM2 layer types: {sorted(unsupported)}")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("the attention head dimension must be even for RoPE")
        if self.conv_cache_length <= 0:
            raise ValueError("conv_cache_length must be positive")
        if self.intermediate_size <= 0 or self.multiple_of <= 0:
            raise ValueError("intermediate_size and multiple_of must be positive")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _Lfm2TorchArgs:
        if config.get("model_type") != "lfm2":
            raise ValueError(
                f"expected an LFM2 checkpoint, got {config.get('model_type')!r}"
            )
        rope = config.get("rope_parameters") or {}
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config.get(
                "num_key_value_heads", config["num_attention_heads"]
            ),
            norm_eps=config.get("norm_eps", config.get("block_norm_eps", 1e-5)),
            conv_bias=config["conv_bias"],
            conv_cache_length=config["conv_L_cache"],
            intermediate_size=config.get("block_ff_dim", config["intermediate_size"]),
            multiple_of=config.get("block_multiple_of", 256),
            auto_adjust_ff_dim=config.get("block_auto_adjust_ff_dim", False),
            ffn_dim_multiplier=config.get("block_ffn_dim_multiplier", 1.0),
            rope_theta=rope.get("rope_theta", config.get("rope_theta", 1_000_000.0)),
            layer_types=tuple(config["layer_types"]),
            pad_token_id=config.get("pad_token_id"),
        )


class _KVCache:
    """A deliberately simple growing K/V cache for one request."""

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


class _ConvCache:
    def __init__(self) -> None:
        self.state = None


class _SequenceCache:
    def __init__(self, torch, layer_types: tuple[str, ...]) -> None:
        self.layers = [
            _KVCache(torch) if layer_type == "full_attention" else _ConvCache()
            for layer_type in layer_types
        ]
        self.offset = 0


def _make_model_class(torch):
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
        def __init__(self, args: _Lfm2TorchArgs) -> None:
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

        def forward(self, hidden, positions, cache: _KVCache):
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
        def __init__(self, args: _Lfm2TorchArgs) -> None:
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

        def forward(self, hidden, cache: _ConvCache):
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
        def __init__(self, args: _Lfm2TorchArgs) -> None:
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
        def __init__(self, args: _Lfm2TorchArgs, layer_type: str) -> None:
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
        def __init__(self, args: _Lfm2TorchArgs) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(
                args.vocab_size, args.hidden_size, padding_idx=args.pad_token_id
            )
            self.layers = nn.ModuleList(
                [DecoderLayer(args, layer_type) for layer_type in args.layer_types]
            )
            self.embedding_norm = RMSNorm(args.hidden_size, args.norm_eps)

        def forward(self, inputs, positions, cache: _SequenceCache):
            hidden = self.embed_tokens(inputs)
            for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
                hidden = layer(hidden, positions, layer_cache)
            return self.embedding_norm(hidden)

    class Lfm2ForCausalLM(nn.Module):
        def __init__(self, args: _Lfm2TorchArgs) -> None:
            super().__init__()
            self.args = args
            self.model = Backbone(args)

        def make_cache(self) -> _SequenceCache:
            return _SequenceCache(torch, self.args.layer_types)

        def materialize_buffers(self, device) -> None:
            for module in self.modules():
                if isinstance(module, Attention):
                    module.materialize_rope(device)

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


def _download_model(repo_id: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "loading an LFM2 model by repository id requires huggingface-hub; "
            "install it or pass a local model directory"
        ) from error
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(_MODEL_FILES),
        )
    )


def _resolve_model(model: str | Path, revision: str | None) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        raise ValueError(f"model path is not a directory: {candidate}")
    return _download_model(str(model), revision)


def _resolve_dtype(torch, dtype):
    if dtype is None or not isinstance(dtype, str):
        return dtype
    name = dtype.removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype!r}")
    return resolved


def _load_model(model_path: Path, *, device, dtype=None):
    try:
        import torch
    except ImportError as error:
        raise ImportError("Lfm2TorchRunner requires PyTorch") from error
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError(
            "loading LFM2 weights requires safetensors; install safetensors"
        ) from error

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError(
            "the PyTorch runner requires an unquantized LFM2 checkpoint; "
            "quantized safetensors use a different physical format"
        )
    args = _Lfm2TorchArgs.from_config(config)
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
    # Module.to(dtype=...) casts floating-point buffers too. RoPE frequencies
    # intentionally remain float32 even when the checkpoint is bfloat16.
    model.materialize_buffers(target)
    model.eval()
    return torch, model, config


def _default_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Lfm2TorchRunner:
    """Greedy LFM2 runner consuming Rust-built integer-token batches."""

    supports_continuation_prefill = True

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        dtype=None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise ImportError("Lfm2TorchRunner requires PyTorch") from error
        self.device = torch.device(device or _default_device(torch))
        self.model_path = _resolve_model(model, revision)
        self._torch, self._model, self.config = _load_model(
            self.model_path, device=self.device, dtype=dtype
        )
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = self.config["vocab_size"]
        self._cache: dict[int, _SequenceCache] = {}
        self.forward_calls = 0

    def forward(self, plan: BatchPlan) -> list[int]:
        torch = self._torch
        token_ids = plan.token_ids
        positions = plan.positions
        indptr = plan.sequence_indptr
        output: list[int] = []
        self.forward_calls += 1
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

        with torch.inference_mode():
            for request_id, sequence_tokens, sequence_positions, cache in work:
                if cache is None:
                    cache = self._model.make_cache()
                    self._cache[request_id] = cache
                inputs = torch.tensor(
                    [sequence_tokens], dtype=torch.long, device=self.device
                )
                position_tensor = torch.tensor(
                    [sequence_positions], dtype=torch.long, device=self.device
                )
                logits = self._model(inputs, cache, position_tensor, logits_to_keep=1)
                output.append(int(logits[:, -1, :].argmax(dim=-1).item()))
        return output

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)

    @property
    def live_cache_count(self) -> int:
        return len(self._cache)
