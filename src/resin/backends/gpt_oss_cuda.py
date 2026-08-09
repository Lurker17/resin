"""Concrete FA4 b22 execution for GPT-OSS on one Hopper GPU.

This is intentionally a model backend, not a public attention abstraction.
Rust owns one flat cache pool and supplies two ordered sparse group plans:
History for full attention and Window for sliding attention.  Python derives
the dense FA4 views from that generic ABI, addresses one shared physical BF16
allocation, and executes the Resin-owned GPT-OSS module hierarchy.

The eager :mod:`resin.backends.gpt_oss` runner remains the independent
correctness implementation.  In particular, this runner has no request-ID
cache map and no Python page allocator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from resin import CacheGroupLayout, CacheLayout

from .gpt_oss import (
    DEFAULT_MODEL,
    _default_device,
    _load_model,
    _resolve_model,
)

if TYPE_CHECKING:
    from resin import BatchPlan


_FA4_VERSION = "4.0.0b22"
_PAGE_SIZE = 128
_SLIDING_WINDOW = 128
_PREFILL_CHUNK_SIZE = 128
_WINDOW_TABLE_WIDTH = 1 + ((_SLIDING_WINDOW - 1 + _PAGE_SIZE - 1) // _PAGE_SIZE)
_CUDA_GRAPH_WARMUP_STEPS = 3


class _PostAcceptCudaError(RuntimeError):
    """Tell the native Engine that accepted CUDA work is not safely retryable."""

    _resin_post_accept = True


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_cuda_graph_decode_shapes(
    value: object,
) -> tuple[tuple[int, int], ...]:
    """Snapshot an immutable, deterministic CUDA-graph shape registry."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw_shapes = tuple(value.items())
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw_shapes = tuple(value)
    else:
        raise TypeError(
            "cuda_graph_decode_shapes must be an iterable of "
            "(batch_size, full_page_table_width) tuples, a mapping from "
            "batch size to full page-table width, or None"
        )

    shapes: list[tuple[int, int]] = []
    for raw_shape in raw_shapes:
        if not isinstance(raw_shape, tuple) or len(raw_shape) != 2:
            raise TypeError(
                "each cuda_graph_decode_shapes entry must be a "
                "(batch_size, full_page_table_width) tuple"
            )
        batch_size, full_page_table_width = raw_shape
        shape = (
            _positive_int("CUDA graph decode batch size", batch_size),
            _positive_int("CUDA graph full page-table width", full_page_table_width),
        )
        if shape in shapes:
            raise ValueError(f"duplicate CUDA graph decode shape: {shape!r}")
        shapes.append(shape)
    return tuple(sorted(shapes))


def _append_slots(
    token_indices: tuple[int, ...],
    pages: tuple[int, ...],
    offsets: tuple[int, ...],
    *,
    page_size: int,
) -> tuple[int, ...]:
    """Turn Rust-provided page/offset pairs into physical slab row indices."""

    if len(pages) != len(offsets):
        raise ValueError("append page and offset arrays must have equal length")
    try:
        selected = tuple(
            pages[token_index] * page_size + offsets[token_index]
            for token_index in token_indices
        )
    except IndexError as error:
        raise ValueError(
            "append metadata does not cover every selected token"
        ) from error
    if any(offsets[token_index] >= page_size for token_index in token_indices):
        raise ValueError("append page offset is outside the physical page")
    return selected


@dataclass(frozen=True, slots=True)
class _HostAttentionBatch:
    """One Rust-frozen decode batch or prefill round before tensorization."""

    phase: str
    round_index: int
    token_indices: tuple[int, ...]
    sequence_indices: tuple[int, ...]
    q_lengths: tuple[int, ...]
    full_page_table_width: int
    full_page_table: tuple[int, ...]
    full_seqused_k: tuple[int, ...]
    sliding_page_table_width: int
    sliding_page_table: tuple[int, ...]
    sliding_seqused_k: tuple[int, ...]
    full_append_slots: tuple[int, ...]
    sliding_append_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("attention batch phase must be prefill or decode")
        if self.round_index < 0:
            raise ValueError("attention batch round index cannot be negative")
        row_count = len(self.sequence_indices)
        if row_count == 0:
            raise ValueError("an attention batch must contain at least one sequence")
        if len(set(self.sequence_indices)) != row_count:
            raise ValueError("an attention round cannot contain a sequence twice")
        if len(self.q_lengths) != row_count or any(
            length <= 0 or length > _PREFILL_CHUNK_SIZE for length in self.q_lengths
        ):
            raise ValueError("query lengths do not match the native chunk rows")
        token_count = sum(self.q_lengths)
        if len(self.token_indices) != token_count:
            raise ValueError("query lengths do not cover the selected tokens")
        if (
            len(self.full_append_slots) != token_count
            or len(self.sliding_append_slots) != token_count
        ):
            raise ValueError("append slots must contain one item per query token")
        _positive_int("full_page_table_width", self.full_page_table_width)
        if len(self.full_page_table) != row_count * self.full_page_table_width:
            raise ValueError("full page table does not match its frozen shape")
        _positive_int("sliding_page_table_width", self.sliding_page_table_width)
        if len(self.sliding_page_table) != row_count * self.sliding_page_table_width:
            raise ValueError("sliding page table does not match its frozen shape")
        for name, lengths, width in (
            ("full", self.full_seqused_k, self.full_page_table_width),
            ("sliding", self.sliding_seqused_k, self.sliding_page_table_width),
        ):
            if len(lengths) != row_count:
                raise ValueError(f"{name} key lengths do not match the table rows")
            if any(length <= 0 or length > width * _PAGE_SIZE for length in lengths):
                raise ValueError(f"{name} key length is outside its frozen page table")


@dataclass(frozen=True, slots=True)
class _HostCacheGroup:
    """Validated generic cache-group metadata before kernel shaping."""

    group_index: int
    block_indptr: tuple[int, ...]
    block_ids: tuple[int, ...]
    first_logical_blocks: tuple[int, ...]
    append_block_ids: tuple[int, ...]
    append_block_offsets: tuple[int, ...]

    def row(self, sequence_index: int) -> tuple[int, ...]:
        start, end = self.block_indptr[sequence_index : sequence_index + 2]
        return self.block_ids[start:end]


@dataclass(frozen=True, slots=True)
class _DeviceAttentionBatch:
    """FA4-ready tensors for one native attention batch."""

    phase: str
    round_index: int
    diagnostic_context: str
    token_indices: Any
    sequence_indices: Any
    last_token_indices: Any
    cu_seqlens_q: Any
    max_seqlen_q: int
    full_page_table: Any
    full_seqused_k: Any
    full_min_seqlen_k: int
    full_max_seqlen_k: int
    sliding_page_table: Any
    sliding_seqused_k: Any
    sliding_min_seqlen_k: int
    sliding_max_seqlen_k: int
    full_append_slots: Any
    sliding_append_slots: Any


@dataclass(slots=True)
class _CudaGraphDecodeState:
    """Long-lived addresses captured by one runner-local decode graph."""

    token_ids: Any
    positions: Any
    attention_batch: _DeviceAttentionBatch
    final_hidden: Any
    output_ids: Any
    graph: Any | None = None


@dataclass(slots=True)
class _CudaGraphDecodeRegistryEntry:
    """One configured shape and its independently resident graph state."""

    shape: tuple[int, int]
    state: _CudaGraphDecodeState | None = None
    capture_count: int = 0
    replay_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    warmup_count: int = 0
    capture_ms: float | None = None
    capture_memory_bytes: int | None = None


class _CudaGraphStagingValidationError(ValueError):
    """Host metadata could not be shaped before any CUDA copy was launched."""


def _fixed_width_full_page_table(
    batch: _HostAttentionBatch, fixed_width: int
) -> tuple[int, ...]:
    """Physically pad Rust-frozen rows without inventing logical pages."""

    _positive_int("fixed full page-table width", fixed_width)
    if batch.full_page_table_width > fixed_width:
        raise ValueError(
            "Rust full page-table width exceeds the configured CUDA graph width"
        )
    padded: list[int] = []
    source_width = batch.full_page_table_width
    for row_index in range(len(batch.sequence_indices)):
        start = row_index * source_width
        row = batch.full_page_table[start : start + source_width]
        padded.extend(row)
        # ``seqused_k`` masks these columns.  Repeating a page already owned by
        # the sequence keeps every physical page ID valid without allocating
        # or exposing graph padding to the native logical allocator.
        padded.extend((row[0],) * (fixed_width - source_width))
    return tuple(padded)


def _invoke_fa4_b22(
    kernel,
    query,
    key_cache,
    value_cache,
    *,
    cu_seqlens_q,
    max_seqlen_q: int,
    page_table,
    seqused_k,
    min_seqlen_k: int,
    max_seqlen_k: int,
    softmax_scale: float,
    learnable_sink,
    sliding: bool,
):
    """Call only the pinned FA4 surface required by official GPT-OSS."""

    result = kernel(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        min_seqlen_k=min_seqlen_k,
        seqused_q=None,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(_SLIDING_WINDOW - 1, 0) if sliding else (None, None),
        learnable_sink=learnable_sink,
        softcap=0.0,
        # FA4 b22's SM90 implementation does not support split-KV.  Do not use
        # its automatic heuristic here because it may choose more than one
        # split for long decode contexts before rejecting that specialization.
        num_splits=1,
        pack_gqa=True,
        deterministic=False,
        return_lse=False,
    )
    return result[0] if isinstance(result, tuple) else result


class GptOssCudaRunner:
    """GPT-OSS BF16 attention on FA4 b22 over one flat native cache pool."""

    # A continuation remains an explicit prefill row even when it contains one
    # token. Sparse History/Window rows are chronological; dense FA4 tables and
    # page-aligned execution rounds are derived locally without any cache state.
    supports_continuation_prefill = True
    physical_batching = True
    physical_decode_batching = True
    supports_true_batching = True
    execution_mode = "flat_cache_fa4_b22"
    prefill_strategy = "derived_128_token_rounds"
    page_size = _PAGE_SIZE
    sliding_window = _SLIDING_WINDOW
    prefill_chunk_size = _PREFILL_CHUNK_SIZE
    batching_capabilities = frozenset(
        {
            "heterogeneous_chunked_prefill",
            "heterogeneous_decode",
            "native_flat_cache_pool",
            "ordered_history_window_groups",
        }
    )

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_parent_capacity: int,
        revision: str | None = None,
        device: str | None = None,
        dtype=None,
        moe_kernel=None,
        validate_first_logits: bool = False,
        diagnose_nonfinite: bool = False,
        cuda_graph_decode_shapes: (
            Iterable[tuple[int, int]] | Mapping[int, int] | None
        ) = None,
    ) -> None:
        if not isinstance(validate_first_logits, bool):
            raise TypeError("validate_first_logits must be a boolean")
        if not isinstance(diagnose_nonfinite, bool):
            raise TypeError("diagnose_nonfinite must be a boolean")
        self._diagnose_nonfinite = diagnose_nonfinite
        self.cache_parent_capacity = _positive_int(
            "cache_parent_capacity", cache_parent_capacity
        )
        self.cache_layout = CacheLayout(
            self.cache_parent_capacity,
            [
                CacheGroupLayout(_PAGE_SIZE),
                CacheGroupLayout(
                    _PAGE_SIZE,
                    retention="window",
                    window_tokens=_SLIDING_WINDOW,
                ),
            ],
        )
        configured_cuda_graph_decode_shapes = _normalize_cuda_graph_decode_shapes(
            cuda_graph_decode_shapes
        )
        if any(
            shape[1] > self.cache_parent_capacity
            for shape in configured_cuda_graph_decode_shapes
        ):
            raise ValueError(
                "CUDA graph full_page_table_width cannot exceed cache_parent_capacity"
            )
        self._cuda_graph_decode_shapes = configured_cuda_graph_decode_shapes
        self._cuda_graph_decode_registry = {
            shape: _CudaGraphDecodeRegistryEntry(shape)
            for shape in configured_cuda_graph_decode_shapes
        }
        self._cuda_graph_capture_count = 0
        self._cuda_graph_replay_count = 0
        self._cuda_graph_hit_count = 0
        self._cuda_graph_miss_count = 0
        self._cuda_graph_eager_fallback_count = 0
        self._cuda_graph_prefill_fallback_count = 0
        self._cuda_graph_shape_fallback_count = 0
        self._cuda_graph_warmup_count = 0
        self._cuda_graph_poisoned_error: str | None = None

        try:
            import torch
        except ImportError as error:
            raise ImportError("GptOssCudaRunner requires PyTorch") from error
        try:
            installed_fa4 = version("flash-attn-4")
        except PackageNotFoundError as error:
            raise ImportError(
                f"GptOssCudaRunner requires flash-attn-4=={_FA4_VERSION}"
            ) from error
        if installed_fa4 != _FA4_VERSION:
            raise RuntimeError(
                f"GptOssCudaRunner requires flash-attn-4=={_FA4_VERSION}, "
                f"got {installed_fa4}"
            )
        from flash_attn.cute import flash_attn_varlen_func

        self._torch = torch
        self._fa4 = flash_attn_varlen_func
        self.device = torch.device(device or _default_device(torch))
        if self.device.type != "cuda":
            raise RuntimeError("GptOssCudaRunner requires CUDA")
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError("the pinned GPT-OSS FA4 runner requires an SM90 GPU")

        resolved_dtype = torch.bfloat16 if dtype is None else dtype
        if isinstance(resolved_dtype, str):
            resolved_dtype = getattr(torch, resolved_dtype.removeprefix("torch."), None)
        if resolved_dtype is not torch.bfloat16:
            raise ValueError("the GPT-OSS FA4 runner requires bfloat16 activations")

        self.model_path = _resolve_model(
            model,
            revision,
            allow_mxfp4=moe_kernel is not None,
        )
        self._torch, self._model, self.config = _load_model(
            self.model_path,
            device=self.device,
            dtype=torch.bfloat16,
            moe_kernel=moe_kernel,
        )
        args = self._model.args
        expected_attention = (64, 8, 64, _SLIDING_WINDOW)
        actual_attention = (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.sliding_window,
        )
        if actual_attention != expected_attention:
            raise ValueError(
                "the pinned GPT-OSS FA4 geometry is "
                f"{expected_attention}, got {actual_attention}"
            )

        self.num_query_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.eos_token_id = self.config.get("eos_token_id")
        self.moe_kernel = moe_kernel
        self.kernel_adapter = (
            "dense" if moe_kernel is None else moe_kernel.capabilities.kernel_id
        )
        self.expert_storage_format = "dense" if moe_kernel is None else "mxfp4"
        self.weight_load_report = getattr(
            self._model, "_resin_weight_load_report", None
        )

        self._full_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "full_attention"
        )
        self._sliding_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "sliding_attention"
        )
        self._full_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._full_layers)
        }
        self._sliding_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._sliding_layers)
        }
        if len(self._full_layers) != len(self._sliding_layers):
            raise ValueError(
                "GPT-OSS History and Window cache groups must have equal byte geometry"
            )
        parent_shape = (
            2,
            len(self._full_layers),
            self.cache_parent_capacity + 1,
            _PAGE_SIZE,
            self.num_kv_heads,
            self.head_dim,
        )
        # FA4's SM90 TMA path can load a complete physical page even when
        # ``seqused_k`` masks its unused tail.  Keep unwritten rows finite so a
        # masked PV operation can never encounter uninitialized NaN/Inf data.
        self._cache_parent_storage = torch.zeros(
            parent_shape, dtype=torch.bfloat16, device=self.device
        )
        self._history_k_cache = self._cache_parent_storage[0]
        self._history_v_cache = self._cache_parent_storage[1]
        # Group membership is a native pool invariant. These views deliberately
        # alias the same parent bytes: a live parent is interpreted as History
        # or Window, never both.
        self._window_k_cache = self._history_k_cache
        self._window_v_cache = self._history_v_cache
        self._cache_pool_id: int | None = None
        self.forward_calls = 0
        self._validate_next_logits = validate_first_logits

        for layer in self._model.model.layers:
            if layer.self_attn.sinks.dtype is not torch.bfloat16:
                raise TypeError("FA4 learnable sink tensors must be bfloat16")
        self._validate_static_finite()

    @staticmethod
    def _bounded_metadata(
        values: tuple[int, ...], *, limit: int = 64
    ) -> tuple[object, ...]:
        """Keep exceptional diagnostic messages useful at large batch sizes."""

        if len(values) <= limit:
            return values
        edge = limit // 2
        return (*values[:edge], "...", *values[-edge:])

    @classmethod
    def _batch_diagnostic_context(cls, batch: _HostAttentionBatch) -> str:
        full_minmax = (min(batch.full_seqused_k), max(batch.full_seqused_k))
        sliding_minmax = (
            min(batch.sliding_seqused_k),
            max(batch.sliding_seqused_k),
        )
        return "; ".join(
            (
                f"sequence_indices={batch.sequence_indices!r}",
                f"q_lengths={batch.q_lengths!r}",
                (f"token_indices={cls._bounded_metadata(batch.token_indices)!r}"),
                f"full_page_table_width={batch.full_page_table_width}",
                (f"full_page_table={cls._bounded_metadata(batch.full_page_table)!r}"),
                f"full_seqused_k={batch.full_seqused_k!r}",
                f"full_seqused_minmax={full_minmax!r}",
                (
                    "full_append_slots="
                    f"{cls._bounded_metadata(batch.full_append_slots)!r}"
                ),
                f"sliding_page_table_width={batch.sliding_page_table_width}",
                (
                    "sliding_page_table="
                    f"{cls._bounded_metadata(batch.sliding_page_table)!r}"
                ),
                f"sliding_seqused_k={batch.sliding_seqused_k!r}",
                f"sliding_seqused_minmax={sliding_minmax!r}",
                (
                    "sliding_append_slots="
                    f"{cls._bounded_metadata(batch.sliding_append_slots)!r}"
                ),
            )
        )

    def _check_finite(
        self,
        tensor,
        name: str,
        *,
        batch: _DeviceAttentionBatch | None = None,
        layer_index: int | None = None,
        layer_type: str | None = None,
    ) -> None:
        """Raise at the first bad CUDA boundary when explicitly enabled.

        The flag is checked before any tensor operation, so normal inference
        never executes ``isfinite`` or introduces a host/device sync.
        """

        if not self._diagnose_nonfinite:
            return
        finite = self._torch.isfinite(tensor)
        if bool(finite.all().item()):
            return

        nonfinite = ~finite
        nonfinite_count = int(nonfinite.sum().item())
        first_index = tuple(
            int(index) for index in nonfinite.nonzero(as_tuple=False)[0].tolist()
        )
        finite_values = tensor.detach().masked_select(finite)
        if finite_values.numel():
            finite_minmax: tuple[float, float] | None = (
                float(finite_values.min().item()),
                float(finite_values.max().item()),
            )
        else:
            finite_minmax = None

        phase = "initialization" if batch is None else batch.phase
        round_index: int | None = None if batch is None else batch.round_index
        details = [
            "GPT-OSS CUDA non-finite diagnostic",
            f"tensor={name}",
            f"phase={phase}",
            f"round={round_index}",
            f"layer={layer_index}",
            f"layer_type={layer_type}",
            f"shape={tuple(tensor.shape)!r}",
            f"dtype={tensor.dtype}",
            f"nonfinite_count={nonfinite_count}",
            f"first_nonfinite_index={first_index!r}",
            f"finite_minmax={finite_minmax!r}",
        ]
        if batch is not None:
            details.append(batch.diagnostic_context)
        raise FloatingPointError("; ".join(details))

    def _validate_static_finite(self) -> None:
        """Scan floating model and prepared-MoE weights once in debug mode."""

        if not self._diagnose_nonfinite:
            return
        torch = self._torch
        for name, parameter in self._model.named_parameters():
            if parameter.is_floating_point():
                self._check_finite(parameter, f"weight:{name}")

        for layer_index, layer in enumerate(self._model.model.layers):
            prepared = getattr(layer.mlp.experts, "_prepared_weights", None)
            for field_name in getattr(prepared, "__slots__", ()):
                value = getattr(prepared, field_name)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    self._check_finite(
                        value,
                        f"prepared_moe_weight:{field_name}",
                        layer_index=layer_index,
                        layer_type=layer.self_attn.layer_type,
                    )

    @property
    def cuda_graph_stats(self) -> dict[str, object]:
        """Return a snapshot of this runner's local decode-graph activity."""

        shape_stats: list[dict[str, object]] = []
        for shape in self._cuda_graph_decode_shapes:
            entry = self._cuda_graph_decode_registry[shape]
            state = entry.state
            captured = state is not None and state.graph is not None
            shape_stats.append(
                {
                    "shape": shape,
                    "captured": captured,
                    "capture_count": entry.capture_count,
                    "replay_count": entry.replay_count,
                    "graph_hit_count": entry.hit_count,
                    "graph_miss_count": entry.miss_count,
                    "warmup_count": entry.warmup_count,
                    "capture_ms": entry.capture_ms,
                    "capture_memory_bytes": entry.capture_memory_bytes,
                }
            )
        captured_shapes = tuple(
            stats["shape"] for stats in shape_stats if stats["captured"]
        )
        return {
            "enabled": bool(self._cuda_graph_decode_shapes),
            "configured_shapes": self._cuda_graph_decode_shapes,
            "captured": bool(captured_shapes),
            "captured_shapes": captured_shapes,
            "shape_stats": tuple(shape_stats),
            "capture_count": self._cuda_graph_capture_count,
            "replay_count": self._cuda_graph_replay_count,
            "graph_hit_count": self._cuda_graph_hit_count,
            "graph_miss_count": self._cuda_graph_miss_count,
            "eager_fallback_count": self._cuda_graph_eager_fallback_count,
            "prefill_fallback_count": self._cuda_graph_prefill_fallback_count,
            "shape_fallback_count": self._cuda_graph_shape_fallback_count,
            "warmup_count": self._cuda_graph_warmup_count,
            "poisoned": self._cuda_graph_poisoned_error is not None,
            "poison_error": self._cuda_graph_poisoned_error,
        }

    def _poison_cuda_graph(self, stage: str, error: Exception) -> None:
        """Make CUDA execution fail-stop after an unsafe graph-path error."""

        if self._cuda_graph_poisoned_error is None:
            self._cuda_graph_poisoned_error = (
                f"{stage}: {type(error).__name__}: {error}"
            )

    def _raise_if_cuda_graph_poisoned(self) -> None:
        error = self._cuda_graph_poisoned_error
        if error is not None:
            raise RuntimeError(
                "this GPT-OSS runner is poisoned after a CUDA graph failure; "
                f"construct a fresh runner ({error})"
            )

    def _cuda_graph_fallback_kind(
        self,
        plan: BatchPlan,
        prefill_batches: tuple[_HostAttentionBatch, ...],
        decode_batch: _HostAttentionBatch | None,
    ) -> str | None:
        """Classify batches that must stay on the ordinary eager path."""

        if not self._cuda_graph_decode_shapes:
            return "disabled"
        if prefill_batches:
            return "prefill"
        if self._diagnose_nonfinite:
            return "shape"
        if decode_batch is None:
            return "shape"

        shape = self._select_cuda_graph_decode_shape(plan, decode_batch)
        if shape is None:
            return "shape"
        batch_size, fixed_full_width = shape
        if (
            len(plan.sequence_ids) != batch_size
            or len(plan.token_ids) != batch_size
            or len(plan.positions) != batch_size
            or len(decode_batch.token_indices) != batch_size
            or len(decode_batch.sequence_indices) != batch_size
            or decode_batch.q_lengths != (1,) * batch_size
            or decode_batch.full_page_table_width > fixed_full_width
            or decode_batch.sliding_page_table_width != _WINDOW_TABLE_WIDTH
        ):
            return "shape"
        return None

    def _select_cuda_graph_decode_shape(
        self,
        plan: BatchPlan,
        decode_batch: _HostAttentionBatch,
    ) -> tuple[int, int] | None:
        """Select the smallest exact registered shape that fits this plan."""

        logical_width = decode_batch.full_page_table_width
        return next(
            (
                shape
                for shape in self._cuda_graph_decode_shapes
                if shape[0] == len(plan.sequence_ids) and shape[1] >= logical_width
            ),
            None,
        )

    def _record_cuda_graph_fallback(self, kind: str) -> None:
        if kind == "disabled":
            return
        self._cuda_graph_eager_fallback_count += 1
        if kind == "prefill":
            self._cuda_graph_prefill_fallback_count += 1
        else:
            self._cuda_graph_shape_fallback_count += 1

    def _new_cuda_graph_decode_state(
        self, shape: tuple[int, int]
    ) -> _CudaGraphDecodeState:
        """Allocate every address whose identity is frozen by capture."""

        batch_size, full_width = shape
        torch = self._torch

        def empty_int32(*tensor_shape: int):
            return torch.empty(tensor_shape, dtype=torch.int32, device=self.device)

        def empty_indices(*tensor_shape: int):
            return torch.empty(tensor_shape, dtype=torch.long, device=self.device)

        attention_batch = _DeviceAttentionBatch(
            phase="decode",
            round_index=0,
            diagnostic_context="",
            token_indices=empty_indices(batch_size),
            sequence_indices=empty_indices(batch_size),
            last_token_indices=torch.arange(
                batch_size, dtype=torch.long, device=self.device
            ),
            cu_seqlens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            ),
            max_seqlen_q=1,
            full_page_table=empty_int32(batch_size, full_width),
            full_seqused_k=empty_int32(batch_size),
            full_min_seqlen_k=1,
            full_max_seqlen_k=full_width * _PAGE_SIZE,
            sliding_page_table=empty_int32(batch_size, _WINDOW_TABLE_WIDTH),
            sliding_seqused_k=empty_int32(batch_size),
            sliding_min_seqlen_k=1,
            sliding_max_seqlen_k=_WINDOW_TABLE_WIDTH * _PAGE_SIZE,
            full_append_slots=empty_indices(batch_size),
            sliding_append_slots=empty_indices(batch_size),
        )
        return _CudaGraphDecodeState(
            token_ids=empty_indices(batch_size),
            positions=empty_indices(batch_size),
            attention_batch=attention_batch,
            final_hidden=torch.empty(
                batch_size,
                self.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            output_ids=empty_indices(batch_size),
        )

    def _stage_cuda_graph_decode(
        self,
        state: _CudaGraphDecodeState,
        plan: BatchPlan,
        host_batch: _HostAttentionBatch,
    ) -> None:
        """Copy one exact Rust plan into the graph's stable device buffers."""

        torch = self._torch
        device_batch = state.attention_batch
        padded_full_page_table = _fixed_width_full_page_table(
            host_batch, device_batch.full_page_table.shape[1]
        )
        values_by_target = (
            (state.token_ids, plan.token_ids),
            (state.positions, plan.positions),
            (device_batch.token_indices, host_batch.token_indices),
            (device_batch.sequence_indices, host_batch.sequence_indices),
            (device_batch.full_page_table, padded_full_page_table),
            (device_batch.full_seqused_k, host_batch.full_seqused_k),
            (device_batch.sliding_page_table, host_batch.sliding_page_table),
            (device_batch.sliding_seqused_k, host_batch.sliding_seqused_k),
            (device_batch.full_append_slots, host_batch.full_append_slots),
            (device_batch.sliding_append_slots, host_batch.sliding_append_slots),
        )
        try:
            staged = tuple(
                (
                    target,
                    torch.as_tensor(values, dtype=target.dtype, device="cpu").view(
                        target.shape
                    ),
                )
                for target, values in values_by_target
            )
        except Exception as error:
            raise _CudaGraphStagingValidationError(
                "CUDA graph host metadata does not match its persistent buffer"
            ) from error
        for target, source in staged:
            target.copy_(source)

    def _cuda_graph_decode_body(self, state: _CudaGraphDecodeState):
        """The complete GPU model step captured for one pure-decode shape."""

        batch = state.attention_batch
        last_hidden = self._forward_batch(state.token_ids, state.positions, batch)
        # Rust's sequence order is logical truth and may differ from attention
        # row order.  Keep the permutation inside the graph before sampling.
        state.final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
        logits = self._model.lm_head(state.final_hidden)
        self._torch.argmax(logits, dim=-1, out=state.output_ids)
        return logits

    def _validate_first_forward_logits(self, logits) -> None:
        if not self._validate_next_logits:
            return
        torch = self._torch
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError(
                "GPT-OSS produced non-finite logits during first-forward validation"
            )
        self._validate_next_logits = False

    def _validate_cuda_graph_decode_output_ids(
        self, eager_output_ids, replay_output_ids
    ) -> None:
        """Require the first graph replay to match its eager warmup exactly."""

        if not bool(self._torch.equal(eager_output_ids, replay_output_ids)):
            raise RuntimeError(
                "CUDA graph validation replay output IDs do not exactly match "
                "the final eager warmup"
            )

    def _capture_cuda_graph_decode(
        self,
        state: _CudaGraphDecodeState,
        *,
        allocation_before: int,
        shape: tuple[int, int],
    ) -> None:
        """Warm the fixed physical shape, capture it, and validate one replay."""

        torch = self._torch
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(_CUDA_GRAPH_WARMUP_STEPS):
                warmup_logits = self._cuda_graph_decode_body(state)
            warmup_output_ids = state.output_ids.clone()
        current_stream.wait_stream(warmup_stream)
        self._cuda_graph_warmup_count += _CUDA_GRAPH_WARMUP_STEPS
        entry = self._cuda_graph_decode_registry[shape]
        entry.warmup_count += _CUDA_GRAPH_WARMUP_STEPS
        self._validate_first_forward_logits(warmup_logits)
        del warmup_logits
        torch.cuda.synchronize(self.device)

        capture_started = perf_counter()
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                self._cuda_graph_decode_body(state)
            # Validate one launch now so instantiation/replay failures surface
            # before Rust commits the plan.  Repeated writes target the same KV
            # slots with identical values and are therefore idempotent.
            graph.replay()
        torch.cuda.synchronize(self.device)
        self._validate_cuda_graph_decode_output_ids(warmup_output_ids, state.output_ids)
        del warmup_output_ids

        capture_ms = (perf_counter() - capture_started) * 1000
        allocation_after = torch.cuda.memory_allocated(self.device)
        capture_memory_bytes = max(0, allocation_after - allocation_before)
        state.graph = graph
        self._cuda_graph_capture_count += 1
        entry.capture_ms = capture_ms
        entry.capture_memory_bytes = capture_memory_bytes
        entry.capture_count += 1

    def _forward_cuda_graph_decode(
        self,
        plan: BatchPlan,
        host_batch: _HostAttentionBatch,
        shape: tuple[int, int],
    ) -> list[int]:
        """Capture once and replay thereafter for one selected registry shape."""

        entry = self._cuda_graph_decode_registry[shape]
        state = entry.state
        if state is None:
            torch = self._torch
            allocation_before = torch.cuda.memory_allocated(self.device)
            state = self._new_cuda_graph_decode_state(shape)
            entry.state = state
            try:
                self._stage_cuda_graph_decode(state, plan, host_batch)
            except (ValueError, _CudaGraphStagingValidationError):
                # Host-shape and staging validation has not launched the graph,
                # so it remains safe to repair the plan and retry.
                if entry.state is state:
                    entry.state = None
                raise
            except Exception as error:
                self._cuda_graph_miss_count += 1
                entry.miss_count += 1
                self._poison_cuda_graph("decode graph staging", error)
                raise
            try:
                self._capture_cuda_graph_decode(
                    state,
                    allocation_before=allocation_before,
                    shape=shape,
                )
                output_ids = state.output_ids.tolist()
            except Exception as error:
                self._cuda_graph_miss_count += 1
                entry.miss_count += 1
                self._poison_cuda_graph("decode graph capture", error)
                self._synchronize_failed_eager_forward()
                raise
            self._cuda_graph_hit_count += 1
            entry.hit_count += 1
            return output_ids

        try:
            self._stage_cuda_graph_decode(state, plan, host_batch)
        except (ValueError, _CudaGraphStagingValidationError):
            raise
        except Exception as error:
            self._cuda_graph_miss_count += 1
            entry.miss_count += 1
            self._poison_cuda_graph("decode graph staging", error)
            raise
        if state.graph is None:  # pragma: no cover - capture errors are raised
            self._cuda_graph_miss_count += 1
            entry.miss_count += 1
            error = RuntimeError("the configured CUDA decode graph was not captured")
            self._poison_cuda_graph("decode graph state", error)
            raise error
        try:
            with self._torch.cuda.device(self.device):
                state.graph.replay()
            output_ids = state.output_ids.tolist()
        except Exception as error:
            self._cuda_graph_miss_count += 1
            entry.miss_count += 1
            self._poison_cuda_graph("decode graph replay", error)
            self._synchronize_failed_eager_forward()
            raise
        self._cuda_graph_replay_count += 1
        self._cuda_graph_hit_count += 1
        entry.replay_count += 1
        entry.hit_count += 1
        return output_ids

    def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
        """Permanently bind this runner's frozen layout to one native pool."""

        if not isinstance(pool_id, int) or isinstance(pool_id, bool) or pool_id <= 0:
            raise ValueError("cache pool id must be a positive integer")
        if layout is not self.cache_layout:
            raise ValueError(
                "cache pool binding must use this runner's frozen cache_layout object"
            )
        if self._cache_pool_id is not None and self._cache_pool_id != pool_id:
            raise ValueError(
                "this GPT-OSS parent allocation is already bound to a different "
                "cache pool"
            )
        self._cache_pool_id = pool_id
        return pool_id

    def _validate_plan_binding(self, plan: BatchPlan) -> None:
        if self._cache_pool_id is None:
            raise ValueError("GPT-OSS cache pool is not bound")
        if plan.cache_pool_id != self._cache_pool_id:
            raise ValueError(
                f"BatchPlan cache pool id {plan.cache_pool_id!r} does not match "
                f"the bound cache pool {self._cache_pool_id!r}"
            )

    @staticmethod
    def _integer_tuple(
        name: str,
        values: object,
        *,
        positive: bool = False,
    ) -> tuple[int, ...]:
        try:
            result = tuple(values)
        except TypeError as error:
            raise ValueError(f"{name} must be iterable") from error
        lower_bound = 1 if positive else 0
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < lower_bound
            for value in result
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must contain {qualifier} integers")
        return result

    @classmethod
    def _cache_group_metadata(
        cls,
        plan: BatchPlan,
        *,
        sequence_count: int,
        token_count: int,
        parent_capacity: int | None,
    ) -> tuple[_HostCacheGroup, _HostCacheGroup]:
        try:
            raw_groups = tuple(plan.cache_groups)
        except (AttributeError, TypeError) as error:
            raise ValueError("BatchPlan cache_groups must be iterable") from error
        if len(raw_groups) != 2:
            raise ValueError(
                "GPT-OSS requires exactly two ordered cache groups: History, Window"
            )

        groups = []
        for expected_index, raw_group in enumerate(raw_groups):
            if raw_group.group_index != expected_index:
                raise ValueError(
                    "GPT-OSS cache groups must be ordered as History then Window"
                )
            block_indptr = cls._integer_tuple(
                f"cache group {expected_index} block_indptr",
                raw_group.block_indptr,
            )
            block_ids = cls._integer_tuple(
                f"cache group {expected_index} block_ids",
                raw_group.block_ids,
                positive=True,
            )
            first_logical_blocks = cls._integer_tuple(
                f"cache group {expected_index} first_logical_blocks",
                raw_group.first_logical_blocks,
            )
            append_block_ids = cls._integer_tuple(
                f"cache group {expected_index} append_block_ids",
                raw_group.append_block_ids,
                positive=True,
            )
            append_block_offsets = cls._integer_tuple(
                f"cache group {expected_index} append_block_offsets",
                raw_group.append_block_offsets,
            )
            if len(block_indptr) != sequence_count + 1:
                raise ValueError(
                    f"cache group {expected_index} block_indptr does not cover "
                    "every sequence"
                )
            if (
                block_indptr[0] != 0
                or block_indptr[-1] != len(block_ids)
                or any(
                    left > right for left, right in zip(block_indptr, block_indptr[1:])
                )
            ):
                raise ValueError(
                    f"cache group {expected_index} block_indptr is malformed"
                )
            if len(first_logical_blocks) != sequence_count:
                raise ValueError(
                    f"cache group {expected_index} first_logical_blocks does not "
                    "cover every sequence"
                )
            if (
                len(append_block_ids) != token_count
                or len(append_block_offsets) != token_count
            ):
                raise ValueError(
                    f"cache group {expected_index} append metadata does not cover "
                    "every query token"
                )
            if any(offset >= _PAGE_SIZE for offset in append_block_offsets):
                raise ValueError(
                    f"cache group {expected_index} append offset is outside its block"
                )
            if parent_capacity is not None and any(
                block_id > parent_capacity
                for block_id in (*block_ids, *append_block_ids)
            ):
                raise ValueError(
                    f"cache group {expected_index} contains a physical block outside "
                    "the bound parent allocation"
                )
            groups.append(
                _HostCacheGroup(
                    group_index=expected_index,
                    block_indptr=block_indptr,
                    block_ids=block_ids,
                    first_logical_blocks=first_logical_blocks,
                    append_block_ids=append_block_ids,
                    append_block_offsets=append_block_offsets,
                )
            )

        history, window = groups
        if set(history.block_ids) & set(window.block_ids):
            raise ValueError(
                "one physical parent cannot be interpreted by History and Window "
                "in the same batch"
            )
        return history, window

    @staticmethod
    def _padded_page_table(
        rows: tuple[tuple[int, ...], ...],
        *,
        minimum_width: int = 1,
    ) -> tuple[int, tuple[int, ...]]:
        width = max(minimum_width, *(len(row) for row in rows))
        table = []
        for row in rows:
            if not row:
                raise ValueError("cache page-table rows must be nonempty")
            table.extend(row)
            table.extend((row[0],) * (width - len(row)))
        return width, tuple(table)

    @classmethod
    def _attention_batches(
        cls,
        plan: BatchPlan,
        *,
        parent_capacity: int | None = None,
    ) -> tuple[tuple[_HostAttentionBatch, ...], _HostAttentionBatch | None]:
        """Derive all dense FA4 views from the generic grouped cache plan."""

        sequence_ids = cls._integer_tuple("sequence_ids", plan.sequence_ids)
        sequence_count = len(sequence_ids)
        if sequence_count == 0:
            raise ValueError("GPT-OSS cannot execute an empty batch")
        token_ids = cls._integer_tuple("token_ids", plan.token_ids)
        token_count = len(token_ids)
        positions = cls._integer_tuple("positions", plan.positions)
        sequence_indptr = cls._integer_tuple("sequence_indptr", plan.sequence_indptr)
        context_lengths = cls._integer_tuple("context_lengths", plan.context_lengths)
        prefill_rows = cls._integer_tuple(
            "prefill_sequence_indices", plan.prefill_sequence_indices
        )
        decode_rows = cls._integer_tuple(
            "decode_sequence_indices", plan.decode_sequence_indices
        )

        if len(positions) != token_count:
            raise ValueError("positions do not cover every query token")
        if len(sequence_indptr) != sequence_count + 1:
            raise ValueError("sequence_indptr does not cover every sequence")
        if (
            sequence_indptr[0] != 0
            or sequence_indptr[-1] != token_count
            or any(
                left >= right
                for left, right in zip(sequence_indptr, sequence_indptr[1:])
            )
        ):
            raise ValueError(
                "sequence_indptr must contain one nonempty query range per sequence"
            )
        if len(context_lengths) != sequence_count:
            raise ValueError("context_lengths do not cover every sequence")
        if (
            tuple(sorted(prefill_rows)) != prefill_rows
            or len(set(prefill_rows)) != len(prefill_rows)
            or tuple(sorted(decode_rows)) != decode_rows
            or len(set(decode_rows)) != len(decode_rows)
            or set(prefill_rows) & set(decode_rows)
            or set(prefill_rows) | set(decode_rows) != set(range(sequence_count))
        ):
            raise ValueError(
                "explicit prefill and decode rows must be sorted, unique, and "
                "partition the batch"
            )

        history, window = cls._cache_group_metadata(
            plan,
            sequence_count=sequence_count,
            token_count=token_count,
            parent_capacity=parent_capacity,
        )
        post_lengths: list[int] = []

        for sequence_index in range(sequence_count):
            token_start, token_end = sequence_indptr[
                sequence_index : sequence_index + 2
            ]
            context_length = context_lengths[sequence_index]
            q_length = token_end - token_start
            if positions[token_start:token_end] != tuple(
                range(context_length, context_length + q_length)
            ):
                raise ValueError(
                    "query positions must be contiguous from each row's context"
                )
            post_length = context_length + q_length
            post_lengths.append(post_length)
            last_logical_block = (post_length - 1) // _PAGE_SIZE

            history_row = history.row(sequence_index)
            if (
                history.first_logical_blocks[sequence_index] != 0
                or len(history_row) != last_logical_block + 1
            ):
                raise ValueError(
                    "History group rows must be the exact post-append prefix"
                )

            expected_window_first = max(0, last_logical_block + 1 - _WINDOW_TABLE_WIDTH)
            window_row = window.row(sequence_index)
            if (
                window.first_logical_blocks[sequence_index] != expected_window_first
                or len(window_row) != last_logical_block - expected_window_first + 1
            ):
                raise ValueError(
                    "Window group rows must be the exact chronological active suffix"
                )

            for group, row in ((history, history_row), (window, window_row)):
                first_logical_block = group.first_logical_blocks[sequence_index]
                for token_index in range(token_start, token_end):
                    position = positions[token_index]
                    relative_block = position // _PAGE_SIZE - first_logical_block
                    if relative_block < 0 or relative_block >= len(row):
                        raise ValueError(
                            f"cache group {group.group_index} does not cover every "
                            "query append"
                        )
                    if (
                        group.append_block_ids[token_index] != row[relative_block]
                        or group.append_block_offsets[token_index]
                        != position % _PAGE_SIZE
                    ):
                        raise ValueError(
                            f"cache group {group.group_index} append destinations "
                            "do not match its chronological row"
                        )

            if sequence_index in prefill_rows:
                if (
                    q_length > _PREFILL_CHUNK_SIZE
                    or context_length // _PAGE_SIZE != (post_length - 1) // _PAGE_SIZE
                ):
                    raise ValueError(
                        "Window prefill dispatch must stay within one logical "
                        "cache block"
                    )
            elif q_length != 1:
                raise ValueError("decode rows must contain exactly one query token")

        def cache_rows(
            group: _HostCacheGroup, row_indices: tuple[int, ...]
        ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
            rows = tuple(group.row(index) for index in row_indices)
            lengths = tuple(
                post_lengths[index] - group.first_logical_blocks[index] * _PAGE_SIZE
                for index in row_indices
            )
            return rows, lengths

        prefill_batches: tuple[_HostAttentionBatch, ...] = ()
        if prefill_rows:
            token_indices = tuple(
                token_index
                for sequence_index in prefill_rows
                for token_index in range(
                    sequence_indptr[sequence_index],
                    sequence_indptr[sequence_index + 1],
                )
            )
            q_lengths = tuple(
                sequence_indptr[index + 1] - sequence_indptr[index]
                for index in prefill_rows
            )
            history_rows, history_lengths = cache_rows(history, prefill_rows)
            window_rows, window_lengths = cache_rows(window, prefill_rows)
            history_width, history_table = cls._padded_page_table(history_rows)
            window_width, window_table = cls._padded_page_table(
                window_rows, minimum_width=_WINDOW_TABLE_WIDTH
            )
            prefill_batches = (
                _HostAttentionBatch(
                    phase="prefill",
                    round_index=0,
                    token_indices=token_indices,
                    sequence_indices=prefill_rows,
                    q_lengths=q_lengths,
                    full_page_table_width=history_width,
                    full_page_table=history_table,
                    full_seqused_k=history_lengths,
                    sliding_page_table_width=window_width,
                    sliding_page_table=window_table,
                    sliding_seqused_k=window_lengths,
                    full_append_slots=_append_slots(
                        token_indices,
                        history.append_block_ids,
                        history.append_block_offsets,
                        page_size=_PAGE_SIZE,
                    ),
                    sliding_append_slots=_append_slots(
                        token_indices,
                        window.append_block_ids,
                        window.append_block_offsets,
                        page_size=_PAGE_SIZE,
                    ),
                ),
            )

        decode_batch = None
        if decode_rows:
            token_indices = tuple(
                sequence_indptr[sequence_index] for sequence_index in decode_rows
            )
            history_rows, history_lengths = cache_rows(history, decode_rows)
            window_rows, window_lengths = cache_rows(window, decode_rows)
            history_width, history_table = cls._padded_page_table(history_rows)
            window_width, window_table = cls._padded_page_table(
                window_rows, minimum_width=_WINDOW_TABLE_WIDTH
            )
            decode_batch = _HostAttentionBatch(
                phase="decode",
                round_index=0,
                token_indices=token_indices,
                sequence_indices=decode_rows,
                q_lengths=(1,) * len(decode_rows),
                full_page_table_width=history_width,
                full_page_table=history_table,
                full_seqused_k=history_lengths,
                sliding_page_table_width=window_width,
                sliding_page_table=window_table,
                sliding_seqused_k=window_lengths,
                full_append_slots=_append_slots(
                    token_indices,
                    history.append_block_ids,
                    history.append_block_offsets,
                    page_size=_PAGE_SIZE,
                ),
                sliding_append_slots=_append_slots(
                    token_indices,
                    window.append_block_ids,
                    window.append_block_offsets,
                    page_size=_PAGE_SIZE,
                ),
            )
        return prefill_batches, decode_batch

    @classmethod
    def _prefill_batches(cls, plan: BatchPlan) -> tuple[_HostAttentionBatch, ...]:
        return cls._attention_batches(plan)[0]

    @classmethod
    def _decode_batch(cls, plan: BatchPlan) -> _HostAttentionBatch | None:
        return cls._attention_batches(plan)[1]

    def _synchronize_failed_eager_forward(self) -> None:
        """Make a launched eager failure quiescent or mark it post-accept."""

        try:
            # Every eager kernel is enqueued on PyTorch's current CUDA stream.
            # Successful synchronization makes the shared parent allocation
            # safe for exact retry or immediate native page retirement.
            self._torch.cuda.synchronize(self.device)
        except BaseException as synchronization_error:
            raise _PostAcceptCudaError(
                "GPT-OSS eager execution failed after launch and CUDA "
                "quiescence could not be established; resources must remain "
                "quarantined"
            ) from synchronization_error

    def _tensorize(self, batch: _HostAttentionBatch) -> _DeviceAttentionBatch:
        torch = self._torch

        def int32(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=self.device)

        cu_seqlens_q = [0]
        for length in batch.q_lengths:
            cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        last_token_indices = tuple(end - 1 for end in cu_seqlens_q[1:])
        row_count = len(batch.sequence_indices)
        return _DeviceAttentionBatch(
            phase=batch.phase,
            round_index=batch.round_index,
            diagnostic_context=(
                self._batch_diagnostic_context(batch)
                if self._diagnose_nonfinite
                else ""
            ),
            token_indices=indices(batch.token_indices),
            sequence_indices=indices(batch.sequence_indices),
            last_token_indices=indices(last_token_indices),
            cu_seqlens_q=int32(cu_seqlens_q),
            max_seqlen_q=max(batch.q_lengths),
            full_page_table=int32(batch.full_page_table).view(
                row_count, batch.full_page_table_width
            ),
            full_seqused_k=int32(batch.full_seqused_k),
            full_min_seqlen_k=min(batch.full_seqused_k),
            full_max_seqlen_k=max(batch.full_seqused_k),
            sliding_page_table=int32(batch.sliding_page_table).view(
                row_count, batch.sliding_page_table_width
            ),
            sliding_seqused_k=int32(batch.sliding_seqused_k),
            sliding_min_seqlen_k=min(batch.sliding_seqused_k),
            sliding_max_seqlen_k=max(batch.sliding_seqused_k),
            full_append_slots=indices(batch.full_append_slots),
            sliding_append_slots=indices(batch.sliding_append_slots),
        )

    def _apply_yarn_rotary(self, hidden, cos, sin):
        """Apply the model's already-materialized YaRN cos/sin tensors."""

        first, second = hidden.chunk(2, dim=-1)
        cos = cos.squeeze(0).unsqueeze(1)
        sin = sin.squeeze(0).unsqueeze(1)
        return self._torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )

    def _attention(
        self,
        layer_index: int,
        attention,
        hidden,
        position_embeddings,
        batch: _DeviceAttentionBatch,
    ):
        torch = self._torch
        token_count = hidden.shape[1]
        flat = hidden.reshape(token_count, self.hidden_size)
        query = attention.q_proj(flat).view(
            token_count, self.num_query_heads, self.head_dim
        )
        key = attention.k_proj(flat).view(token_count, self.num_kv_heads, self.head_dim)
        value = attention.v_proj(flat).view(
            token_count, self.num_kv_heads, self.head_dim
        )
        cos, sin = position_embeddings
        query = self._apply_yarn_rotary(query, cos, sin)
        key = self._apply_yarn_rotary(key, cos, sin)
        layer_type = attention.layer_type
        if self._diagnose_nonfinite:
            self._check_finite(
                query,
                "query",
                batch=batch,
                layer_index=layer_index,
                layer_type=layer_type,
            )
            self._check_finite(
                key,
                "key",
                batch=batch,
                layer_index=layer_index,
                layer_type=layer_type,
            )
            self._check_finite(
                value,
                "value",
                batch=batch,
                layer_index=layer_index,
                layer_type=layer_type,
            )
        if query.dtype is not torch.bfloat16:
            raise TypeError("FA4 query tensor must be bfloat16")
        if key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
            raise TypeError("FA4 K/V tensors must be bfloat16")
        if attention.sinks.dtype is not torch.bfloat16:
            raise TypeError("FA4 learnable sink tensor must be bfloat16")

        sliding = layer_type == "sliding_attention"
        if sliding:
            slot = self._sliding_layer_slots[layer_index]
            key_cache = self._window_k_cache[slot]
            value_cache = self._window_v_cache[slot]
            append_slots = batch.sliding_append_slots
            page_table = batch.sliding_page_table
            seqused_k = batch.sliding_seqused_k
            min_seqlen_k = batch.sliding_min_seqlen_k
            max_seqlen_k = batch.sliding_max_seqlen_k
        else:
            slot = self._full_layer_slots[layer_index]
            key_cache = self._history_k_cache[slot]
            value_cache = self._history_v_cache[slot]
            append_slots = batch.full_append_slots
            page_table = batch.full_page_table
            seqused_k = batch.full_seqused_k
            min_seqlen_k = batch.full_min_seqlen_k
            max_seqlen_k = batch.full_max_seqlen_k

        key_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, append_slots, key
        )
        value_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, append_slots, value
        )
        output = _invoke_fa4_b22(
            self._fa4,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=batch.cu_seqlens_q,
            max_seqlen_q=batch.max_seqlen_q,
            page_table=page_table,
            seqused_k=seqused_k,
            min_seqlen_k=min_seqlen_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            learnable_sink=attention.sinks,
            sliding=sliding,
        )
        if output.dtype is not torch.bfloat16:
            raise TypeError("FA4 output tensor must be bfloat16")
        if self._diagnose_nonfinite:
            self._check_finite(
                output,
                "fa4_output",
                batch=batch,
                layer_index=layer_index,
                layer_type=layer_type,
            )
        attention_out = attention.o_proj(output.reshape(1, token_count, -1))
        if self._diagnose_nonfinite:
            self._check_finite(
                attention_out,
                "attention_out",
                batch=batch,
                layer_index=layer_index,
                layer_type=layer_type,
            )
        return attention_out

    def _forward_batch(self, token_ids, positions, batch: _DeviceAttentionBatch):
        selected_tokens = token_ids.index_select(0, batch.token_indices)
        selected_positions = positions.index_select(0, batch.token_indices)
        hidden = self._model.model.embed_tokens(selected_tokens).unsqueeze(0)
        position_embeddings = self._model.model.rotary_emb(
            selected_positions.unsqueeze(0), hidden.dtype
        )
        if self._diagnose_nonfinite:
            self._check_finite(hidden, "round_input_hidden", batch=batch)
        for layer_index, layer in enumerate(self._model.model.layers):
            layer_type = layer.self_attn.layer_type
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if self._diagnose_nonfinite:
                self._check_finite(
                    normalized,
                    "normalized_attention_input",
                    batch=batch,
                    layer_index=layer_index,
                    layer_type=layer_type,
                )
            attention_out = self._attention(
                layer_index,
                layer.self_attn,
                normalized,
                position_embeddings,
                batch,
            )
            hidden = residual + attention_out
            if self._diagnose_nonfinite:
                self._check_finite(
                    hidden,
                    "post_attention_residual",
                    batch=batch,
                    layer_index=layer_index,
                    layer_type=layer_type,
                )
            residual = hidden
            normalized_mlp_input = layer.post_attention_layernorm(hidden)
            if self._diagnose_nonfinite:
                self._check_finite(
                    normalized_mlp_input,
                    "normalized_mlp_input",
                    batch=batch,
                    layer_index=layer_index,
                    layer_type=layer_type,
                )
            moe_output = layer.mlp(normalized_mlp_input)
            if self._diagnose_nonfinite:
                self._check_finite(
                    moe_output,
                    "moe_output",
                    batch=batch,
                    layer_index=layer_index,
                    layer_type=layer_type,
                )
            hidden = residual + moe_output
            if self._diagnose_nonfinite:
                self._check_finite(
                    hidden,
                    "post_mlp_hidden",
                    batch=batch,
                    layer_index=layer_index,
                    layer_type=layer_type,
                )
        hidden = self._model.model.norm(hidden).squeeze(0)
        if self._diagnose_nonfinite:
            self._check_finite(hidden, "round_output_hidden", batch=batch)
        return hidden.index_select(0, batch.last_token_indices)

    def forward(self, plan: BatchPlan) -> list[int]:
        self._raise_if_cuda_graph_poisoned()
        self._validate_plan_binding(plan)
        if any(token_id >= self.vocab_size for token_id in plan.token_ids):
            raise ValueError("token id is out of range")
        if any(
            position >= self._model.args.max_position_embeddings
            for position in plan.positions
        ):
            raise ValueError("position exceeds max_position_embeddings")

        prefill_batches, decode_batch = self._attention_batches(
            plan, parent_capacity=self.cache_parent_capacity
        )
        host_batches = prefill_batches + (
            () if decode_batch is None else (decode_batch,)
        )
        covered = {index for batch in host_batches for index in batch.sequence_indices}
        if covered != set(range(len(plan.sequence_ids))):
            raise ValueError(
                "native attention batches do not cover every plan sequence"
            )

        self.forward_calls += 1
        graph_fallback = self._cuda_graph_fallback_kind(
            plan, prefill_batches, decode_batch
        )
        if graph_fallback is None:
            assert decode_batch is not None
            cuda_graph_shape = self._select_cuda_graph_decode_shape(plan, decode_batch)
            assert cuda_graph_shape is not None
            return self._forward_cuda_graph_decode(plan, decode_batch, cuda_graph_shape)
        self._record_cuda_graph_fallback(graph_fallback)

        torch = self._torch
        token_ids = torch.tensor(plan.token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(plan.positions, dtype=torch.long, device=self.device)
        final_hidden = torch.empty(
            len(plan.sequence_ids),
            self.hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        try:
            with torch.inference_mode():
                for host_batch in prefill_batches:
                    batch = self._tensorize(host_batch)
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                if decode_batch is not None:
                    batch = self._tensorize(decode_batch)
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                if self._diagnose_nonfinite:
                    self._check_finite(final_hidden, "assembled_final_hidden")
                logits = self._model.lm_head(final_hidden)
                if self._diagnose_nonfinite:
                    self._check_finite(logits, "logits")

            self._validate_first_forward_logits(logits)
            return logits.argmax(dim=-1).tolist()
        except BaseException:
            self._synchronize_failed_eager_forward()
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Physical page reuse is entirely controlled by the native pool."""

        del sequence_ids


__all__ = ["GptOssCudaRunner"]
