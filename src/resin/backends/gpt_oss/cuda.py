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

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from resin import BatchPhase, CacheGroupLayout, CacheLayout

from .._common import default_device
from .graphs import DecodeGraphs, GraphInputError, GraphShape
from .plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_TABLE_WIDTH,
    AttentionBatch,
    build_attention_batches,
)
from .reference import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)

if TYPE_CHECKING:
    from resin import BatchPlan


_FA4_VERSION = "4.0.0b22"
_PAGE_SIZE = PAGE_SIZE
_SLIDING_WINDOW = SLIDING_WINDOW
_PREFILL_CHUNK_SIZE = PREFILL_CHUNK_SIZE
_WINDOW_TABLE_WIDTH = WINDOW_TABLE_WIDTH


class _PostAcceptCudaError(RuntimeError):
    """Tell the native Engine that accepted CUDA work is not safely retryable."""

    _resin_post_accept = True


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class _DeviceAttentionBatch:
    """FA4-ready tensors for one native attention batch."""

    phase: BatchPhase
    round_index: int
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
class _GraphDecodeState:
    token_ids: Any
    positions: Any
    batch: _DeviceAttentionBatch
    final_hidden: Any
    output_ids: Any


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
    page_size = _PAGE_SIZE
    sliding_window = _SLIDING_WINDOW
    prefill_chunk_size = _PREFILL_CHUNK_SIZE

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_parent_capacity: int,
        revision: str | None = None,
        device: str | None = None,
        dtype=None,
        moe_kernel=None,
        cuda_graph_decode_shapes: object = None,
    ) -> None:
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
        self.device = torch.device(device or default_device(torch, require_cuda=True))
        if self.device.type != "cuda":
            raise RuntimeError("GptOssCudaRunner requires CUDA")
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError("the pinned GPT-OSS FA4 runner requires an SM90 GPU")
        self._graphs = DecodeGraphs(
            torch,
            self.device,
            cuda_graph_decode_shapes,
            capacity=self.cache_parent_capacity,
        )

        resolved_dtype = torch.bfloat16 if dtype is None else dtype
        if isinstance(resolved_dtype, str):
            resolved_dtype = getattr(torch, resolved_dtype.removeprefix("torch."), None)
        if resolved_dtype is not torch.bfloat16:
            raise ValueError("the GPT-OSS FA4 runner requires bfloat16 activations")

        self.model_path = resolve_model_path(
            model,
            revision,
            allow_mxfp4=moe_kernel is not None,
        )
        self._torch, self._model, self.config = load_model(
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

        for layer in self._model.model.layers:
            if layer.self_attn.sinks.dtype is not torch.bfloat16:
                raise TypeError("FA4 learnable sink tensors must be bfloat16")

    @property
    def cuda_graph_stats(self) -> dict[str, object]:
        return self._graphs.stats

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

    def _tensorize(self, batch: AttentionBatch) -> _DeviceAttentionBatch:
        torch = self._torch

        def int32(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=self.device)

        cu_seqlens_q = [0]
        for length in batch.query_lengths:
            cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        last_token_indices = tuple(end - 1 for end in cu_seqlens_q[1:])
        row_count = len(batch.sequence_indices)
        return _DeviceAttentionBatch(
            phase=batch.phase,
            round_index=0,
            token_indices=indices(batch.token_indices),
            sequence_indices=indices(batch.sequence_indices),
            last_token_indices=indices(last_token_indices),
            cu_seqlens_q=int32(cu_seqlens_q),
            max_seqlen_q=max(batch.query_lengths),
            full_page_table=int32(batch.full_table).view(row_count, batch.full_width),
            full_seqused_k=int32(batch.full_lengths),
            full_min_seqlen_k=min(batch.full_lengths),
            full_max_seqlen_k=max(batch.full_lengths),
            sliding_page_table=int32(batch.sliding_table).view(
                row_count, batch.sliding_width
            ),
            sliding_seqused_k=int32(batch.sliding_lengths),
            sliding_min_seqlen_k=min(batch.sliding_lengths),
            sliding_max_seqlen_k=max(batch.sliding_lengths),
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
        return attention.o_proj(output.reshape(1, token_count, -1))

    def _forward_batch(self, token_ids, positions, batch: _DeviceAttentionBatch):
        selected_tokens = token_ids.index_select(0, batch.token_indices)
        selected_positions = positions.index_select(0, batch.token_indices)
        hidden = self._model.model.embed_tokens(selected_tokens).unsqueeze(0)
        position_embeddings = self._model.model.rotary_emb(
            selected_positions.unsqueeze(0), hidden.dtype
        )
        for layer_index, layer in enumerate(self._model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention_out = self._attention(
                layer_index,
                layer.self_attn,
                normalized,
                position_embeddings,
                batch,
            )
            hidden = residual + attention_out
            residual = hidden
            normalized_mlp_input = layer.post_attention_layernorm(hidden)
            moe_output = layer.mlp(normalized_mlp_input)
            hidden = residual + moe_output
        hidden = self._model.model.norm(hidden).squeeze(0)
        return hidden.index_select(0, batch.last_token_indices)

    @staticmethod
    def _pad_page_table(batch: AttentionBatch, width: int) -> tuple[int, ...]:
        """Pad each row by repeating one page already owned by that row."""

        if batch.full_width > width:
            raise GraphInputError("decode page table exceeds CUDA graph shape")
        padded = []
        source_width = batch.full_width
        for row in range(len(batch.sequence_indices)):
            pages = batch.full_table[row * source_width : (row + 1) * source_width]
            if not pages:
                raise GraphInputError("decode page-table row is empty")
            padded.extend(pages)
            padded.extend((pages[0],) * (width - source_width))
        return tuple(padded)

    def _new_graph_state(self, shape: GraphShape) -> _GraphDecodeState:
        batch_size, full_width = shape
        torch = self._torch

        def empty(dtype, *shape):
            return torch.empty(shape, dtype=dtype, device=self.device)

        batch = _DeviceAttentionBatch(
            phase=BatchPhase.DECODE,
            round_index=0,
            token_indices=empty(torch.long, batch_size),
            sequence_indices=empty(torch.long, batch_size),
            last_token_indices=torch.arange(
                batch_size, dtype=torch.long, device=self.device
            ),
            cu_seqlens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            ),
            max_seqlen_q=1,
            full_page_table=empty(torch.int32, batch_size, full_width),
            full_seqused_k=empty(torch.int32, batch_size),
            full_min_seqlen_k=1,
            full_max_seqlen_k=full_width * _PAGE_SIZE,
            sliding_page_table=empty(torch.int32, batch_size, _WINDOW_TABLE_WIDTH),
            sliding_seqused_k=empty(torch.int32, batch_size),
            sliding_min_seqlen_k=1,
            sliding_max_seqlen_k=_WINDOW_TABLE_WIDTH * _PAGE_SIZE,
            full_append_slots=empty(torch.long, batch_size),
            sliding_append_slots=empty(torch.long, batch_size),
        )
        return _GraphDecodeState(
            token_ids=empty(torch.long, batch_size),
            positions=empty(torch.long, batch_size),
            batch=batch,
            final_hidden=empty(torch.bfloat16, batch_size, self.hidden_size),
            output_ids=empty(torch.long, batch_size),
        )

    def _stage_graph_state(
        self, state: _GraphDecodeState, plan: BatchPlan, host: AttentionBatch
    ) -> None:
        batch = state.batch
        rows = len(plan.sequence_ids)
        if (
            host.phase != BatchPhase.DECODE
            or host.query_lengths != (1,) * rows
            or host.sliding_width != _WINDOW_TABLE_WIDTH
        ):
            raise GraphInputError("plan is not a fixed-shape decode batch")
        values = (
            (state.token_ids, plan.token_ids),
            (state.positions, plan.positions),
            (batch.token_indices, host.token_indices),
            (batch.sequence_indices, host.sequence_indices),
            (
                batch.full_page_table,
                self._pad_page_table(host, batch.full_page_table.shape[1]),
            ),
            (batch.full_seqused_k, host.full_lengths),
            (batch.sliding_page_table, host.sliding_table),
            (batch.sliding_seqused_k, host.sliding_lengths),
            (batch.full_append_slots, host.full_append_slots),
            (batch.sliding_append_slots, host.sliding_append_slots),
        )
        try:
            staged = tuple(
                (
                    target,
                    self._torch.as_tensor(
                        source, dtype=target.dtype, device="cpu"
                    ).view(target.shape),
                )
                for target, source in values
            )
        except Exception as error:
            raise GraphInputError("plan does not fit CUDA graph buffers") from error
        for target, source in staged:
            target.copy_(source)

    def _graph_body(self, state: _GraphDecodeState) -> None:
        hidden = self._forward_batch(state.token_ids, state.positions, state.batch)
        state.final_hidden.index_copy_(0, state.batch.sequence_indices, hidden)
        logits = self._model.lm_head(state.final_hidden)
        self._torch.argmax(logits, dim=-1, out=state.output_ids)

    def _graph_forward(self, plan: BatchPlan, host: AttentionBatch) -> list[int] | None:
        if not self._graphs.shapes:
            return None
        shape = self._graphs.select(len(plan.sequence_ids), host.full_width)
        if shape is None:
            self._graphs.fallback("shape")
            return None
        return self._graphs.run(
            shape,
            allocate=self._new_graph_state,
            stage=lambda state: self._stage_graph_state(state, plan, host),
            body=self._graph_body,
            output=lambda state: state.output_ids,
            synchronize_failure=self._synchronize_failed_eager_forward,
        )

    def forward(self, plan: BatchPlan) -> list[int]:
        self._graphs.check()
        self._validate_plan_binding(plan)
        if any(token_id >= self.vocab_size for token_id in plan.token_ids):
            raise ValueError("token id is out of range")
        if any(
            position >= self._model.args.max_position_embeddings
            for position in plan.positions
        ):
            raise ValueError("position exceeds max_position_embeddings")

        prefill_batches, decode_batch = build_attention_batches(
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

        if prefill_batches:
            self._graphs.fallback("prefill")
        elif decode_batch is not None:
            graph_output = self._graph_forward(plan, decode_batch)
            if graph_output is not None:
                return graph_output

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
                logits = self._model.lm_head(final_hidden)

            return logits.argmax(dim=-1).tolist()
        except BaseException:
            self._synchronize_failed_eager_forward()
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Physical page reuse is entirely controlled by the native pool."""

        del sequence_ids


__all__ = ["GptOssCudaRunner"]
