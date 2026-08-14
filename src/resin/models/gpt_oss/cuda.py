"""Concrete FA4 b22 execution for GPT-OSS on one Hopper GPU.

This is intentionally a model backend, not a public attention abstraction.
Rust owns one flat cache pool and supplies two ordered sparse group plans:
History for full attention and Window for sliding attention.  Python derives
the dense FA4 views from that generic ABI, addresses one shared physical BF16
allocation, and executes the Resin-owned GPT-OSS module hierarchy.

The eager :mod:`resin.models.gpt_oss` runner remains the independent
correctness implementation.  In particular, this runner has no request-ID
cache map and no Python page allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from resin import BatchPhase, CacheGroupLayout, CacheLayout
from resin.backend.checks import positions as check_positions
from resin.backend.checks import positive_int
from resin.backend.checks import token_ids as check_token_ids
from resin.backend.cuda.graphs import (
    CudaGraphCache,
    CudaGraphStats,
    CudaGraphStep,
    GraphInputError,
    GraphShape,
    graph_shapes,
)
from resin.backend.cuda.paged_kv import PagedKvArena, PagedKvTable
from resin.backend.cuda.staging import TensorStager
from resin.backend.loading import default_device
from resin.backend.sampling import GreedySampler
from resin.models.gpt_oss.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    WINDOW_TABLE_WIDTH,
    AttentionBatch,
    build_attention_batches,
)
from resin.models.gpt_oss.reference import (
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
_NUM_KV_HEADS = 8
_HEAD_DIM = 64
_BF16_BYTES = 2


class _PostAcceptCudaError(RuntimeError):
    """Tell the native Engine that accepted CUDA work is not safely retryable."""

    _resin_post_accept = True


@dataclass(frozen=True, slots=True)
class GptOssKvCapacityPlan:
    """Advisory shared-pool sizing from an SGLang-style static fraction."""

    total_memory_bytes: int
    model_memory_bytes: int
    static_memory_fraction: float
    static_memory_bytes: int
    kv_budget_bytes: int
    bytes_per_parent: int
    parent_capacity: int
    allocation_bytes: int
    full_parent_target: int
    swa_parent_target: int
    full_token_target: int
    swa_token_target: int
    swa_full_tokens_ratio: float


def plan_gpt_oss_kv_capacity(
    total_memory_bytes: int,
    model_memory_bytes: int,
    static_memory_fraction: float,
    *,
    attention_layers_per_group: int = 18,
    swa_full_tokens_ratio: float = 0.8,
) -> GptOssKvCapacityPlan:
    """Calculate GPT-OSS KV capacity without imposing an allocation policy.

    ``static_memory_fraction`` follows SGLang's definition: model weights plus
    KV cache divided by total GPU memory. Resin uses one fungible parent pool
    for full and sliding-window attention, so the SWA ratio is returned as an
    advisory token-capacity split rather than two physical allocations.
    """

    for name, value in (
        ("total_memory_bytes", total_memory_bytes),
        ("model_memory_bytes", model_memory_bytes),
        ("attention_layers_per_group", attention_layers_per_group),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if model_memory_bytes >= total_memory_bytes:
        raise ValueError("model_memory_bytes must be smaller than total GPU memory")
    if (
        isinstance(static_memory_fraction, bool)
        or not isinstance(static_memory_fraction, (int, float))
        or not 0.0 < static_memory_fraction < 1.0
    ):
        raise ValueError("static_memory_fraction must be between zero and one")
    if (
        isinstance(swa_full_tokens_ratio, bool)
        or not isinstance(swa_full_tokens_ratio, (int, float))
        or not 0.0 <= swa_full_tokens_ratio <= 1.0
    ):
        raise ValueError("swa_full_tokens_ratio must be between zero and one")

    static_memory_bytes = int(total_memory_bytes * static_memory_fraction)
    kv_budget_bytes = static_memory_bytes - model_memory_bytes
    if kv_budget_bytes <= 0:
        raise ValueError("the static memory target leaves no room for KV cache")
    bytes_per_parent = (
        2
        * attention_layers_per_group
        * _PAGE_SIZE
        * _NUM_KV_HEADS
        * _HEAD_DIM
        * _BF16_BYTES
    )
    # The device slab contains parent zero as a permanent null/graph-capture
    # page in addition to the positive parent IDs owned by the native pool.
    parent_capacity = kv_budget_bytes // bytes_per_parent - 1
    if parent_capacity <= 0:
        raise ValueError("the KV budget cannot hold a usable parent plus null page")
    full_parent_target = max(1, int(parent_capacity / (1.0 + swa_full_tokens_ratio)))
    swa_parent_target = parent_capacity - full_parent_target
    return GptOssKvCapacityPlan(
        total_memory_bytes=total_memory_bytes,
        model_memory_bytes=model_memory_bytes,
        static_memory_fraction=float(static_memory_fraction),
        static_memory_bytes=static_memory_bytes,
        kv_budget_bytes=kv_budget_bytes,
        bytes_per_parent=bytes_per_parent,
        parent_capacity=parent_capacity,
        allocation_bytes=(parent_capacity + 1) * bytes_per_parent,
        full_parent_target=full_parent_target,
        swa_parent_target=swa_parent_target,
        full_token_target=full_parent_target * _PAGE_SIZE,
        swa_token_target=swa_parent_target * _PAGE_SIZE,
        swa_full_tokens_ratio=float(swa_full_tokens_ratio),
    )


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
    full: PagedKvTable
    sliding: PagedKvTable


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
    max_batch_tokens = _PREFILL_CHUNK_SIZE
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
        self.cache_parent_capacity = positive_int(
            "cache_parent_capacity", cache_parent_capacity
        )
        self.cache_layout = CacheLayout(
            self.cache_parent_capacity,
            [
                CacheGroupLayout(_PAGE_SIZE),
                CacheGroupLayout(
                    _PAGE_SIZE,
                    retention="window",
                    window_tokens=WINDOW_RETENTION_TOKENS,
                ),
            ],
        )
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
        self._sampler = GreedySampler(torch)
        self._fa4 = flash_attn_varlen_func
        self.device = torch.device(device or default_device(torch, require_cuda=True))
        if self.device.type != "cuda":
            raise RuntimeError("GptOssCudaRunner requires CUDA")
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError("the pinned GPT-OSS FA4 runner requires an SM90 GPU")
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)
        if any(width > self.cache_parent_capacity for _, width in decode_graph_shapes):
            raise ValueError("CUDA graph table width exceeds cache capacity")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=self._synchronize_failed_eager_forward,
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
        # FA4's SM90 TMA path can load a complete physical page even when
        # ``seqused_k`` masks its unused tail.  Keep unwritten rows finite so a
        # masked PV operation can never encounter uninitialized NaN/Inf data.
        self._kv = PagedKvArena(
            torch,
            self.device,
            self.cache_layout,
            layer_count=len(self._full_layers),
            page_size=_PAGE_SIZE,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            resource="GPT-OSS KV",
        )

        for layer in self._model.model.layers:
            if layer.self_attn.sinks.dtype is not torch.bfloat16:
                raise TypeError("FA4 learnable sink tensors must be bfloat16")

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    @staticmethod
    def _graph_capture_inputs(shape: GraphShape) -> dict[str, tuple[int, ...]]:
        """Return valid synthetic decode inputs backed only by null page zero."""

        batch_size, full_width = shape
        rows = tuple(range(batch_size))
        zeros = (0,) * batch_size
        return {
            "token_ids": zeros,
            "positions": zeros,
            "token_indices": rows,
            "sequence_indices": rows,
            "full_page_table": (0,) * (batch_size * full_width),
            "full_seqused_k": (1,) * batch_size,
            "sliding_page_table": (0,) * (batch_size * _WINDOW_TABLE_WIDTH),
            "sliding_seqused_k": (1,) * batch_size,
            "full_append_slots": zeros,
            "sliding_append_slots": zeros,
        }

    def capture_cuda_graphs(self) -> CudaGraphStats:
        """Capture every configured decode graph before accepting requests."""

        already_captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape not in already_captured:
                self._graphs.capture(shape, self._graph_capture_inputs(shape))
        if self._graphs.shapes:
            # Capture writes synthetic K/V to parent zero, which the native
            # allocator permanently reserves. Restore its null-page contract
            # before a request can reach the runner.
            self._kv.storage[:, :, 0].zero_()
            self._torch.cuda.synchronize(self.device)
        return self._graphs.stats

    def bind_cache_pool(self, pool_id: int, layout: CacheLayout) -> int:
        """Permanently bind this runner's frozen layout to one native pool."""

        return self._kv.bind(pool_id, layout)

    def _validate_plan_binding(self, plan: BatchPlan) -> None:
        self._kv.check(plan)

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

    def _tensorize(
        self, batch: AttentionBatch, staging: TensorStager, prefix: str
    ) -> _DeviceAttentionBatch:
        torch = self._torch

        def device(name, values, dtype):
            return staging.device(f"{prefix}.{name}", values, dtype)

        cu_seqlens_q = [0]
        for length in batch.query_lengths:
            cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        last_token_indices = tuple(end - 1 for end in cu_seqlens_q[1:])
        row_count = len(batch.sequence_indices)

        def table(name, values, width, lengths, append_slots):
            return PagedKvTable(
                page_table=device(f"{name}.pages", values, torch.int32).view(
                    row_count, width
                ),
                lengths=device(f"{name}.lengths", lengths, torch.int32),
                append_slots=device(f"{name}.append", append_slots, torch.long),
                min_length=min(lengths),
                max_length=max(lengths),
            )

        return _DeviceAttentionBatch(
            phase=batch.phase,
            round_index=0,
            token_indices=device("token_indices", batch.token_indices, torch.long),
            sequence_indices=device(
                "sequence_indices", batch.sequence_indices, torch.long
            ),
            last_token_indices=device(
                "last_token_indices", last_token_indices, torch.long
            ),
            cu_seqlens_q=device("cu_seqlens_q", cu_seqlens_q, torch.int32),
            max_seqlen_q=max(batch.query_lengths),
            full=table(
                "full",
                batch.full_table,
                batch.full_width,
                batch.full_lengths,
                batch.full_append_slots,
            ),
            sliding=table(
                "sliding",
                batch.sliding_table,
                batch.sliding_width,
                batch.sliding_lengths,
                batch.sliding_append_slots,
            ),
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
            group = batch.sliding
        else:
            slot = self._full_layer_slots[layer_index]
            group = batch.full
        key_cache = self._kv.keys[slot]
        value_cache = self._kv.values[slot]

        key_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, key
        )
        value_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, value
        )
        output = _invoke_fa4_b22(
            self._fa4,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=batch.cu_seqlens_q,
            max_seqlen_q=batch.max_seqlen_q,
            page_table=group.page_table,
            seqused_k=group.lengths,
            min_seqlen_k=group.min_length,
            max_seqlen_k=group.max_length,
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

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
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
            full=PagedKvTable(
                empty(torch.int32, batch_size, full_width),
                empty(torch.int32, batch_size),
                empty(torch.long, batch_size),
                1,
                full_width * _PAGE_SIZE,
            ),
            sliding=PagedKvTable(
                empty(torch.int32, batch_size, _WINDOW_TABLE_WIDTH),
                empty(torch.int32, batch_size),
                empty(torch.long, batch_size),
                1,
                _WINDOW_TABLE_WIDTH * _PAGE_SIZE,
            ),
        )
        token_ids = empty(torch.long, batch_size)
        positions = empty(torch.long, batch_size)
        final_hidden = empty(torch.bfloat16, batch_size, self.hidden_size)
        output_ids = empty(torch.long, batch_size)

        def launch() -> None:
            hidden = self._forward_batch(token_ids, positions, batch)
            final_hidden.index_copy_(0, batch.sequence_indices, hidden)
            logits = self._model.lm_head(final_hidden)
            self._sampler(logits, out=output_ids)

        return CudaGraphStep(
            inputs={
                "token_ids": token_ids,
                "positions": positions,
                "token_indices": batch.token_indices,
                "sequence_indices": batch.sequence_indices,
                "full_page_table": batch.full.page_table,
                "full_seqused_k": batch.full.lengths,
                "sliding_page_table": batch.sliding.page_table,
                "sliding_seqused_k": batch.sliding.lengths,
                "full_append_slots": batch.full.append_slots,
                "sliding_append_slots": batch.sliding.append_slots,
            },
            output=output_ids,
            launch=launch,
        )

    def _graph_forward(self, plan: BatchPlan, host: AttentionBatch) -> list[int] | None:
        if not self._graphs.shapes:
            return None
        rows = len(plan.sequence_ids)
        if (
            host.phase != BatchPhase.DECODE
            or host.query_lengths != (1,) * rows
            or host.sliding_width != _WINDOW_TABLE_WIDTH
        ):
            raise GraphInputError("plan is not a fixed-shape decode batch")
        shape = self._graphs.match(
            lambda candidate: candidate[0] == rows and candidate[1] >= host.full_width
        )
        if shape is None:
            self._graphs.fallback("shape")
            return None
        return self._graphs.run(
            shape,
            {
                "token_ids": plan.token_ids,
                "positions": plan.positions,
                "token_indices": host.token_indices,
                "sequence_indices": host.sequence_indices,
                "full_page_table": self._pad_page_table(host, shape[1]),
                "full_seqused_k": host.full_lengths,
                "sliding_page_table": host.sliding_table,
                "sliding_seqused_k": host.sliding_lengths,
                "full_append_slots": host.full_append_slots,
                "sliding_append_slots": host.sliding_append_slots,
            },
        )

    def forward(self, plan: BatchPlan) -> list[int]:
        self._graphs.check()
        self._validate_plan_binding(plan)
        check_token_ids(plan.token_ids, self.vocab_size)
        check_positions(plan.positions, self._model.args.max_position_embeddings)

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
        staging = TensorStager(torch, self.device)
        token_ids = staging.device("token_ids", plan.token_ids, torch.long)
        positions = staging.device("positions", plan.positions, torch.long)
        final_hidden = torch.empty(
            len(plan.sequence_ids),
            self.hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        try:
            with torch.inference_mode():
                for index, host_batch in enumerate(prefill_batches):
                    batch = self._tensorize(host_batch, staging, f"prefill.{index}")
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                if decode_batch is not None:
                    batch = self._tensorize(decode_batch, staging, "decode")
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                logits = self._model.lm_head(final_hidden)

            return self._sampler(logits).tolist()
        except BaseException:
            self._synchronize_failed_eager_forward()
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Physical page reuse is entirely controlled by the native pool."""

        del sequence_ids


__all__ = ["GptOssCudaRunner"]
