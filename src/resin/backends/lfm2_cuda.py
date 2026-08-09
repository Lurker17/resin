"""Paged FA4/FlashInfer execution for the Resin-owned PyTorch LFM2 model.

This deliberately specializes one model and one CUDA stack. Rust supplies the
complete immutable page, append, prefill, and decode subplans. This runner only
tensorizes that metadata and addresses the corresponding physical K/V slab while
mapping short-convolution state into one native-StatePool-owned physical slab.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, NamedTuple

from resin import CacheGroupLayout, CacheLayout, StateMappingConfig

from .lfm2_torch import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    _load_model,
    _resolve_model,
)

if TYPE_CHECKING:
    from resin import BatchPlan


# A failed CUDA wait means Python cannot prove that enqueued work has stopped
# touching its allocations.  Keep those allocations process-live instead of
# letting destructors return them to CUDA/PyTorch for unsafe reuse.
_CUDA_COMPLETION_QUARANTINE: list[tuple[object, ...]] = []
_ABANDON_QUERY_TIMEOUT_SECONDS = 0.010
_ABANDON_QUERY_INTERVAL_SECONDS = 0.001
_RECURRENT_STATE_MAPPING_PROTOCOL = "lfm2.conv_state.v1"


def _cache_slab_shape(
    num_layers: int,
    max_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int]:
    """Return [layer, page, token, KV-head, head-dim] slab geometry.

    Positive page IDs address the native pool. Row zero is the permanent null
    page used for safe kernel-table padding, so it is present in the physical
    tensor but excluded from ``max_pages``.
    """

    return (num_layers, max_pages + 1, page_size, num_kv_heads, head_dim)


class _PrefillKernelMetadata(NamedTuple):
    """Kernel-shaped views derived only from one frozen Rust batch plan."""

    dense_token_indices: tuple[int, ...]
    dense_cu_seqlens: tuple[int, ...]
    dense_max_q_len: int
    continuation_token_indices: tuple[int, ...]
    continuation_cu_seqlens: tuple[int, ...]
    continuation_max_q_len: int
    continuation_page_table: tuple[int, ...]
    continuation_page_table_width: int
    continuation_seqused_k: tuple[int, ...]
    continuation_min_seqlen_k: int
    continuation_max_seqlen_k: int


class _CacheKernelMetadata(NamedTuple):
    """FlashInfer-shaped metadata derived from one generic cache group."""

    block_indptr: tuple[int, ...]
    block_ids: tuple[int, ...]
    last_block_lengths: tuple[int, ...]
    append_batch_indices: tuple[int, ...]
    append_positions: tuple[int, ...]
    decode_token_indices: tuple[int, ...]
    decode_block_indptr: tuple[int, ...]
    decode_block_ids: tuple[int, ...]
    decode_last_block_lengths: tuple[int, ...]
    prefill: _PrefillKernelMetadata


class _RecurrentStateRow(NamedTuple):
    """One validated row's physical recurrent-state address and owner."""

    slot: int
    generation: int
    engine_id: int
    request_id: int


class _RecurrentStateArena(NamedTuple):
    """Process-owned physical interpretation of one mapped native StatePool."""

    arena_id: int
    pool_id: int
    capacity: int
    protocol: str
    layer_indices: tuple[int, ...]
    shape: tuple[int, int, int, int]
    device_key: str
    dtype_key: str
    slab: object
    generations: list[int | None]
    owners: list[tuple[int, int] | None]


# A native arena id is a permanent physical identity, not merely a matching
# integer acknowledgement. Keep its storage and sidecars strongly process-live
# so a later runner can either reuse the exact object or reject the mapping.
_RECURRENT_STATE_ARENAS: dict[int, _RecurrentStateArena] = {}


class _InPlaceConvCache:
    """ShortConv cache facade that can never detach from its slab view."""

    __slots__ = ("_state",)

    def __init__(self, state) -> None:
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value) -> None:
        if value is None:
            raise ValueError("mapped convolution state cannot be cleared")
        self._state.copy_(value)


def _cache_kernel_metadata(plan: BatchPlan, page_size: int) -> _CacheKernelMetadata:
    """Derive all LFM2 kernel views from the generic History-group plan.

    Context length chooses the kernel only *after* Rust has explicitly labeled a
    row as prefill. In particular, a one-token prefill continuation remains on
    the paged prefill path rather than being inferred to be decode. The native
    group's exact append destinations are also checked against the row tables
    before any device work; FlashInfer then receives the equivalent derived
    batch/position inputs.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    sequence_count = len(plan.sequence_ids)
    sequence_indptr = tuple(plan.sequence_indptr)
    context_lengths = tuple(plan.context_lengths)
    prefill_rows = tuple(plan.prefill_sequence_indices)
    decode_rows = tuple(plan.decode_sequence_indices)
    positions = tuple(plan.positions)
    token_count = len(plan.token_ids)
    groups = tuple(plan.cache_groups)
    if len(groups) != 1:
        raise ValueError("LFM2 requires exactly one cache group")
    group = groups[0]
    if group.group_index != 0:
        raise ValueError("LFM2 cache group must have group_index zero")
    block_indptr = tuple(group.block_indptr)
    block_ids = tuple(group.block_ids)
    first_logical_blocks = tuple(group.first_logical_blocks)
    frozen_append_block_ids = tuple(group.append_block_ids)
    frozen_append_block_offsets = tuple(group.append_block_offsets)

    if len(sequence_indptr) != sequence_count + 1:
        raise ValueError("sequence_indptr does not cover every batch row")
    if sequence_indptr[0] != 0 or sequence_indptr[-1] != token_count:
        raise ValueError("sequence_indptr does not cover every query token")
    if any(start > end for start, end in zip(sequence_indptr, sequence_indptr[1:])):
        raise ValueError("sequence_indptr must be monotonic")
    if len(positions) != token_count:
        raise ValueError("positions does not cover every query token")
    if len(context_lengths) != sequence_count:
        raise ValueError("context_lengths does not cover every batch row")
    if len(block_indptr) != sequence_count + 1:
        raise ValueError("cache block_indptr does not cover every batch row")
    if block_indptr[0] != 0 or block_indptr[-1] != len(block_ids):
        raise ValueError("cache block_indptr does not cover every block ID")
    if any(start > end for start, end in zip(block_indptr, block_indptr[1:])):
        raise ValueError("cache block_indptr must be monotonic")
    if len(first_logical_blocks) != sequence_count:
        raise ValueError("first_logical_blocks does not cover every batch row")
    if (
        len(frozen_append_block_ids) != token_count
        or len(frozen_append_block_offsets) != token_count
    ):
        raise ValueError("cache append metadata does not cover every query token")
    if (
        tuple(sorted(prefill_rows)) != prefill_rows
        or len(set(prefill_rows)) != len(prefill_rows)
        or tuple(sorted(decode_rows)) != decode_rows
        or len(set(decode_rows)) != len(decode_rows)
    ):
        raise ValueError("explicit phase row indices must be sorted and unique")
    prefill_row_set = set(prefill_rows)
    decode_row_set = set(decode_rows)
    if prefill_row_set & decode_row_set or prefill_row_set | decode_row_set != set(
        range(sequence_count)
    ):
        raise ValueError("explicit prefill and decode rows must partition the batch")

    dense_tokens: list[int] = []
    dense_cu = [0]
    dense_max_q_len = 0
    continuation_tokens: list[int] = []
    continuation_cu = [0]
    continuation_max_q_len = 0
    continuation_page_rows: list[tuple[int, ...]] = []
    continuation_seqused_k: list[int] = []
    last_block_lengths: list[int] = []
    append_batch_indices: list[int] = []
    append_positions: list[int] = []
    decode_token_indices: list[int] = []
    decode_block_indptr = [0]
    decode_block_ids: list[int] = []
    decode_last_block_lengths: list[int] = []

    for sequence_index in range(sequence_count):
        token_start, token_end = sequence_indptr[sequence_index : sequence_index + 2]
        q_length = token_end - token_start
        if q_length <= 0:
            raise ValueError("LFM2 cache rows must contain at least one query token")
        context_length = context_lengths[sequence_index]
        if context_length < 0:
            raise ValueError("context lengths must be nonnegative")
        row_positions = positions[token_start:token_end]
        expected_positions = tuple(range(context_length, context_length + q_length))
        if row_positions != expected_positions:
            raise ValueError("query positions do not match row context and length")

        block_start, block_end = block_indptr[sequence_index : sequence_index + 2]
        blocks = tuple(block_ids[block_start:block_end])
        first_logical_block = first_logical_blocks[sequence_index]
        if first_logical_block != 0:
            raise ValueError("LFM2 History cache rows must start at logical block zero")
        post_append_length = context_length + q_length
        expected_block_count = (post_append_length + page_size - 1) // page_size
        if len(blocks) != expected_block_count:
            raise ValueError(
                "cache block table is not the exact post-append History view"
            )
        if not blocks or any(block_id <= 0 for block_id in blocks):
            raise ValueError("cache block IDs must be positive")
        last_block_length = (post_append_length - 1) % page_size + 1
        last_block_lengths.append(last_block_length)

        append_batch_indices.extend([sequence_index] * q_length)
        append_positions.extend(row_positions)
        token_cursor = token_start
        while token_cursor < token_end:
            relative_position = positions[token_cursor]
            logical_offset = relative_position // page_size
            block_offset = relative_position % page_size
            span = min(token_end - token_cursor, page_size - block_offset)
            span_end = token_cursor + span
            if frozen_append_block_ids[token_cursor:span_end] != (
                blocks[logical_offset],
            ) * span or frozen_append_block_offsets[token_cursor:span_end] != tuple(
                range(block_offset, block_offset + span)
            ):
                raise ValueError(
                    "cache append destinations do not match the frozen group plan"
                )
            token_cursor = span_end

        if sequence_index in decode_row_set:
            if q_length != 1:
                raise ValueError("decode rows must contain exactly one query token")
            decode_token_indices.append(token_start)
            decode_block_ids.extend(blocks)
            decode_block_indptr.append(len(decode_block_ids))
            decode_last_block_lengths.append(last_block_length)

        if sequence_index not in prefill_row_set:
            continue
        row_tokens = list(range(token_start, token_end))

        if context_length == 0:
            dense_tokens.extend(row_tokens)
            dense_cu.append(len(dense_tokens))
            dense_max_q_len = max(dense_max_q_len, q_length)
            continue

        continuation_tokens.extend(row_tokens)
        continuation_cu.append(len(continuation_tokens))
        continuation_max_q_len = max(continuation_max_q_len, q_length)
        continuation_page_rows.append(blocks)
        continuation_seqused_k.append(post_append_length)

    page_table_width = max(map(len, continuation_page_rows), default=0)
    page_table: list[int] = []
    for pages in continuation_page_rows:
        page_table.extend(pages)
        page_table.extend((pages[0],) * (page_table_width - len(pages)))

    return _CacheKernelMetadata(
        block_indptr=block_indptr,
        block_ids=block_ids,
        last_block_lengths=tuple(last_block_lengths),
        append_batch_indices=tuple(append_batch_indices),
        append_positions=tuple(append_positions),
        decode_token_indices=tuple(decode_token_indices),
        decode_block_indptr=tuple(decode_block_indptr),
        decode_block_ids=tuple(decode_block_ids),
        decode_last_block_lengths=tuple(decode_last_block_lengths),
        prefill=_PrefillKernelMetadata(
            dense_token_indices=tuple(dense_tokens),
            dense_cu_seqlens=tuple(dense_cu),
            dense_max_q_len=dense_max_q_len,
            continuation_token_indices=tuple(continuation_tokens),
            continuation_cu_seqlens=tuple(continuation_cu),
            continuation_max_q_len=continuation_max_q_len,
            continuation_page_table=tuple(page_table),
            continuation_page_table_width=page_table_width,
            continuation_seqused_k=tuple(continuation_seqused_k),
            continuation_min_seqlen_k=min(continuation_seqused_k, default=0),
            continuation_max_seqlen_k=max(continuation_seqused_k, default=0),
        ),
    )


def _prefill_kernel_metadata(plan: BatchPlan, page_size: int) -> _PrefillKernelMetadata:
    """Return the prefill subset of generic cache-group metadata."""

    return _cache_kernel_metadata(plan, page_size).prefill


def _invoke_paged_fa4_b22(
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
):
    """Invoke the exact paged-varlen surface pinned by the LFM2 CUDA image."""

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
        window_size=(None, None),
        softcap=0.0,
        num_splits=1,
        pack_gqa=True,
        deterministic=False,
        return_lse=False,
    )
    return result[0] if isinstance(result, tuple) else result


class _Lfm2CudaCompletion:
    """Private lifetime capsule for one asynchronously enqueued model step."""

    __slots__ = (
        "_owner",
        "_plan",
        "_resources",
        "_private_stream",
        "_quarantine",
        "_staging",
        "_ephemeral",
        "_output_host",
        "_start_event",
        "_done_event",
        "_done_recorded",
        "_dispatch_error",
        "_accepted",
        "_terminal",
        "_retired",
    )

    def __init__(
        self,
        owner,
        plan,
        start_event,
        done_event,
        resources,
        private_stream,
        quarantine,
    ) -> None:
        self._owner = owner
        self._plan = plan
        self._resources = list(resources)
        self._private_stream = private_stream
        self._quarantine = quarantine
        self._staging = None
        self._ephemeral = []
        self._output_host = None
        self._start_event = start_event
        self._done_event = done_event
        self._done_recorded = False
        self._dispatch_error = None
        self._accepted = False
        self._terminal = False
        self._retired = False

    def _retire(self) -> None:
        """Drop launch lifetimes only after poll extracted a terminal result."""

        if not self._terminal:
            raise RuntimeError("a pending CUDA completion cannot be retired")
        self._plan = None
        self._resources = None
        self._private_stream = None
        self._quarantine = None
        self._staging = None
        self._ephemeral = None
        self._output_host = None
        self._start_event = None
        self._done_event = None
        self._dispatch_error = None
        self._retired = True

    def _quarantine_lifetimes(self, wait_error) -> None:
        """Fail-stop ownership transfer when CUDA completion is unknowable."""

        quarantine = self._quarantine
        payload = (
            self._plan,
            self._resources,
            self._private_stream,
            self._staging,
            self._ephemeral,
            self._output_host,
            self._start_event,
            self._done_event,
            self._dispatch_error,
            wait_error,
        )
        # Append before clearing a single reference. If append itself fails, the
        # caller leaves this completion intact rather than deliberately retiring
        # resources whose CUDA lifetime is unknown.
        quarantine.append(payload)
        self._terminal = True
        self._retired = True
        self._plan = None
        self._resources = None
        self._private_stream = None
        self._quarantine = None
        self._staging = None
        self._ephemeral = None
        self._output_host = None
        self._start_event = None
        self._done_event = None
        self._dispatch_error = None

    def _abandon(self) -> None:
        """Briefly query, then retire or quarantine an abandoned completion."""

        if self._retired:
            return
        if not self._accepted:
            self._terminal = True
            self._retire()
            return

        if self._done_recorded:
            fence = self._done_event
            fence_name = "terminal CUDA event"
        else:
            # Recording the terminal event can itself fail after earlier kernels
            # were accepted, so query the retained private stream as the fallback
            # fence. Never synchronize from a destructor: driver teardown may hang.
            fence = self._private_stream
            fence_name = "private CUDA stream"

        deadline = monotonic() + _ABANDON_QUERY_TIMEOUT_SECONDS
        while True:
            try:
                if fence.query():
                    break
            except BaseException as error:
                self._quarantine_lifetimes(error)
                return

            remaining = deadline - monotonic()
            if remaining <= 0:
                self._quarantine_lifetimes(
                    TimeoutError(
                        f"{fence_name} remained pending while abandoning completion"
                    )
                )
                return
            sleep(min(_ABANDON_QUERY_INTERVAL_SECONDS, remaining))

        self._terminal = True
        self._retire()

    def __del__(self) -> None:
        # Destructors must never surface errors or perform an unbounded CUDA wait.
        # _abandon either observes readiness in its short query window or transfers
        # every lifetime into the fail-stop quarantine.
        try:
            self._abandon()
        except BaseException as error:
            try:
                self._quarantine_lifetimes(error)
            except BaseException:
                # The completion remains structurally intact if even quarantine
                # allocation fails. There is no safer Python-level recovery path.
                pass


class Lfm2CudaRunner:
    """Greedy BF16 LFM2 runner with FA4 prefill and FlashInfer paged decode."""

    supports_continuation_prefill = True
    supports_recurrent_state_mapping = True
    recurrent_state_mapping = StateMappingConfig(
        _RECURRENT_STATE_MAPPING_PROTOCOL,
        ["write"],
    )

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        page_size: int = 16,
        max_pages: int = 256,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        import flashinfer
        import torch
        from flash_attn.cute import flash_attn_varlen_func

        if not torch.cuda.is_available():
            raise RuntimeError("Lfm2CudaRunner requires CUDA")
        if torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("FlashAttention-4 requires an SM90-or-newer GPU")
        if page_size <= 0 or max_pages <= 0:
            raise ValueError("page_size and max_pages must be positive")

        self._torch = torch
        self._flashinfer = flashinfer
        self._fa4 = flash_attn_varlen_func
        self.device = torch.device("cuda")
        self._stream = torch.cuda.Stream(device=self.device)
        self.page_size = page_size
        self.max_pages = max_pages
        self.cache_layout = CacheLayout(
            max_pages,
            [CacheGroupLayout(page_size)],
        )
        self.model_path = _resolve_model(model, revision)
        # Persistent CUDA state is created on the same private stream used for
        # every launch, so dispatch never needs an implicit default-stream handoff.
        with torch.cuda.stream(self._stream):
            self._torch, self._model, self.config = _load_model(
                self.model_path, device=self.device
            )
            self.eos_token_id = self.config.get("eos_token_id")
            self.vocab_size = self.config["vocab_size"]
            args = self._model.args
            self.num_query_heads = args.num_attention_heads
            self.num_kv_heads = args.num_key_value_heads
            self.head_dim = args.hidden_size // args.num_attention_heads
            self.scale = self.head_dim**-0.5
            self._attention_layers = [
                index
                for index, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ]
            self._attention_slot = {
                layer_index: slot
                for slot, layer_index in enumerate(self._attention_layers)
            }
            self._conv_layers = [
                index
                for index, layer_type in enumerate(args.layer_types)
                if layer_type != "full_attention"
            ]
            self._conv_slot = {
                layer_index: slot for slot, layer_index in enumerate(self._conv_layers)
            }
            self.hidden_size = args.hidden_size
            self.conv_cache_length = args.conv_cache_length

            cache_shape = _cache_slab_shape(
                len(self._attention_layers),
                max_pages,
                page_size,
                self.num_kv_heads,
                self.head_dim,
            )
            # FA4 b22's SM90 paged path may issue TMA loads for the full final
            # physical page and mask scores afterwards. Zero the unwritten tail
            # so a masked lane can never feed NaN/garbage into the PV reduction.
            self._k_cache = torch.zeros(
                cache_shape, dtype=torch.bfloat16, device=self.device
            )
            self._v_cache = torch.zeros_like(self._k_cache)
            self._decode_workspace = torch.zeros(
                workspace_bytes, dtype=torch.uint8, device=self.device
            )
            self._decode = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._decode_workspace, "NHD"
            )
        self._cache_pool_id: int | None = None
        self._state_mapping_binding: tuple[int, int, str] | None = None
        self._state_mapping_arena_id: int | None = None
        self._conv_state_slab = None
        self._conv_state_generations: list[int | None] | None = None
        self._conv_state_owners: list[tuple[int, int] | None] | None = None
        self._completion_owner = object()
        self._inflight: _Lfm2CudaCompletion | None = None
        self.forward_calls = 0

    def _validate_cache_pool_binding(self, pool_id: int, layout: CacheLayout) -> None:
        if not isinstance(pool_id, int) or isinstance(pool_id, bool) or pool_id <= 0:
            raise ValueError("cache pool id must be a positive integer")
        if layout is not self.cache_layout:
            raise ValueError(
                "cache pool binding must use this runner's frozen cache_layout object"
            )
        if self._cache_pool_id is not None and self._cache_pool_id != pool_id:
            raise ValueError(
                "this LFM2 physical slab is already bound to a different cache pool"
            )

    def _validate_recurrent_state_mapping_binding(
        self, pool_id: int, capacity: int, protocol: str
    ) -> tuple[int, int, str]:
        if not isinstance(pool_id, int) or isinstance(pool_id, bool) or pool_id <= 0:
            raise ValueError("state pool id must be a positive integer")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("state pool capacity must be a positive integer")
        if protocol != _RECURRENT_STATE_MAPPING_PROTOCOL:
            raise ValueError(
                "unsupported recurrent-state mapping protocol: "
                f"expected {_RECURRENT_STATE_MAPPING_PROTOCOL!r}, got {protocol!r}"
            )

        binding = (pool_id, capacity, protocol)
        if self._state_mapping_binding is not None:
            if binding != self._state_mapping_binding:
                raise ValueError(
                    "this LFM2 convolution slab is already bound to a different "
                    "recurrent-state mapping"
                )
        return binding

    def _recurrent_state_arena_descriptor(
        self, binding: tuple[int, int, str]
    ) -> tuple[
        int,
        int,
        str,
        tuple[int, ...],
        tuple[int, int, int, int],
        str,
        str,
    ]:
        pool_id, capacity, protocol = binding
        shape = (
            len(self._conv_layers),
            capacity,
            self.hidden_size,
            self.conv_cache_length - 1,
        )
        return (
            pool_id,
            capacity,
            protocol,
            tuple(self._conv_layers),
            shape,
            str(self.device),
            str(self._torch.bfloat16),
        )

    def _validate_recurrent_state_arena(
        self,
        arena: _RecurrentStateArena,
        binding: tuple[int, int, str],
    ) -> None:
        expected = self._recurrent_state_arena_descriptor(binding)
        actual = (
            arena.pool_id,
            arena.capacity,
            arena.protocol,
            arena.layer_indices,
            arena.shape,
            arena.device_key,
            arena.dtype_key,
        )
        if actual != expected:
            raise ValueError(
                f"required recurrent-state arena {arena.arena_id} has incompatible "
                "pool identity or physical geometry"
            )

    def _allocate_recurrent_state_mapping(
        self,
        binding: tuple[int, int, str],
        required_arena_id: int | None = None,
    ) -> _RecurrentStateArena:
        if required_arena_id is not None:
            arena = _RECURRENT_STATE_ARENAS.get(required_arena_id)
            if arena is None:
                raise ValueError(
                    f"required recurrent-state arena {required_arena_id} is not "
                    "registered in this process"
                )
            self._validate_recurrent_state_arena(arena, binding)
            return arena

        # Import only when a new recurrent mapping is actually requested.
        from resin import mint_state_mapping_arena_id

        arena_id = mint_state_mapping_arena_id()
        if not isinstance(arena_id, int) or isinstance(arena_id, bool) or arena_id <= 0:
            raise RuntimeError("native recurrent-state arena id must be positive")
        if arena_id in _RECURRENT_STATE_ARENAS:
            raise RuntimeError(
                f"native recurrent-state arena id {arena_id} is already registered"
            )

        (
            pool_id,
            capacity,
            protocol,
            layer_indices,
            shape,
            device_key,
            dtype_key,
        ) = self._recurrent_state_arena_descriptor(binding)
        generations: list[int | None] = [None] * capacity
        owners: list[tuple[int, int] | None] = [None] * capacity
        with self._torch.cuda.stream(self._stream):
            slab = self._torch.zeros(
                shape,
                dtype=self._torch.bfloat16,
                device=self.device,
            )
        return _RecurrentStateArena(
            arena_id=arena_id,
            pool_id=pool_id,
            capacity=capacity,
            protocol=protocol,
            layer_indices=layer_indices,
            shape=shape,
            device_key=device_key,
            dtype_key=dtype_key,
            slab=slab,
            generations=generations,
            owners=owners,
        )

    def _commit_recurrent_state_mapping(
        self,
        binding: tuple[int, int, str],
        arena: _RecurrentStateArena,
    ) -> int:
        self._validate_recurrent_state_arena(arena, binding)
        registered = _RECURRENT_STATE_ARENAS.get(arena.arena_id)
        if registered is None:
            _RECURRENT_STATE_ARENAS[arena.arena_id] = arena
        elif registered is not arena:
            raise RuntimeError(
                f"recurrent-state arena {arena.arena_id} was registered concurrently"
            )

        self._conv_state_slab = arena.slab
        self._conv_state_generations = arena.generations
        self._conv_state_owners = arena.owners
        self._state_mapping_arena_id = arena.arena_id
        self._state_mapping_binding = binding
        return arena.arena_id

    def bind_cache_and_recurrent_state(
        self,
        cache_pool_id: int,
        layout: CacheLayout,
        state_pool_id: int,
        state_capacity: int,
        protocol: str,
        required_arena_id: int | None,
    ) -> tuple[int, int, int, str, int]:
        """Atomically bind the cache and recurrent-state physical arenas."""

        self._validate_cache_pool_binding(cache_pool_id, layout)
        state_binding = self._validate_recurrent_state_mapping_binding(
            state_pool_id, state_capacity, protocol
        )
        if required_arena_id is not None and (
            not isinstance(required_arena_id, int)
            or isinstance(required_arena_id, bool)
            or required_arena_id <= 0
        ):
            raise ValueError(
                "required recurrent-state arena id must be positive or None"
            )
        prepared = None
        if self._state_mapping_binding is None:
            # Arena lookup/allocation is the only fallible physical step. Finish
            # it before committing either runner-side identity.
            prepared = self._allocate_recurrent_state_mapping(
                state_binding, required_arena_id
            )
            arena_id = prepared.arena_id
        else:
            arena_id = self._state_mapping_arena_id
            if arena_id is None:
                raise RuntimeError("bound LFM2 recurrent-state arena id disappeared")
            if required_arena_id is not None and arena_id != required_arena_id:
                raise ValueError(
                    "this LFM2 recurrent-state arena does not match the StatePool's "
                    "required physical arena"
                )

        if prepared is not None:
            self._commit_recurrent_state_mapping(state_binding, prepared)
        self._cache_pool_id = cache_pool_id
        return (
            cache_pool_id,
            state_pool_id,
            state_capacity,
            protocol,
            arena_id,
        )

    def _validate_cache_binding(self, plan: BatchPlan) -> None:
        if self._cache_pool_id is None or plan.cache_pool_id != self._cache_pool_id:
            raise ValueError(
                f"BatchPlan cache pool {plan.cache_pool_id!r} does not match the "
                f"bound physical pool {self._cache_pool_id!r}"
            )
        groups = tuple(plan.cache_groups)
        if len(groups) != 1 or groups[0].group_index != 0:
            raise ValueError("BatchPlan does not contain LFM2 cache group zero")

    def _preflight_cache(self, plan: BatchPlan) -> _CacheKernelMetadata:
        """Validate and derive the complete cache launch before CUDA acceptance."""

        self._validate_cache_binding(plan)
        derived = _cache_kernel_metadata(plan, self.page_size)
        if any(block_id > self.max_pages for block_id in derived.block_ids):
            raise ValueError("BatchPlan cache block ID is outside the physical slab")
        return derived

    def _preflight_recurrent_state(
        self, plan: BatchPlan
    ) -> tuple[_RecurrentStateRow, ...]:
        """Validate the complete native CSR and ownership map without CUDA work."""

        binding = self._state_mapping_binding
        slab = self._conv_state_slab
        generations = self._conv_state_generations
        owners = self._conv_state_owners
        if binding is None or slab is None or generations is None or owners is None:
            raise ValueError("LFM2 recurrent-state mapping is not bound")

        pool_id, capacity, protocol = binding
        if plan.state_pool_id != pool_id:
            raise ValueError(
                f"BatchPlan state pool id {plan.state_pool_id!r} does not match "
                f"the bound recurrent-state pool {pool_id!r}"
            )
        if protocol != _RECURRENT_STATE_MAPPING_PROTOCOL:
            raise RuntimeError("bound LFM2 recurrent-state protocol diverged")

        engine_id = plan.engine_id
        if (
            not isinstance(engine_id, int)
            or isinstance(engine_id, bool)
            or engine_id <= 0
        ):
            raise ValueError("BatchPlan engine_id must be a positive integer")

        request_ids = tuple(plan.sequence_ids)
        row_count = len(request_ids)
        state_indptr = tuple(plan.state_indptr)
        slots = tuple(plan.state_slots)
        state_generations = tuple(plan.state_generations)
        access_modes = tuple(plan.state_access_modes)
        if state_indptr != tuple(range(row_count + 1)):
            raise ValueError(
                "LFM2 recurrent-state CSR must contain exactly one handle per row"
            )
        if not (len(slots) == len(state_generations) == len(access_modes) == row_count):
            raise ValueError(
                "LFM2 recurrent-state metadata must contain exactly one handle per row"
            )
        if any(mode != "write" for mode in access_modes):
            raise ValueError("LFM2 recurrent-state handles must all use write access")

        rows = []
        seen_pairs: set[tuple[int, int]] = set()
        seen_slots: set[int] = set()
        for request_id, slot, generation in zip(
            request_ids, slots, state_generations, strict=True
        ):
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise ValueError("recurrent-state slot must be an integer")
            if slot < 0 or slot >= capacity:
                raise ValueError(
                    f"recurrent-state slot {slot!r} is outside capacity {capacity}"
                )
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation <= 0
            ):
                raise ValueError("recurrent-state generation must be positive")
            pair = (slot, generation)
            if pair in seen_pairs:
                raise ValueError(
                    "duplicate recurrent-state (slot, generation) in one batch"
                )
            if slot in seen_slots:
                raise ValueError("one recurrent-state slot cannot serve two batch rows")
            seen_pairs.add(pair)
            seen_slots.add(slot)

            current_generation = generations[slot]
            current_owner = owners[slot]
            if (current_generation is None) != (current_owner is None):
                raise RuntimeError("LFM2 recurrent-state sidecars diverged")
            owner = (engine_id, request_id)
            if current_generation is not None and generation < current_generation:
                raise ValueError("recurrent-state generation is stale")
            if current_generation == generation and current_owner != owner:
                raise ValueError(
                    "recurrent-state slot/generation is already owned by a "
                    "different Engine request"
                )
            rows.append(_RecurrentStateRow(slot, generation, engine_id, request_id))

        return tuple(rows)

    def _prepare_recurrent_state(self, rows: tuple[_RecurrentStateRow, ...]) -> None:
        """Zero newly generated slots in stream order, then publish tombstones."""

        slab = self._conv_state_slab
        generations = self._conv_state_generations
        owners = self._conv_state_owners
        if slab is None or generations is None or owners is None:
            raise RuntimeError("LFM2 recurrent-state mapping disappeared")
        for row in rows:
            if generations[row.slot] == row.generation:
                continue
            slab[:, row.slot].zero_()
            generations[row.slot] = row.generation
            owners[row.slot] = (row.engine_id, row.request_id)

    def _metadata(
        self,
        plan: BatchPlan,
        derived: _CacheKernelMetadata | None = None,
    ):
        torch = self._torch
        if derived is None:
            derived = self._preflight_cache(plan)
        prefill = derived.prefill

        pinned = {}
        metadata = {"_pinned": pinned}

        def host(name, values, dtype):
            tensor = torch.tensor(
                values,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            pinned[name] = tensor
            return tensor

        def device(name, values, dtype):
            tensor = host(name, values, dtype)
            return tensor.to(device=self.device, non_blocking=True)

        # Append and PyTorch-indexing arguments are CUDA tensors. Their pinned
        # sources must outlive the nonblocking H2D copies.
        metadata.update(
            {
                "block_indptr": device(
                    "block_indptr", derived.block_indptr, torch.int32
                ),
                "block_ids": device("block_ids", derived.block_ids, torch.int32),
                "last_block_lengths": device(
                    "last_block_lengths", derived.last_block_lengths, torch.int32
                ),
                "append_batch_indices": device(
                    "append_batch_indices",
                    derived.append_batch_indices,
                    torch.int32,
                ),
                "append_positions": device(
                    "append_positions", derived.append_positions, torch.int32
                ),
                "prefill_token_indices": device(
                    "prefill_token_indices",
                    prefill.dense_token_indices,
                    torch.long,
                ),
                "prefill_cu_seqlens": device(
                    "prefill_cu_seqlens", prefill.dense_cu_seqlens, torch.int32
                ),
                "prefill_max_q_len": prefill.dense_max_q_len,
                "decode_token_indices": device(
                    "decode_token_indices",
                    derived.decode_token_indices,
                    torch.long,
                ),
            }
        )

        if prefill.continuation_max_q_len:
            continuation_page_table = device(
                "continuation_prefill_page_table",
                prefill.continuation_page_table,
                torch.int32,
            ).view(
                len(prefill.continuation_seqused_k),
                prefill.continuation_page_table_width,
            )
            metadata.update(
                {
                    "continuation_prefill_token_indices": device(
                        "continuation_prefill_token_indices",
                        prefill.continuation_token_indices,
                        torch.long,
                    ),
                    "continuation_prefill_cu_seqlens": device(
                        "continuation_prefill_cu_seqlens",
                        prefill.continuation_cu_seqlens,
                        torch.int32,
                    ),
                    "continuation_prefill_page_table": continuation_page_table,
                    "continuation_prefill_seqused_k": device(
                        "continuation_prefill_seqused_k",
                        prefill.continuation_seqused_k,
                        torch.int32,
                    ),
                    "continuation_prefill_max_q_len": (prefill.continuation_max_q_len),
                    "continuation_prefill_min_seqlen_k": (
                        prefill.continuation_min_seqlen_k
                    ),
                    "continuation_prefill_max_seqlen_k": (
                        prefill.continuation_max_seqlen_k
                    ),
                }
            )
        else:
            metadata["continuation_prefill_max_q_len"] = 0

        # FlashInfer's planner consumes these host views. Supplying the pinned
        # Rust-derived values directly avoids its CUDA-to-host planning copy and
        # synchronization; append still uses the independent CUDA views above.
        metadata.update(
            {
                "decode_block_indptr": host(
                    "decode_block_indptr", derived.decode_block_indptr, torch.int32
                ),
                "decode_block_ids": host(
                    "decode_block_ids", derived.decode_block_ids, torch.int32
                ),
                "decode_last_block_lengths": host(
                    "decode_last_block_lengths",
                    derived.decode_last_block_lengths,
                    torch.int32,
                ),
            }
        )
        return metadata

    def _stage_device_list(self, staging, name, values, dtype):
        host = self._torch.tensor(
            values,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        device = host.to(device=self.device, non_blocking=True)
        staging[f"{name}_host"] = host
        staging[name] = device
        return device

    def _plan_decode(self, plan: BatchPlan, metadata) -> bool:
        torch = self._torch
        if not plan.decode_sequence_indices:
            return False
        self._decode.plan(
            metadata["decode_block_indptr"],
            metadata["decode_block_ids"],
            metadata["decode_last_block_lengths"],
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            sm_scale=self.scale,
            data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
        )
        return True

    def _attention(self, layer_index, layer, hidden, positions, metadata, decode):
        torch = self._torch
        token_count = hidden.shape[0]
        queries = layer.q_proj(hidden).view(
            token_count, self.num_query_heads, self.head_dim
        )
        keys = layer.k_proj(hidden).view(token_count, self.num_kv_heads, self.head_dim)
        values = layer.v_proj(hidden).view(
            token_count, self.num_kv_heads, self.head_dim
        )
        queries = layer.q_layernorm(queries)
        keys = layer.k_layernorm(keys)
        queries, keys = layer._apply_rope(
            queries.transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            positions.unsqueeze(0),
        )
        queries = queries.squeeze(0).transpose(0, 1).contiguous()
        keys = keys.squeeze(0).transpose(0, 1).contiguous()

        cache_slot = self._attention_slot[layer_index]
        k_cache = self._k_cache[cache_slot]
        v_cache = self._v_cache[cache_slot]
        self._flashinfer.append_paged_kv_cache(
            keys,
            values,
            metadata["append_batch_indices"],
            metadata["append_positions"],
            (k_cache, v_cache),
            metadata["block_ids"],
            metadata["block_indptr"],
            metadata["last_block_lengths"],
            kv_layout="NHD",
        )

        output = torch.empty_like(queries)
        if metadata["prefill_max_q_len"]:
            prefill_token_indices = metadata["prefill_token_indices"]
            prefill_output = self._fa4(
                queries[prefill_token_indices],
                keys[prefill_token_indices],
                values[prefill_token_indices],
                cu_seqlens_q=metadata["prefill_cu_seqlens"],
                cu_seqlens_k=metadata["prefill_cu_seqlens"],
                max_seqlen_q=metadata["prefill_max_q_len"],
                max_seqlen_k=metadata["prefill_max_q_len"],
                softmax_scale=self.scale,
                causal=True,
                pack_gqa=True,
            )
            if isinstance(prefill_output, tuple):
                prefill_output = prefill_output[0]
            output[prefill_token_indices] = prefill_output

        if metadata["continuation_prefill_max_q_len"]:
            continuation_token_indices = metadata["continuation_prefill_token_indices"]
            continuation_output = _invoke_paged_fa4_b22(
                self._fa4,
                queries[continuation_token_indices],
                k_cache,
                v_cache,
                cu_seqlens_q=metadata["continuation_prefill_cu_seqlens"],
                max_seqlen_q=metadata["continuation_prefill_max_q_len"],
                page_table=metadata["continuation_prefill_page_table"],
                seqused_k=metadata["continuation_prefill_seqused_k"],
                min_seqlen_k=metadata["continuation_prefill_min_seqlen_k"],
                max_seqlen_k=metadata["continuation_prefill_max_seqlen_k"],
                softmax_scale=self.scale,
            )
            output[continuation_token_indices] = continuation_output

        if decode:
            decode_token_indices = metadata["decode_token_indices"]
            decode_output = self._decode.run(
                queries[decode_token_indices], (k_cache, v_cache)
            )
            output[decode_token_indices] = decode_output

        return layer.out_proj(output.reshape(token_count, -1))

    def _convolution(self, layer_index, layer, hidden, plan, recurrent_rows, ephemeral):
        slab = self._conv_state_slab
        if slab is None:
            raise RuntimeError("LFM2 recurrent-state mapping disappeared")
        conv_slot = self._conv_slot[layer_index]
        outputs = []
        for index, row in enumerate(recurrent_rows):
            start, end = plan.sequence_indptr[index : index + 2]
            state = slab[conv_slot, row.slot].unsqueeze(0)
            cache = _InPlaceConvCache(state)
            # The facade turns ShortConv's ``cache.state = next_state`` into an
            # in-place copy and therefore cannot detach this generation from its
            # fixed physical slab address. Retain both view and facade until the
            # terminal event proves every dependent kernel complete.
            ephemeral.extend((state, cache))
            outputs.append(layer(hidden[start:end].unsqueeze(0), cache).squeeze(0))
        return self._torch.cat(outputs, dim=0)

    def _run_model(
        self,
        plan,
        tokens,
        positions,
        final_indices,
        metadata,
        decode,
        recurrent_rows,
        ephemeral,
    ):
        torch = self._torch
        hidden = self._model.model.embed_tokens(tokens)
        for layer_index, layer in enumerate(self._model.model.layers):
            normalized = layer.operator_norm(hidden)
            if layer.is_attention_layer:
                residual = self._attention(
                    layer_index,
                    layer.self_attn,
                    normalized,
                    positions,
                    metadata,
                    decode,
                )
            else:
                residual = self._convolution(
                    layer_index,
                    layer.conv,
                    normalized,
                    plan,
                    recurrent_rows,
                    ephemeral,
                )
            hidden = hidden + residual
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))
        hidden = self._model.model.embedding_norm(hidden)
        logits = torch.nn.functional.linear(
            hidden[final_indices], self._model.model.embed_tokens.weight
        )
        output = logits.argmax(dim=-1)
        # Keep the terminal activation chain and device result explicitly alive.
        # PyTorch's stream-aware allocator safely recycles earlier layer-local
        # temporaries in order on this same stream.
        ephemeral.extend((hidden, logits))
        return output

    def _completion_resources(self):
        """Return the persistent CUDA objects one accepted launch must lease."""

        return (
            self._torch,
            self._flashinfer,
            self._fa4,
            self._model,
            self._k_cache,
            self._v_cache,
            self._decode_workspace,
            self._decode,
            self._stream,
            self._conv_state_slab,
            self._conv_state_generations,
            self._conv_state_owners,
            self._state_mapping_binding,
        )

    def dispatch(self, plan: BatchPlan):
        """Enqueue one model step on the private stream and return its capsule."""

        if self._inflight is not None:
            raise RuntimeError("Lfm2CudaRunner already has an in-flight batch")
        if not plan.sequence_ids:
            raise ValueError("BatchPlan is empty")
        if any(token < 0 or token >= self.vocab_size for token in plan.token_ids):
            raise ValueError("token id is out of range")
        cache_metadata = self._preflight_cache(plan)
        recurrent_rows = self._preflight_recurrent_state(plan)

        torch = self._torch
        start_event = torch.cuda.Event()
        done_event = torch.cuda.Event()
        completion = _Lfm2CudaCompletion(
            self._completion_owner,
            plan,
            start_event,
            done_event,
            self._completion_resources(),
            self._stream,
            _CUDA_COMPLETION_QUARANTINE,
        )
        # Install the guard before any accelerator operation can be enqueued. If
        # dispatch raises after this point, retaining the capsule is safer than
        # allowing a second launch to race partially submitted work.
        self._inflight = completion
        accepted = False
        try:
            with torch.cuda.stream(self._stream), torch.inference_mode():
                start_event.record(self._stream)
                accepted = True
                completion._accepted = True
                self.forward_calls += 1
                try:
                    self._prepare_recurrent_state(recurrent_rows)
                    metadata = self._metadata(plan, cache_metadata)
                    staging = {
                        "metadata": metadata,
                        "recurrent_rows": recurrent_rows,
                    }
                    completion._staging = staging
                    tokens = self._stage_device_list(
                        staging, "tokens", plan.token_ids, torch.long
                    )
                    positions = self._stage_device_list(
                        staging, "positions", plan.positions, torch.long
                    )
                    final_indices = self._stage_device_list(
                        staging,
                        "final_indices",
                        [end - 1 for end in plan.sequence_indptr[1:]],
                        torch.long,
                    )
                    completion._ephemeral.extend((tokens, positions, final_indices))
                    decode = self._plan_decode(plan, metadata)
                    output_device = self._run_model(
                        plan,
                        tokens,
                        positions,
                        final_indices,
                        metadata,
                        decode,
                        recurrent_rows,
                        completion._ephemeral,
                    )
                    completion._ephemeral.append(output_device)
                    output_host = torch.empty(
                        output_device.shape,
                        dtype=output_device.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    output_host.copy_(output_device, non_blocking=True)
                    staging["output_host"] = output_host
                    completion._output_host = output_host
                except BaseException as error:
                    # Once the start event is accepted, a Python exception cannot
                    # prove that no CUDA work was submitted. Convert it into a
                    # poll error so Rust quarantines the batch instead of retrying.
                    completion._dispatch_error = error
                try:
                    done_event.record(self._stream)
                    completion._done_recorded = True
                except BaseException as error:
                    if completion._dispatch_error is None:
                        completion._dispatch_error = error
        except BaseException as error:
            if not accepted:
                self._inflight = None
                raise
            if completion._dispatch_error is None:
                completion._dispatch_error = error

        return completion

    def poll(self, completion) -> list[int] | None:
        """Return None while pending, then retire and return one token per row."""

        if not isinstance(completion, _Lfm2CudaCompletion):
            raise TypeError("completion was not created by Lfm2CudaRunner.dispatch")
        if completion._owner is not self._completion_owner:
            raise ValueError("completion belongs to a different Lfm2CudaRunner")
        if self._inflight is not completion or completion._retired:
            raise ValueError("completion is stale or is not the in-flight batch")
        if not completion._done_recorded:
            raise completion._dispatch_error
        if not completion._done_event.query():
            return None
        if completion._dispatch_error is not None:
            raise completion._dispatch_error

        # Event readiness includes the private-stream D2H copy. Do not clear the
        # one-in-flight guard if extraction raises; the same handle can be polled
        # again without losing its plan or tensor lifetimes.
        output = completion._output_host.tolist()
        completion._terminal = True
        completion._retire()
        self._inflight = None
        return output

    def release(self, request_ids: list[int]) -> None:
        # Recurrent state is StatePool-owned. Request release cannot invalidate a
        # physical generation or its owner tombstone; only a new generation may
        # zero and reuse that slot.
        del request_ids

    @property
    def live_cache_count(self) -> int:
        # This runner no longer owns request-keyed cache objects. Its persistent
        # convolution slab is native-StatePool-mapped physical storage.
        return 0
