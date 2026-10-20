from collections.abc import Sequence
from typing import Any, Final, final
import numpy as np
import numpy.typing as npt
from resin.protocols import ConversationCodec, MediaInput, Runner

__version__: Final[str]

@final
class BatchPhase:
    DECODE: Final[BatchPhase]
    PREFILL: Final[BatchPhase]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class BatchPlan:
    """
    Immutable execution plan for one prefill or decode step.

    This runs one forward pass on a batch of sequences. `token_ids` contains
    the newly scheduled input tokens, packed by sequence into one flat array;
    `token_indptr` is the CSR row pointer, so sequence `i` owns
    `token_ids[token_indptr[i]:token_indptr[i + 1]]`.

    `context_lengths[i]` is the number of tokens processed before this step,
    so that sequence's new tokens occupy consecutive absolute positions
    beginning there. `cache_groups` contains one placement plan per declared
    `CacheLayout` group, and every group plan covers every sequence.
    """
    def __new__(
        cls,
        /,
        sequence_ids: Sequence[int] | npt.NDArray[np.uint64],
        context_lengths: Sequence[int] | npt.NDArray[np.uint32],
        token_ids: Sequence[int] | npt.NDArray[np.uint32],
        token_indptr: Sequence[int] | npt.NDArray[np.uint32],
        *,
        phase: BatchPhase = ...,
        batch_id: int = 0,
        batch_key: int | None = None,
        cache_groups: Sequence[CacheGroupPlan] | None = None,
        sampling_mask: Sequence[bool] | npt.NDArray[np.bool_] | None = None,
        sampling_params: Sequence[SamplingParams] | None = None,
        media: Sequence[MediaInput | None] | None = None,
    ) -> BatchPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def batch_id(self, /) -> int:
        """
        Monotonic identifier for tracing and correlating runner steps.
        """
    @property
    def batch_key(self, /) -> int | None:
        """
        Optional key that prevents incompatible requests from sharing a batch.
        """
    @property
    def cache_groups(self, /) -> tuple[CacheGroupPlan, ...]:
        """
        Ordered physical placement plans corresponding to CacheLayout groups.
        There is one plan per layout group, not one plan per sequence; each plan
        contains one block-table row for every sequence in this batch.
        """
    @property
    def context_lengths(self, /) -> npt.NDArray[np.uint32]:
        """
        Number of tokens already processed for each sequence.
        """
    @property
    def media(self, /) -> tuple[MediaInput | None, ...]:
        """
        Opaque runner inputs, one per sequence. The scheduler never inspects them.
        """
    @property
    def num_sequences(self, /) -> int:
        """
        Number of sequences (batch size).
        """
    @property
    def num_tokens(self, /) -> int:
        """
        Number of flattened scheduled tokens.
        """
    @property
    def phase(self, /) -> BatchPhase:
        """
        Whether this is a prefill or decode batch. Mixed-chunk is not supported.
        """
    @property
    def positions(self, /) -> npt.NDArray[np.uint32]:
        """
        Absolute position of every flattened scheduled token.
        """
    @property
    def sampling_mask(self, /) -> npt.NDArray[np.bool_]:
        """
        Which sequences should generate a sampled token. Used for chunked
        prefill, when some sequences only advance the cache.
        """
    @property
    def needs_logprobs(self, /) -> bool:
        """Whether any sampled row requests scores, checked without boxing policies."""
    @property
    def sampling_params(self, /) -> tuple[SamplingParams, ...]:
        """
        Effective sampling parameters, one per sequence.
        """
    @property
    def sequence_ids(self, /) -> npt.NDArray[np.uint64]:
        """
        Stable request identifier for each sequence.
        """
    @property
    def token_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        Flattened scheduled token IDs, grouped by sequence.
        """
    @property
    def token_indptr(self, /) -> npt.NDArray[np.uint32]:
        """
        Offsets into `token_ids` for each sequence.
        """
    def token_range(self, /, sequence_index: int) -> tuple[int, int]:
        """
        Return the half-open range in `token_ids` for one sequence.
        """

@final
class CacheGroupFamily:
    """
    Whether a group stores token history or recurrent checkpoints.
    """

    HISTORY: Final[CacheGroupFamily]
    STATE: Final[CacheGroupFamily]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class CacheGroupLayout:
    """
    Geometry and retention policy for one type of cache.

    Cache groups share a pool of equal-sized slabs. `blocks_per_slab` is how
    many of this group's blocks fit in one slab.
    """
    def __new__(
        cls,
        /,
        block_tokens: int,
        *,
        blocks_per_slab: int = 1,
        retention: CacheRetention = ...,
        window_tokens: int | None = None,
        family: CacheGroupFamily = ...,
        read_window_tokens: int | None = None,
    ) -> CacheGroupLayout: ...
    def __repr__(self, /) -> str: ...
    @property
    def block_tokens(self, /) -> int:
        """
        Number of token positions represented by one logical block.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab.
        """
    @property
    def family(self, /) -> CacheGroupFamily:
        """
        Whether this group stores token history or recurrent checkpoints.
        """
    @property
    def read_window_tokens(self, /) -> int | None:
        """
        Logical read window including the current token. Defaults to the full
        retained window; reserve extra `window_tokens` for simultaneous writes.
        """
    @property
    def retention(self, /) -> CacheRetention:
        """
        Lifetime policy for logical blocks in this group.
        """
    @property
    def window_resident_block_count(self, /) -> int | None:
        """
        Maximum resident logical blocks per sequence for a window group.
        """
    @property
    def window_tokens(self, /) -> int | None:
        """
        Retained suffix length for `WINDOW` groups, otherwise `None`.
        """

@final
class CacheGroupPlan:
    """
    Plan for one type of cache in a BatchPlan.

    A model can have one or more types of cache, such as full attention,
    sliding-window attention, or recurrent state, each described by one
    `CacheGroupLayout`. Cache memory is organized into fungible slabs. Each
    slab is assigned to one cache group and holds `blocks_per_slab` blocks.

    A block is indexed by `(slab_id, slotnum)`: `block_slab_ids` contains
    one-based slab IDs and `block_slotnums` contains zero-based slot numbers. A
    cached token location is `(block, block_offset)`, equivalently
    `(slab_id, slotnum, block_offset)`.

    The plan covers every sequence in the batch. Adjacent `block_indptr`
    values select each sequence's chronological block-table row.
    `block_table_start_indices[i]` is sequence `i`'s block-table start index, so
    row entry `j` represents block `block_table_start_indices[i] + j`; block
    `b` begins at token position `b * CacheGroupLayout.block_tokens`. Full-
    attention tables start at zero; sliding-window and recurrent tables advance
    as old blocks leave their resident rings.

    For flattened batch token `k`, the `append_*` arrays supply exactly that
    cached-token location: `(append_slab_ids[k], append_slotnums[k],
    append_block_offsets[k])`.
    """
    def __new__(
        cls,
        /,
        group_index: int,
        block_indptr: Sequence[int] | npt.NDArray[np.uint32],
        block_slab_ids: Sequence[int] | npt.NDArray[np.uint32],
        block_slotnums: Sequence[int] | npt.NDArray[np.uint32],
        block_table_start_indices: Sequence[int] | npt.NDArray[np.uint32],
        append_slab_ids: Sequence[int] | npt.NDArray[np.uint32],
        append_slotnums: Sequence[int] | npt.NDArray[np.uint32],
        append_block_offsets: Sequence[int] | npt.NDArray[np.uint32],
        *,
        blocks_per_slab: int = 1,
    ) -> CacheGroupPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def append_block_offsets(self, /) -> npt.NDArray[np.uint32]:
        """
        Token offset within each token's destination block. Entry `k`
        corresponds to `BatchPlan.token_ids[k]` and is less than the matching
        `CacheGroupLayout.block_tokens`.
        """
    @property
    def append_slab_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        One-based slab ID of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def append_slotnums(self, /) -> npt.NDArray[np.uint32]:
        """
        Zero-based slot number of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def block_indptr(self, /) -> npt.NDArray[np.uint32]:
        """
        CSR row pointer into the block-address arrays. Its length is one more
        than the batch sequence count.
        """
    @property
    def block_slab_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        One-based slab ID for each flattened block-table entry. Pair with
        `block_slotnums`; slab zero is reserved as null.
        """
    @property
    def block_slotnums(self, /) -> npt.NDArray[np.uint32]:
        """
        Zero-based slot number for each flattened block-table entry. Pair with
        `block_slab_ids`.
        """
    @property
    def block_table_start_indices(self, /) -> npt.NDArray[np.uint32]:
        """
        Absolute logical-block index at which each sequence's table row starts.
        This is zero for full-attention history and advances for sliding-window
        and recurrent rings. It has one entry per sequence and is unrelated to
        the physical slab ID.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab. Valid slot numbers
        are `0 <= slotnum < blocks_per_slab`.
        """
    @property
    def group_index(self, /) -> int:
        """
        Zero-based index of this plan's matching CacheGroupLayout. BatchPlan
        orders group plans by this index and rejects gaps or reordering.
        """

@final
class CacheLayout:
    """
    Immutable grouped-cache geometry owned by one Engine.

    `parent_capacity` is the number of fungible physical slabs shared by every
    group. Group order is stable and identifies each group in BatchPlan.
    """
    def __new__(
        cls, /, parent_capacity: int, groups: Sequence[CacheGroupLayout]
    ) -> CacheLayout: ...
    def __repr__(self, /) -> str: ...
    @property
    def group_count(self, /) -> int:
        """
        Number of logical cache groups.
        """
    def group_kernel_page_capacity(self, /, group_index: int) -> int: ...
    def group_kernel_page_count(self, /, group_index: int) -> int: ...
    @property
    def groups(self, /) -> tuple[CacheGroupLayout, ...]:
        """
        Ordered immutable cache-group descriptors.
        """
    @property
    def parent_capacity(self, /) -> int:
        """
        Number of fungible physical slabs in the pool.
        """

@final
class CacheRetention:
    """
    Cache-block lifetime policy.

    For history groups, `HISTORY` retains every logical block and `WINDOW`
    retains the suffix selected by `CacheGroupLayout.window_tokens`. Recurrent
    state groups use `HISTORY` and retain their two-page checkpoint ring.
    """

    HISTORY: Final[CacheRetention]
    WINDOW: Final[CacheRetention]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class Engine:
    def __enter__(self, /) -> Engine: ...
    def __exit__(self, /, _exc_type: Any, _exc_value: Any, _traceback: Any) -> bool: ...
    def __new__(
        cls,
        /,
        runner: Runner,
        *,
        max_running_requests: int = 32,
        context_length: int = 32768,
        scheduler_config: SchedulerConfig | None = None,
        eos_token_id: int | None = None,
        prefix_cache: bool = False,
    ) -> Engine: ...
    def __repr__(self, /) -> str: ...
    @property
    def active_request_count(self, /) -> int: ...
    @property
    def cache_free_parent_count(self, /) -> int | None: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_occupied_block_count(self, /) -> int | None: ...
    @property
    def cache_parent_capacity(self, /) -> int | None: ...
    @property
    def cache_pool_id(self, /) -> int | None: ...
    def cancel(self, /, request: RequestLease) -> bool: ...
    def clear_prefix_cache(self, /) -> None:
        """
        Drop all reusable cache prefixes.
        """
    def close(self, /) -> None: ...
    @property
    def closed(self, /) -> bool: ...
    @property
    def context_length(self, /) -> int: ...
    def has_work(self, /) -> bool: ...
    @property
    def max_running_requests(self, /) -> int: ...
    @property
    def new_token_ratio(self, /) -> float: ...
    @property
    def poison_error(self, /) -> str | None: ...
    @property
    def poisoned(self, /) -> bool: ...
    @property
    def prefix_cache_stats(self, /) -> tuple[int, int, int, int] | None:
        """
        `(entries, hits, misses, evictions)` when prefix caching is enabled.
        """
    @property
    def ready_request_count(self, /) -> int: ...
    @property
    def scheduler_config(self, /) -> SchedulerConfig: ...
    def step(self, /, *, phase: BatchPhase | None = None) -> list[TokenEvent]: ...
    def next_phase(self, /) -> BatchPhase | None:
        """Read-only admission-aware proposal; reserves no requests or resources."""
    def poison(self, /, reason: str) -> None:
        """Permanently contain a peer/runtime failure without releasing resources."""
    def submit(
        self,
        /,
        token_ids: Sequence[int] | None = None,
        max_new_tokens: int | None = None,
        *,
        priority: int = 0,
        batch_key: int | None = None,
        sampling_params: SamplingParams | None = None,
        stop_token_ids: Sequence[int] | None = None,
        media: MediaInput | None = None,
    ) -> RequestLease: ...

@final
class RequestLease:
    """
    A caller-owned lifetime handle for one submitted request.

    The Engine keeps the shared state only while the request is active. Once
    the request retires, its generated tokens are moved into this lease. If
    the caller drops the lease without taking the result, the token buffer is
    released when the last internal reference disappears.
    """
    def __eq__(self, other: object, /) -> bool: ...
    def __index__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    @property
    def cached_tokens(self, /) -> int:
        """
        Number of prompt tokens restored from the prefix cache at admission.
        """
    @property
    def cancelled(self, /) -> bool: ...
    @property
    def done(self, /) -> bool: ...
    @property
    def id(self, /) -> int: ...
    @property
    def request_id(self, /) -> int: ...
    def take_result(self, /) -> list[int]:
        """
        Consume the completed token buffer exactly once.
        """

@final
class ResourceBinding:
    """
    Concrete Engine-owned resource identities passed to `bind_resources`.
    """
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_pool_id(self, /) -> int | None: ...

@final
class RunnerConfig:
    """
    Frozen scheduling limits and persistent-resource declarations for a runner.

    Runners without special requirements may omit `runner_config` entirely.
    """
    def __new__(
        cls,
        /,
        *,
        max_batch_tokens: int | None = None,
        prefill_chunk_size: int | None = None,
        continuation_prefill: bool = False,
        cache_write_lookahead_tokens: int = 0,
        cache_layout: CacheLayout | None = None,
    ) -> RunnerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_write_lookahead_tokens(self, /) -> int: ...
    @property
    def continuation_prefill(self, /) -> bool: ...
    @property
    def max_batch_tokens(self, /) -> int | None: ...
    @property
    def prefill_chunk_size(self, /) -> int | None: ...

@final
class SamplingParams:
    """
    Per-request output-token sampling policy.

    A zero temperature selects the maximum-logit token. Positive temperatures
    enable stochastic sampling after optional top-k and top-p filtering.
    """
    def __new__(
        cls,
        /,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        extra: Any | None = None,
    ) -> SamplingParams: ...
    def __repr__(self, /) -> str: ...
    @property
    def extra(self, /) -> Any | None: ...
    @property
    def seed(self, /) -> int | None: ...
    @property
    def logprobs(self, /) -> bool: ...
    @property
    def top_logprobs(self, /) -> int: ...
    @property
    def temperature(self, /) -> float: ...
    @property
    def top_k(self, /) -> int | None: ...
    @property
    def top_p(self, /) -> float: ...

@final
class SchedulerConfig:
    """
    Native admission and token-budget policy for an Engine.
    """
    def __new__(
        cls,
        /,
        *,
        max_batch_tokens: int | None = None,
        prefill_chunk_size: int = 8192,
        schedule_conservativeness: float = 1.0,
    ) -> SchedulerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def max_batch_tokens(self, /) -> int | None: ...
    @property
    def prefill_chunk_size(self, /) -> int: ...
    @property
    def schedule_conservativeness(self, /) -> float: ...

@final
class TokenEvent:
    @property
    def logprobs(self, /) -> TokenLogprobs | None: ...
    def __repr__(self, /) -> str: ...
    @property
    def finish_reason(self, /) -> str | None: ...
    @property
    def finished(self, /) -> bool: ...
    @property
    def index(self, /) -> int: ...
    @property
    def request_id(self, /) -> int: ...
    @property
    def token_id(self, /) -> int: ...

@final
class _HttpServer:
    def __new__(
        cls,
        /,
        engine: Engine,
        codec: ConversationCodec,
        *,
        model: str,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
    ) -> _HttpServer: ...
    def __repr__(self, /) -> str: ...
    @property
    def model(self, /) -> str: ...
    @property
    def running(self, /) -> bool: ...
    def serve(self, /, *, host: str = "127.0.0.1", port: int = 8000) -> None: ...
    def shutdown(self, /) -> None:
        """
        Request shutdown from another thread; serve returns after the current
        inference step has finished and its requests have been cancelled.
        """

@final
class TokenLogprobs:
    def __new__(cls, logprob: float, top_logprobs: Sequence[tuple[int, float]] = ()) -> TokenLogprobs: ...
    @property
    def logprob(self) -> float: ...
    @property
    def top_logprobs(self) -> list[tuple[int, float]]: ...

@final
class ForwardOutput:
    """Borrowed NumPy SoA output, valid until the synchronous step returns.

    All arrays must have their exact declared dtype, one dimension, and contiguous
    storage. Construction retains the arrays without copying. Do not mutate or
    recycle their storage until step() returns. Token rows use token_indptr;
    score rows use logprob_indptr (defaults to token_indptr). An empty score row
    opts out; every other score row must cover the entire raw output token row.
    top_indptr partitions top_token_ids/top_logprobs by scored token.
    """
    def __new__(
        cls, token_ids: npt.NDArray[np.uint32], token_indptr: npt.NDArray[np.uint32], *,
        logprobs: npt.NDArray[np.float32] | None = None,
        logprob_indptr: npt.NDArray[np.uint32] | None = None,
        top_indptr: npt.NDArray[np.uint32] | None = None,
        top_token_ids: npt.NDArray[np.uint32] | None = None,
        top_logprobs: npt.NDArray[np.float32] | None = None,
    ) -> ForwardOutput: ...
    @property
    def token_ids(self) -> npt.NDArray[np.uint32]: ...
    @property
    def token_indptr(self) -> npt.NDArray[np.uint32]: ...
    @property
    def logprobs(self) -> npt.NDArray[np.float32] | None: ...
    @property
    def logprob_indptr(self) -> npt.NDArray[np.uint32] | None: ...
    @property
    def top_indptr(self) -> npt.NDArray[np.uint32] | None: ...
    @property
    def top_token_ids(self) -> npt.NDArray[np.uint32] | None: ...
    @property
    def top_logprobs(self) -> npt.NDArray[np.float32] | None: ...
