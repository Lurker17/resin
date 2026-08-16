from collections.abc import Sequence
from typing import Any, Final, final

from resin.protocols import ConversationCodec, Runner

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
    Immutable Engine-issued execution plan for one homogeneous batch phase.

    ``token_ids`` contains every sequence's new tokens in one flat array.
    Adjacent values in ``token_indptr`` select each sequence's tokens.
    ``context_lengths[i]`` is the number of tokens already processed for
    sequence ``i``; positions are therefore implicit and start at that value.
    ``cache_groups`` supplies the pages used by those sequences.
    """
    def __new__(
        cls,
        /,
        sequence_ids: Sequence[int],
        context_lengths: Sequence[int],
        token_ids: Sequence[int],
        token_indptr: Sequence[int],
        *,
        phase: BatchPhase = ...,
        batch_id: int = 0,
        batch_key: int | None = None,
        cache_groups: Sequence[CacheGroupPlan] | None = None,
        sample_sequence_indices: Sequence[int] | None = None,
        sampling_params: Sequence[SamplingParams] | None = None,
    ) -> BatchPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def batch_id(self, /) -> int:
        """
        Monotonic identifier used to match an async dispatch with its result.
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
        """
    @property
    def context_lengths(self, /) -> list[int]:
        """
        Number of tokens already processed for each sequence.
        """
    @property
    def num_sequences(self, /) -> int:
        """
        Number of sequences (batch size).
        """
    @property
    def phase(self, /) -> BatchPhase:
        """
        Whether this is a prefill or decode batch. Mixed-chunk is not supported.
        """
    @property
    def sample_sequence_indices(self, /) -> list[int]:
        """
        Which sequence indices should generate a sampled token. Used for
        chunked prefill, when some sequences only advance the cache.
        """
    @property
    def sampling_params(self, /) -> tuple[SamplingParams, ...]:
        """
        Effective sampling parameters, one per sequence.
        """
    @property
    def sequence_ids(self, /) -> list[int]:
        """
        Stable request identifier for each sequence.
        """
    @property
    def token_ids(self, /) -> list[int]:
        """
        Flattened scheduled token IDs, grouped by sequence.
        """
    @property
    def token_indptr(self, /) -> list[int]:
        """
        Offsets into ``token_ids`` for each sequence.
        """
    def token_range(self, /, sequence_index: int) -> tuple[int, int]:
        """
        Return the half-open range in ``token_ids`` for one sequence.
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
    Geometry and retention policy for one logical cache group.

    Groups with different tensor layouts share the parent pool. A group packs
    ``blocks_per_parent`` logical blocks into each equally-sized parent.
    """
    def __new__(
        cls,
        /,
        block_tokens: int,
        *,
        blocks_per_parent: int = 1,
        retention: CacheRetention = ...,
        window_tokens: int | None = None,
        family: CacheGroupFamily = ...,
    ) -> CacheGroupLayout: ...
    def __repr__(self, /) -> str: ...
    @property
    def block_tokens(self, /) -> int:
        """
        Number of token positions represented by one logical block.
        """
    @property
    def blocks_per_parent(self, /) -> int:
        """
        Number of this group's logical blocks packed into one parent allocation.
        """
    @property
    def family(self, /) -> CacheGroupFamily:
        """
        Whether this group stores token history or recurrent checkpoints.
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
        Retained suffix length for ``WINDOW`` groups, otherwise ``None``.
        """

@final
class CacheGroupPlan:
    """
    Physical page placement for one cache group in a BatchPlan.

    Adjacent ``block_indptr`` values select each sequence's physical pages from
    ``block_ids``. ``first_logical_blocks`` gives the logical number represented
    by each sequence's first physical block. The two ``append_*`` arrays contain
    one write target per flattened batch token.
    """
    def __new__(
        cls,
        /,
        group_index: int,
        block_indptr: Sequence[int],
        block_ids: Sequence[int],
        first_logical_blocks: Sequence[int],
        append_block_ids: Sequence[int],
        append_block_offsets: Sequence[int],
    ) -> CacheGroupPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def append_block_ids(self, /) -> list[int]:
        """
        Physical destination block for every flattened batch token.
        """
    @property
    def append_block_offsets(self, /) -> list[int]:
        """
        In-block destination offset for every flattened batch token.
        """
    @property
    def block_ids(self, /) -> list[int]:
        """
        Flattened physical block IDs, grouped chronologically by sequence.
        """
    @property
    def block_indptr(self, /) -> list[int]:
        """
        Offsets into ``block_ids`` for each sequence.
        """
    @property
    def first_logical_blocks(self, /) -> list[int]:
        """
        Logical block number corresponding to each sequence's first block.
        """
    @property
    def group_index(self, /) -> int:
        """
        Zero-based index into the Engine's CacheLayout groups.
        """

@final
class CacheLayout:
    """
    Immutable grouped-cache geometry owned by one Engine.

    ``parent_capacity`` is the number of fungible physical parents shared by
    every group. Group order is stable and identifies each group in BatchPlan.
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
        Number of fungible physical parent allocations in the pool.
        """

@final
class CacheRetention:
    """
    Cache-block lifetime policy.

    For history groups, ``HISTORY`` retains every logical block and ``WINDOW``
    retains the suffix selected by ``CacheGroupLayout.window_tokens``. Recurrent
    state groups use ``HISTORY`` and retain their two-page checkpoint ring.
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
    def asynchronous(self, /) -> bool: ...
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
    def dispatch(self, /) -> int | None: ...
    def has_work(self, /) -> bool: ...
    @property
    def in_flight_batch_count(self, /) -> int: ...
    @property
    def in_flight_batch_id(self, /) -> int | None: ...
    @property
    def in_flight_request_count(self, /) -> int: ...
    @property
    def max_running_requests(self, /) -> int: ...
    @property
    def new_token_ratio(self, /) -> float: ...
    @property
    def poison_error(self, /) -> str | None: ...
    @property
    def poisoned(self, /) -> bool: ...
    def poll(self, /) -> list[TokenEvent]: ...
    @property
    def prefix_cache_stats(self, /) -> tuple[int, int, int, int] | None:
        """
        ``(entries, hits, misses, evictions)`` when prefix caching is enabled.
        """
    @property
    def ready_request_count(self, /) -> int: ...
    @property
    def scheduler_config(self, /) -> SchedulerConfig: ...
    def step(self, /) -> list[TokenEvent]: ...
    def submit(
        self,
        /,
        token_ids: Sequence[int],
        max_new_tokens: int | None = None,
        *,
        priority: int = 0,
        batch_key: int | None = None,
        sampling_params: SamplingParams | None = None,
        stop_token_ids: Sequence[int] | None = None,
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
    Concrete Engine-owned resource identities passed to ``bind_resources``.
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

    Runners without special requirements may omit ``runner_config`` entirely.
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
        extra: Any | None = None,
    ) -> SamplingParams: ...
    def __repr__(self, /) -> str: ...
    @property
    def extra(self, /) -> Any | None: ...
    @property
    def seed(self, /) -> int | None: ...
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
