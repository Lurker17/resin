# Resin architecture and API

Resin is a Rust control plane for token-ID-in/token-ID-out inference. Rust owns
requests, scheduling, logical progress, native storage ownership, and cleanup.
Python runners own model weights, physical tensor layouts, kernels, streams,
and completion events.

```text
token IDs -> Engine -> immutable BatchPlan -> model runner -> token IDs
                ^                             |
                +---- validate and commit ----+
```

The crossing is deliberately coarse: one immutable plan handoff per batch, not
one call per layer or kernel. An asynchronous runner may then be polled for that
handoff's completion. Tensor payloads never pass through Rust.

## Responsibility boundary

| Rust control plane | Python execution plane |
| --- | --- |
| Request IDs, priorities, cursors, EOS, cancellation, retirement | Model architecture, weights, and sampling |
| Cohort selection, token budgets, phase and sample masks | Tensorization and forward execution |
| Flat cache allocation, group placement, references, and writer leases | Physical cache slabs addressed by plan page IDs |
| State-slot generations, references, pins, and access leases | Model-specific recurrent-state tensors |
| Frozen and validated plan metadata | Kernels, streams, and completion events |
| Completion validation and fail-stop quarantine | Optional physical request cleanup |

The runner cannot admit work or commit logical progress. It executes a plan and
returns one token ID per sequence row. Resin decides which returned IDs become
events through the plan's sampling mask.

## Native crate boundary

`resin-core` is PyO3-free. It owns typed request identities, homogeneous
phase selection, token-range planning, and resource lifetimes. The root
`_resin` crate is the Python/runner adapter and translates core views into the
public `BatchPlan` ABI.

The Engine uses the same small generic shape at its two ownership boundaries:
`WithResources<Owner, ResourceSet<Attention, Recurrent>>`. The concrete fields
make durable request state visibly different from temporary accepted-batch
access:

| Resource | Request-owned state | Accepted-batch state | Executor view |
| --- | --- | --- | --- |
| Full-context attention | Private pages for maximum context | Writer lease on the active prefix | Chronological CSR block table and append targets |
| Sliding-window attention | Bounded private cyclic pages | Writer lease on the resident suffix | Chronological suffix plus current append targets |
| Recurrent state | Generation-checked handles and access modes | Pinned handles and shared-read/exclusive-write lease | Slot/generation CSR and access modes |

The paged-KV coordinator returns its executor view and exact writer lease
together and rejects a second acquisition until release. Recurrent access is
similarly retained by the accepted batch. Resin intentionally does not support
overlap scheduling, future-batch planning, or speculative recurrent-state
locks/residency. A quarantined accepted batch therefore keeps every capability
that could still be addressed by device work.

## Request lifecycle

1. `submit()` validates the prompt and optional state handles and returns a
   caller-owned `RequestLease`. If the runner
   declared a cache layout, Resin atomically reserves every request-owned cache
   block needed for the request's declared maximum lifetime.
2. The scheduler chooses a homogeneous cohort by priority, optional `batch_key`,
   queue age, phase policy, token budget, and state-access compatibility.
3. Resin freezes the selected token ranges, cache rows, append destinations,
   and state handles into a `BatchPlan`. It acquires exact state and cache writer
   leases before runner acceptance.
4. A synchronous runner receives `forward(plan)`. An asynchronous runner receives
   `dispatch(plan)` followed by zero or more `poll(completion)` calls. One batch
   may be in flight per Engine; this is an intentional resource contract, not an
   overlap-scheduling feature gap.
5. On success, optional runner cleanup runs first. Resin then unpins state,
   releases cache/state access, and retires terminal physical storage. Only
   after those steps succeed does it expose logical progress, requeue survivors,
   and return `TokenEvent`s.

Errors before runner acceptance restore the selected requests without changing
their queue age. Errors after work may have reached the device poison the Engine
and quarantine the accepted batch; potentially addressed storage is not returned
to an allocator. A runner-call exception may set `_resin_post_accept = True` to
select that fail-stop path. A malformed result after synchronous `forward`
returns is always post-accept. Ready cancellation retires immediately; in-flight
cancellation becomes terminal after accepted work completes. `close()` drains
accepted work, cancels queued work, and permanently closes a healthy Engine.

## Public API

Importing `resin` exports native descriptors only and does not initialize an
accelerator framework.

```python
SchedulerConfig(
    *,
    max_batch_tokens=None,
    prefill_chunk_size=8192,
    schedule_conservativeness=1.0,
)

CacheGroupLayout(
    block_tokens,
    *,
    blocks_per_parent=1,
    retention=CacheRetention.HISTORY,
    window_tokens=None,
)

CacheLayout(parent_capacity, groups)

StateMappingConfig(protocol, access_modes)
StatePool(capacity)

Engine(
    runner,
    *,
    max_running_requests=32,
    context_length=32768,
    scheduler_config=None,
    eos_token_id=None,
    state_pool=None,
)
```

`context_length` is the Engine's maximum model/KV context and defaults to
32768 tokens. A submission whose prompt plus requested completion could exceed
it is rejected before admission. `max_running_requests` is the scheduler's
request-count admission cap; `prefill_chunk_size` controls how many prompt
tokens one execution plan may contain.

The runner, not the Engine caller, owns its frozen `CacheLayout`. The Engine
discovers that declaration, creates exactly one matching native pool, and binds
it once. There is no second cache configuration surface.

The Engine lifecycle is:

```python
request = engine.submit(
    token_ids,
    max_new_tokens=None,
    *,
    state_handles=None,
    state_access_modes=None,
    priority=0,
    batch_key=None,
)

batch_id_or_none = engine.dispatch()
events = engine.poll()
events = engine.step()           # one dispatch attempt, then one poll
engine.cancel(request)
result = request.take_result()  # transfers and clears the token buffer
engine.has_work()
engine.close()
```

`TokenEvent` exposes `request_id`, `token_id`, zero-based generated-token
`index`, and `finished`. Engine diagnostics include ready/in-flight counts,
closed/poison state, the frozen cache layout and pool ID, free parents, and
occupied blocks.

`StatePool.allocate(count=1)` returns generation-checked `StateHandle`s.
`retain`/`release` manage caller references; `pin`/`unpin` delay physical reuse;
`counts`, `access_counts`, and pool counters expose diagnostics. The Engine takes
its own reference at submission, so a caller may keep a sharing reference or
release it to transfer ownership.

## One flat cache pool

A `CacheLayout` is an ordered list of cache groups over one pool of equal-byte
physical parents. Group position is its identity; there are no names or global
registries. A free parent is fungible. While occupied it is bound to one group
and divided into that group's `blocks_per_parent` logical blocks.

Kernel page ID zero is reserved for a runner-owned null/padding row. For positive
parent ID `p`, packing `K`, and zero-based slot `s`, a group's positive page ID
is:

```text
1 + (p - 1) * K + s
```

Consequently group `g` needs
`1 + parent_capacity * groups[g].blocks_per_parent` physical slab rows,
including row zero. The runner chooses each group's tensor interpretation; Resin
only owns placement and lifetime.

Retention describes request ownership:

- **History** grows transactionally to each scheduled token range and exposes
  the growing logical prefix.
- **Window** grows toward a bounded private set of blocks. Physical storage cycles
  as logical blocks advance, while each plan row projects the retained suffix
  back into chronological order. Prefill ranges stop at a Window block boundary
  so one accepted plan never overwrites a block it still reads.

Recurrent model state is not a cache-retention policy. It uses the separate
generation-checked `StatePool` API.

The live cache path has no content index. Requests own private blocks, so there
is no cross-request block sharing or prefix reuse in the current Engine.

See the [flat cache-pool contract](cache-pool.md) for allocation and plan details.

## BatchPlan ABI

`BatchPlan` is frozen. `phase` is one `BatchPhase.PREFILL` or
`BatchPhase.DECODE` value for the complete batch. Base arrays describe flattened
tokens and positions, sequence CSR (`sequence_ids`, `sequence_indptr`),
pre-append context lengths, and `sample_sequence_indices`. Each sequence row also
has its request's `SamplingParams`, with an effective seed resolved into it, plus
the current random offset. Optional state arrays provide generation-checked
slots, access modes, CSR, and a pool ID.

Managed cache metadata is only:

```python
plan.cache_pool_id
plan.cache_groups  # ordered tuple[CacheGroupPlan, ...]
```

Each `CacheGroupPlan` contains `group_index`, row CSR
(`block_indptr`, `block_ids`), `first_logical_blocks`, and one exact
`append_block_ids`/`append_block_offsets` destination per flattened batch token.
Rows always appear in sequence order and represent cache blocks chronologically,
even when a Window group's physical slots have cycled. Runners may tensorize
these arrays, but must not reconstruct a competing allocator or placement policy.

## Runner protocols

Runners are duck typed:

```python
class SyncRunner:
    def forward(self, plan: BatchPlan) -> list[int]: ...
    def release(self, request_ids: list[int]) -> None: ...  # optional


class AsyncRunner:
    def dispatch(self, plan: BatchPlan) -> object: ...
    def poll(self, completion: object) -> list[int] | None: ...
    def release(self, request_ids: list[int]) -> None: ...  # optional
```

`dispatch` and `poll` must be supplied together. Callable bindings are
snapshotted at Engine construction. `supports_continuation_prefill = True` is
required when scheduler token controls are enabled.

A cache-only runner declares `cache_layout` and implements:

```python
bind_cache_pool(pool_id, same_layout_object) -> pool_id
```

The acknowledgement must be an exact positive integer. A runner using mapped
recurrent state declares `supports_recurrent_state_mapping = True` and a frozen
`recurrent_state_mapping`. A cache-plus-state runner uses one atomic
`bind_cache_and_recurrent_state(...)` handshake; it does not publish two
independently fallible resource bindings. The exact state protocol is documented
in [state ownership](state-pool.md).

## Implemented paths and limits

`Lfm2TorchRunner` is the synchronous reference implementation.
`Lfm2CudaRunner` uses one History group plus a mapped convolution-state slab. It
exposes only the atomic cache-plus-state binder and asynchronous `dispatch` /
`poll`; there are no separate binding or synchronous execution paths. Its H100
continuation harness is driven by the native Engine rather than fabricated
runner-facing plans.
`GptOssCudaRunner` uses ordered History and Window groups over one parent pool,
with FA4 attention and fused MoE execution. Its optional tensor-parallel context
shards attention and KV heads plus the MXFP4 expert intermediate dimension, then
reduces row-parallel attention and expert outputs through NCCL. The native
Engine remains topology-blind: each rank executes the same frozen plan and owns
an identically shaped logical cache pool, while physical KV bytes shrink with
the local KV-head count. The Modal harness maps TP 1, 2, 4, and 8 to matching
multi-H200 containers and verifies exact output-token agreement across ranks.

Current limits are one in-flight batch per Engine, private rather than shared
request cache, no expert parallel or data-parallel attention, and no multi-node
serving. Tensor parallelism is currently specific to the GPT-OSS CUDA backend
rather than a native scheduling concept.
Tokenization, chat formatting, detokenization, and text streaming remain outside
Resin. A completed token buffer is transferred to the caller's `RequestLease`;
the Engine does not retain completed outputs indefinitely.

Reusable runner building blocks live under `resin.backend`. The device-neutral
package provides plan, binding, and sampling helpers. `resin.backend.cuda`
provides injected-framework CUDA graphs, asynchronous completion, tensor
staging, paged KV, and mapped recurrent-state storage. Concrete architectures
and their runners live in the separate `resin_models_cuda` and
`resin_models_torch` distributions under `models/`. Importing those package
namespaces does not load PyTorch,
FlashInfer, or FlashAttention; importing a concrete implementation loads the
framework it uses.

## Code map

- `crates/resin-py/src/lib.rs` — Python API, lifecycle, and transactional orchestration
- `crates/resin-py/src/planning.rs` — token-range and grouped-plan construction
- `crates/resin-py/src/runner.rs` — runner discovery and resource binding
- `crates/resin-py/src/cache.rs` — layout and plan descriptors
- `crates/resin-py/src/state_pool.rs` — state ownership, access, and physical mappings
- `resin/backend/` — reusable execution and resource gadgets
- `models/` — separately installable model definitions, loaders, and runners
