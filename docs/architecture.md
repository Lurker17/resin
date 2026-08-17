# Architecture

Resin separates control-plane policy from model-specific kernels.

The Rust Engine owns:

- request admission, priority, batching, and prefill chunking;
- request progress and completion;
- one unified paged cache allocator;
- full-history, sliding-window, and recurrent-state retention;
- exact-token LRU prefix reuse;
- cache reader/writer leases across asynchronous execution;
- retry or quarantine decisions after runner failures.

The Python runner owns:

- model weights and device allocations;
- conversion of `BatchPlan` page indices into kernel metadata;
- attention, recurrent, collective, sampling, and speculative-decode kernels;
- CUDA graphs and completion polling.

The boundary is intentionally asymmetric: scheduling complexity belongs in the
Engine, while a forward pass receives immutable token and page metadata and can
execute straightforwardly.

## Lifecycle

1. The runner publishes a frozen `RunnerConfig`.
2. The Engine creates the declared `CacheLayout` and calls
   `bind_resources(ResourceBinding)` once.
3. `submit()` records tokens, generation limits, priority, batch key, and
   sampling policy. With prefix caching enabled, it attaches the longest aligned
   cached prefix.
4. The scheduler selects a phase- and key-compatible cohort, respecting token
   and cache capacity.
5. The cache coordinator grows each request, creates ordered group page tables,
   and acquires one batch lease.
6. The Engine emits a `BatchPlan` and calls `forward()` or `dispatch()`.
7. Once execution is terminal, generated tokens are committed, aligned prefixes
   may be published, retired pages return to the common pool, and requests are
   requeued or completed.

Only one batch may be in flight per Engine today. Multiple requests within that
batch execute continuously and may decode concurrently.

## Cache geometry

Every group uses equal-sized physical slabs. `blocks_per_slab` lets smaller
states share one slab, while the allocator remains free to assign each slab
to whichever group needs it. There is no fixed VRAM percentage for recurrent
state versus attention history.

`CacheGroupFamily.HISTORY` covers full and sliding token history.
`CacheGroupFamily.STATE` covers recurrent checkpoints and retains a two-page
ring per live sequence. See [unified paged cache](cache-pool.md).

CUDA runners normally determine slab capacity after loading weights and
capturing graphs: they reserve the configured model/static memory fraction,
measure remaining VRAM, divide by bytes per slab, and construct the final
layout. This mirrors TokenSpeed's geometry-first sizing rather than guessing
separate cache ratios.

## Prefix reuse

The prefix cache stores immutable page snapshots in an exact-token block trie.
Its checkpoint alignment is the least common multiple of every group block size,
so a hit restores attention, sliding-window, and recurrent state at the same
token boundary. Cached pages consume the same fungible pool and are evicted LRU
when live allocation needs capacity.

## Speculative decoding

The generic Engine schedules one decode anchor per sequence. A runner such as
DeepSeek-V4 may draft and verify several tokens, then return the accepted token
row. `cache_write_lookahead_tokens` reserves enough destinations for the maximum
verified block. DSpark state updates and target verification remain inside the
model runner and its fused kernels.

## Failure boundary

Before acceptance, a failure restores the exact selected cohort and permits a
retry with a new batch ID. After acceptance, Resin does not guess whether device
work is still touching memory: the Engine is poisoned and the resources are
quarantined. This keeps allocator reuse conservative.

## Current limits

- one in-flight batch per Engine;
- no prefill/decode disaggregation;
- no expert parallel or data-parallel attention;
- no multi-node serving;
- tensor parallelism is implemented by model backends rather than the native
  scheduler.
