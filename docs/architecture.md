# Architecture

Resin separates request and resource ownership from model-specific execution.
The dependency-free `resin-core` crate contains the complete
`EngineCore<P>` lifecycle. `P` is opaque frontend metadata: native code schedules
tokens without knowing about Python sampling objects or model media. Payloads
are cloned once per dispatched row; native applications should use `Arc` for
large shared data.

`resin-py` converts Python inputs, binds the runner once, and adapts native plans,
errors, handles, and events. Torch remains an installed dependency and the Rust
HTTP server remains bundled; neither changes the ownership boundary. Native
applications can use `resin-core` without Python, Torch, or HTTP.

The Rust Engine owns:

- request admission, priority, batching, and prefill chunking;
- request progress and completion;
- one unified paged cache allocator;
- full-history, sliding-window, and recurrent-state retention;
- exact-token LRU prefix reuse;
- cache reader/writer leases across runner execution and commit;
- retry or quarantine decisions after runner failures.

The Python runner owns:

- model weights and device allocations;
- conversion of `BatchPlan` page indices into kernel metadata;
- attention, recurrent, collective, sampling, and speculative-decode kernels;
- CUDA graphs and execution synchronization.

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
6. The Engine emits a `BatchPlan` and submits execution. It retains the selected
   requests and their writer lease until result collection finishes.
7. The Engine validates the complete raw result against the leased write span
   before applying stop tokens or generation limits. It prepares events, releases
   terminal model state, publishes aligned prefixes, retires cache leases, and
   commits progress before requeuing survivors.

`step()` remains the synchronous driver over the same submission and completion
path. Python callers may use `enqueue_step()` and `complete_step()` separately;
an optional runner `enqueue(plan)` returns an owned object with `result()`.
Returning that object accepts execution, so all collection failures quarantine
resources. Ordinary `forward()` runners work through this path too. Cancellation
requires collection first, preserving all token events; `close()` drains pending
work before cleanup. There is no public API that commits arbitrary page IDs.
Rust callers use `begin_step()` and `finish_step()`; Rust runners implement
`engine::Runner<P>` and receive an owned `ExecutionPlan<P>`. The native
[conformance tests](../crates/resin-core/tests/engine.rs) demonstrate this API
without a Python interpreter. Native applications bind their arena to the core's
cache pool and use the same logical runner throughout its lifetime. They call
`close(&mut runner)` for orderly shutdown; the core does not own the device
runtime or call back into a runner when it is dropped.

One batch may be in flight per Engine. Multiple requests within that batch
execute together and may decode concurrently. GPT-OSS implements deferred CUDA
completion with `CudaStep`, retaining the native plan, pinned staging, device
tokens and a completion event covering the result copy. Its synchronous
`forward()` collects that same step. The runner prevents another launch from
overwriting borrowed graph outputs before collection.

The owned step follows DeepInfer's
[`WorkerStepState`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/zos_status.py#L10).
This prototype does not launch the next decode before committing the previous
one. That requires device-side sampled-token/accepted-length feedback, sequence
remapping and coordinated reservation of the next write span, as in DeepInfer's
[`Worker.execute_model_step`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L89).

## Cache geometry

Every group uses equal-sized physical slabs. `blocks_per_slab` lets smaller
states share one slab, while the allocator remains free to assign each slab
to whichever group needs it. There is no fixed VRAM percentage for recurrent
state versus attention history.

`CacheGroupFamily.HISTORY` covers full and sliding token history.
`CacheGroupFamily.STATE` covers recurrent checkpoints and retains a two-page
ring per live sequence. A Window group declares both physical `window_tokens`
and logical `read_window_tokens`. A step may write at most
`window_tokens - read_window_tokens + 1` tokens so its earliest query can still
read the preceding window. The Engine caps prefill and checks speculative
lookahead against this bound. See [unified paged cache](cache-pool.md).

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
when live allocation needs capacity. Successive History snapshots share immutable
page-table segments instead of copying every ancestor's references. Window and
State snapshots remain bounded. Trie nodes carry direct LRU links, so eviction
does not scan all cached entries.

## Speculative decoding

The generic Engine schedules one decode anchor per sequence. A runner such as
DeepSeek-V4 may draft and verify several tokens, then return the accepted token
row. `cache_write_lookahead_tokens` reserves enough destinations for the maximum
verified block. DSpark state updates and target verification remain inside the
model runner and its fused kernels.

## Failure boundary

A rejected execution must have no persistent effects, or must have rolled them
back. It restores the exact selected cohort and permits a retry with a new batch
ID. After execution may have changed state, failure poisons the Engine and
quarantines its resources. Successful device synchronization cannot undo an
in-place recurrent update. Native callbacks that panic also take this failure
path. See the [runner contract](kernel-boundary.md) for Python error handling.

## Current limits

- one batch in flight per Engine; next-step device feedback is not implemented;
- no prefill/decode disaggregation;
- no expert parallel or data-parallel attention;
- no multi-node serving;
- tensor parallelism is implemented by model backends rather than the native
  scheduler.
