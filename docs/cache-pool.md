# Flat cache-pool contract

Resin has one managed cache path: a runner declares one ordered `CacheLayout`,
and the Engine creates one matching flat pool. The pool is native and private to
the Engine. Python sees its identity and frozen placement metadata, not allocator
methods.

## Layout declaration

```python
from resin import CacheGroupLayout, CacheLayout, CacheRetention

layout = CacheLayout(
    parent_capacity=4096,
    groups=[
        CacheGroupLayout(128),
        CacheGroupLayout(128, retention=CacheRetention.WINDOW, window_tokens=128),
    ],
)
```

`CacheLayout` is frozen. `groups` is an immutable tuple, and group order is the
complete group identity. There are no group names or process-wide registrations.
Every value is structural:

- `parent_capacity` is the number of equal-byte physical parents;
- `block_tokens` is one group's logical block width;
- `blocks_per_parent` is that group's packing factor;
- retention is the `CacheRetention.HISTORY` or `CacheRetention.WINDOW` enum.

Window groups require `window_tokens`. Request-sized recurrent state normally
uses `StatePool`; rolling state that has a token position, retention window, and
page representation may instead be an ordinary Window group. DeepSeek-V4 uses
that form for its compressor and indexer tails.

## Physical addressing

All parents are fungible while free. An occupied parent is bound to exactly one
group and subdivided according to that group's packing. This makes groups with
different tensor interpretations compete honestly for one memory budget without
requiring independent arenas.

Page ID zero is never allocated. The runner provides a zero/null row there for
safe padded kernel tables. For positive parent ID `p`, group packing `K`, and
slot `s` in `[0, K)`, Resin emits:

```text
kernel_page_id = 1 + (p - 1) * K + s
```

The corresponding Python helpers are:

```python
layout.group_kernel_page_capacity(group_index)  # parent_capacity * K
layout.group_kernel_page_count(group_index)  # 1 + parent_capacity * K
```

Each runner allocates its own physical group view with exactly the required row
count. Resin does not describe layers, K/V planes, dtypes, or strides.

## Retention today

History and Window share the allocator but have different request tables.

History starts with an empty request table. Immediately before dispatch, it
grows transactionally to the end of the frozen token range. Its plan row starts
at logical block zero and grows as the request advances.

Window also starts empty and grows toward its private bounded width:

```text
resident blocks = 1 + ceil((window_tokens - 1) / block_tokens)
```

Logical blocks cycle over those physical slots. A plan never exposes this cyclic
order: it emits the retained suffix chronologically and sets
`first_logical_blocks[row]` to its first logical block. Prefill is split at a
Window block boundary, preventing one accepted plan from overwriting bytes that
the same plan still needs for attention. A restored prefix holds immutable
Window pages. When its cyclic slot is needed again, Resin replaces the shared
page before granting the writer lease.

## Sparse runner ABI

Every managed `BatchPlan` has one positive `cache_pool_id` and one ordered
`CacheGroupPlan` per layout group:

```python
group.group_index
group.block_indptr
group.block_ids
group.first_logical_blocks
group.append_block_ids
group.append_block_offsets
```

`block_indptr` and `block_ids` form a CSR table in batch sequence order.
`first_logical_blocks` has one value per row. The append arrays have one entry per
flattened `plan.token_ids` entry and are exact destinations in the group view.
Every append page belongs to its sequence's emitted row, every page ID is
positive, and every offset is within that group's block width.

The runner may convert the sparse rows into the dense or backend-specific
metadata its kernels require. It should treat page order and append destinations
as authoritative rather than deriving a separate physical placement scheme.

## Binding

A managed runner owns one exact layout object:

```python
class Runner:
    runner_config = RunnerConfig(cache_layout=layout)

    def bind_resources(self, binding):
        assert binding.cache_layout is layout
        self.cache.bind(binding.cache_pool_id, binding.cache_layout)
```

The Engine completes native allocation before calling the binder. The runner
must accept its same frozen layout object and bind its storage permanently to
the positive pool ID. Repeating the same pair may be idempotent; rebinding the
same physical storage to another pool is an error.

If a runner also maps `StatePool` slots, it implements the combined atomic
binding described in [state ownership](state-pool.md). Resin deliberately avoids
two independently fallible publication callbacks for the two resources.

## Ownership and failure

Submission creates empty ordered group tables and consumes no physical parent.
Before dispatch, all additional group blocks for one request are placed
atomically. If growth cannot be satisfied, its existing tables are unchanged.
Request tables retain allocated blocks until completion, cancellation, or
memory retraction.

Before dispatch, Resin prepares and acquires an exact writer lease over the
blocks addressable by that plan. The lease spans runner acceptance, asynchronous
polling, and runner cleanup. Successful completion releases the lease before a
terminal request table is retired. An uncertain post-accept failure quarantines
the batch and lease instead of making possibly addressed memory reusable.

Engine diagnostics expose `cache_pool_id`, `cache_parent_capacity`,
`cache_free_parent_count`, and `cache_occupied_block_count`. The allocator itself
is not public Python API.

Under memory pressure, eligible cache-only requests can release all pages while
retaining their allocator identity. The scheduler later replays committed
inputs into newly grown tables. Separately mapped `StatePool` state currently
makes a request ineligible for this reset path; cache-addressed rolling state
does not.

## Exact-token LRU prefix reuse

Prefix caching is opt-in at `Engine` construction. A token trie finds the
longest block-complete prefix shared with an earlier request. Published pages
become immutable, a hit attaches references to those same physical pages, and
the first subsequent write uses copy-on-write where necessary. Entries consume
the ordinary pool budget and an LRU victim is evicted automatically when live
allocation needs capacity. `Engine.prefix_cache_stats` reports entries, hits,
misses, and evictions. `Engine.clear_prefix_cache()` explicitly releases every
cached page and recurrent checkpoint while leaving live requests untouched.

The alignment is the least common multiple of every group's `block_tokens` and,
when applicable, a mapped recurrent checkpoint granularity. Thus a prefix hit
never exposes a partially written History block. A model whose recurrent tail
is represented by cache groups needs no separate checkpoint object: its aligned
state pages participate in the same immutable snapshot.

## Model mappings

`Lfm2CudaRunner` declares one History group. `GptOssCudaRunner` declares one
History group followed by one Window group, both over the same parent pool. The
model runner decides which layers interpret each group; the native allocator
only needs their ordered identities and equal parent-byte geometry.

### DeepSeek-V4

DeepSeek-V4 declares six groups in one pool:

| Group | Logical block | Retention | Blocks per parent |
| --- | ---: | --- | ---: |
| sliding-attention KV | 128 tokens | Window | 1 |
| c4 compressed history + index keys | 128 input tokens | History | 4 |
| c4 compressor tail | 4 tokens | Window | 8 |
| c128 compressed history | 128 input tokens | History | 256 |
| c128 compressor tail | 8 tokens | Window | 8 |
| c4 indexer tail | 4 tokens | Window | 32 |

These are byte-compatible interpretations of a 5.75 MiB parent on each TP
rank for the released 43-layer checkpoint plus its three DSpark stages. There are no
fixed SWA/full/recurrent allocation ratios. A free parent binds to whichever
group currently needs it; small groups pack multiple child blocks into that
parent, and it becomes fungible again when its last child is released.

The runner allocates after loading weights and static collective workspaces. It
queries CUDA total and free memory and applies:

```text
protected bytes  = ceil(total bytes * (1 - gpu_memory_utilization))
cache bytes      = free bytes - protected bytes
physical parents = floor(cache bytes / bytes per parent)
```

One physical parent is reserved as the zero/null target used by masked runtime
rows, and six more provide group-private graph-capture scratch space; the
remainder becomes the Engine's `parent_capacity`. The scratch parents prevent
synthetic captures from aliasing six incompatible tensor interpretations. The
calculation uses the actual current BF16 KV and FP32 recurrent-state byte
geometry. It does not assume FP8 KV. CUDA graphs are captured after this
allocation inside the protected headroom.

For DSpark-5, c4 compressor correctness needs twelve retained token positions.
cuTile kernels require a power-of-two static iteration extent, so the runner
presents a masked sixteen-position view while only the final twelve positions
have allocator-backed pages. The four masked positions do not consume cache.
