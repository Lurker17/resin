# Flat cache-pool contract

Resin has one managed cache path: a runner declares one ordered `CacheLayout`,
and the Engine creates one matching flat pool. The pool is native and private to
the Engine. Python sees its identity and frozen placement metadata, not allocator
methods.

## Layout declaration

```python
from resin import CacheGroupLayout, CacheLayout

layout = CacheLayout(
    parent_capacity=4096,
    groups=[
        CacheGroupLayout(128, retention="history"),
        CacheGroupLayout(128, retention="window", window_tokens=128),
    ],
)
```

`CacheLayout` is frozen. `groups` is an immutable tuple, and group order is the
complete group identity. There are no group names or process-wide registrations.
Every value is structural:

- `parent_capacity` is the number of equal-byte physical parents;
- `block_tokens` is one group's logical block width;
- `blocks_per_parent` is that group's packing factor;
- retention is exactly `"history"` or `"window"`.

Window groups require `window_tokens`. Recurrent model state is deliberately
outside this layout and uses `StatePool`.

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

History reserves enough blocks at submission for the maximum possible input
length, `len(prompt) + max_new_tokens - 1`. Its plan row starts at logical block
zero and grows as the request advances.

Window reserves a private bounded width:

```text
resident blocks = 1 + ceil((window_tokens - 1) / block_tokens)
```

Logical blocks cycle over those physical slots. A plan never exposes this cyclic
order: it emits the retained suffix chronologically and sets
`first_logical_blocks[row]` to its first logical block. Prefill is split at a
Window block boundary, preventing one accepted plan from overwriting bytes that
the same plan still needs for attention. The current Window path is private
request storage; it does not share immutable pages and does not perform snapshot
restore copies.

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
    cache_layout = layout

    def bind_cache_pool(self, pool_id, bound_layout):
        assert bound_layout is self.cache_layout
        # Allocate or expose physical group slabs here.
        return pool_id
```

The Engine completes native allocation before calling the binder. The runner
must accept its same frozen layout object, bind permanently to the positive pool
ID, and return that exact integer. Repeating the same pair may be idempotent;
rebinding the same physical storage to another pool is an error.

If a runner also maps `StatePool` slots, it implements the combined atomic
binding described in [state ownership](state-pool.md). Resin deliberately avoids
two independently fallible publication callbacks for the two resources.

## Ownership and failure

Submission reserves all groups atomically. If any group cannot be satisfied, no
partial request allocation becomes visible. Request tables retain their blocks
until completion or cancellation.

Before dispatch, Resin prepares and acquires an exact writer lease over the
blocks addressable by that plan. The lease spans runner acceptance, asynchronous
polling, and runner cleanup. Successful completion releases the lease before a
terminal request table is retired. An uncertain post-accept failure quarantines
the batch and lease instead of making possibly addressed memory reusable.

Engine diagnostics expose `cache_pool_id`, `cache_parent_capacity`,
`cache_free_parent_count`, and `cache_occupied_block_count`. The allocator itself
is not public Python API.

## Content reuse status

The current path has no content index, cache admission, or physical block
sharing. Every request receives private cache blocks and therefore receives no
prefix reuse.

## Model mappings

`Lfm2CudaRunner` declares one History group. `GptOssCudaRunner` declares one
History group followed by one Window group, both over the same parent pool. The
model runner decides which layers interpret each group; the native allocator only
needs their ordered identities and equal parent-byte geometry.
