# Unified paged cache

Each Engine owns one pool of equal-sized physical parents. A runner declares an
ordered `CacheLayout`; every `CacheGroupLayout` explains how one model state is
packed into those parents:

- `family=HISTORY` stores token-addressed attention or rolling history.
- `family=STATE` stores recurrent checkpoints.
- `retention=HISTORY` keeps the full logical history. State groups retain only
  their current and preceding checkpoints.
- `retention=WINDOW` keeps a bounded chronological suffix.
- `blocks_per_parent` says how many smaller blocks fit in one parent.

Parents are fungible. They are not divided into fixed History, Window, or State
regions. Allocation binds a free parent to whichever group currently needs it,
and retirement returns it to the common pool. Admission estimates the parent
demand for every request and group, including packing, before admitting work.

The runner allocates the matching device arena once and receives its pool ID and
frozen layout through `bind_resources`. A `BatchPlan` does not repeat that
identity. It contains only the ordered `CacheGroupPlan`s whose structured
physical addresses refer to the already-bound arena.

## Batch plans

`BatchPlan.cache_groups` has one plan per declared block type—such as full
attention, sliding-window attention, or recurrent state—not one per sequence.
The cache is a pool of equally sized physical slabs. Each slab in use is
assigned to one cache group and fits that group's `blocks_per_parent` blocks.
A physical block is identified by `(slab_id, slotnum)`; the API exposes those
components as a parent ID and child slot. A cached token location is
`(block, block_offset)`, equivalently `(slab_id, slotnum, block_offset)`. Each
group plan covers every batch sequence and has one CSR block table:

- `block_indptr` slices the parallel `block_parent_ids` and
  `block_child_slots` arrays by sequence.
- `first_logical_blocks[i]` is the absolute chunk index represented by sequence
  `i`'s first table entry. For a group with 64-token blocks, logical block 37
  covers token positions 2368 through 2431. Full-history rows begin at zero;
  sliding-window rows advance when earlier chunks are evicted.
- `append_parent_ids`, `append_child_slots`, and `append_block_offsets` give one
  exact cached-token write location per scheduled token.

The Engine allocates pages, performs copy-on-write decisions, and holds the
reader/writer lease until asynchronous execution is terminal. Model code only
turns each `(physical_parent, child_slot)` pair into its model-specific tensor
row. Parent zero remains reserved for inactive/null rows; live parent IDs are
one-based. Keeping the address structured prevents runners from silently
assuming a different child packing than the allocator.

## Prefix caching

`Engine(..., prefix_cache=True)` enables an exact-token block trie with LRU
entries. Prefixes are published only at an alignment shared by every cache
group. A hit attaches immutable references to the cached pages and schedules
only the unmatched suffix.

Live requests have priority over cached entries. When allocation needs more
parents, the least-recently-used prefix is evicted. Advancing a shared Window or
State ring uses copy-on-write so one request cannot overwrite another request's
prefix.

State groups use two checkpoint pages per live sequence. LFM2 checkpoints every
128 tokens and caps each prefill chunk to the same interval. At a restored
boundary, the cached input page remains read-only and the next checkpoint page
is private and writable. Within a checkpoint interval, decode updates the
private page in place. This keeps recurrent state prefix-compatible without a
second slot allocator.

## Physical sizing

DeepSeek chooses parent capacity from free VRAM after model load and before
cache allocation and CUDA graph capture, subject to its memory-utilization
target. Other runners may supply a fixed capacity. The parent byte size is the
largest group allocation. Smaller groups set `blocks_per_parent` from their
exact byte ratio, so the allocator can compare all demand in parent units.

For LFM2, one BF16 attention KV page is 98,304 elements and one recurrent
checkpoint is 16,384 elements, so six state checkpoints fit in one parent. The
ratio is derived from model geometry rather than reserved as a fixed percentage
of VRAM. DeepSeek-V4 similarly packs its smaller compressed-history and rolling
state groups into the same parent size. GPT-OSS uses full-history and sliding
attention groups with equal-sized pages.

Page zero is reserved as the device null page. Positive page IDs are stable for
the lifetime of the Engine-issued lease.
