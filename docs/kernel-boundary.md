# Runner boundary

The Engine owns request scheduling, cache allocation, cache pressure, prefix
reuse, and asynchronous resource lifetime. A runner owns model execution and
kernel scheduling.

## `BatchPlan`

The independent sequence metadata is deliberately small:

- `sequence_ids`: stable request IDs, one per sequence.
- `context_lengths`: tokens processed before this dispatch, one per sequence.
- `token_ids`: scheduled inputs flattened in sequence order.
- `token_indptr`: offsets that slice `token_ids` for each sequence.
- `phase`: prefill or decode. A dispatch never mixes phases.

Positions are derived rather than transported independently. `plan.positions`
memoizes the flattened ranges beginning at each `context_lengths[i]`.

The numeric fields on `BatchPlan` are read-only NumPy arrays backed directly by
the frozen native plan: `sequence_ids` is `uint64`; token, position, pointer,
and sequence-index arrays are `uint32`. Repeated property access creates a new
view, not a new data buffer. A runner should pass these arrays directly to its
pinned host staging path rather than calling `list()` or `tuple()` first.

`batch_id` correlates an asynchronous dispatch with its later poll result.
`batch_key` is an optional caller-defined compatibility key; the scheduler does
not mix different keys in one batch.

`sampling_mask` has one Boolean per sequence and marks which sequences should
generate a sampled token. It is used for chunked prefill, when some sequences
only advance the cache. `sampling_params` contains one effective sampling policy
per sequence. Random draw offsets are derived from the absolute sampled
position rather than transported separately.

## Cache metadata

`cache_groups` has the same order as the runner's bound `CacheLayout`. Each
`CacheGroupPlan` contains a CSR physical page table and exact append locations.
The Engine has already allocated, retained, and leased those pages. The runner
must not allocate pages or keep a request-ID cache map.

Its pointer, block, and append fields are likewise read-only, zero-copy
`uint32` arrays. Constructors accept ordinary integer sequences or arrays with
the exact native dtype; the Engine-generated execution path does not perform a
Python element-by-element conversion.

The cache pool ID appears only in `ResourceBinding`, which permanently binds the
runner's physical arena to one Engine. Repeating it in every `BatchPlan` would
not add information.

History, sliding-window history, and recurrent checkpoints use the same page
allocator. Their different retention and packing rules are properties of the
bound `CacheLayout`, not parallel plan formats.

## Execution contracts

A synchronous runner implements:

```python
def forward(plan: BatchPlan) -> Sequence[int] | Sequence[Sequence[int]]: ...
```

An asynchronous runner implements:

```python
def dispatch(plan: BatchPlan) -> object: ...
def poll(completion: object) -> Sequence[int] | Sequence[Sequence[int]] | None: ...
```

`None` means the completion is still pending. The Engine keeps the exact cache
lease until polling returns a terminal result and runner cleanup completes.

Prefill runners return one token for each sampled sequence. A decode runner may
return multiple tokens for each sequence; this is how speculative decoding
commits an accepted block. The runner declares its maximum additional cache writes through
`RunnerConfig.cache_write_lookahead_tokens`, so the Engine reserves those pages
before dispatch. DSpark and other speculative algorithms remain model-runner
details rather than fields in the generic plan.

An exception before work is accepted is retryable. An exception marked
`_resin_post_accept = True`, a malformed result after execution, or a polling
failure quarantines the in-flight resources because GPU use may still exist.
