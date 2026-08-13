# Runner and kernel boundary

Resin crosses from Rust into Python once per model batch. Rust freezes logical
work and physical identities; the runner converts that metadata into tensors and
launches the model-specific kernels.

```text
Engine ownership -> BatchPlan -> runner tensorization -> device work
       ^                                                   |
       +-------------- completion and cleanup -------------+
```

There is no runner base class and no per-layer callback.

## Execution protocols

A synchronous runner implements:

```python
class SyncRunner:
    def forward(self, plan):
        return [next_token_id for _ in plan.sequence_ids]

    def release(self, request_ids):  # optional
        ...
```

An asynchronous runner implements both methods:

```python
class AsyncRunner:
    def dispatch(self, plan):
        return opaque_completion

    def poll(self, opaque_completion):
        return None  # still pending
        # or one token ID per row
```

Supplying only one of `dispatch` and `poll` is an error. When both are callable,
the asynchronous protocol is selected. Resin snapshots the callables at Engine
construction, so later monkey-patching does not change an active Engine's
contract.

The output is dense: exactly one integer token ID per sequence row. Only rows in
`sample_sequence_indices` commit that ID or emit an event. An unsampled
continuation row still has an output position, which keeps the runner protocol
uniform.

`release(request_ids)` is optional physical cleanup for terminal requests. It
runs while native state and cache capabilities remain protected and before those
resources are retired.

## Base BatchPlan

`BatchPlan` is frozen and validates these groups of fields:

- `token_ids`, `positions`: equally sized flattened input arrays;
- `sequence_ids`, `sequence_indptr`: nonempty sequence CSR;
- `context_lengths`: one pre-append logical length per sequence;
- `phase`: one `BatchPhase.PREFILL` or `BatchPhase.DECODE` value applying to
  every row;
- `sample_sequence_indices`: sorted rows whose returned IDs commit;
- `batch_id`, optional `batch_key`, and optional positive `engine_id`.

For sequence row `i`, flattened tokens occupy
`sequence_indptr[i]:sequence_indptr[i + 1]`. Their positions are absolute logical
positions. Runners should consume the explicit batch phase and sample set rather
than reconstructing policy from positions or context lengths.

The public constructor remains useful for reference runners and focused tests.
It defaults to `BatchPhase.PREFILL`; decode plans must pass
`phase=BatchPhase.DECODE`. Engine-built managed plans add state and cache
identities and should be treated as authoritative.

## Cache metadata

When a runner declares `cache_layout`, a plan contains the matching positive
`cache_pool_id` and an ordered tuple of `CacheGroupPlan`s. For each group:

```python
group.group_index
group.block_indptr
group.block_ids
group.first_logical_blocks
group.append_block_ids
group.append_block_offsets
```

The block arrays are sequence CSR. Each row is chronological; for a Window group
`first_logical_blocks[i]` may be nonzero even though its private physical storage
has cycled. The append arrays align one-for-one with flattened `plan.token_ids`
and give exact positive page IDs and within-block offsets.

A runner normally tensorizes this sparse representation once, then derives the
backend's page-table format. It must validate the plan's pool ID against its
permanent binding and keep page zero as a null/padding row. Placement, lifetime,
and chronological projection belong to Resin; layers, planes, dtype, and stride
belong to the runner.

The only cache handshake is:

```python
bind_cache_pool(pool_id, runner.cache_layout) -> pool_id
```

The exact frozen layout object is passed back to its owner. The runner may
allocate its physical slabs before acknowledging, then permanently associates
them with that pool ID. See [flat cache pool](cache-pool.md).

## State metadata

Optional state arrays are `state_pool_id`, `state_slots`, `state_generations`,
`state_access_modes`, and `state_indptr`. They form row CSR just like sequence
metadata. The generation must participate in physical ownership checks; a slot
index alone is not a stable identity.

A model-specific recurrent-state runner declares a frozen
`StateMappingConfig`. A state-only runner uses `bind_recurrent_state_mapping`.
A runner with both cache and recurrent state uses one
`bind_cache_and_recurrent_state` callback so it cannot expose half of a physical
resource pair. Exact signatures and acknowledgement rules are in
[StatePool ownership](state-pool.md).

## Acceptance and fail-stop behavior

Resin acquires the exact state and cache writer capabilities before calling the
runner. A normal exception raised by synchronous `forward` or asynchronous
`dispatch` is treated as pre-accept: capabilities are released and queue order
is restored. Either exception may set `_resin_post_accept = True` when device
work may already address the plan; that exact marker triggers quarantine. Once
synchronous `forward` returns, malformed result type or cardinality is also
post-accept: Resin cannot safely retry stateful work whose result violated the
protocol.

Once an asynchronous dispatch returns a completion object, any polling, cleanup,
or commit error is post-accept. The Engine becomes poisoned and does not recycle
the accepted resources. This is stricter than attempting to infer device state
from a Python traceback.

Runners should therefore follow three rules:

1. Validate all plan and binding invariants before launching device work.
2. Do not return from `dispatch` until ownership of a valid completion is clear.
3. Make `poll` report `None` only while the same completion remains live.

## Current model boundaries

`Lfm2CudaRunner` declares one History cache group and a mapped convolution-state
slab. Because it always owns both resources, its only binding method is
`bind_cache_and_recurrent_state`; because its CUDA work is asynchronous, its
only execution methods are `dispatch` and `poll`. It tensorizes sparse cache
rows for its attention kernels and validates native append destinations before
launching device append work.

`GptOssCudaRunner` declares ordered History and Window groups with 128-token
blocks. Its runner maps full-attention layers to History and sliding-attention
layers to Window, while both groups draw from one flat parent pool. FA4 attention,
FlashInfer fused MoE, and optional runner-local decode graph code remain
execution-plane details.

The local runner implementations and GPT-OSS deployment harness have been
migrated to this plan shape. The harness has not been launched, so this document
intentionally makes no new GPU correctness or performance claim for the path.

## What is not in the boundary

Resin does not standardize a tensor library, attention backend, compiler cache,
CUDA graph policy, model checkpoint format, tokenizer, or sampling algorithm.
Those can evolve independently as long as the runner honors the frozen plan,
binding acknowledgements, dense output contract, and completion lifetime.
