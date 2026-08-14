# StatePool ownership and mapping

`StatePool` is Resin's generic handle pool for state that outlives one forward
call. It owns identities and lifetimes, not tensor bytes. A runner may interpret
each slot as a row in a model-specific physical slab.

This API is separate from cache retention. `CacheGroupLayout` describes only
History or Window token cache; recurrent model state uses `StatePool`.

## Handles and references

```python
from resin import StatePool

pool = StatePool(capacity=1024)
handle = pool.allocate()[0]

assert pool.contains(handle)
assert pool.counts(handle) == (1, 0)  # references, completion pins

pool.retain([handle])
pool.release([handle])
pool.release([handle])
```

`allocate(count=1)` atomically returns one or more frozen `StateHandle`s. A
handle exposes `slot` and `generation`. Reusing a slot increments its generation,
so a stale handle cannot address a newer allocation.

Caller operations are bulk-atomic:

- `retain(handles)` adds external references;
- `release(handles)` removes external references;
- `pin(handles)` delays reuse across external completion work;
- `unpin(handles)` removes those pins.

References express ownership. Pins do not own a slot; they only delay reuse
after references reach zero. Caller operations cannot consume references or pins
held internally by an Engine.

`counts(handle)` returns total references and completion pins.
`access_counts(handle)` returns active Engine read-lease count and whether a write
lease is active. Pool diagnostics are `pool_id`, `capacity`, `free_count`,
`active_count`, `retiring_count`, and `physical_mapping`. A retiring slot has no
references but remains unavailable because a pin or access lease still protects
it.

## Engine ownership

```python
from resin import Engine, StatePool

pool = StatePool(256)
handle = pool.allocate()[0]
engine = Engine(runner, state_pool=pool)

request = engine.submit(
    [1, 2, 3],
    max_new_tokens=8,
    state_handles=[handle],
    state_access_modes=["write"],
)

pool.release([handle])  # Optional: transfer sole ownership to the Engine.
```

Submission validates that all handles belong to the Engine's pool and takes an
Engine-owned reference. The caller may keep its original reference for sharing
or release it as above. Terminal completion and ready cancellation release the
Engine reference. Accepted asynchronous work also holds completion protection
until polling and physical cleanup finish.

If `state_access_modes` is omitted, every handle defaults to `"write"`. Reads are
shared; writes are exclusive. Duplicates in one row are collapsed for conflict
checking with write dominance, but the frozen `BatchPlan` preserves every slot,
generation, and requested mode in row CSR.

The scheduler retains the oldest compatible prefix of a proposed cohort. At
dispatch, the pool atomically checks access already held by other Engines. Leases
remain active through runner acceptance, polling, and cleanup. This makes a
shared pool safe without requiring the model runner to implement its own host
lock table.

## Frozen mapping descriptor

A runner that gives each StatePool slot a persistent physical meaning declares:

```python
from resin import StateMappingConfig


class Runner:
    supports_recurrent_state_mapping = True
    recurrent_state_mapping = StateMappingConfig(
        "model.state.v1",
        ["write"],
    )
```

`protocol` is a nonblank opaque string owned by the runner. `access_modes` is the
exact ordered handle pattern required for every submitted sequence. For example,
`["read", "write"]` requires exactly two state handles in that order on every
request using the Engine.

The Engine validates the descriptor and supplied pool before invoking a binder.
For a state-only runner, the handshake is:

```python
bind_recurrent_state_mapping(pool_id, capacity, protocol)
    -> (pool_id, capacity, protocol, positive_arena_id)
```

The runner allocates or exposes its physical slab, permanently associates it
with the native pool, and acknowledges the exact descriptor. The pool records
`(protocol, arena_id)` in `physical_mapping`. Exact rebinding is idempotent; a
different protocol or arena is rejected. This permanent claim prevents a later
Engine from changing the physical interpretation of surviving slot generations.

A backend calls `resin.mint_state_mapping_arena_id()` when it creates a new
physical slab. Despite the long name, the function only returns a process-unique
positive integer; it does not allocate state or device memory itself.

## Atomic cache-plus-state binding

A runner that declares both `cache_layout` and recurrent-state mapping uses one
combined callback:

```python
bind_cache_and_recurrent_state(
    cache_pool_id,
    same_cache_layout_object,
    state_pool_id,
    state_capacity,
    protocol,
    required_arena_id,
) -> (
    cache_pool_id,
    state_pool_id,
    state_capacity,
    protocol,
    positive_arena_id,
)
```

`required_arena_id` is `None` for an unmapped pool or the pool's permanent
existing arena ID. The runner completes all fallible physical allocation before
publishing either identity. Resin validates the single acknowledgement and only
then records the native mapping. There is no two-callback partial-binding state.

`Lfm2CudaRunner` uses this contract for one managed History cache group and its
`lfm2.conv_state.v1` convolution-state slab. Its arena ID names one exact
process-live slab plus generation/owner sidecars. A later LFM runner presented
with that ID reuses the same objects after checking pool identity, geometry,
device, and dtype; an ID not registered in the process is rejected rather than
silently attached to newly allocated bytes. The runner does not expose the
state-only binder: cache and recurrent state are always acknowledged together.

Generic `StatePool` references can be shared, but the meaning of that sharing is
protocol-specific. LFM convolution state is request-continuation state: one live
slot generation is tombstoned to its first `(engine_id, request_id)` and rejects
substitution by another request. Distinct slots in the same mapped pool may be
used by different Engines and still address the one shared physical arena.

## Failure and reuse

Invalid handles, stale generations, foreign pools, count underflow, and access
conflicts fail before mutation. If a runner rejects work before acceptance,
Resin releases temporary pins and access and restores the request. After
acceptance, uncertain failure quarantines the batch; protected slots are not
recycled.

A slot returns to `free_count` only when its references, completion pins, and
access leases are all zero. Generation checks then ensure that any old Python
handle remains invalid after physical reuse.
