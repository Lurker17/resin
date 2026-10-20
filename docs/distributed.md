# Distributed execution prototype

`resin.Parallelism` defines rank placement in the Rust core and exposes that
immutable configuration to Python. `resin.backend.topology.ParallelGroups` owns
the Python process-group handles. `resin.backend.distributed` coordinates
synchronous Engine steps. Each rank owns a model runtime; each attention-DP
replica has one native request Engine, on its lowest global rank. The other
ranks in that replica follow its driver.

The boundaries are:

| Component | Responsibility |
| --- | --- |
| `Parallelism` | Rust-defined dimensions, validation, rank placement, and partitions |
| `ParallelGroups` | Explicit device groups and Gloo control groups |
| `DistributedEngine` | Phase agreement, normal local Engine steps, empty participation, failure propagation |
| Native `Engine` | Local admission, requests, pages, leases, output validation, commit |
| `DistributedWorker` | Replica commands and model callback invocation |
| `DistributedRunner` | Weight placement, CP partitioning, expert dispatch/combine, device synchronization |

This preserves the existing checkpoint-to-runtime preparation boundary. Create
groups before preparing model weights and inject the appropriate group into each
prepared runtime. An attention layer uses `groups.attn_tp`; a sharded expert uses
`groups.expert_tp`; expert dispatch uses `groups.ep`. The coordinator never
infers model weight layouts or uses the default world group for those operations.

CUDA runners declare `parallelism_capabilities`; the shared base defaults to
single-rank execution. GPT-OSS and DeepSeek-V4 declare EP, attention DP,
and attention replication alongside their existing TP paths. Both new
explicit-group paths currently require `attn_tp=cp=expert_tp=1`.
GPT-OSS also declares expert TP for its legacy TP entry point. These
declarations describe supported axes; preparation also checks combinations,
graph policies, and speculative modes. Both explicit-group paths provide
captured prefill FFN regions and full decode graphs. GPT-OSS EP additionally
requires a kernel advertising FP32 local expert partials so the cross-rank
combine does not add an intermediate activation rounding step.
`DistributedWorker` validates a declared capability set collectively before
serving begins. Custom distributed runners may supply the same declaration.

See [DeepSeek-V4 EP and attention DP](deepseek-v4-parallelism.md) and
[GPT-OSS EP and attention DP](gpt-oss-parallelism.md) for the implemented
model paths and real-GPU validation commands.

## CUDA graphs across expert owners

Graph-enabled CUDA runners call `runner.capture_cuda_graphs()` on every rank
before constructing workers or entering serving loops. All ranks must configure
the same graph catalog and capture shapes in the same order. The shared CUDA
runner checks those catalogs collectively, owns capture/replay, and chooses
common physical buckets from the logical sizes in `StepContext`.

Prefill captures the FFN regions while attention executes outside those regions.
Decode captures the model forward and sampling. Each step votes on graph
eligibility and capacity across the serving domain before issuing device work.
If any replica needs an unsupported shape, every replica uses the eager path
for that step. This keeps expert collectives in the same order even when request
queues and batch sizes differ. Graph counters expose captures, replays, and
eager fallback reasons.

After workers stop, call `runner.close_cuda_graphs()` on every rank before
destroying process groups. This waits for device work and releases captured
NCCL references; keeping graphs alive during group destruction can hang shutdown.

`ExpertParallelDispatch` has two transports behind the same local expert
callback. Eager execution packs only rows with selected experts on each owner,
reading split counts on the host. Captured execution all-gathers a fixed token
capacity per source, executes local experts on the gathered rows, and
reduce-scatters their weighted partials back to the source. Routing IDs and a
device valid-row scalar determine active assignments without host reads or
dynamic tensor shapes. Attention replication still dispatches once and
broadcasts the combined rows. GPT-OSS retains FP32 partials through the combine.

This dense graph transport is a correctness prototype. Every expert owner
receives `EP * capacity` rows, including padding and rows whose selected experts
are elsewhere. It adds bandwidth, scratch memory, and padded kernel work;
capture does not establish efficient wide-EP dispatch or communication overlap.
The model and kernel adapters remain responsible for capture-safe local expert
execution and their own weight layouts.

Idle replicas replay the same expert collectives. Empty prefill replays FFN
regions without attention. Empty decode replays padded model rows using reserved
null or scratch cache storage, leaving live request pages untouched. Graph
decode can still return the normal NumPy logprob columns after replay.

## Rank placement

`world_size` is the total number of ranks. The five configuration fields are
`world_size`, `attn_dp`, `attn_tp`, `cp`, and `ep`, each defaulting to one.
`Parallelism` is immutable and accepts keyword arguments only:

```python
from resin import Parallelism

parallelism = Parallelism(world_size=4, attn_dp=4)
```

Attention TP is explicit; attention replication and expert TP are derived:

```python
attn_replication = world_size // (attn_dp * cp * attn_tp)
expert_tp = world_size // ep
```

All sizes must be positive integers. Both `attn_dp * cp * attn_tp` and `ep` must
divide `world_size`. Attention uses an
`[ATTN_DP, CP, ATTN_REPLICATION, ATTN_TP]` mesh, keeping each attention-TP group
contiguous. The derived replication axis repeats the same attention shard and
receives the same logical plan; it does not create additional request Engines.
Experts use `[EP, EXPERT_TP]` over the same ranks. Expert weights are sharded
across that complete mesh, with no separate expert-replica axis. CP describes
attention token partitioning within those ranks; it is not an additional
multiplier in the expert mesh. For example,
`Parallelism(world_size=8, ep=8, attn_tp=8)` gives attention TP eight and
expert TP one on eight ranks.

`parallelism.coordinates(rank)` returns `(dp_index, cp_index, replication_index,
tp_index)`, and `parallelism.rank(...)` maps those four coordinates back to the
global rank. `parallelism.attn_replication` is the derived replication count;
`groups.attn_replication` contains ranks with the same attention shard at fixed
DP, CP, and TP coordinates. The runner owns any device communication across
that group.

For `Parallelism(world_size=8, ep=4, cp=2, attn_dp=2, attn_tp=2)`, both
attention TP and expert TP are two, and attention replication is one. The groups
containing rank zero are:

| Group | Ranks |
| --- | --- |
| Attention TP | 0, 1 |
| CP | 0, 2 |
| Attention DP | 0, 4 |
| Attention replication | 0 |
| EP | 0, 2, 4, 6 |
| Expert TP | 0, 1 |
| Request replica | 0, 1, 2, 3 |
| Serving domain | 0–7 |
| Domain drivers | 0, 4 |

`Parallelism(world_size=4, attn_dp=4)` gives four attention replicas with
attention TP one, CP one, and expert TP four. Each rank drives its own request
Engine while the experts span all four ranks.

`Parallelism(world_size=8, attn_tp=2)` gives attention replication four and
expert TP eight. There is one request replica and one driver for all eight
ranks. Its four attention-TP groups receive the same logical plan. In general,
each request replica contains `cp * attn_replication * attn_tp` ranks, and there
are exactly `attn_dp` request Engines.

The complete expert mesh connects every attention replica, so all drivers
synchronize their serving loops. This also holds with `ep=1`, where expert TP
spans the entire world. A replica with no requests participates in empty
execution while its peers have work. The serving domain is the complete world;
no connected-component inference is needed.

The Rust `Parallelism` value contains numeric configuration and rank layouts;
Python `ParallelGroups` creates and owns the communication handles.
`ParallelGroups.create(parallelism)` is collective over an initialized default
world group, checks agreement on configuration/backend, creates groups in
deterministic order, and reuses identical groups. Its `.topology` field refers
to the supplied `Parallelism`. Device collectives default to NCCL; control
collectives use Gloo. The CPU conformance test uses Gloo for both.
`groups.close()` destroys only the groups it created, after workers stop; the
caller owns the default world.

Each attention-DP replica still has its own native scheduler, request queue, and
logical cache pool. Request limits, token budgets, and cache-parent capacity are
per replica; `attn_dp` does not scale one shared cache pool. The runner supplies
a logical cache layout and capacity backed by its physical TP/CP shards on all
ranks in that replica. The native configuration type does not add a global
request router or move the current Python serving coordinator into Rust.

## One synchronous step

Drivers in the same world call `has_work()`, `step()`, and `close()` in matching
order. `submit()` and `cancel()` remain local operations; there is no request router.

1. Drivers exchange read-only `Engine.next_phase()` proposals. When replicas
   propose different phases, drivers alternate prefill and decode. The proposal shares the native admission
   fallback used by actual dispatch and reserves nothing.
2. Each driver calls `Engine.step(phase=agreed_phase)`. A mismatched local cohort
   remains queued. Cache pressure and retraction still belong to the native Engine.
3. Actual plans rendezvous with empty or failed preparation on other drivers.
   This second exchange determines whether there is real work, computes logical
   replica capacities, and catches plan serialization errors before execution.
4. Each driver broadcasts its logical plan to all ranks in its request replica. All workers
   validate inputs and vote readiness before calling the model. An idle replica
   receives `plan=None` and participates in the model's empty execution path.
5. Workers finish the forward callback and exchange execution status. The native
   Engine validates and commits the local result using its existing synchronous
   transaction, including NumPy [logprob columns](logprobs.md).
6. Drivers exchange commit status before returning events. A failure permanently
   poisons every native Engine in the domain and stops further execution.

Each batch still has one synchronous call and return. There is no pending-step
handle, deferred commit, or previous-step output bookkeeping.

Replica commands also propagate resource binding and release. Page allocation
exists only on the driver; followers bind their device arenas to the same logical
pool identity and receive its page tables. Request IDs and pool IDs are local to
the request replica; cross-replica device metadata must preserve that namespace.
Bind and release callbacks are replica-local commands and must not issue
collectives spanning other drivers. Cross-replica collectives belong in forward.

## Model callback

A CP, EP, or attention-DP runtime implements:

```python
def forward_distributed(self, plan: BatchPlan | None, context: StepContext):
    # Model-owned CP partitioning and expert communication, including plan=None.
    # Return ordinary token rows or ForwardOutput on a nonempty driver.
    ...
```

`context` carries the local groups, agreed phase, step ID, local logical sequence
and token counts, and `(sequence_count, token_count)` for each replica in driver
rank order. `max_num_sequences` and `max_num_tokens` describe logical per-replica
capacity. The model derives physical shard sizes and graph buckets from them.
CP groups are static across phases. The runner uses `context.phase` to shard
attention tokens across CP ranks during prefill and replicate attention
execution across those ranks during decode. The coordinator broadcasts the same
logical plan within a replica; phase-specific CP behavior and expert routing
belong to the model callback.

An optional `validate_distributed(plan, context)` performs host checks before the
readiness vote. The forward callback must preserve collective ordering and finish
its device work before returning. Mid-collective process/device failures rely on
process-group timeouts and process teardown. The coordinator propagates errors;
it cannot roll back in-place model state or globally undo a peer's completed local
commit. Failure marks all native Engines poisoned, preserving resources and
preventing cleanup/reuse through normal Engine operations.

Existing TP-only runtimes that use an injected TP group can continue implementing
`forward(plan)` with `Parallelism(world_size=2, attn_tp=2)`, for example when
running GPT-OSS 120B over two ranks. Here `cp`, `ep`, and `attn_dp` remain one,
and attention TP and expert TP both equal `world_size`. GPT-OSS's
`TensorParallelContext` can be constructed from
`groups.attn_tp.local_rank`, `groups.attn_tp.size`, and `groups.attn_tp.handle`
(use `None` for a single-rank context). The ordinary `forward(plan)` callback
also remains allowed whenever `cp=ep=attn_dp=1`, including attention replication
greater than one; the runtime owns weight preparation and communication across
its attention and expert groups. This prototype does not change current
deployment entry points or rewrite default-world assumptions inside existing models. Each model
must use its supplied groups before it can support a composed TP/CP/EP/DP layout.

## Process integration

After initializing distributed communication and preparing each rank's runtime:

```python
# For a CUDA runner with graphs enabled, all ranks capture before serving.
runner.capture_cuda_graphs()
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker, max_running_requests=256)
    try:
        # Admit this replica's requests here. Every driver stays in the same loop.
        while engine.has_work():
            events = engine.step()
            consume_local_events(events)
    finally:
        engine.close()
else:
    worker.serve()
# Once all ranks have left their serving loops:
groups.close()
```

The prototype uses object collectives for control metadata. It establishes the
ownership and execution contract; it does not claim DeepInfer's control-path
performance. Shared-memory command transport, CUDA/NCCL performance validation,
model-specific CP/EP kernels, HTTP integration, expert load balancing, and a
replica request router remain separate work.

## Source references and validation

The design follows DeepInfer's explicit device/CPU group separation in
[`parallel_state.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/parallel_utils/parallel_state.py),
its read-only phase vote followed by an actual-scheduling vote in
[`scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L299),
and model-level empty execution in
[`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py).
Resin keeps synchronous execution and explicit injected groups; it does not adopt
DeepInfer's deferred previous-step postprocessing or global process-group registry.

[`test_distributed.py`](../tests/test_distributed.py) launches eight real processes
and checks numerical TP/CP reductions and variable-sized EP dispatch/combine,
including empty senders, mixed phases, unequal request queues, cancellation,
array score output, shutdown, idle DP replicas sharing expert TP, TP-only
compatibility, and failures in initialization, preparation, serialization,
preflight, execution, output validation, release, and control failure after model
execution. These are CPU Gloo conformance tests; GPU kernel and multi-node
behavior are not established by them.
