# Distributed execution prototype

`resin.backend.topology` describes rank placement and owns explicit process
groups. `resin.backend.distributed` coordinates synchronous Engine steps. Each
rank owns a model runtime; each attention-DP replica has one native request
Engine, on its lowest global rank. The other attention-TP/CP ranks follow that driver.

The boundaries are:

| Component | Responsibility |
| --- | --- |
| `ParallelTopology` | Total TP width and derived attention/expert meshes |
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

## Rank placement

`tp` is the total number of ranks. The four configuration fields are `tp`, `ep`,
`cp`, and `attn_dp`; the two tensor-parallel widths are derived:

```python
world_size = tp
expert_tp = tp // ep
attn_tp = tp // (attn_dp * cp)
```

All sizes must be positive integers, and `ep` and `attn_dp * cp` must divide `tp`.
Attention uses an `[ATTN_DP, CP, ATTN_TP]` mesh; experts use `[EP, EXPERT_TP]`
over the same ranks. There is no separate expert-replica axis. For example,
`tp=8, ep=8` gives attention TP eight and expert TP one on eight ranks.

For `ParallelTopology(tp=8, ep=4, cp=2, attn_dp=2)`, both derived TP widths are
two. The groups containing rank zero are:

| Group | Ranks |
| --- | --- |
| Attention TP | 0, 1 |
| CP | 0, 2 |
| Attention DP | 0, 4 |
| EP | 0, 2, 4, 6 |
| Expert TP | 0, 1 |
| Request replica | 0, 1, 2, 3 |
| TP-world serving domain | 0–7 |
| Domain drivers | 0, 4 |

The complete expert mesh connects every attention replica in this TP world, so
all its drivers synchronize their serving loops. This also holds with `ep=1`,
where expert TP spans the entire world. A replica with no requests participates
in empty execution while its peers have work. The serving domain is therefore
the TP world directly; no connected-component inference is needed.

`ParallelGroups.create()` is collective over an initialized default world group,
checks agreement on topology/backend, creates groups in deterministic order, and
reuses identical groups. Device collectives default to NCCL; control collectives
use Gloo. The CPU conformance test uses Gloo for both. `groups.close()` destroys
only the groups it created, after workers stop; the caller owns the default world.

## One synchronous step

Drivers in the TP world call `has_work()`, `step()`, and `close()` in matching
order. `submit()` and `cancel()` remain local operations; there is no request router.

1. Drivers exchange read-only `Engine.next_phase()` proposals. When replicas
   propose different phases, drivers alternate prefill and decode. The proposal shares the native admission
   fallback used by actual dispatch and reserves nothing.
2. Each driver calls `Engine.step(phase=agreed_phase)`. A mismatched local cohort
   remains queued. Cache pressure and retraction still belong to the native Engine.
3. Actual plans rendezvous with empty or failed preparation on other drivers.
   This second exchange determines whether there is real work, computes logical
   replica capacities, and catches plan serialization errors before execution.
4. Each driver broadcasts its logical plan to its TP/CP followers. All workers
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
The coordinator broadcasts a logical plan; it does not mistake copying a plan
for implementing CP attention or EP routing.

An optional `validate_distributed(plan, context)` performs host checks before the
readiness vote. The forward callback must preserve collective ordering and finish
its device work before returning. Mid-collective process/device failures rely on
process-group timeouts and process teardown. The coordinator propagates errors;
it cannot roll back in-place model state or globally undo a peer's completed local
commit. Failure marks all native Engines poisoned, preserving resources and
preventing cleanup/reuse through normal Engine operations.

Existing TP-only runtimes that use an injected TP group can continue implementing
`forward(plan)` when `cp=ep=attn_dp=1`; attention TP and expert TP then both equal
the total `tp`. GPT-OSS's `TensorParallelContext` can be constructed from
`groups.attn_tp.local_rank`, `groups.attn_tp.size`, and `groups.attn_tp.handle`
(use `None` for a single-rank context). This prototype does not change current
deployment entry points or rewrite default-world assumptions inside existing models. Each model
must use its supplied groups before it can support a composed TP/CP/EP/DP layout.

## Process integration

After initializing distributed communication and preparing each rank's runtime:

```python
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
