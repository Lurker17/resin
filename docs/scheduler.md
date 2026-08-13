# Native scheduling and batch phases

Resin keeps ready-queue ordering and request progress in Rust. The scheduler
selects work; the runner executes the already-frozen token ranges.

```python
from resin import Engine, SchedulerConfig

engine = Engine(
    runner,
    max_running_requests=32,
    context_length=32768,
    scheduler_config=SchedulerConfig(
        max_batch_tokens=4096,
        prefill_chunk_size=8192,
        schedule_conservativeness=1.0,
    ),
)

request = engine.submit(
    token_ids,
    max_new_tokens=64,  # optional hard stop, not a KV reservation
    priority=5,
    batch_key=7,
)
```

`context_length` is an Engine-level model limit, not a batching preference. It
defaults to 32768 and rejects requests whose requested sequence would exceed
that bound. If `max_new_tokens` is omitted, generation may use the remaining
model context. The value is always a terminal generation cap; Resin never
reserves that entire lifetime at submission.

`prefill_chunk_size` defaults to 8192 tokens and is the scheduler's execution
unit for prompt or replay work. A runner may declare a smaller hard
`prefill_chunk_size`; the Engine uses the smaller bound and never silently
re-chunks an accepted plan. A runner must declare exact
`supports_continuation_prefill = True` when the selected range stops before the
end of a prompt; the Engine rejects such a range before runner acceptance when
the runner does not support continuation.

## Cohort selection

The scheduler distinguishes admitted requests from waiting requests. Admission
is sticky: successful survivors remain admitted until completion, cancellation,
or memory retraction. There is no FIFO/round-robin mode that rotates a decoder
back behind waiting work.

Selection is deterministic:

1. Keep admitted prefill chunks running and fill compatible vacant slots.
2. Otherwise, admit waiting prefills into vacant slots and run that prefill
   batch before resident decoders.
3. When no prefill can run, execute the resident decode batch.
4. Within one phase, choose the highest priority and oldest compatible
   `batch_key` cohort.
5. Never exceed `max_running_requests` admitted requests.

The scheduler does not fill spare rows with a lower priority, another key, or an
incompatible phase. `batch_key=None` is a real cohort and differs from zero.
Priorities are signed 32-bit integers; larger values run first within the
preferred phase.

Prefill and decode never share a plan. `BatchPlan.phase` is the explicit
`BatchPhase.PREFILL` or `BatchPhase.DECODE` value for the complete batch. A
one-token continuation prefill therefore remains unambiguously prefill.

Prefill can pause resident decoders, but does not evict them: their KV and
admission remain live. A full admitted set is not displaced merely to accept a
new prefill.

Native turn ordinals never cross the Python boundary. A pre-accept rejection
restores the selected requests and their exact order.

## KV admission and retraction

Managed KV starts empty and grows transactionally to the end of each frozen
plan. Admission uses an estimated completion length, not the request's entire
generation cap. `schedule_conservativeness` scales that estimate:

- larger values reserve more admission headroom and reduce retractions;
- smaller values admit more concurrent work and may replay more often;
- `1.0` is the default, while `0.0` is maximally aggressive.

The estimated new-token ratio begins at
`min(0.7 * schedule_conservativeness, 1.0)`, decays during successful decode,
and is raised from observed progress after a retraction. `Engine.new_token_ratio`
exposes the current diagnostic value. This estimate controls admission only;
actual pages are allocated on demand, and allocation failure cannot corrupt an
existing request table.

If the next step cannot grow, Resin keeps at least one running request and
retracts an eligible attention-only request with the least completed output
(longer prompts lose ties). It asks the runner to release request-local state,
frees the request's KV, returns it to the waiting tail, and later replays its
prompt plus committed generated inputs. Replay prefill never samples a duplicate
token.

Requests with recurrent-state handles are not currently retracted. Their state
cannot be discarded safely without an explicit reset protocol; they remain
sticky and an unrecoverable KV shortfall is reported instead.

## State compatibility

Each submitted state handle has a `"read"` or `"write"` access mode. Omission
defaults to write. Reads may coexist; writes are exclusive. Duplicate handles in
one row collapse for conflict checking, with write dominating read, while the
plan preserves the submitted occurrences.

After selecting the homogeneous cohort, Resin keeps its longest state-compatible
prefix. It stops immediately before the first conflicting row and never skips
that row to backfill a younger one. At dispatch, `StatePool` also checks leases
already held by another Engine sharing the pool. A conflict before runner
acceptance rejects the dispatch without advancing requests.

Access remains leased through asynchronous completion and runner cleanup. See
[state ownership](state-pool.md) for lifetime details.

## Token-range planning

Without token controls, a prefill row consumes its remaining prompt or replay
input and a decode row consumes its one input token. With controls:

- `prefill_chunk_size` limits each selected prefill row;
- `max_batch_tokens` is consumed in row order across flattened plan tokens;
- decode rows always request one token and later rows are omitted when the
  decode batch exhausts its token budget;
- the scheduler keeps an oldest prefix of the selected cohort rather than
  skipping a row.

For a managed Window cache group, prefill also stops at the next group block
boundary. That makes each accepted range safe for the current private cyclic
Window storage: no plan overwrites a physical block that it still exposes for
attention. Multiple Window groups compose by taking the earliest boundary.

`BatchPlan.sequence_indptr` describes the final flattened range for each row.
`context_lengths` contains each row's pre-append logical length. Initial prefill
is sampled only when it reaches the end of the prompt; replay prefill is never
sampled; decode rows are always sampled. `sample_sequence_indices` makes this
event/commit mask explicit. The runner still returns one token ID per row,
including an unsampled continuation or replay row.

## Dispatch and completion

One Engine currently accepts at most one in-flight batch. `dispatch()` returns
the new batch ID, or `None` when work cannot be dispatched because the queue is
empty or a batch is already in flight. `poll()` returns an empty list while an
asynchronous completion is pending. `step()` performs one dispatch attempt and
then one poll.

On successful completion Resin:

1. validates exactly one output token ID per plan row;
2. asks the runner to release terminal physical request state, if supported;
3. releases state and cache writer leases;
4. commits prefill/decode progress;
5. retires completed or cancelled requests;
6. returns survivors to the sticky admitted set;
7. emits events for sampled rows.

EOS and `max_new_tokens` are terminal conditions. A ready cancellation completes
without dispatch. In-flight cancellation records intent and retires only after
the accepted batch finishes.

## Failure semantics

Failures before runner acceptance are transactional: temporary access is
released and selected requests return to their original positions. Once the
runner may have addressed cache or state storage, a protocol, polling, cleanup,
or commit failure poisons the Engine. The accepted batch and its capabilities are
quarantined rather than retried against reusable memory.

This distinction is the scheduler's central safety rule: logical retry is only
allowed while Resin can prove that physical execution did not begin.

## Current limits

Scheduling does not choose CUDA graph buckets, kernel shapes, cache admission,
or sampling algorithms. Those remain runner concerns. The native scheduler is
single-Engine and deliberately single-in-flight; `StatePool` leases are the
mechanism that coordinates state access across Engines. Resin does not support
overlap scheduling: it never plans the next batch ahead, takes future-batch
locks, or duplicates recurrent-state residency to hide scheduler overhead.
