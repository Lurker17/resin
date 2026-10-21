# Architecture review and implementation

Date: 2026-09-11

Resin's goal is to provide the reusable parts of an inference engine: request
lifecycle, scheduling, cache ownership, and a precise execution contract. A
model author supplies a forward pass and kernel choices. That boundary should
also support a Rust application or a future worker executor without requiring
it to copy the Python extension's state machine.

## Decisions

- Keep native ownership authoritative and retain the synchronous `step()` path.
- Make `resin-core` the complete substrate and `resin-py` its language adapter.
- Keep Torch required and the Rust HTTP server bundled. Lazy imports still
  prevent importing Torch or initializing CUDA through `import resin`.
- Share execution utilities whose invariants are actually common. Keep GPT-OSS
  transformer semantics in the GPT-OSS package.
- Preserve prefix sharing and resource leases; simplify their representation
  and remove repeated work rather than weakening ownership checks.

## Implemented changes

**Complete native lifecycle.** `EngineCore<P>` now owns submission, request
progress, admission, retraction, preparation, output validation, completion,
cancellation, and cache quarantine. Opaque payloads let frontends supply model
media and sampling metadata without teaching the core about Python conventions.
The Python adapter resolves callbacks and converts inputs, plans, errors,
handles, and events. Native conformance tests exercise the public interface,
including rejected execution, malformed results, callback failures, and panics.

**Execution failure semantics.** A device can synchronize successfully after
partially updating recurrent state. Retrying that step would use the wrong
starting state. CUDA failure guards now quarantine any failure after execution
begins; host-only validation remains retryable. Native runners explicitly
return `Error::rejected` only when execution had no persistent effects or rolled
them back. Complete raw output is checked against phase and leased speculative
capacity before EOS or length truncation, so stopping cannot conceal an invalid
verified block.

**Window read/write geometry.** Logical read history is now distinct from
physical retention. A Window group's maximum simultaneous write span is
`window_tokens - read_window_tokens + 1`. Native prefill planning observes the
smallest group bound, and invalid speculative lookahead is rejected before
execution. CPU cache simulations cover early queries after prefix copy-on-write
and cyclic reuse. Built-in runners declare their actual read windows; DeepSeek
also reserves a configurable 4096-token prefill guard band. That guard band
costs about 1.6 GiB per resident request per rank for the current checkpoint;
it is counted by native admission. A smaller chunk reduces this cost. DeepSeek
sparse attention metadata now clips local history separately for each query.

**Control-plane work.** Page uniqueness checks use reusable linear-time scratch
storage. Ready-queue selection and reinsertion use sorted merges. Progressively
longer History prefix snapshots share immutable page-table segments, and prefix
LRU operations use direct links. Window/State snapshots remain bounded and
physical page retirement still requires explicit ownership validation. The
native-to-Python execution path transfers owned plan arrays instead of cloning
and revalidating the public constructor representation.

**Serving lifecycle.** The HTTP bridge bounds queued jobs, active requests,
admissions before stepping, and each client's event buffer. Admission overload
returns 429. A slow client is cancelled without stalling inference for other
clients. Actor failure stops the listener, readiness reflects actor state, and
`shutdown()` requests orderly cancellation and closure after the current model
step. Rust protocol tests and actual local HTTP tests cover overload, stalled
clients, admission fairness, fatal execution, and shutdown.

## CPU measurements

The reproducible harness is
[`control_plane_bench.rs`](../crates/resin-core/examples/control_plane_bench.rs):

```bash
cargo run -p resin-core --release --example control_plane_bench
```

Measured on the same local macOS host, with rustc 1.97.1 optimized builds and the
median of three process runs. The comparison isolates the allocator, prefix,
and scheduler changes against commit `4775fd3`; timings exclude workload setup.

| Workload | Before | After |
| --- | ---: | ---: |
| Validate 4096 pages, 100 times | 409.369 ms | 15.194 ms |
| Select/return 512 rows from 100,000, 20 times | 444.537 ms | 5.915 ms |
| Publish 2048 progressively longer checkpoints | 1451.125 ms | 174.228 ms |
| Clear those checkpoints | 15.699 ms | 0.148 ms |
| Evict 16,384 prefix entries | 494.500 ms | 1.750 ms |
| Peak RSS for the snapshot workload | 98,713,600 bytes | 3,096,576 bytes |

These are CPU microbenchmarks, not end-to-end decode speedups. Prefix publication
still scans a request's current pages for ownership validation, so aggregate
publication CPU work across every growing checkpoint is not claimed to be
linear. Metadata storage no longer duplicates every ancestor's page vector.

## Validation

All 96 native tests and 340 Python tests pass, including nine native tests of
the complete engine through its public Rust interface. Five GPU-only Python
tests are skipped locally. Ruff lint/format, Rust formatting, whitespace checks,
and the repository code-size report also pass.

The Modal runs use the repository's deployment scripts and local source, with
bounded generation lengths and the existing model volumes:

| Workload | GPU validation |
| --- | --- |
| [GPT-OSS 20B](https://modal.com/apps/ekzhang/main/ap-ztWbQBejSeODhSRHQFSIrs) | H200 TP2; 32-token prompt run, graph execution and rank agreement |
| [GPT-OSS 120B](https://modal.com/apps/ekzhang/main/ap-TOW7Ue1eqyipe76Su91djK) | H200 TP2; 9216-token prompts across prefill chunks; batches 1/4, exactly 16/64 output events, two captured graph shapes |
| [Qwen 3.5](https://modal.com/apps/ekzhang/main/ap-KfeSuS2co6MUioKndbQKcR) | B200; speculative text and image smoke |
| [Gemma 4](https://modal.com/apps/ekzhang/main/ap-QTvQb9CrkhbwfDu1XdXdRj) | RTX PRO 6000; speculative image smoke and five graph replays |
| [Inkling](https://modal.com/apps/ekzhang/main/ap-P5TOxtuO6oefhxPiZe8HQY) | B300; 32-token speculative smoke |
| [DeepSeek window metadata](https://modal.com/apps/ekzhang/main/ap-FmJuId2KIma96ZnZrv1rxM) | B300; all three actual-kernel tests pass for cold prefill, continuation, and speculative writes |
| [DeepSeek-V4 full model](https://modal.com/apps/ekzhang/main/ap-mFyB85EYbPa0H1JZ2gP4mU) | B300 TP4; 8192 input tokens, 32 outputs, speculation 5; actual prefix hit agrees with cold first-token continuation; rank outputs agree; 31 graph replays and no graph breaks |

These runs exercise healthy execution; host-side fault injection
covers quarantine semantics. They do not establish independent-reference
numerical parity, arbitrary device-fault recovery, or production-load latency.
The GPT benchmark leaves prefix caching disabled; its batch-four aggregate
interval includes prefill scheduling and is not a pure decode measurement.
DeepSeek's final timed request has a populated prefix cache, so it is not a
cold-prefill comparison either.

The final DSpark small-chunk guard was tested locally after the full-model run.
It changes the default read declaration from 128 to 129 and retention from
4223 to 4224 tokens; both layouts use 34 blocks and allow 4096 writes, leaving
the tested device maps, kernels, and graph shapes unchanged. Its regression
checks every block alignment for write spans 2, 6, 130, and 4096. The extra block
needed by certain small spans preserves the oldest pre-anchor draft input.

Reproduction commands (from the repository root with Modal authenticated):

```bash
uv run modal run deploy/modal_gpt_oss.py --mode run --model 20b --max-new-tokens 32 --temperature 0
uv run modal run deploy/modal_gpt_oss.py --mode bench --model 120b --cache-mode prewarm --prompt-tokens 9216 --max-new-tokens 16 --batch-sizes 1,4 --context-length 16384 --profile-decode-steps 2
uv run modal run deploy/modal_qwen3_5.py --mode smoke --max-new-tokens 32
uv run modal run deploy/modal_gemma4.py --mode smoke --max-new-tokens 16
uv run modal run deploy/modal_inkling.py --mode smoke --max-new-tokens 32
uv run modal run deploy/modal_deepseek_v4.py --mode validate-window
uv run modal run deploy/modal_deepseek_v4.py --mode profile --input-tokens 8192 --max-new-tokens 32 --temperature 0 --speculative-tokens 5 --batch-size 1
```

## Next steps toward a full engine substrate

1. Measure mixed-arrival latency with separate queue, planning, execution, and
   commit timing. Bound prefill interference using a concrete latency target.
   Move costly prompt rendering out of the engine actor through a bounded
   preparation queue when profiling justifies it. Stock codecs use
   `resin-recipe` for conversation formatting and incremental parsing, and the
   HTTP actor consumes native parser events directly. Applications can supply
   another codec through the same structural Python protocol.
2. Expose a narrow scheduling-policy seam while retaining compatibility,
   capacity, progress, and lease validation in the core. Keep the current
   scheduler as the reference policy.
3. Introduce typed completion progress when an algorithm needs more than the
   current ordinary/speculative token-row contract.
4. Add asynchronous completion only when profiling demonstrates useful overlap.
   Prepared work must retain ownership through delayed completion, cancellation,
   and failure; a background Python thread alone does not supply those semantics.
5. Put topology, collectives, and worker transport behind an executor. Add cache
   identity namespaces for model revisions/adapters before sharing across them.
6. Add explicit resource export/import and completion-backed cache transfer when
   prefill/decode disaggregation or multi-node execution needs it.

The current system remains synchronous and single-dispatch. It does not yet
provide a public policy plugin, asynchronous completion, distributed cache
transfer, or a watchdog that can interrupt a hung GPU call. Those are deliberate
follow-on projects, rather than extra abstractions needed for this hardening.
