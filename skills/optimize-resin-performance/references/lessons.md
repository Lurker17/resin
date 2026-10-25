# Resin performance lessons and evidence

These cases were assembled on 2026-10-25 from repository documentation, selected
diffs after scanning all 471 commits reachable through `git log --all`, and
accessible Resin conversations. The history spans 2026-08-09 through 2026-10-25.
This is not a claim to have read every diff or every conversation. Recheck current
code before applying a historical lesson, especially retired DeepSeek V4 paths.

The tracked docs and commits are the durable evidence. Many raw `artifacts/`
directories are local and ignored; paths below are retrieval hints and may not
exist in another checkout. A missing artifact does not invalidate a documented
result, but limits independent reproduction. Use `git show HASH` for a change or
`git show HASH:path` for a removed report; do not recreate its retired Modal probe.

## Choose the relevant case

- Exposed host time or serialization: cases 1–3.
- Cold stalls or scaling with context: case 4.
- Graphs, fusion or replacing an upstream kernel: cases 5–6.
- Speculation, cache boundaries, crashes or serving: cases 7–8.
- Prefill/reference gap: case 9.
- Multi-GPU regression or numerical drift: cases 10–11.
- Benchmark ambiguity: case 12 and the provenance notes.

## 1. Fast GPU kernels can still sit behind slow host coordination

Historical DeepSeek V4.1 on eight B300s, EP8/ADP8, 128K shared/private prompts,
balanced routes and forced acceptance:

| Source | B8 Engine interval | What changed |
| --- | ---: | --- |
| `de438e4` | 60.920 ms | Prepared-forward baseline; device phases 21.640 ms. |
| `cd5ecfd` | 31.099 ms | Prepared metadata, CPU planning and Engine overlap bundle. |
| `40a44fd` | 25.683 ms | Compact graph-selection vote; longer measurement window. |
| `43b9333` | 15.897 ms | Fixed-record gather/broadcast plus lightweight graph counters. |

These are successive qualified snapshots, not isolated causal estimates for each
change. The overlap bundle reduced B8 kernels/forward from 5,116 to 1,436. The last
control-path change retained those 1,436 kernels and all six rank agreements; it
did not remove failure coordination. A separate paired trace showed the graph-body
gap shrinking from 10.895 to 0.170 ms. Do not subtract those profiler intervals from
the separately measured Engine times or call all of the difference CPU execution.

Relevant commits: `8e1ea23` and `69345e4` prepare/share metadata; `cc940b6` adds
Engine overlap; `4c38de5` changes control transport. Detailed graph statistics
cost about 2.4 ms/step in a local diagnostic versus about 0.004 ms for aggregate
counters; that is harness overhead, not faster model math.

Read [recorded V4.1 measurements](../../../docs/benchmarks.md#recorded-v41-measurements)
and [distributed contracts](../../../docs/distributed.md). First inspect repeated
work and transport between graph bodies when device work is already close to a
reference. Preserve exact catalog revalidation and readiness/commit agreement.

## 2. CPU algorithmic complexity matters at long context and large concurrency

`29d4478` removed quadratic cache validation, shared immutable prefix page-table
segments, and improved scheduler batch selection. The historical report at
`01be552:docs/architecture-review.md` compares medians of three process runs on one
macOS host against `4775fd3`:

- Validate 4,096 pages 100 times: 409.369 → 15.194 ms.
- Select/return 512 requests from 100,000, 20 times: 444.537 → 5.915 ms.
- Publish 2,048 growing checkpoints: 1,451.125 → 174.228 ms.
- Snapshot peak RSS: 98,713,600 → 3,096,576 bytes.

These are CPU microbenchmarks, not measured decode improvements. Preserve ownership,
invalidation, eviction and retirement checks while changing their complexity.
Current anchors: [paged resources](../../../crates/resin-core/src/resources/paged.rs),
[prefix resources](../../../crates/resin-core/src/resources/prefix.rs) and
[scheduler](../../../docs/scheduler.md).

## 3. Repeated preparation and object conversion accumulate

The October 24 cleanup found GPT-OSS preparing attention metadata three times in
distributed graph decode and four times in eager fallback. Whole-batch packed
block IDs and post-lengths were also recomputed inside per-row loops. Commits
`be46b18` and `c6da8dd` reuse prepared metadata/native validation; `902ad82` and
`e02036d` keep device graph views and borrowed flat output columns.

`d5706c3` retains request-lifetime media/sampling on followers and transports typed
step buffers. The subsequent native HTTP packet work (`a51e819`) reported CPU
serialization roundtrips at 512 pages/request improving from 54 to 17 µs at B8
and 597 to 356 µs at B256. No isolated GPU speedup was measured for these cleanups.

Avoid lists, per-token Python scalars, repeated hashing and copies at a boundary
that already validated the data. Retain ownership until asynchronous copies and
receives finish. `40a44fd` corrected reusable buffers that became inference tensors;
`9ffe6f3` retains pinned storage through asynchronous copies. See
[the runner boundary](../../../docs/kernel-boundary.md).

## 4. Runtime geometry prevents first-use compilation stalls

The live EP4 V4.1 conversation reported roughly 19-second stalls on unseen prompt
lengths. `826deb0` removed metadata query-length specializations, with GPU
qualification in `24d76a8`. Earlier `a77b14a`, `bc812eb` and `041a19f` moved history,
table-width and context-capacity dependence out of metadata/candidate JIT keys.
Qualification varied table widths 1, 3 and 1,025 and contexts through 128K while
compiled-kernel counts remained unchanged.

The full-model EP8 128K prefill experiments measured 9.774 s with 1K chunks,
4.009 s with 4K and 3.691 s with 8K. This combines execution/configuration changes;
it is not an isolated speedup from disabling compilation. See
[benchmark evidence](../../../docs/benchmarks.md#recorded-v41-measurements).

Older precedents: `31ae8d8` keeps only model-fixed trailing cuTile dimensions
static; `76f8665` separates variable prefill from tiny decode convolution;
`e092a75` uses device-fixed persistent grids rather than request-dependent grid
caps. Their code changes alone do not prove stable compiler caches. Check every
rank with unseen lengths, runtime sampling changes and changed-input graph replay.
Do not replace the problem with exhaustive startup warming.

## 5. Graph padding can add work and change numerical results

The original prefill policy padded every input to a power of two. At 129 live rows
and a 256-row graph, Qwen's first FFN region regressed 585 → 789 µs and Gemma's
617 → 743 µs; Inkling failed numerical parity. Small GPT-OSS regions improved
249.43 → 116.27 µs at three rows, while 8,192 rows were essentially unchanged.

Commits `fa0a664`, `308fbfb` and `2ab268f` preserve useful graph regions and actual
GEMM geometry. GPT-OSS's separate output-projection capture cost 24–30 µs versus
16–19 µs eager, so it remained eager; QKV plus RoPE amortized staging over more
operations. DeepSeek exact buckets avoid a demonstrated 17 → 32 row parity failure.

These are region timings, not whole-model speedups. Read
[graph policy](../../../docs/cuda-graphs.md), including serial shared-workspace use.
Maintain graphability, report actual capture scope, and distinguish intentional
prefill partial-chunk fallback from a missing capture. Do not loosen numerical
checks simply to improve a replay counter.

## 6. Benchmark the whole kernel adapter and preserve rounding boundaries

The retired report `4ea6384:docs/deepseek-kernel-reuse.md` contains both outcomes:

- Pinned TileKernels FP8 quantization emitted GEMM's required scale layout directly:
  at width 4,096 and 512 rows, scale preparation fell 36.854 → 2.565 µs with exact
  bytes/scales across 38 layout cases.
- A correct FlashInfer `hash_topk` substitution lost to the local one-row kernel:
  1.484 µs local, 3.588 µs ordinary adapter, 2.347 µs compiled adapter. The required
  scale-before-BF16-conversion boundary prevented moving that work downstream.

The same work retained left-associated routing normalization and fixed all-negative
softplus tails: 33,516 reference BF16 weights matched exactly; the old path
produced 9,576 nonfinite weights. Earlier `c79fa23` found that an accepted tensor
shape still had the wrong FP8 scale-major interpretation. Shape acceptance alone
does not validate packing, scale axes, strides or accumulation precision.

Recent positive primitive cases (`41ad813`, `5bd97f0`, `d2a4fc3`) include GPT-OSS
router 14.39 → 4.12 µs at eight rows and Inkling 4K convolution 496.74 → 122.11 µs.
They remain primitive claims until a matched end-to-end run demonstrates impact.
Read [primitive checks](../../../docs/benchmarks.md#cuda-model-primitive-checks).

## 7. Overlap requires ownership and device feedback, not just an async API

Older commits `5fc98c9` and `7471736` removed pending/async machinery that could not
launch the next decode before committing the previous result. This is not advice
to remove today's measured overlap: `cc940b6` later supplied device feedback,
owned results and Engine continuation.

The live EP4 endpoint initially disabled overlap for stop tokens and idle replicas.
`b46e1c6` retains stopped rows until launched successors finish; `b70331d` keeps
idle attention replicas participating. The paired HTTP experiment reported
59 → 82 tokens/s at temperature zero and 63 → 87 at 0.7, with matching output and
acceptance. Startup verification recorded 15 enqueues and 14 continuations for one
active chat with three idle ranks (`eaa4152`). These are historical chat-reported
results, not a new benchmark run or a universal 38% speedup.

The EP8 B256 sweep also exposed page-boundary intervals of 89.96, 97.68 and
87.20 ms against a 39.95 ms median; the 42.507 ms mean includes them. An earlier
39.794 ms window stayed within one page. Include those drains, and preserve leases,
page retirement, prefix publication and failure quarantine when reducing them.

`535e5a4` derives safe simultaneous writes as physical retention minus logical
read window plus one. `8335f67` adds speculative pre-anchor history. Recompute the
complete read/write envelope when changing chunk size, draft width or capacity.
Graph padding needs reserved scratch storage; masks alone did not prevent the
unscheduled-row corruption fixed by `c79fa23`. See
[the scheduler](../../../docs/scheduler.md) and
[model correctness contract](../../../docs/model-forward-pass.md).

## 8. Boundary tests expose bugs that ordinary generation misses

The live V4.1 stress failure was compressed-KV page-table row ordering for batched
speculative queries. A request crossing 128 tokens could read another request's
table despite correct scheduler allocation. `1c1d4dc` fixes flattened query order;
`35f0c1b` corrects the independent probe oracle. The real DeepGEMM regression and
128-request mixed full-model replay passed before the HTTP concurrency retest.

Use distinct physical pages per request, unequal histories, boundary-crossing query
blocks, dirty/NaN unused tails, cached continuations and changed inputs. A one-row
test cannot establish a batched mapping. `f786a1a` requires exact greedy
target-only/speculative parity and a sequential oracle for every verification
column's logits and final full/sliding/recurrent state, not just plausible text
or an acceptance rate. Include rejected drafts, EOS and unrelated preceding work.

## 9. Faster prefill may mean less or approximate computation

V4.1 bounded consumer replay (`43c3db4`, documented by `ca300db`) reduced EP8
128K prefill from 3.6910 to 1.9509 s, about 67.2K input tokens/s/GPU. The first
20 layers still process the full prompt; a boundary prepares full compressed KV;
later layers process retained queries. The historical 140-token consumer SWA tail
is approximate; exact reconstruction required 2,560 tokens for that configuration.

This does not establish equality to full prefill. Approximate consumer state stays
out of prefix caching and prompt scoring takes the full path. Separate deterministic
qualification compared compact/masked replay and prefix restoration, including
unequal/idle cohorts; that does not turn the algorithm into exact full computation.
See [the model's validation notes](../../../models/cuda/resin_models_cuda/deepseek_v41/README.md#validation).

Inspect a reference's actual work before tuning toward its published curve.
Larger chunks alone cannot reproduce this algorithm, and consumer-sized graph
buckets, full boundary writes and absolute draft positions are part of the change.

## 10. Topology and precision can dominate a seeming kernel regression

The October 23 dense-model work found exact graph/eager parity within each B300
topology but Gemma cross-topology prompt-score drift up to 4.68 nats. Single-GPU
emulation of two BF16 FFN shards and BF16 addition reproduced it exactly, isolating
partial-result rounding from communication/cache/graph bugs. Raw local evidence:
`artifacts/cuda-model-improvements/gemma-ffn-emulation/`. Do not call within-topology
parity cross-topology correctness; compare sensitive logprobs as well as tokens.

Loaders also needed to slice CPU-backed tensors before GPU transfer: loading the
full model and then sharding defeated TP memory savings. Parallelism support must
cover actual head/channel/state geometry and collective order, not only flags.

Historical GPT-OSS 120B H100 sweeps at `b2f7c3c` measured best tokens/s/GPU of
574.750 for TP1, 488.915 TP2, 286.481 TP4, 988.479 EP2 and 1,013.632 EP4. These
workload-specific curves argue for measuring the mesh; they are not a general
ranking. See [recorded decode sweeps](../../../docs/benchmarks.md#recorded-decode-sweeps).

## 11. Communication overlap and NUMA need controlled evidence

DeepGEMM persistent barriers and asynchronous NCCL Engram fetches reproduced a
grid-sync timeout on B300 EP2. Serial warmup did not fix composition. `e655d01`
and `f90ae8c` join fetches on the device stream before the persistent kernel;
eager, capture and 30 replays then passed at 4 and 1,280 tokens. This is a required
dependency for that composition, not a measured throughput win or justification
for host synchronization everywhere.

The local October 24 NUMA experiment had fewer B256 kernels (1,428 → 1,386) and
1.235 ms less changed-kernel work, yet GPU-active union rose 38.495 → 48.471 ms.
NCCL work rose 0.390 → 10.604 ms and its grid changed from 32 to two blocks.
The intended paired NUMA A/B stalled; a different-allocation comparison did not
establish a benefit. NCCL duration includes waiting, so this is evidence to inspect
topology/configuration, not a complete causal diagnosis.

Local evidence is under
`artifacts/benchmarks/2026-10-24-deepseek-v41-b300-ep8-numa/`.
The API commit `3f2893b` and [NUMA guide](../../../docs/numa.md) explain placement
before child startup, container CPU masks/quotas and fallback reporting. Affinity
does not reserve CPUs or migrate existing pages. Do not claim that API presence
establishes a measured performance improvement.

## 12. Changed outputs and projected acceptance invalidate casual comparisons

The retired `4ea6384` report records identical baselines at 160.14 and
212.62 tokens/s with different outputs. Candidate 209.14 and an old-softplus variant
273.87 did not establish a speedup because paths and accepted tokens differed.
Use fixed-token/first-divergence diagnosis for numerical changes, then measure
natural serving performance separately from controlled synthetic work.

Similarly, a live EP4 chat's natural draft acceptance was about 33–37%, roughly
two committed tokens/step. A published Resin EP8 chart projected 75%, or 3.25 with
three drafts. The user-visible 80 versus expected 200 tokens/s gap could not be
attributed entirely to overlap or kernels. Match topology and acceptance first.

The historical sampler bundle (1.501 ms B8 and 5.405 ms B256 of summed kernel work)
was a candidate opportunity, not a completed speedup. Those sources used
top-k-before-top-p (`580dde5`, `b2f7c3c`). The later fused sampler deliberately
changed to top-p on the full scaled distribution followed by the top-k cap,
retaining cutoff ties, and CDF sampling changed seeded stochastic choices.
See the current [sampling contract](../../../docs/benchmarks.md#shared-sampler-checks)
before choosing an oracle; a faster upstream sampler with another contract is
not a drop-in replacement.

## 13. A graphed adapter or rollback can dominate the work it surrounds

The October 25 GLM-5.3-Flash MTP3 traces at 8K context used four B200s,
EP4/attention-DP4, 8192-token chunks and controlled greedy prompts. Source
`878d83d4` is recorded in the local artifact manifests; these are observations
from that implementation, not measured optimization wins.

At B8, four captured sampler calls took 1.35 ms per cycle, including 1.09 ms
in top-p processing even though the final selection was greedy. The eager
sampler's greedy branch did not apply inside the captured tensor path. Fusing
all 28 final selection kernels could address only 43 microseconds. Inspect
which branches actually execute in the graph before choosing a sampler fusion;
runtime changes to sampling parameters must remain valid on replay.

Every MoE dispatch had 32 preparation kernels after routing: buffer padding,
owner expansion, sorting, selection and payload masks. Across 46 dispatches
that was 1,472 kernels and 2.21 ms at B8. This identifies a shared transport
adapter as a fusion target without changing expert GEMMs or router tie order.
Preserve expert-TP fanout, duplicate destinations, source-replica rules, idle
participation, sentinels, wire precision and multipass behavior.

The contiguous post-sampling KDA block was accepted-prefix rollback: another
34 recurrences after the target's 34. At B64, target recurrence took 11.25 ms
and rollback recurrence 11.06 ms per cycle. The latter discarded its attention
outputs. A state-only recurrence is a candidate, subject to independent state
parity and graph tests; retaining every intermediate state instead would add
roughly 34 GiB at width four for this model and batch. Separate necessary state
repair from avoidable output computation and quantify the memory tradeoff.

An accepted-prefix implementation also changes how cost depends on acceptance.
Projecting 75% acceptance from a lower-acceptance run scales output tokens while
holding cycle time fixed; it does not execute the additional recurrent updates.
Label that projection accordingly and base the optimization claim on measured
throughput with matched outputs and acceptance.

Evidence: `artifacts/glm53/wide-8k-878d83d4/analysis/sampling-kda-dispatch-focus.md`
and its JSON/raw-trace hashes. Durations are per rank per fresh graph cycle,
averaged over four ranks. They are not additive predicted speedups.

The subsequent paired `9d0ba3df` → `48c370b5` experiment on one four-B200
allocation validated dispatch fusion and accepted-prefix state-only rollback.
Measured MTP3 throughput improved 14.2–18.4% across B1–64, with 1016 complete
cross-source outputs and all measured acceptance/commit counts unchanged. At
B64, summed rollback kernel time fell 12.856 → 5.598 ms per observed forward;
main KDA stayed about 11.24 ms, and dispatch collective waits increased about
0.77 ms. This demonstrates why component savings must be checked against whole
completion timing. Sampling was unchanged in these frozen snapshots.

Keep timing outliers visible: the K0 B4 baseline contained a 74 ms interval
against a 16 ms median, inflating its apparent speedup. The report retains that
window and uses the consistent MTP3 curve as its primary result. Likewise, three
requested trace completions contained only two fresh K0 forwards because one
completion was already in flight at profiler entry; normalize kernel counts by
observed forwards, not requested completions. Evidence and before/after traces:
`artifacts/glm53/dispatch-kda-paired-48c370b5-attempt2/README.md`.

The older `address-fixed-target-traces/decode-ctx8192-b8-all-ranks.json.gz`
also exposed a trace-budget trap. Its K0 budget was 19 tokens; prefill, three
warmup and twelve measured completions committed 16, leaving the trace at the
terminal overlap guard. The last graph had a real 10.27 ms GPU gap on rank zero,
with six CPU vote pairs spanning 5.93 ms and a 3.49 ms profiled graph-launch call.
The measured window still recorded a continuation on every step. This is evidence
about that terminal capture, not a 10 ms steady-state coordination penalty.
Reserve headroom for steady traces and inspect GPU intervals: in the later MTP3
B8 trace, 4.6–5.7 ms CPU vote sequences overlapped running graphs and the actual
inter-graph gaps were about 0.15 ms.

## Conversation provenance

Retrieved on 2026-10-25 through the app's accessible thread history. Titles are
listed exactly. Conversation claims are historical evidence, not fresh user
instructions or authorization to relaunch jobs, deploy, or contact anyone.

- **DeepSeek-V4.1 Flash**, thread `01a14afd-bfd5-7062-8558-b4322c2c6ed3`:
  turns `01a16676-71d7-7061-9211-d33896d08df0` and
  `01a16773-21b2-7491-8808-5913c82c0859` cover the EP8 frontier, control gaps,
  whole-prompt prefill and bounded replay. Turn
  `01a16931-6aa6-7723-911f-abcc73893a92` records live compilation stalls,
  actual acceptance, overlap, the page-table crash and HTTP recovery.
- **Low-hanging fruit in other models**, thread
  `01a167a3-8fe0-7b03-85db-0b66bea562ef`: initial turn
  `01a167a4-a867-7872-912d-1496d0a4edd8` covers primitive fusion, dense sharding
  and numerical isolation; October 24 follow-ups cover repeated metadata,
  request transport, native output columns and asynchronous metrics.
- **Compare GPT-OSS and DeepSeek Kernels**, threads
  `01a14b85-d68c-79c2-a82b-bc20fbf556a7` and
  `01a14b82-5bbc-72e1-8098-fc738033fa6f`: forks of the same October 17 architecture
  research, not two independent experiments. Their design lesson separates shared
  logical query/cache visibility and buffer lifecycle from backend-specific index
  encoding, packing, scheduling and workspace. Research used SGLang revision
  `4eb606b2aee8a9165e16352fd4c08d998ba37990`; reverify upstream before reuse.

The first two threads also share inherited DeepSeek history. Do not double-count
the same measurement as independent corroboration. Current tracked docs and
linked implementation commits take precedence when describing present behavior.
