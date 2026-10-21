# DeepSeek-V4 Flash: composable attention and expert parallelism

The CUDA runner accepts explicit groups prepared from `resin.Parallelism`:

```python
from resin import Parallelism
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

# Initialize the default NCCL world first, with one process per GPU.
groups = ParallelGroups.create(
    Parallelism(world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2)
)
runner = DeepSeekV4CudaRunner(
    checkpoint_path,
    max_seq_len=4096,
    max_batch_size=16,
    parallel_groups=groups,
    device=f"cuda:{local_rank}",
    cuda_graph_prefill="breakable",
    cuda_graph_decode="full",
    speculative_tokens=5,  # Set zero for ordinary decode.
)
runner.capture_cuda_graphs()  # Collective startup, before workers serve.
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker)
    # Submit local requests; collectively step and close every driver.
else:
    worker.serve()
# Every rank reaches this after its serving loop has stopped.
runner.close_cuda_graphs()
groups.close()
```

Attention TP, attention DP, CP, EP, expert TP, and attention replication compose
through this path. Expert TP is `world_size / ep`, independent of attention TP
and CP. The legacy `tensor_parallel=` path remains available and is mutually
exclusive with explicit groups. DSpark speculative decoding uses these same
groups; set `speculative_tokens` within the checkpoint's supported block size.

Model dimensions and quantization geometry still constrain valid layouts.
Attention head, indexer, vocabulary, and grouped projection slices must satisfy
their sharding rules. EP must divide the routed expert count. Expert TP partitions
complete intermediate quantization blocks: 128-value boundaries for FP4 experts
or FP8 model storage, with unequal aligned shard widths permitted. Preparation
checks these boundaries before allocating model storage or preparing kernels.

## Ownership and execution

Attention projections, vocabulary shards, indexers, and logits use injected
attention-TP contexts. Routed experts use separate EP ownership and expert-TP
intermediate coordinates. Checkpoint loading reads owned experts and slices
packed weights/scales on their logical intermediate axes before converting them
to prepared kernel storage. Execution does not infer these layouts from the
default process group.

The shared `ExpertParallelDispatch` preserves every attention rank's token
origins across `groups.device_domain`. Selected rows visit every intermediate
shard of their expert owner; returned partials combine across EP and expert TP.
The returned FFN rows are complete, so the model does not apply another
attention-TP reduction to them. Shared experts remain local. Sampling broadcasts
use `groups.device_replica`, covering all ranks executing the same request
replica. Independent attention-DP requests retain their own origin rows.

Each attention-DP replica has its own Rust scheduler, logical cache pool, and
request namespace. Physical cache capacity is agreed within that replica.
Prefill executes one sequence per model call and aligns the call count across
replicas. Missing calls and entirely idle replicas join matching expert rounds.
Eager empty rounds do not run attention or modify KV state. Captured empty
prefill replays only FFN regions; captured empty decode uses reserved scratch
cache rows.

## Context parallelism

DeepSeek's recurrent MLA compressors require ordered updates. Prefill therefore
keeps full hidden rows, MLA KV writes, compressor tails, and indexer compressor
state replicated across CP ranks. It splits query/output projection work,
indexer query scoring, and sparse attention over contiguous query segments,
then gathers attention output before the transformer block's full FFN.
Zero-query ranks still participate in the output gather.

Decode executes replicated attention on CP ranks. Complete KV and recurrent
state copies remain present on every CP rank in both phases; CP does not
increase logical cache capacity. This conservative query partition preserves
the cache/compressor boundary and duplicates projection, state, and FFN work.
GPT-OSS uses a different CP boundary with local hidden and FFN rows; see its
[model documentation](gpt-oss-parallelism.md#context-parallelism).

## CUDA graphs and current costs

The explicit-group runner uses shared graph machinery for prefill FFN regions
and full decode. Every rank captures matching catalogs at startup. Graph
readiness is validated before the coordinator's execution vote. Replicas agree
on common buckets before GPU work; unsupported shapes force the entire domain
to execute eagerly. Requested decode logprobs use retained graph logits.

Captured expert dispatch all-gathers a fixed token capacity across the device
world and reduce-scatters weighted partials to their origins. Device valid-row
scalars mask padding. Every callback receives `world_size * capacity` rows,
including foreign routes. Each attention copy dispatches its own origin, even
when another copy has identical hidden rows. Eager dispatch packs selected
owners and all their expert-TP lanes, reading split counts on the host.

This dense transport is a correctness prototype. It adds communication,
padding, scratch storage, and duplicated expert work; sparse graph-safe routing
and communication overlap remain separate work. The DeepSeek adapter retains
its existing FP4 kernel and arithmetic, so shared transport validation does not
establish DeepSeek-specific numerical correctness.

Sparse MLA metadata follows FlashInfer's fixed 128-column SWA prefix. Short
windows retain invalid slots in that prefix; compressed rows start at column
128 and active lengths include the reserved prefix. Per-layer compressed-index
capacity comes from the configured maximum context, with selected logical rows
ordered consistently before physical page mapping. Prefill, decode, and
verification therefore present the same physical reduction geometry even when
another attention-DP replica requires a larger graph bucket. Indexer scoring
still uses the current context size.

Fixed capacity adds padding for short contexts. At the default 4,096-token
maximum, C4 layers reserve up to 512 compressed columns. A 1,048,576-token
maximum makes C128 layers reserve 8,192 compressed columns, about 32.5 KiB of
indices per query including SWA. Device lengths mask inactive entries, but the
physical width can still affect kernel scheduling and scratch requirements.
This tradeoff needs throughput and memory measurements before wide-context
production use.

## Speculative decoding

DSpark's draft stages use the same attention and expert meshes as the target.
Each decode step runs draft stages, target verification, and local acceptance
and state commit. An idle attention-DP replica joins the draft expert rounds
before the target expert rounds. During prefill, it joins only target FFNs;
DSpark's prefill initialization updates cache state without expert collectives.
Ranks sharing expert communication must agree on the speculative block size;
worker startup checks the lookahead before execution, and graph startup includes
it in the shared capture signature. Replica cache capacities remain independent.

Captured draft dispatch reserves `batch_capacity * K` token rows; target
verification reserves `batch_capacity * (K + 1)`. Device masks scale the live
replica batch by each phase's width, including zero for idle replicas. Graph
selection includes the complete verification lookahead. Accepted lengths may
differ between requests and attention-DP replicas.

Graphs retain verification logits. The shared output formatter selects only
accepted positions, including the bonus/correction token, and returns logprobs
through the ordinary NumPy SoA output. Requesting speculative logprobs does not
force eager fallback. Rejected recurrent state uses the existing cleanup path.
Compressor rings preserve the committed history plus the verification tail.
Pooling masks unused values as well as scores, so arbitrary recycled memory
cannot enter a live window. Compressed indexer writes apply the page mapping's
row scale before adding offsets into the shared allocation.
Padded replay requests use the same group-private scratch parents as capture;
only absent history columns in live requests remain zero. Dummy verification
writes therefore preserve the permanently zeroed null parent.

## Validation

```bash
uv run pytest -q
uv run modal run deploy/modal_deepseek_v4.py --mode validate-parallel \
  --parallel-baseline --report-path artifacts/deepseek-v4-parallel-validation.json

uv run modal run deploy/modal_deepseek_v4.py --mode validate-composable \
  --parallel-cases ep-attention-tp,attention-tp-expert-tp

uv run modal run deploy/modal_deepseek_v4.py --mode validate-composable \
  --parallel-speculative-tokens 5 --parallel-cases ep-attention-tp,ep-attention-dp \
  --report-path artifacts/deepseek-v4-speculative-graphs2.json

uv run modal run deploy/modal_deepseek_v4.py --mode validate-composable \
  --parallel-world-size 4 --parallel-gpu-type B200 \
  --parallel-cases ep-context-attention-tp,ep-attention-dp-tp,ep-attention-dp-context

uv run modal run deploy/modal_deepseek_v4.py --mode validate-kernels \
  --report-path artifacts/deepseek-v4-native-kernels-validation.json

uv run modal run deploy/modal_deepseek_v4.py --mode probe-experts \
  --report-path artifacts/deepseek-v4-expert-full-checkpoint-probe.json
```

The existing DeepSeek validation command uses the complete pinned Flash
checkpoint and eager four-GPU EP/attention-DP cases. It exercises independent
request engines, unequal queues, idle ranks, continuation prefill, cancellation,
cache reuse, and generated-token logprobs. An optional TP4 run reports token
comparisons separately because quantized layouts can change greedy choices.
It is not a completed GPU validation of the new composed layouts or graphs.

Blackwell runs use the complete 155.43-GiB pinned Flash checkpoint, compare
fixed prefill and decode contexts against an independent legacy-TP reference,
and check captured-NCCL teardown. Validation exposed and corrected an existing
short-window sparse-metadata ABI error: compressed rows must follow the fixed
128-column SWA prefix. Earlier numerical reports predate that correction and
must not serve as accuracy evidence for the current implementation.

The earlier full-model reports below also predate the compressed indexer
addressing correction described under speculative validation. Keep their
numerical results as historical diagnostics; they do not qualify the current
implementation or establish that every difference came from quantization.

The [native B300 regression run](https://modal.com/apps/ekzhang/main/ap-0CratjzJslkPeebbuvwIKZ)
passes all 44 tests, including dense FP32 attention references with separate
SWA/compressed pools, short and mixed windows, sink logits, replay with changed
queries, and negative index padding with nonzero offsets. This isolates the
kernel metadata contract; full-model numerical checks remain separate.

The corrected [two-Blackwell full-checkpoint run](https://modal.com/apps/ekzhang/main/ap-7xdCOphT67GAcKMriz4SEO)
passes the EP2 + attention-TP2 control bit-for-bit against legacy eager TP2:
all four prefill contexts and all four same-history decode observations are
exact, including genuine sampler output. Each rank captures 258 prefill FFN
regions and two decode shapes, replays 559 prefill regions and 12 decode graphs,
and records zero eager fallbacks. Cache reclamation, isolated replay, scores,
and captured-NCCL teardown pass.

Expert TP2 in the same run completes those execution checks but exceeds the
unchanged logit limits. Its prefill relative L2 error is 0.13435 (max absolute
6.0606); the largest same-history decode relative L2 error is 0.29297, with
KL 0.59791. The four fixed prefill top tokens and four genuine sampler choices
still match. The full report remains failed; these observations do not establish
model-quality equivalence across quantized tensor-parallel layouts. Diagnostics
are in `artifacts/deepseek-v4-static-pairs-diagnostics.json` and the profile volume
directory `composable-20261021T020507Z`.

The [four-B200 composed run](https://modal.com/apps/ekzhang/main/ap-lNAJndyyC3dGhHZpoBA7S2)
completes the following layouts; each uses EP2 and expert TP2:

| Attention layout | Request engines | Coordinated steps | FFN replays per rank | Decode replays per rank |
| --- | ---: | ---: | ---: | ---: |
| CP2, attention TP2 | 1 | 24 | 559 | 12 |
| Attention DP2, attention TP2 | 2 | 26 | 559 | 14 |
| Attention DP2, CP2 | 2 | 26 | 559 | 14 |

Each rank captures 258 FFN regions and two decode graphs, with zero fallbacks or
poisoned graphs. Phase agreement, finite logprobs, cancellation, cache reclaim,
and identical isolated replay pass. Each attention-DP layout exercises seven
idle calls in one request replica and two in the other. Every driver's greedy
probe and four genuine fixed-history sampler choices match legacy TP4. All
three layouts still fail the strict numerical limits. The retained report is
`artifacts/deepseek-v4-static-composed4-diagnostics.json`, with durable profile
`composable-20261021T021100Z`.

The [remaining two-GPU CP/DP run](https://modal.com/apps/ekzhang/main/ap-Y37UA1Sl0fBJeDSOiTqweF)
also completes all four layouts: CP2 or attention DP2 with EP2/ETP1, and CP2 or
attention DP2 with EP1/ETP2. Comparing CP against attention DP while holding
attention TP, EP, and expert TP fixed gives bit-exact prefill and decode logits
on both request drivers. Their comparisons against legacy TP2 remain failed.
On four GPUs, the equivalent CP2+TP2 versus DP2+TP2 comparison differs by prefill
relative L2 0.063–0.079 and decode up to 0.103; it is not numerically qualified.
All comparisons retain the original limits in
`artifacts/deepseek-v4-static-mesh-pairwise-diagnostics.json`. The full two-GPU
report is `artifacts/deepseek-v4-static-cp-dp-diagnostics.json`, durable profile
`composable-20261021T021205Z`.

The one-Blackwell expert probe loads all 256 first-layer experts. EP2 is exact;
expert TP2 differs by relative L2 about 0.0037. Padding and duplicate inputs are
exact. FlashInfer emits BF16 GEMM2 results before finalizing expert contributions,
so FP32 transport cannot undo intermediate-shard rounding. A separate DeepGEMM
probe verifies true FP32 ordinary GEMM output and CUDA-graph replay, but the
pinned grouped API rejects FP32 outputs. This is diagnostic evidence, not an
implemented FP32 expert backend or a model-quality equivalence result.

Failed runs persist `baseline.pt`, per-case logits, genuine-sampler decode
traces, KL/offset/error metrics, and `report.json` under the printed directory in
the `resin-deepseek-v4-profiles-v1` Modal volume. Reuse the exact baseline and
disable graphs to isolate the device path:

```bash
uv run modal run deploy/modal_deepseek_v4.py --mode validate-composable \
  --parallel-cases ep-attention-tp --no-parallel-graphs \
  --parallel-baseline-path /profiles/<run>/baseline.pt
```

Decode traces observe genuine logits/tokens before advancing the Engine with
the baseline token. They therefore compare identical histories without changing
the captured sampler. Legacy baseline files explicitly report skipped decode
comparison. Existing CPU tests cover composed expert shards and CP/cache control
flow; they do not replace Blackwell numerical validation.

Speculative validation compares complete 16-token greedy probes against both
eager target-only and legacy DSpark references. It records token differences as
numerical diagnostics rather than forcing one token per decode step. Feature
checks require actual accepted drafts, valid output block lengths, contiguous
stream indices, scored requests, idle/resume, cache recovery, and graph replay
without fallback. Baseline identity includes the speculative block size.

The [B300 native regression run](https://modal.com/apps/ekzhang/main/ap-43zIVOqvQYM3QlwkHjUPBt)
passes 54 tests. New cases cover poisoned unused compressor history, rejected
tails, cache reuse, block sizes 1/2/5, and scaled packed indexer writes in eager
execution and CUDA graphs. The latter check every cache row outside the intended
destination: missing the indexer's fourfold row scale previously corrupted other
fields in the shared allocation. Unused compressor values also previously
allowed `0 * NaN` to poison pooling. Fresh baseline identities reject results
from before these fixes.

The corrected [four-B200 DSpark graph run](https://modal.com/apps/ekzhang/main/ap-hHBExVMCBpSBYApmCbaZXw)
passes all four layouts with `K=5`, EP2 and expert TP2:

| Attention layout | Verified blocks | Accepted draft tokens | Decode replays per rank |
| --- | ---: | ---: | ---: |
| CP2, replication2 | 14 | 22 | 13 |
| CP2, attention TP2 | 24 | 12 | 23 |
| Attention DP2, attention TP2 | 36 | 30 | 24 |
| Attention DP2, CP2 | 36 | 36 | 22 |

Block and acceptance counts sum request drivers. Each rank captures 258 prefill
regions and two decode graphs, with 344 prefill replays and zero fallbacks.
Every driver reproduces all 16 probe token IDs after cache reuse with finite
logprobs, recovers all 64 cache parents, and completes cancellation. Both
attention-DP layouts exercise idle replicas resuming work. The report is
`artifacts/deepseek-v4-speculative-graphs4-scaled.json`; durable diagnostics are
in `composable-20261021T050825Z`. Cross-layout token comparisons remain mixed,
so this validates execution and cache behavior rather than numerical parity.

The corresponding two-B300 [graph](https://modal.com/apps/ekzhang/main/ap-r3gi0zB0nAOp5RXZVk3oQp)
and [eager](https://modal.com/apps/ekzhang/main/ap-CDdnKrsYNa9xGUy7bbvV8O)
runs pass EP2 with attention TP2 and EP2 with attention DP2. The TP control
matches all 16 legacy DSpark token IDs. Both modes exercise accepted drafts,
scored replay, cache recovery, cancellation, and attention-DP idle/resume.
Reports are `artifacts/deepseek-v4-speculative-graphs2-scaled.json` and
`artifacts/deepseek-v4-speculative-eager2-scaled.json`.

The [four-B200 eager rerun](https://modal.com/apps/ekzhang/main/ap-BpvBJgbBZsasqTEDY8Ym8D)
passes the same four layouts after eager combine was changed to preserve expert
owner summation order. Before that correction, CUDA atomic indexed addition
changed prefill scores across repeated requests and one attention-DP replica
changed its greedy output. All six driver probe pairs now reproduce token IDs
and chosen/top-3 FP32 score bits exactly. The shared NCCL tests also repeat a
cancellation-sensitive owner sum 24 times per rank, including empty sources and
replicated attention. The report is
`artifacts/deepseek-v4-speculative-eager4-ordered.json`, with durable diagnostics
in `composable-20261021T052302Z`.

The targeted [four-B200 scratch-page recheck](https://modal.com/apps/ekzhang/main/ap-OTFblir1fPSMoeXBhQFCKQ)
passes both attention-DP layouts with exact scored replay, zero fallbacks, full
cache reclamation, and the null parent observed unchanged on all four ranks.
Its report is `artifacts/deepseek-v4-speculative-graphs4-scratch.json`.

The final [two-B300 live-padding run](https://modal.com/apps/ekzhang/main/ap-YfbNxt9uDrkig2v15CVqzL)
forces one live request on one attention replica and two on the other. Both
ranks replay the `(2, 512)` decode graph six times with asymmetric live batches;
the null parent is still zero after the completed workload. All three added
requests emit eight tokens, both isolated 16-token probes repeat their token
IDs and chosen/top-3 score bits exactly, all 64 cache parents are reclaimed, and
no eager fallback occurs. This run verifies actual padded replay rather than
capture alone. It records 64 verified blocks and 42 accepted draft tokens in
`artifacts/deepseek-v4-speculative-live-padding.json`, with durable diagnostics
in `composable-20261021T053758Z`.

Asynchronous communication overlap and a cross-replica HTTP request router
remain separate work.
