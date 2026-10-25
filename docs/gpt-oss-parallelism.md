# GPT-OSS: composable attention and expert parallelism

The Hopper runner supports GPT-OSS 20B/120B with BF16 activations and MXFP4
experts. Model structure and checkpoint preparation live in
`resin_models_cuda.gpt_oss`; shared Resin owns scheduling, page allocation,
staging, sampling, graph lifetime, recipes, and serving.

The Hopper runner accepts explicit attention and expert meshes:

```python
from resin import Parallelism
from resin.backend.topology import ParallelGroups
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

# Initialize NCCL first, with one process per GPU.
groups = ParallelGroups.create(
    Parallelism(world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2)
)
runner = GptOssCudaRunner(
    checkpoint_path,
    device=f"cuda:{local_rank}",
    parallel_groups=groups,
    kernel_moe=FlashInferSM90MXFP4Kernel(),
    cuda_graph_prefill="breakable",
    cuda_graph_decode="full",
)
runner.capture_cuda_graphs()  # Collective startup, before workers serve.
```

This layout has two request engines, attention TP two, CP two, and expert TP
four. The attention and expert meshes share the same eight ranks. Attention
replication is the remaining factor in `world_size / (attn_dp * cp * attn_tp)`;
CP does not add another factor to expert TP, which is `world_size / ep`.
See [distributed execution](distributed.md) for drivers, workers, and stepping.

All declared axes compose through `parallel_groups=`. The legacy
`tensor_parallel=` construction remains available and is mutually exclusive
with explicit groups. Preparation checks model and kernel geometry: attention
TP must divide query/KV heads, EP must divide the global expert count, and expert
TP partitions complete 32-value MXFP4 groups. Intermediate shards may have
unequal aligned widths. The SM90 adapter requires at least four local experts.
A numerically valid `Parallelism` value alone does not guarantee valid model
geometry or an accepted kernel configuration.

After serving has stopped on every rank:

```python
runner.close_cuda_graphs()  # Release captured NCCL references first.
groups.close()
```

## Cache and admission

One History group and one Window group share a physical slab pool. Both use
128-token blocks. Sliding attention reads 128 tokens; physical window retention
is `8192 + 128 - 1` tokens so an 8192-token prefill chunk preserves its earliest
query's history. The native Engine enforces that write bound and supplies
chronological page rows and exact append destinations.

`cache_parent_capacity` caps the total parent budget across both groups. The
default budget comes from available device memory and `gpu_memory_utilization`;
page zero is reserved for inactive rows. `prefix_cache=True` enables aligned,
exact-token reuse with copy-on-write as the window advances. The Modal HTTP
deployment enables it. See [cache ownership](cache-pool.md) and the current
[benchmark measurements](benchmarks.md).

## Preparation and execution boundaries

The router retains every global expert. Checkpoint preparation independently
slices attention heads, the expert ownership axis, and expert intermediate
channels. All six MXFP4 weight/scale/bias tensors receive their appropriate
slices before kernel packing. Gate/up biases follow intermediate channels;
down-projection bias belongs only to expert-TP lane zero. GPT-OSS's local
`MXFP4Config` describes the expert count, hidden and intermediate widths, top-k,
and SwiGLU limit. `MXFP4Weights` carries the six checkpoint tensors into
FlashInfer preparation; the prepared weights retain their configuration.

`ExpertParallelDispatch` is shared with DeepSeek-V4.1. With explicit groups it
preserves each attention rank as an independent origin, sends selected rows to
every intermediate shard of their expert owner, and combines all expert/TP
partials back into complete origin rows. This result must not be reduced again
across attention TP. Sampling broadcasts across `groups.device_replica` so all
ranks executing the same logical requests agree on selected tokens.

The GPT callback maps global IDs to local IDs and marks foreign routes inactive
without renormalizing weights. Its concrete FlashInfer `forward_partial` takes
hidden rows, expert IDs, routing probabilities, and prepared weights. It retains
BF16 GEMMs, expands fixed assignment slots, applies output bias and routing
probabilities in FP32, and preserves FP32 owner sums through communication. Only
the combined origin result is cast to the activation dtype. Packing and kernel
invocation live in the GPT-OSS implementation.

## Context parallelism

Prefill partitions the packed token axis into balanced contiguous CP segments.
Each rank embeds, normalizes, and executes FFNs on its local hidden rows. Each
attention layer gathers rotated local K/V back into global packed order and
writes the full cache on every CP copy. Local queries use filtered sequence
metadata with K lengths ending at their own query segment; this preserves
causal and sliding-window attention without changing global positions. A rank
with no local queries still joins K/V and expert exchanges. Final hidden rows
are gathered before selecting request-final tokens.

Decode runs the ordinary replicated attention path on CP ranks. KV storage is
replicated across CP in both phases, so adding CP does not increase logical
cache capacity. Each attention-DP replica retains its own scheduler, request
namespace, and cache pool.

## CUDA graphs and current costs

The shared runner captures prefill FFN regions and QKV projection with RoPE.
`prefill_attention` reports those projection regions separately; FA4 attention,
CP gathering, KV writes, and the output projection remain eager. Projection
graphs use exact token buckets to preserve eager GEMM row geometry.
Full decode captures expert communication and sampling. All ranks capture
matching catalogs before serving. Replicas agree on an expert/decode bucket or
collectively fall back to eager execution. Projection buckets are local to the
attention-TP group. Device valid-row scalars mask expert padding. Graph readiness is checked
before the coordinator's execution vote. Requested logprobs use retained decode
logits after replay.

Eager dispatch sends each row only to its selected expert owners and their
expert-TP lanes, reading split counts on the host. Captured dispatch uses
GPU-driven sparse NVLink all-to-all and FP32 combine. Static receive slots remain
`world_size * capacity`, but inactive sentinels prevent padding from selecting
expert GEMMs, and only selected owner ranks receive payloads. One attention
TP/replication copy dispatches each origin, then shares the completed result;
CP prefill segments remain independent. CP decode copies still dispatch separate
origins. See [transport limits](distributed.md)
for the 64-rank NVLink-domain bound and cross-host restrictions. Communication
overlap and a native FP32-partial kernel remain separate work.

Idle prefill replays only matching FFN regions. Idle decode replays padded rows
using reserved null cache storage, without writing live request pages.

## Hopper validation

The [prefill QKV graph run](https://modal.com/apps/ekzhang/main/ap-UxCximYPdg8xJM0R4gsUap)
passes on two H200s with EP2+CP2 and EP2+attention-DP2. Each rank captures
144 QKV/RoPE shapes. The CP ranks replay 593 regions each; the independent DP
replicas replay 449 and 593. All first/last-layer projection probes match eager
execution exactly on nonzero inputs at 1, 16, and 32 rows, including changed
inputs at 16 rows. Partial tails use the intentional `padding` fallback.
The workloads vary contexts and physical page maps, preserve finite logprobs,
and recover all 128 cache parents after cancellation and reuse. Full-model
comparisons pass the existing numerical limits in both layouts.

The report is `artifacts/gpt-oss-prefill-qkv-graphs.json`, stored durably at
`resin-gpt-oss-profiles/gpt-oss-composable2-20261021T063209Z.json`.
An intermediate TP2 probe still exceeds the full-model numerical limits;
those values vary from the earlier TP diagnostics. Exact same-bucket projection
checks do not qualify full-model accuracy across quantized TP layouts.

The completed two-H200 runs below establish eager and graph-enabled EP2 with
attention DP one and two for the preceding implementation. The expanded
composable implementation has real two/eight-H100 full-model graph runs.
All greedy probes match, including the combined attention DP2, attention TP2,
CP2, EP2, expert TP4 layout. Tensor-parallel layouts exceed the fixed numerical
limits, so these runs establish graph/lifecycle execution rather than a complete
accuracy qualification. Those results are separate from the older two-GPU proof
and from DeepSeek-specific Blackwell validation.

The reports below are historical qualification evidence. Their dedicated Modal
validation launcher has been retired. The runs used the pinned full
GPT-OSS-20B checkpoint, independent single-GPU baselines, and graph-enabled
layouts combining attention TP, attention DP, CP, EP, and expert TP. Their
checks covered logits, greedy output, logprobs, unequal queues, idle/resumed
replicas, cancellation, cache reuse, and graph teardown; they did not establish
throughput or MFU.

Use the shared benchmark CLI for current performance measurements:

```bash
uv run scripts/bench.py --backend modal --model gpt-oss-20b \
  --gpu H100 --world-size 2 --attn-dp 2 --attn-tp 1 --cp 1 --ep 2 \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/gpt-oss-ep2.jsonl --trace-dir bench/gpt-oss-ep2-traces
```

This measures the requested topology; it does not repeat the retired
cross-topology numerical qualification. Shared transport and runtime contracts
remain covered by the normal GPU tests.

### Quantized tensor-parallel numerics

The two-H100 composed pair matrix passes EP2+CP2 and EP2+attention-DP2 within
the fixed max-absolute 0.5 / relative-L2 0.02 logit limits. Layouts involving
attention TP or expert TP exceed some limits despite matching greedy probes;
the diagnostic report retains those failures.

The single-H100 real-weight expert probe reproduces the captured first-layer
output exactly. EP2/EP4 also reproduce it exactly; expert TP2/TP4 differ by
relative L2 0.00236/0.00210 (max absolute 0.25). An independently dequantized
dense reference shows comparable sensitivity when only FC2 shard rounding
changes. FlashInfer's raw FC2 output is BF16; FP32 routing/communication cannot
recover rounding already performed inside each intermediate shard. Attention
TP uses FP32 GEMM partials and reduction, casting once before residual addition.


`artifacts/gpt-oss-expert-probe.json` records the real-weight comparison.
`artifacts/gpt-oss-composable-pairs-fp32-attention-diagnostics.json` retains the
complete pair-matrix failures. These checks do not establish model-quality
equivalence across quantized TP layouts.

The final [eight-H100 all-axes run](https://modal.com/apps/ekzhang/main/ap-1WwRBajIhnJ3VleKXrA8FQ)
uses `Parallelism(world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2)`.
Every rank captures 144 FFN regions and two full decode shapes, then replays
792 prefill regions and 26 decode graphs with zero eager fallbacks. Both
request engines recover all 128 cache parents; all four eight-token greedy
probes match the independent single-GPU baseline. Five fixed logit comparisons
still fail. The complete diagnostic report is
`artifacts/gpt-oss-composable-all-axes-diagnostics.json`, also stored at
`resin-gpt-oss-profiles/gpt-oss-composable8-20261021T013452Z.json`.

### Sparse transport qualification

The [four-H100 sparse full-model run](https://modal.com/apps/ekzhang/main/ap-N0wMpfwNbuzaeQUuQH7jft)
completes EP2+attention-DP2+attention-TP2 and EP2+CP2+attention-TP2 with
matching greedy probes and actual prefill/decode graph replays. Nine fixed
numerical comparisons fail, retained in
`artifacts/gpt-oss-sparse-expert-parallel.json`; this is not a full-model accuracy
pass. A [bounded real-checkpoint expert probe](https://modal.com/apps/ekzhang/main/ap-EvzcAxZHhzbykqH3zrQWC8)
separately verifies the same prepared shards: sparse eager equals variable-size
eager exactly, and three captured replays are bitwise identical on every rank,
including idle origins. Its strict unsplit expert-block comparison also passes.

Unchunked prompt logprobs match the actual raw output-head distribution within 1.91e-6.
However, [single-GPU chunked versus unchunked scoring](https://modal.com/apps/ekzhang/main/ap-1Roe4kUsfpXJXV9f3dMY8x)
differs by up to 5.15 nats, independently of EP or TP; cross-chunk numerical
equivalence remains unqualified. A [first-layer attention check](https://modal.com/apps/ekzhang/main/ap-FasvATBT14nPXXMCnH0stM)
using actual Q/K/V and FP32 causal/window attention with the learned sink passes
all seven chunk shapes (maximum absolute error 0.01993, relative L2 below 0.00174).
These checks isolate transport and first-layer attention, without identifying
the cause of the full-model batch-shape sensitivity.

Durable reports in `resin-gpt-oss-profiles` are
`gpt-oss-composable4-20261022T031241Z.json`,
`expert-transport-isolation-20261022T032942Z.json`, and
`prompt-scoring-isolation-20261022T033120Z.json`. The last report's matching `.pt`
contains the first-layer attention inputs and outputs for replay without another
checkpoint load. These files preserve evidence from the retired one-off
diagnostics; current benchmarks do not rerun those diagnostic workflows.

### Completed graph run

On October 20, 2026, the [graph validation run](https://modal.com/apps/ekzhang/main/ap-xZU7DoXI7fIgUyVCtk8vJj)
passed on two H200s with the same pinned full checkpoint:

| Layout | Coordinated steps | Prefill region replays per rank | Full decode replays per rank | Eager fallbacks |
| --- | ---: | ---: | ---: | ---: |
| EP2, attention DP1 | 54 | 720 | 24 | 0 |
| EP2, attention DP2 | 59 | 792 | 26 | 0 |

Every rank captured 144 prefill regions and two decode shapes. The DP2 case
included 17 idle-rank calls: idle replicas replayed 288 FFN regions and five
full decode graphs, then resumed independent requests. Queued and active
cancellation, finite generated-token logprobs, and cache reuse passed. All three
request engines reclaimed 128 of 128 cache parents. All six ordinary eight-token
greedy probes matched their single-GPU baseline.

Checkpoint logit comparisons passed the existing fixed tolerances: worst
absolute error was 0.125 and worst relative L2 error was 0.00936, below the 0.02
limit. Five independent GPU tests passed, covering eager and captured NCCL
dispatch, FP32 cancellation across owners, and real SM90 partial-expert replay.
Captured resources were explicitly released and process-group shutdown completed.

The local report is `artifacts/gpt-oss-parallel-graphs-validation.json`.
A durable copy is in the `resin-gpt-oss-profiles` Modal volume at
`gpt-oss-ep-graphs-20261020T204020Z.json`. These bounded correctness checks do
not measure wide-EP throughput or MFU.

### Completed eager run

On October 20, 2026, the [Modal run](https://modal.com/apps/ekzhang/main/ap-wpLVi6yZjYV2ELZOqqylBp)
passed on two H200s with the full checkpoint revision
`6cee5e81ee83917806bbde320786a8fb61efebee`:

| Layout | Coordinated steps | Captured first-layer expert max error | Eight-step same-history logit max error | Worst separate prefill logit max error |
| --- | ---: | ---: | ---: | ---: |
| EP2, attention DP1 | 51 | 0 | 0 | 0.125 |
| EP2, attention DP2 | 56 | 0 | 0 | 0.125 |

Worst prefill relative L2 error was 0.00873, within the fixed 0.02 limit.
All six ordinary eight-token greedy replays matched the single-GPU baseline.
Independent single-GPU baseline repeats were exact. Both NCCL reference tests
passed, including all-empty exchanges and replicated attention.

The DP2 workload exercised unequal queues, 14 idle-rank calls, queued and active
cancellation, and finite generated-token logprobs. Every request cache reclaimed
all 128 parents after the workload and repeated probes. These are bounded
correctness checks, not throughput or MFU measurements.

The local report is `artifacts/gpt-oss-parallel-validation.json`. A durable copy
is in the `resin-gpt-oss-profiles` Modal volume at
`gpt-oss-ep-20261020T200558Z.json`.
