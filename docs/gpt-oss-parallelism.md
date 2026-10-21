# GPT-OSS: composable attention and expert parallelism

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

## Preparation and execution boundaries

The router retains every global expert. Checkpoint preparation independently
slices attention heads, the expert ownership axis, and expert intermediate
channels. All six MXFP4 weight/scale/bias tensors receive their appropriate
slices before kernel packing. Gate/up biases follow intermediate channels;
down-projection bias belongs only to expert-TP lane zero. The kernel config
describes the local expert count and intermediate width.

`ExpertParallelDispatch` is shared with DeepSeek-V4. With explicit groups it
preserves each attention rank as an independent origin, sends selected rows to
every intermediate shard of their expert owner, and combines all expert/TP
partials back into complete origin rows. This result must not be reduced again
across attention TP. Sampling broadcasts across `groups.device_replica` so all
ranks executing the same logical requests agree on selected tokens.

The GPT callback maps global IDs to local IDs and marks foreign routes inactive
without renormalizing weights. EP kernels advertise `supports_fp32_partial` and
implement `forward_partial`. The Hopper adapter retains its BF16 GEMMs but
expands fixed assignment slots, applies output bias and routing probabilities
in FP32, and preserves FP32 owner sums through communication. Only the combined
origin result is cast to the activation dtype. Packing and kernel selection
remain inside the prepared kernel adapter.

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

The shared runner captures prefill FFN regions and full decode, including expert
communication and sampling. All ranks capture matching catalogs before serving.
Replicas agree on a physical bucket or collectively fall back to eager
execution. Device valid-row scalars mask padding. Graph readiness is checked
before the coordinator's execution vote. Requested logprobs use retained decode
logits after replay.

Eager dispatch sends each row only to its selected expert owners and their
expert-TP lanes, reading split counts on the host. Captured dispatch uses a
fixed-capacity world all-gather and FP32 reduce-scatter; every local expert
callback receives `world_size * capacity` rows, including padding and foreign
routes. Replicated attention and attention-TP ranks currently dispatch their own
origins even when their hidden rows are identical. This is a correctness
prototype with extra communication, scratch memory, activation expansion, and
duplicate expert work. Sparse graph-safe routing, communication overlap, and a
native FP32-partial kernel remain separate work.

Idle prefill replays only matching FFN regions. Idle decode replays padded rows
using reserved null cache storage, without writing live request pages.

## Hopper validation

The completed two-H200 runs below establish eager and graph-enabled EP2 with
attention DP one and two for the preceding implementation. The expanded
composable implementation has real two/eight-H100 full-model graph runs.
All greedy probes match, including the combined attention DP2, attention TP2,
CP2, EP2, expert TP4 layout. Tensor-parallel layouts exceed the fixed numerical
limits, so these runs establish graph/lifecycle execution rather than a complete
accuracy qualification. Those results are separate from the older two-GPU proof
and from DeepSeek-specific Blackwell validation.

```bash
uv run modal run deploy/modal_gpt_oss_parallel.py --graphs \
  --report-path artifacts/gpt-oss-parallel-graphs-validation.json

uv run modal run deploy/modal_gpt_oss_parallel.py --composable --world-size 8 --gpu-type H100 \
  --report-path artifacts/gpt-oss-composable-graphs-validation.json

# Recheck only the layout using every axis.
uv run modal run deploy/modal_gpt_oss_parallel.py --composable --world-size 8 --gpu-type H100 \
  --parallel-cases all-axes
```

The composable command uses the pinned full GPT-OSS-20B checkpoint, independent
single-GPU baselines, and graph-enabled layouts combining attention TP, attention
DP, CP, EP, and expert TP. It also runs NCCL transport, real SM90 partial-expert,
and real FA4 CP attention comparisons. The full-model workload checks same-history
logits, greedy output, logprobs, unequal queues, idle/resumed replicas,
cancellation, cache reuse, capture/replay counts, and orderly graph teardown.
These are bounded correctness checks; they do not establish throughput or MFU.

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

```bash
uv run modal run deploy/modal_gpt_oss_parallel.py --expert-probe --gpu-type H100 \
  --expert-reference-path /profiles/resin-gpt-oss/<run>/baseline-0-expert.pt
```

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
