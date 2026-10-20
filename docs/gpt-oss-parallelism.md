# GPT-OSS: EP and attention DP

The Hopper runner accepts the same explicit process groups as DeepSeek-V4:

```python
from resin import Parallelism
from resin.backend.topology import ParallelGroups
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

# Initialize NCCL first, with one process per GPU.
groups = ParallelGroups.create(Parallelism(world_size=2, ep=2, attn_dp=2))
runner = GptOssCudaRunner(
    checkpoint_path,
    device=f"cuda:{local_rank}",
    parallel_groups=groups,
    kernel_moe=FlashInferSM90MXFP4Kernel(),
    cuda_graph_prefill="breakable",
    cuda_graph_decode="full",
)
# Collective startup operation: every rank captures before workers serve.
runner.capture_cuda_graphs()
```

Use `DistributedWorker` and `DistributedEngine` as described in
[distributed execution](distributed.md). Each attention-DP replica owns a
scheduler and cache namespace. With `attn_dp=1`, the remaining ranks replicate
the same attention and requests; they still own different expert weights.

After every rank has left its driver loop or `worker.serve()`, release captured
NCCL graphs before destroying process groups:

```python
runner.close_cuda_graphs()  # All ranks, after serving has stopped.
groups.close()
```

The explicit-group path requires `ep == world_size` and `attn_tp=cp=expert_tp=1`.
It supports eager execution and CUDA graphs. The existing `tensor_parallel=`
path is separate.
`parallelism_capabilities` declares the supported axes, and construction checks
their combinations. The local fused kernel must also support the resulting
expert count; the Hopper adapter requires at least four local experts.

## Preparation and execution boundaries

The model retains a router over all experts. Checkpoint preparation slices the
expert axis of all six MXFP4 weight, scale, and bias tensors before converting
them to the local kernel's storage. Attention TP and expert ownership use
separate coordinates. The kernel config describes only locally stored experts.

`ExpertParallelDispatch` in `resin.backend.expert_parallel` is shared with
DeepSeek-V4. It sends each row once per selected expert owner, passes global
routes to a local callback, and returns combined rows in their original order.
GPT-OSS's callback maps owned expert IDs to local IDs and marks other routes
inactive without renormalization. The existing kernel adapter retains ownership
of MXFP4 packing and the fused kernel call.

EP kernels must advertise `supports_fp32_partial` and implement
`forward_partial`: weighted local expert sums in FP32, including for empty
inputs. Dispatch preserves that dtype through the return exchange and global
combine, then casts to the activation dtype. Rounding each owner's sum to BF16
before the exchange introduces an extra rounding step and can change routing in
later layers.

The Hopper adapter keeps its existing BF16 GEMMs. Internally it expands fixed
assignment slots into rows with one expert each, then applies the prepared output
bias, routing weights, and reduction in FP32. Packed weights are shared with
ordinary execution; only a small zero-bias tensor is added during preparation.
Inactive assignments use the kernel's expert sentinel. Expansion and chunking
have static shapes, respecting the kernel's token limit without reading route
counts on the host.

GPT-OSS packs one attention batch per step, so every rank executes one expert
round per layer. Eager idle replicas execute empty expert rounds without
attention or KV writes. Captured idle prefill replays only FFN regions; captured
idle decode uses padded model rows backed by reserved null cache storage.
Routing tensors have a consistent collective dtype even when the model's router
implementation changes.

## CUDA graph execution

The shared CUDA runner captures prefill FFN regions and full decode, including
expert communication and sampling. All ranks call `capture_cuda_graphs()` before
serving with matching catalogs. At each step, replicas agree on a bucket or
collectively fall back to eager execution. Unequal batches and idle replicas
use a device valid-row scalar to mask padding. Requested logprobs are formatted
from the retained decode logits after replay.

Captured expert dispatch uses a fixed-capacity all-gather and FP32
reduce-scatter, followed by attention-replica broadcast. This sends padded source
rows to every owner; each owner receives `EP * capacity` rows. It avoids
host routing-count reads and dynamic shapes, at the cost of dense communication,
scratch storage, and padded kernel work. The eager transport still packs only
selected owners and reads variable split counts on the host. The Hopper adapter
also copies expanded activations. Communication overlap, sparse graph-safe
dispatch, and a native FP32-partial kernel remain future work.

## Hopper validation

```bash
uv run modal run deploy/modal_gpt_oss_parallel.py \
  --report-path artifacts/gpt-oss-parallel-validation.json
```

The command requests two H200s with H100 fallback and uses a pinned GPT-OSS-20B
checkpoint. It runs an independent NCCL routing reference, real fused expert
comparisons, and full-model EP2 with attention DP one and two. Single-GPU runs
provide the same-context reference logits. It also exercises unequal prompts,
idle replicas, cancellation, cache reuse, and generated-token logprobs.
The JSON report records numerical errors against fixed tolerances and greedy
token agreement. This validates the shared transport on Hopper; DeepSeek-V4's
Blackwell-specific model and kernels need their own GPU validation.

To exercise captured prefill and decode, including idle-rank replay:

```bash
uv run modal run deploy/modal_gpt_oss_parallel.py --graphs \
  --report-path artifacts/gpt-oss-parallel-graphs-validation.json
```

This mode also captures and replays the independent NCCL transport reference
and real Hopper partial-expert kernel. Full-model graph cases retain logprob
output, compare greedy decode logits with matching baseline contexts, and
require both graph modes to replay without eager fallback. The validation uses
32-token prefill buckets and bounded decode shapes.

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
