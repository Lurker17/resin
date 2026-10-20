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
    cuda_graph_prefill="disabled",
    cuda_graph_decode="disabled",
)
```

Use `DistributedWorker` and `DistributedEngine` as described in
[distributed execution](distributed.md). Each attention-DP replica owns a
scheduler and cache namespace. With `attn_dp=1`, the remaining ranks replicate
the same attention and requests; they still own different expert weights.

The explicit-group path requires `ep == world_size`, `attn_tp=cp=expert_tp=1`,
and disabled CUDA graphs. The existing `tensor_parallel=` path is separate.
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

The Hopper adapter keeps its existing BF16 GEMMs. Internally it expands active
assignments into rows with one expert each, then applies the prepared output
bias, routing weights, and reduction in FP32. Packed weights are shared with
ordinary execution; only a small zero-bias tensor is added during preparation.
The adapter chunks expanded rows to respect the kernel's token limit.

GPT-OSS packs one attention batch per step, so every rank executes one expert
round per layer. Idle request replicas execute empty expert rounds without
attention or KV writes. Routing tensors have a consistent collective dtype even
when the model's router implementation changes.

This is an eager correctness path: split counts synchronize with the host and
communication buffers are allocated per layer. The Hopper precision adapter
also copies expanded activations and returns FP32 partials. Communication
overlap, a native FP32-partial kernel, and CUDA graph capture are future work.

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

### Completed run

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
