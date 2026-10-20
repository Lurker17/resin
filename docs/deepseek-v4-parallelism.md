# DeepSeek-V4 Flash: EP and attention DP

The CUDA runner accepts explicit groups prepared from `resin.Parallelism`:

```python
from resin import Parallelism
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

# Initialize the default NCCL world first, with one process per GPU.
groups = ParallelGroups.create(Parallelism(world_size=4, ep=4, attn_dp=4))
runner = DeepSeekV4CudaRunner(
    checkpoint_path,
    max_seq_len=4096,
    max_batch_size=16,
    parallel_groups=groups,
    device=f"cuda:{local_rank}",
    cuda_graph_prefill="disabled",
    cuda_graph_decode="disabled",
    speculative_tokens=0,
)
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker)
    # Submit this replica's requests, then collectively step/close all drivers.
else:
    worker.serve()
```

The current explicit-group path requires full-width EP (`ep == world_size`),
attention TP one, and CP one. Expert TP is one. Attention DP may be any divisor
of the world size; remaining ranks replicate attention within the same request
replica. The existing `tensor_parallel=` entry point retains its TP behavior.
The two construction modes are mutually exclusive.

## Ownership and execution

Attention projections, vocabulary shards, indexers, and logits use their
injected attention-TP context. Expert ownership uses the EP coordinate.
Checkpoint loading reads only the owned expert tensors and applies the separate
attention-TP slices. Preparation replaces checkpoint expert containers with
local kernel storage and an EP dispatcher. Model execution no longer reads
process-global attention rank or width.

For each MoE layer, the dispatcher sends a hidden row once to each distinct
selected expert owner. The owner executes its existing local FP4 kernel and
returns partial rows; the source combines them in its own token order. Replicated
attention dispatches from one copy and broadcasts the combined result to the
other copies. Shared experts remain local. Independent DP requests are never
summed together.

Every attention-DP replica has its own Rust scheduler, logical cache pool, and
request namespace. Physical cache capacity is agreed within that request
replica. Prefill aligns the number of per-sequence model calls across replicas;
missing rows and completely idle replicas join zero-token expert rounds without
running attention or modifying KV state. Decode retains each replica's own
batch size.

`parallelism_capabilities` declares supported axes. Preparation additionally
rejects unsupported combinations before CUDA work, including CP, expert TP,
attention TP in the explicit-group path, CUDA graphs, and speculative decoding.

## Validation

```bash
uv run pytest -q
uv run modal run deploy/modal_deepseek_v4.py --mode validate-parallel \
  --parallel-baseline --report-path artifacts/deepseek-v4-parallel-validation.json
```

The Modal command prefers four B300 GPUs and can fall back to four B200s. It uses
the complete pinned Flash checkpoint.
It checks NCCL routing against an independent arithmetic reference, EP-only
replicated attention, and independent attention-DP request engines. The model
workload includes unequal prompts and queues, idle ranks, continuation prefill,
active and queued cancellation, cache reuse, and generated-token logprobs.
An optional eager TP4 run records a baseline token comparison; different
quantized execution layouts can change greedy tokens, so that comparison is
reported separately from the routing and lifecycle assertions.

This is an eager correctness implementation. Dispatch reads variable split
counts on the host and allocates communication buffers per layer. CUDA graph
capture, asynchronous communication overlap, expert TP, CP, DSpark, and a
cross-replica HTTP request router remain outside this path.

Blackwell validation remains pending: the attempted four-GPU jobs did not
receive an allocation. The shared dispatcher and independent request engines
have passed [full GPT-OSS-20B validation on two H200s](gpt-oss-parallelism.md#completed-run).
That evidence does not validate DeepSeek-V4's model, cache state, or Blackwell
kernel integration.
