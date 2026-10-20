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
    cuda_graph_prefill="breakable",
    cuda_graph_decode="full",
    speculative_tokens=0,
)
# Collective startup operation, before any rank starts serving requests.
runner.capture_cuda_graphs()
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker)
    # Submit this replica's requests, then collectively step/close all drivers.
else:
    worker.serve()
# All ranks reach this after their serving loops have stopped.
runner.close_cuda_graphs()
groups.close()
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
missing rows and completely idle replicas join matching expert rounds. Eager
empty rounds do not run attention or modify KV state. Captured empty prefill
replays only FFN regions; captured empty decode runs padded rows backed by
reserved null/scratch cache storage. Decode retains each replica's own logical
batch size.

`parallelism_capabilities` declares supported axes. Preparation additionally
rejects unsupported combinations before CUDA work, including CP, expert TP,
attention TP in the explicit-group path, and speculative decoding.

## CUDA graphs

The explicit-group runner uses the shared CUDA graph machinery for prefill FFN
regions and full decode. Every rank captures the same catalog collectively at
startup. Replicas agree on physical buckets before GPU work; if any replica
cannot use a configured shape, all use eager execution for that step. DeepSeek's
per-sequence prefill calls and empty participation remain aligned across the
domain. Requested decode logprobs use logits retained from graph replay.

Graph expert dispatch all-gathers a fixed capacity of source rows and
reduce-scatters local weighted expert partials back to their sources. A device
valid-row scalar masks padding, and routing metadata stays on the GPU. Attention
replication still dispatches once and broadcasts complete rows. This transport
is shared with GPT-OSS; the DeepSeek model retains its own FP4 kernel and cache
adapters.

The dense transport sends every padded row to every expert owner, giving each
owner `EP * capacity` rows. It makes shapes and collectives capturable but adds
bandwidth, scratch memory, and padded expert work. Sparse graph-safe routing and
communication overlap remain separate work. The eager path continues to pack
only selected owners and reads variable split counts on the host.

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

The current DeepSeek Modal validation command runs eager execution. Captured
DeepSeek execution is implemented but has not been validated on Blackwell.
Asynchronous communication overlap, expert TP, CP, DSpark, and a cross-replica
HTTP request router remain outside this explicit-group path.

Blackwell validation remains pending: the attempted four-GPU jobs did not
receive an allocation. The shared dispatcher and independent request engines
have passed [eager GPT-OSS-20B validation on two H200s](gpt-oss-parallelism.md#completed-eager-run).
The shared graph orchestration and transport also passed
[captured GPT-OSS-20B validation](gpt-oss-parallelism.md#completed-graph-run),
including independent attention-DP queues and idle-rank replay in both phases.
That evidence does not validate DeepSeek-V4's model, cache state, or Blackwell
kernel integration.
