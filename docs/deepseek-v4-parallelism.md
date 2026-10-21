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
    speculative_tokens=0,
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
exclusive with explicit groups. Explicit-group execution currently requires
`speculative_tokens=0`.

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

## Validation

```bash
uv run pytest -q
uv run modal run deploy/modal_deepseek_v4.py --mode validate-parallel \
  --parallel-baseline --report-path artifacts/deepseek-v4-parallel-validation.json
```

The existing DeepSeek validation command uses the complete pinned Flash
checkpoint and eager four-GPU EP/attention-DP cases. It exercises independent
request engines, unequal queues, idle ranks, continuation prefill, cancellation,
cache reuse, and generated-token logprobs. An optional TP4 run reports token
comparisons separately because quantized layouts can change greedy choices.
It is not a completed GPU validation of the new composed layouts or graphs.

Blackwell model/kernel validation remains pending; the earlier four-GPU jobs
did not receive an allocation. The shared transport and independent request
engines passed the preceding [eager two-H200 GPT-OSS run](gpt-oss-parallelism.md#completed-eager-run).
Shared graph orchestration also passed the preceding
[captured GPT-OSS run](gpt-oss-parallelism.md#completed-graph-run), including idle
attention-DP replay in both phases. The expanded eight-GPU GPT-OSS graph matrix
is pending. CPU tests cover composed expert shards and DeepSeek CP/cache control
flow, but none of this substitutes for DeepSeek-specific Blackwell validation.

Asynchronous communication overlap, explicit-group DSpark, and a cross-replica
HTTP request router remain separate work.
