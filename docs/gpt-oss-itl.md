# GPT-OSS runner status

Resin contains a synchronous GPT-OSS reference runner and a Hopper-specific CUDA
runner for released GPT-OSS 20B and 120B checkpoints. This document describes
the current code path, not historical benchmark results.

## Reference runner

`resin.backends.gpt_oss.reference.GptOssRunner` is the readable PyTorch path. It owns
per-request model cache in Python and implements synchronous `forward(plan)`.
Equal-length rows execute as a physical batch; unsupported heterogeneous shapes
fail rather than silently running serially under a batched label.

The reference runner is useful for model loading, greedy-output comparison, and
small correctness work. It does not exercise Resin's managed cache pool.

## CUDA runner

`resin.backends.gpt_oss.cuda.GptOssCudaRunner` is the managed path. Its relevant
constructor surface is:

```python
runner = GptOssCudaRunner(
    model,
    cache_parent_capacity=...,
    revision=None,
    device=None,
    dtype=None,
    moe_kernel=None,
    cuda_graph_decode_shapes=((1, full_page_table_width),),
)

engine = Engine(runner, eos_token_id=runner.eos_token_id)
```

The runner requires SM90 and BF16 activations. It loads the model explicitly;
importing `resin` alone does not import PyTorch or initialize CUDA.

## One cache path

The CUDA runner owns this frozen layout:

```python
CacheLayout(
    cache_parent_capacity,
    [
        CacheGroupLayout(128, retention="history"),
        CacheGroupLayout(128, retention="window", window_tokens=128),
    ],
)
```

Group zero is History for full-attention layers. Group one is Window for
sliding-attention layers. Both use one block per parent and have equal byte
geometry. The runner allocates one physical parent slab with a zero/null row;
the History and Window views alias those same bytes. Native group binding ensures
that a live parent is interpreted as one group, never both simultaneously.

The Engine passes one `cache_pool_id` and two ordered sparse
`CacheGroupPlan`s. History rows contain the chronological prefix. Window rows
contain the chronological retained suffix even though their private physical
blocks cycle. The runner derives FA4 tables and bounded prefill rounds from these
rows and validates every append page and offset before launching device work.

There is no second full/sliding allocation protocol, no restore-copy metadata,
and no separate cache configuration on `Engine`.

The current Window storage remains private to each request. There is no content
index or publication path, so GPT-OSS performs no cross-request prefix reuse
today.

## Execution path

The managed runner keeps policy boundaries narrow:

- Resin schedules phase-homogeneous prefill or decode batches and freezes
  physical cache destinations.
- The runner groups compatible work into derived 128-token execution rounds.
- Full and sliding attention consume their corresponding group views through
  FlashAttention-4.
- The model's MXFP4 expert path may use the configured FlashInfer fused-MoE
  adapter.
- Greedy sampling remains runner-owned.
- Configured decode shapes use Resin's reusable CUDA graph cache: the backend
  declares stable tensors and one device launch, then the shared cache stages,
  warms, captures, validates, and replays it automatically.

CUDA graphs are strictly a current-batch execution optimization. The common
graph layer stages the current Rust-built decode plan into stable buffers; it
does not plan a batch ahead, reserve future pages or recurrent state, or enable
overlap scheduling. Prefill and unconfigured decode shapes use the eager path.

The runner implements synchronous `forward(plan)`. It materializes the dense
token-ID result before returning; CUDA failures establish device
quiescence before reporting a retryable exception, or mark the exception
post-accept when quiescence cannot be proven. Resin retains the accepted result
and ownership leases until `Engine.poll()` commits it.

## Capacity and shapes

`cache_parent_capacity` is one shared physical-parent budget, not a capacity per
attention type. A positive page ID addresses one row in the aliased parent slab;
page zero is the initialized null row.

Each `cuda_graph_decode_shapes` entry is `(batch_size,
full_page_table_width)`. The runner selects the smallest configured width that
fits the current decode batch. `cuda_graph_stats` reports capture, replay, and
fallback counts.

GPT-OSS currently pins 128-token cache blocks, a 128-token sliding window,
64 query heads, eight KV heads, and head dimension 64. Unsupported checkpoint or
attention geometry is rejected during runner construction.

## Validation status

The local code, host-side contract tests, and Modal GPT-OSS harness have been
migrated to the flat History/Window plan. The harness has not been launched.
Earlier measurements exercised a different cache protocol and are intentionally
not carried forward as evidence.

A new GPU proof should establish, on the current tree:

1. cold prefill and multi-step decode correctness against the reference runner;
2. long prefill across repeated private Window reuse;
3. separate prefill/decode scheduling and request retirement;
4. flat-pool exhaustion and complete parent recovery;
5. post-accept failure quarantine;
6. graph/eager output equivalence for every configured decode shape;
7. only then, fresh latency and throughput measurements.

Until that run exists, the accurate claim is that GPT-OSS is implemented on the
new local code path, not that the migrated path has a new remote performance
result.
