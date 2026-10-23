# DeepSeek V4.1 Flash text runner

`DeepSeekV41CudaRunner` loads indexed safetensors from
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Serving imports Resin code;
checkpoint Python is a development reference.

The prepared forward pass follows [DeepInfer 774ce244](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01).
Its CUDA operator sources in `upstream/` retain Apache-2.0 attribution and license.
The pinned kernel stack is DeepGEMM `1fc9c669`, FlashMLA `2e5429fc`,
DeepSelect `bfa4507d`, and TileKernels 1.0.0; full revisions live in
[the CUDA package configuration](../../pyproject.toml).

## Boundaries

| Component | Responsibility |
| --- | --- |
| `model.py` | Shifted HC, shared KV/index ownership, Engram, and DSpark architecture. |
| `linear.py`, `attention.py`, `compressor.py`, `experts.py` | Lower loaded checkpoint tensors into fused projections, packed scales and prepared operators. |
| `indexer.py`, `upstream/` | Paged FP4 MQA, candidate selection and fused cache transforms. |
| `batch.py` | Prepare shared store slots, attention indices, compression lengths and state-page mappings once per forward. |
| `cache.py` | Typed zero-copy views over Engine-owned byte slabs and live page tables. |
| `cuda.py` | Adapt the shared V4 serving runtime and prepare graph regions. |
| Shared Resin runtime | Scheduling, cache admission, graph lifecycle, sampling, scores and distributed steps. |

Preparation fuses Q-A/KV and Q-B/index-Q. BF16 vocabulary/Markov GEMMs return FP32
logits; FP8 projections use per-token block32 casts and packed UE8M0 TMA scales.
C2 combines BF16 gate/value projections with FP32 accumulation and pools on the
Engine's paged FP32 state. Only completed compression windows update packed KV
or index keys; rejected speculative rows cannot overwrite accepted earlier state.

Each target or draft forward prepares one batch metadata object from live positions
and Engine parent/child page tables. Native kernels expand SWA windows, prepare
compression/state mappings and adapt selected compressed indices once per KV
source; consuming layers reuse those tensors. Graph replay reads updated inputs,
including page tables and masks. Full-history table width is a runtime dimension,
so growing context reuses the same compiled metadata and index-conversion kernels.
Prefill also shares address geometry and unpacked compressed KV across consumers
of the same owner.

FlashMLA fuses query RoPE, sparse attention, inverse output RoPE and FP8 output
casting. KV stores fuse RMS normalization, RoPE and quantization. Sliding KV uses
528-byte FP8 rows, compressed KV 288-byte FP4 rows, and index keys 68-byte FP4
rows. Decode uses DeepGEMM's paged MQA scorer; CED consumers score candidate blocks
without gathering complete key histories or forming per-head history matrices.

With `expert_backend="deepgemm"`, DeepGEMM fuses gate selection, sparse EP
transport, routed/shared experts and combine. Lazy shifted HC carries residual
updates across both attention and FFN boundaries, materializing them for Engram,
DSpark feature taps and the final output. Serial layers reuse prepared collective
workspaces. `expert_backend="flashinfer"` retains the existing expert adapter;
selecting DeepGEMM never silently falls back.

Engram tables are row-sharded over the device domain. Ordered fetch hooks issue
all lookups together. FlashInfer execution waits at each consuming layer;
DeepGEMM joins all NCCL fetches before persistent computation because its grid
barriers require the entire GPU. These are device-stream dependencies, without
host synchronization. Modal uses NCCL owner lookups with GPU-resident shards;
host-resident RDMA Engram fetch is not implemented.

## Parallelism and graphs

The runner declares attention DP, CP, EP and expert TP support. Attention TP means
KV-head sharding; this one-KV-head MLA implementation requires `attn_tp=1`.
The DeepGEMM expert backend currently requires `attn_dp=ep=world_size`, CP1 and
expert TP1. Other supported meshes use the FlashInfer expert path.

Decode uses full CUDA graphs. Prefill graphs cover projections and FFNs;
attention, indexing, cache updates and Engram hashing execute between regions.
CP partitions query attention while retaining replicated KV, compressor and FFN
work. There is no pipeline or sequence parallelism.

## Validation

CPU tests cover geometry, physical aliases, child slots, rollback, shifted HC,
prepared weight layouts, projection fusion and independent paged-address equations.
Run the family tests with:

```bash
uv run pytest models/cuda/tests/test_deepseek_v41*.py -q
```

Independent B300 operator checks cover paged indexing/CED, projections, compressor
pooling, packed attention, fused stores, vocabulary heads, HC/Engram residuals and
shared batch metadata, including graph replay:

```bash
uv run modal run deploy/modal_v41_forward_probe.py \
  --output artifacts/benchmarks/deepinfer-v41-forward-probe.json
uv run modal run deploy/modal_deepinfer_kernels.py --mode mega-moe \
  --output artifacts/benchmarks/deepinfer-v41-mega-moe-probe.json
```

Validate the full checkpoint with natural expert routing and acceptance:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 4 --attn-dp 4 --ep 4 \
  --expert-backend deepgemm --draft-tokens 3 --natural-acceptance \
  --no-balanced-experts --validate-model --prefill-chunk-size 8 \
  --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepinfer-v41-forward-ep4-k3-validation.json
```

The [B300 forward probes](https://modal.com/apps/ekzhang/main/ap-AMkTFWk8QDJEM5Dg1cRdQd)
and [head probes](https://modal.com/apps/ekzhang/main/ap-CRERPEvWiUcxDXtsYWV0Wn) pass
all six families, including merged/unmerged Q layouts, C1/C2 paged indexing, C2
state across chunk boundaries, packed attention, stores, FP32 logits and exact replay.
The [direct FP8 output probe](https://modal.com/apps/ekzhang/main/ap-AeDmb4KbJfjlNfVVL0wA0M)
checks H64/G8/R1024 at 1/5/1,024/1,280 rows: O-A values, packed scales and O-B
outputs match BF16-plus-cast bitwise, including five graph replays.

The [shared metadata B300 probe](https://modal.com/apps/ekzhang/main/ap-m1vDGbDRZUwEdde9ROc8Fd)
at `a4e62a1` checks seven shapes, including batch256 with four target queries or
three draft queries at 128K context and a 1,024-query masked prefill. All 18 output
fields match the CPU equations, including irregular masks and both position
dtypes. Five graph replays per case update positions, parent/child tables, masks
and selected indices. Table widths 1, 3 and 1,025 also match the equations while
reusing identical compiled metadata and conversion kernels. Its report is
`artifacts/benchmarks/deepinfer-v41-dynamic-metadata-probe.json`.

At clean commit `cd5ecfd`, the [full-checkpoint EP4/ADP4 B300 check](https://modal.com/apps/ekzhang/main/ap-G0MLxyy7133CqWY3HJwRjo)
passes on all four drivers with the released 384 experts and natural routing.
Each driver runs four prefill and three decode steps, replays three full decode
graphs, 160 prefill compute regions and 480 attention-projection regions. Eager
and captured execution produce identical greedy IDs, prompt logprobs,
generated-token logprobs and top-3 logprobs (maximum absolute difference zero).
This includes the shared metadata and prepared forward operators. Natural K3
verification accepts 9/9 drafts on this short counting prompt; this does not
establish model quality or representative acceptance. The report is
`artifacts/benchmarks/2026-10-22-deepseek-v41-prepared/overlap-validation.json`.

The same run checks Engine-owned GPU feedback with greedy sampling, scores and
token stops disabled. All four drivers emit 160 tokens identical to synchronous
captured execution, with 57 enqueues and 55 GPU continuations per driver. Generated
positions 26 through 185 cross a 128-token cache-page boundary, exercising the
transition back through page planning as well as continued graph replay.

The [native FP32 route EP2 probe](https://modal.com/apps/ekzhang/main/ap-72WXKqszkwgN9zSWQIn3gv)
at `14af36b` covers synthetic 16-expert, top-6 routing with widths 5,120/2,304 and
1,280 rows/rank, rather than the full 384-expert checkpoint. It checks route
precision, FFN numerical error and exact graph replay; its reports live beside
the joined-Engram probes below.

A B300 EP2 composition test with 384 experts and hidden/intermediate widths
5,120/2,304 reproduces the NCCL/persistent-grid conflict after all ranks finish
serial warmup: asynchronous Engram fetches cause a DeepGEMM grid-sync timeout
at four tokens. Joining both fetches first passes
eager execution, capture and 30 graph replays at four and 1,280 tokens, with exact
MoE and Engram outputs. [Joined results](https://modal.com/apps/ekzhang/main/ap-rZUgGEwWmUWHh5XBuCBcxC),
[overlap failure](https://modal.com/apps/ekzhang/main/ap-XVDWkUOmoD41DVkMtwSvL2), traces
and reproduction sources are archived under
`artifacts/benchmarks/2026-10-22-v41-forward-port-microprobes/`.

## Benchmark status and remaining work

The [historical 128K EP8 sweep](https://modal.com/apps/ekzhang/main/ap-zYfGn4MBfzHHNDRSsHqnXO)
used commit `41e7814`, K4 with three forced draft acceptances and balanced experts.
It reached 1,430.90 output tokens/s/GPU at batch256/GPU. Those measurements predate
the paged FP4 indexer, fused dense projections, cross-block lazy HC and Engram
prefetch; they do not measure the new forward pass. Raw records and PNG/SVG plots
remain under `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-128k/`.
The [prepared-forward EP8 baseline](https://modal.com/apps/ekzhang/main/ap-lfw5WzD1XFQrU1UZ06dAXN)
at `de438e4` completes all six 128K points with K3, two forced draft acceptances
and balanced experts. At batch256/GPU, Engine steps take 140.85 ms and device work
takes 46.47 ms; projecting 75% acceptance yields 5,906.86 output tokens/s/GPU.
At batch8/GPU, the corresponding times are 60.92/21.64 ms, exposing substantial
host coordination and planning overhead. Raw records, PNG/SVG plots and all-rank
Perfetto traces live under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-deepinfer/`.
The [Engine-overlap EP8 sweep](https://modal.com/apps/ekzhang/main/ap-gnaa8Li1vQmHzzujs8tZT4)
at `cd5ecfd` measures the same six workloads with prepared batch metadata and
continuous completion timing. Batch256/GPU takes 39.79 ms: 20,907.62 projected
tokens/s/GPU at 75% acceptance, or 19,299.34 actual tokens/s/GPU with two forced
acceptances. This is 3.54× the historical baseline. Each point records 160 native
continuations across eight drivers and zero eager fallbacks. The 48 raw rank
traces each contain two forwards; batch8/GPU drops from 5,116 to 1,436 CUDA
kernels per forward. Records, plots, source provenance and merged Perfetto traces
are under `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-overlap/`.
See [the benchmark protocol](../../../../docs/benchmarks.md) for all six points,
timing scope and the distinction between measured and projected acceptance.

The runner accepts text token IDs and rejects image tokens. DSpark uses fixed-width
greedy proposals up to the released five-token block; its confidence head is loaded
but does not choose proposal length. Decoder bounded prefill replay, adaptive draft
length, host RDMA Engram, wider EP and broader quality workloads remain separate work.
