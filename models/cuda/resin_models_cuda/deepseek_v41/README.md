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
| `cache.py` | Typed zero-copy views over Engine-owned byte slabs and live page tables. |
| `cuda.py` | Adapt the shared V4 serving runtime and prepare graph regions. |
| Shared Resin runtime | Scheduling, cache admission, graph lifecycle, sampling, scores and distributed steps. |

Preparation fuses Q-A/KV and Q-B/index-Q. BF16 vocabulary/Markov GEMMs return FP32
logits; FP8 projections use per-token block32 casts and packed UE8M0 TMA scales.
C2 combines BF16 gate/value projections with FP32 accumulation and pools on the
Engine's paged FP32 state. Only completed compression windows update packed KV
or index keys; rejected speculative rows cannot overwrite accepted earlier state.

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
prepared weight layouts and projection fusion. Run the family tests with:

```bash
uv run pytest models/cuda/tests/test_deepseek_v41*.py -q
```

Independent B300 operator checks cover paged indexing/CED, projections, compressor
pooling, packed attention, fused stores and vocabulary heads, including graph replay:

```bash
uv run modal run deploy/modal_v41_forward_probe.py \
  --output artifacts/benchmarks/deepinfer-v41-forward-probe.json
uv run modal run deploy/modal_deepinfer_kernels.py --mode mega-moe \
  --output artifacts/benchmarks/deepinfer-v41-mega-moe-probe.json
```

Validate the full checkpoint with natural expert routing and acceptance:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --ep 8 \
  --expert-backend deepgemm --draft-tokens 3 --natural-acceptance \
  --no-balanced-experts --validate-model --prefill-chunk-size 8 \
  --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepinfer-v41-ep8-k3-validation.json
```

The [B300 forward probes](https://modal.com/apps/ekzhang/main/ap-AMkTFWk8QDJEM5Dg1cRdQd)
and [head probes](https://modal.com/apps/ekzhang/main/ap-CRERPEvWiUcxDXtsYWV0Wn) pass
all six families, including merged/unmerged Q layouts, C1/C2 paged indexing, C2
state across chunk boundaries, packed attention, stores, FP32 logits and exact
graph replay. Full-checkpoint parity and throughput qualification remain in progress.
Short-prompt parity does not establish model quality or representative acceptance.

## Benchmark status and remaining work

The [historical 128K EP8 sweep](https://modal.com/apps/ekzhang/main/ap-zYfGn4MBfzHHNDRSsHqnXO)
used commit `41e7814`, K4 with three forced draft acceptances and balanced experts.
It reached 1,430.90 output tokens/s/GPU at batch256/GPU. Those measurements predate
the paged FP4 indexer, fused dense projections, cross-block lazy HC and Engram
prefetch; they do not measure the new forward pass. Raw records and PNG/SVG plots
remain under `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-128k/`.
New throughput measurements are pending; see [the benchmark protocol](../../../../docs/benchmarks.md).

The runner accepts text token IDs and rejects image tokens. DSpark uses fixed-width
greedy proposals up to the released five-token block; its confidence head is loaded
but does not choose proposal length. Decoder bounded prefill replay, adaptive draft
length, host RDMA Engram, wider EP and broader quality workloads remain separate work.
