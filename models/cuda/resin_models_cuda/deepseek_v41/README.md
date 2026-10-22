# DeepSeek V4.1 Flash text runner

`DeepSeekV41CudaRunner` loads the original, indexed safetensors for
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, pinned at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Checkpoint Python is read as a
reference during development; serving imports only Resin-owned implementations.
The pinned configuration fixtures record both public and inference geometry.

The family owns shifted hyper-connections, block32 FP8 projections, the changed
DSpark stages, Engram, and shared compressed attention. The existing V4 serving
runtime supplies scheduling, page-table staging, CUDA graph lifecycle, sampling,
prompt scores, speculative acceptance, failure handling, and distributed steps.
Prepared FP4 experts and sparse expert transport also reuse the V4 adapters.

`expert_backend="deepgemm"` selects the pinned DeepGEMM mega-MoE operation,
including sparse EP dispatch/combine and the shared expert, plus the fused gate
and attention-to-FFN shifted-HC transition. Checkpoint weights
are transformed once during preparation, and serial layers share one symmetric
workspace per expert layout. This backend currently requires
`attn_dp=ep=world_size` with expert TP1; unsupported meshes fail before loading.
`expert_backend="flashinfer"` selects the existing prepared expert path and is
the current default. Selecting DeepGEMM never silently falls back to it.

The actual prepared adapter passes on two B300s against independently decoded
FP4/FP8 FFNs, including shared experts, changing routes, balanced routes, empty
participation, and bitwise graph replay. The actual Block HC wiring also passes
the unfused checkpoint equations and graph replay. Reproduce these checks with:

```bash
uv run modal run deploy/modal_deepinfer_kernels.py --mode mega-moe \
  --output artifacts/benchmarks/deepinfer-megakernels-ep2-validation.json
```

This small synthetic expert mesh validates the integration; full checkpoint and
throughput qualification remain separate below. The current attention path uses
pinned FlashMLA fused query-RoPE, sparse attention, inverse output-RoPE and FP8
output casting, with prepared query/output projection layouts. Sliding KV stores
528-byte FP8 rows; compressed KV stores 288-byte FP4 rows. Both use typed strided
views over the Engine's shared byte slabs, preserving native page ownership.
Prefill gathers a transient BF16 history view for FlashMLA;
decode reads the packed pages directly. The packed-cache and sharded-Engram
path passes the microprobes and full-checkpoint K0/K4 checks described below.

The DeepGEMM path also fuses the BF16 gate projection and top-k directly into
the expert workspace, preserving BF16-rounded route weights and the exact
balanced benchmark routes. Within each block, `mega_mhc` fuses attention's
residual update with the shifted HC mixes, collapse and FFN RMSNorm. Block
boundaries retain the full HC residual for Engram and DSpark feature collection.

Four cache groups share the ordinary Engine slab pool: each layer's sliding KV,
source-owned compressed KV plus index keys, rolling FP32 compressor inputs, and
compressed token IDs for Engram lookback. KV consumers map to their declared
source's storage. Index-result ownership is independent of KV ownership. Future
verification rows stay causally invisible and are overwritten before corrected
execution consumes them. No Python request-to-cache map is introduced.
Index keys remain BF16, with 128 values per compressed position. They occupy
separate bytes alongside packed compressed KV; the dense index scorer has not
been replaced by a packed index kernel. Compressor value/score state remains
FP32 and token history remains int64. CPU tests verify source aliases, byte
ranges, nonzero child slots, and scratch isolation.

For the pinned 43 attention layers (40 target plus three draft layers), a
128-token prefill chunk and maximum batch 256 give 3,244,032-byte parents
(3.09375 MiB), with child packings `(1, 16, 1, 2048)` for sliding, compressed,
compressor and token groups. The graph reservation is 530 parents, including
one trailing TMA/draft guard; parent zero adds one more. At eight ranks with
ADP8/EP8 and K4, the native estimator gives the following **per-GPU declared
cache demand**, rather than measured allocations:

| Workload, 256 requests per replica | Usable parents at prompt end | Usable parents at decode bound | Bytes including null and graph reservation |
| --- | ---: | ---: | ---: |
| 4,096 independent prompt tokens | 1,793 | 1,809 | 7,591,034,880 (7.06970 GiB) |
| 131,072 total tokens, 126,976 shared + 4,096 private | 1,855 | 1,871 | 7,792,164,864 (7.25702 GiB) |

The decode bound uses two warmup steps, four measured steps, four draft tokens,
and the benchmark's extra request/write headroom: prompt length plus 39.
Shared full-history pages count once per replica; window and recurrent write
capacity stays private. Prefix-cache snapshots and copy-on-write transients can
need additional space. The arena actually preallocates the full budget selected
by `cache_capacity_plan`; the table does not include model weights, projection
scratch, collective workspaces or other CUDA graph allocations.

The runner declares its supported parallelisms. Attention TP shards KV heads;
this MLA model has one KV head, so `attn_tp>1` is rejected. Attention DP, CP,
EP, and expert TP have implementation paths. CP partitions query attention
while retaining replicated KV, compressor, and FFN work, as in V4.

Engram tables are now row-sharded over the full device domain, independently of
attention and expert ownership. On eight ranks, the two pinned tables allocate
48,000,771 and 48,002,086 rows per rank. Each 256-wide FP8 row has eight UE8M0
block32 scale bytes: 264 bytes per row. Their exact combined table allocation is
25,344,754,248 bytes (23.60414 GiB) per GPU. All ranks allocate equal shard
widths; the final rank has six padded rows in the second table. This is table storage only, excluding Engram's
projection weights, hash metadata and lookup workspaces.

The current lookup transport is an explicit NCCL dense fallback: gather all
ranks' IDs, perform owner-masked local FP8 lookup, then reduce-scatter BF16
responses. Serial Engram layers share the same reusable buffers. At eight ranks,
capacity 1,280 (batch 256 times five verification positions), 24 hashes/token and
head dimension 256, one buffer set occupies 143,769,600 bytes (137.109375 MiB)
per GPU. Each additional prepared capacity retains its own buffer set. This
removes replicated tables, but communication and response buffers scale with
world size and padded capacity; eager lookup also negotiates row counts. It is
not DeepEP's sparse RDMA or host-resident Engram fetch, and wide-mesh performance
is unqualified.

The sharded Engram microprobe passes on two B300s with full, partial, idle,
changed, unequal and empty participation, including 12 exact graph replays per
driver ([Modal run](https://modal.com/apps/ekzhang/main/ap-KQVUzpENnEAToeLVnsCFw9)).
The composed packed-cache/Engram microprobe also passes
([Modal run](https://modal.com/apps/ekzhang/main/ap-gWOjxiW0vwCIa8FWK2FmiE)):
FP8/FP4 gathers and direct query/output projections with varied scales match
their independent references exactly. Composed prefill/decode relative L2 is
0.0093559 and draft relative L2 is 0.00809238, both below 0.02. FP8+FP4 and
FP8+FP8 each pass five exact graph replays with changing inputs. The local report
is `artifacts/benchmarks/deepseek-v41-packed-engram-native-validation.json`.

Full-checkpoint validation with packed KV, world-sharded Engram and DeepGEMM
passes on the following B300 layouts, with `attn_tp=cp=1`:

| Layout | Draft width | Load / capture seconds | Evidence |
| --- | ---: | ---: | --- |
| ADP8 / EP8 | K0 | 121.48 / 44.73 | [Modal run](https://modal.com/apps/ekzhang/main/ap-B5WSsCGNVDsA16rpWawkLf) |
| ADP2 / EP2 | K0 | 180.43 / 29.14 | [Modal run](https://modal.com/apps/ekzhang/main/ap-sEUo0lfftzIdLgOLcyLmOH) |
| ADP8 / EP8 | K4 | 91.80 / 26.07 | [Modal run](https://modal.com/apps/ekzhang/main/ap-9LrJvUe1BrPFjyR1a5xXfs) |

Every driver has exactly equal eager/captured output IDs, selected and top-three
logprobs, and all 25 scored positions of the same 26-token counting prompt.
Four prefill steps with chunk size eight replay 160 FFN and 480
attention-projection regions per driver. K0 replays seven full decode graphs;
K4 replays three.
Eager execution replays none. K4 produces 16 tokens counting from 1 through 8,
with its first eight IDs equal to K0, and naturally accepts 12 of 12 draft
tokens across three blocks on each driver. These are short execution/parity
checks, not a model-quality or representative acceptance-rate measurement.
Reports are saved locally as
`artifacts/benchmarks/deepseek-v41-packed-engram-target-validation.json`,
`artifacts/benchmarks/deepseek-v41-packed-engram-adp2ep2-target-validation.json`
and `artifacts/benchmarks/deepseek-v41-packed-engram-dspark-validation.json`;
generated reports are ignored by Git.

The eight-GPU K0 run reported 208,370,466,816 free bytes (194.06012 GiB) per
GPU before cache reservation. Its validation harness requested 128 usable
parents plus six reserved parents, allocating 434,700,288 bytes (414.5625 MiB)
per GPU. This small validation allocation is separate from the declared B256
demand above and from a benchmark's budget-sized arena. Long-context, wider EP
and throughput qualification of the new memory path remain separate targets.
In the tested Modal GPU image build, `nvidia-smi` saw the B300s, while `lspci`
failed and the sysfs PCI device inventory was empty.

The earlier full-checkpoint K0/K4 validation and prefill/decode sweeps used
replicated Engram and BF16 KV. The original short validation reports remain at
`artifacts/benchmarks/deepseek-v41-adp8ep8-deepgemm-{target,dspark}-validation.json`.
Their metadata estimate was 237.24 GiB of weights/GPU at
ADP8/EP8; that estimate does not describe the new sharded path. The completed
4K decode baseline actually allocated 16,029,057,024 cache bytes (14.92822 GiB)
per GPU: 2,423 usable plus 421 reserved 5.375 MiB parents at memory utilization
0.96. It passed batches 1/4/16/64/256 with exact scheduled batches, fixed 3/4
draft acceptance, full decode replay and zero fallbacks. Its local records are
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-adp8ep8-deepgemm/decode.jsonl`
([Modal run](https://modal.com/apps/ekzhang/main/ap-8YRB4M4VQug1LRneccR4bY)).
The 4K prefill baseline also completed batches 1/4/16/64; its records are
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-adp8ep8-deepgemm/prefill.jsonl`
([Modal run](https://modal.com/apps/ekzhang/main/ap-ohMUUnG9MOyeL8nNauQsXZ)).
Both baseline charts are complete. Those throughput and allocation results
predate world-sharded Engram and packed KV.

CED currently reproduces candidate selection and causal masking using dense
index scores; it does not yet avoid scoring keys outside candidate blocks.
Initial prefill executes all 40 backbone layers at every scheduled query
position; the bounded decoder-tail replay optimization is not implemented.
Sparse attention/indexing, cache updates, and Engram hashing execute outside
prefill projection/FFN graphs. Full decode uses the existing captured path.
Large-context metadata, wide meshes, and throughput remain separate
qualification targets; implementing those paths does not establish performance.

DSpark uses a requested fixed width of greedy proposals. The checkpoint's
confidence head is loaded, but its policy does not choose proposal lengths.
The released draft block supports five tokens; benchmarks default to four.
Acceptance and performance qualification therefore apply to the recorded width.

This runner accepts text token IDs. Image tokens are explicitly rejected; the
checkpoint's vision tower and image-specific router biases are not loaded.
Use the V4.1 recipe selector for prompts and responses: its numeric reasoning
budget, system marker, and leading-space DSML tags differ from V4.

Run the small native CUDA checks before allocating the full checkpoint:
`deploy.deepseek_v41_validation.run_probes()` checks quantization, GEMM, shifted
HC, shared paged attention against a Torch oracle, and graph replay. CPU tests
cover checkpoint geometry, physical source sharing, n-gram history and rollback,
source ownership, and the shifted residual/draft contracts. GPU and numerical
qualification results belong in the deployment reports, including any failures.

Earlier four-GPU TP4 debug runs sliced
query heads while replicating KV. That layout violates the current attention-TP
contract, so those reports and charts are historical debugging artifacts,
not supported-layout validation. The legacy TP-only `modal_deepseek_v4.py`
run/serve/benchmark paths reject these models; its checkpoint download remains
usable. Use the [distributed benchmark launcher](../../../../docs/benchmarks.md)
for the explicit attention-DP/EP layout.
