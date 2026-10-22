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
throughput qualification remain separate below. FlashMLA's fused attention is
not integrated yet: it requires packed 528-byte FP8 sliding KV, 288-byte FP4
compressed KV, and matching prepared projections. Those can use strided views
over the existing shared slab pool without changing scheduler ownership.

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

The runner declares its supported parallelisms. Attention TP shards KV heads;
this MLA model has one KV head, so `attn_tp>1` is rejected. Attention DP, CP,
EP, and expert TP have implementation paths. CP partitions query attention
while retaining replicated KV, compressor, and FFN work, as in V4.

The initial deployment target is `Parallelism(world_size=8, attn_dp=8, ep=8)`
on eight B300s, with attention TP1 and expert TP1. Engram tables are currently
replicated across attention-DP replicas; sharded table lookup is not implemented.
A metadata-only estimate puts weights at 237.24 GiB per GPU on this layout,
before caches, workspaces, and graph buffers. Four B300s with ADP4/EP4 require
271.70 GiB of weights per GPU and cannot fit those additional allocations.
These estimates establish neither runtime memory headroom nor qualification.

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

Full-checkpoint target, natural DSpark, and throughput qualification on the
eight-GPU ADP8/EP8 layout remain pending. Earlier four-GPU TP4 debug runs sliced
query heads while replicating KV. That layout violates the current attention-TP
contract, so those reports and charts are historical debugging artifacts,
not supported-layout validation. The legacy TP-only `modal_deepseek_v4.py`
run/serve/benchmark paths reject these models; its checkpoint download remains
usable. Use the [distributed benchmark launcher](../../../../docs/benchmarks.md)
for the explicit attention-DP/EP layout.
