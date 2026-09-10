# CUDA model package boundaries

CUDA model families are deliberately small accelerator frontends. Backend-
neutral configuration, prompt codecs, prepared media, and scheduler lowering
live in `resin.models.<family>` so CUDA and AMD implementations can share them.
A CUDA family owns its runner, optimized model, physical cache geometry,
checkpoint materialization, draft implementation, and concrete kernels. All
of those files live together in `resin_models_cuda.<family>`; the package does
not split implementations into `runners`, `kernels`, `drafts`, `media`, or
`loading` directories. Importers use the owning module directly, without
compatibility aliases or implementation registries.

Resin itself also owns reusable transformer semantics and CUDA host mechanics:
graph capture/replay, fixed-buffer staging, sampling inputs, and slab page-
table addressing. Concrete CUDA, cuTile, TileLang, FlashInfer, and model-
specific fused kernels remain directly importable from
`resin_models_cuda.<family>`; they are never moved into `resin.backend.cuda` or
selected through a registry. A small cross-model CUDA primitive may live as a
root module, such as `paged_kv.py` or `paged.py`, when it has multiple model
owners and does not justify another package layer.

Every CUDA model runner derives from `resin.backend.cuda.runner.CudaRunner`.
That base owns device normalization, KV precision selection, failure
containment, sampling, Engine cache binding, graph modes, capture hooks, shape
selection, fallback accounting, replay, poisoning checks, and mode-scoped
statistics while remaining unaware of model layers and physical cache layout.
Each model-owned cache arena decides which precision and heterogeneous state
layouts it can actually support.

`PagedCudaRunner` adds only the shared text-decoder paging, FA4, sampling, and
tensor-parallel execution. Its default feed-forward hook calls the model's
module directly; a family such as GPT-OSS may override that hook for a concrete
router or MoE kernel without making MoE part of the runner's identity. Gemma 4,
Qwen3.5, Inkling, and DeepSeek V4 derive directly from the common graph base and
retain their different cache views, media/speculation paths, and architecture
layers without adding branches to `PagedCudaRunner`.

## Graph execution modes

A `CudaRunner` configures one internal `CudaGraphExecutor` with semantic
modes such as `prefill` and `decode`. Each `CudaGraphMode` supplies only its
shape buckets, fixed-input factory, and program builder. The builder returns
either one `CudaGraphStep` (a full graph) or a `CudaGraphProgram` containing
captured and eager regions (a breakable graph). Capture, staging, replay,
poisoning, validation, and metrics are shared by every model, every mode, and
both graph styles.

Phase code must keep the model operation separate from graph orchestration.
Eager execution and graph builders call the same model/attention helpers; a
graph region is a capture boundary around those helpers, not a second
implementation of the phase. Consequently, adding breakable prefill means
registering a `prefill` mode and choosing its region boundaries. It must not
add a second graph runner, duplicate the model forward pass, or fork cache and
sampling semantics. Decode can independently remain full-graph or become
breakable in the same runner.

`tests/test_architecture_size.py` records the committed pre-refactor baseline
and requires the complete CUDA model package to remain at least twenty percent
smaller. It also rejects implementation-category directories and backend-
neutral compatibility shims.
