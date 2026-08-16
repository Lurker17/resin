# Resin

_Inspired by SGLang, vLLM, TokenSpeed, FlashInfer, and FlashAttention-4._

> [!WARNING]
> Almost the entire repository is vibe coded right now. Big chunks of code
> (~10K lines or more) will be added and removed in the coming days, mostly
> removed, as I iterate to make it ready for human programmers.

Resin is an inference engine _engine_. It provides modular, low-level
building blocks for making your own LLM inference engine. At its core is the
`Engine` class: a fast, native scheduler and KV cache allocator, which then
hands off to a `Runner` – you write the forward pass yourself in Python.

```text
token IDs -> Rust Engine -> frozen BatchPlan -> Python Runner -> token IDs
                 ^                                  |
                 +------- ownership and commit -----+
```

The package is deliberately token-ID in/token-ID out. Tokenizers, chat
templates, text streaming, and serving APIs belong above Resin. This allows
you to _write your own code_ to control request admission.

Importing `resin` does not import PyTorch or initialize CUDA.

Start with the two-page [architecture and API overview](docs/architecture.md).
The focused contracts are:

- [flat cache pool](docs/cache-pool.md)
- [scheduler](docs/scheduler.md)
- [state ownership](docs/state-pool.md)
- [runner/kernel boundary](docs/kernel-boundary.md)
- [GPT-OSS runner status](docs/gpt-oss-itl.md)
- [persistent kernel artifacts](docs/kernel-cache.md)

## Install for development

Resin requires Python 3.11+ and Rust.

```bash
uv sync
uv run maturin develop --generate-stubs
uv run python scripts/normalize_stubs.py
uv run ruff format resin/_resin.pyi
```

The control-plane tests do not require a GPU. Importing a concrete model module
loads PyTorch; importing `resin` or a model package namespace does not.

Install `resin-models-cuda`; it pulls in `resin[gpu]` for the shared
accelerator runtime. The uv lock intentionally
overrides FlashAttention-4's stale CUTLASS
DSL metadata pin so FA4 and Quack use one tested toolchain.
Reusable device-neutral helpers live in `resin.backend`; CUDA helpers
are explicit imports from `resin.backend.cuda`.

## Developing

The Modal harnesses are launched from the repository root. Install the
development dependencies first, then authenticate with Modal once if this is a
new machine:

```bash
uv sync
uv run maturin develop --generate-stubs
uv run python scripts/normalize_stubs.py
uv run ruff format resin/_resin.pyi
uv run modal setup
```

`modal setup` is interactive and only needs to be repeated when the local
Modal credentials change.

Run the local LFM2 harness on an H100 with the native Engine and asynchronous
dispatch/poll path:

```bash
uv run modal run deploy/modal_lfm2.py --mode run
```

Useful LFM2 validation modes are:

```bash
# Compare the eager baseline.
uv run modal run deploy/modal_lfm2.py --mode baseline

# Populate, then verify, the persistent FlashInfer kernel cache.
uv run modal run deploy/modal_lfm2.py --mode prewarm
uv run modal run deploy/modal_lfm2.py --mode verify
uv run modal run deploy/modal_lfm2.py --mode probe

# Exercise asynchronous completion and native continuation scheduling.
uv run modal run deploy/modal_lfm2.py --mode async
uv run modal run deploy/modal_lfm2.py --mode native-continuation-probe
```

Run GPT-OSS from the flat-cache path as follows. The checkpoint step is
explicit: GPU modes fail closed if the pinned model and tokenizer are not
already in the Modal volume.

```bash
# Download and validate the pinned 20B checkpoint once.
uv run modal run deploy/modal_gpt_oss.py --mode download --model 20b

# Run one prompt on H200. Quote the prompt so Modal receives it as one argument.
uv run modal run deploy/modal_gpt_oss.py \
  --mode run --model 120b \
  --prompt "Explain why batching improves language-model throughput." \
  --temperature 1.0 --top-p 0.95 --top-k 40 --seed 7

# Sweep 8k prompt / 256 output at batch sizes 1,2,8,16,32.
uv run modal run deploy/modal_gpt_oss.py \
  --mode bench --model 120b --cache-mode prewarm

# Re-run the benchmark without changing saved kernel artifacts.
uv run modal run deploy/modal_gpt_oss.py \
  --mode bench --model 120b --cache-mode verify
```

Both GPU modes also accept `--schedule-conservativeness`; larger values leave
more estimated KV headroom, while smaller values admit more work and may cause
retraction/replay. Run mode accepts per-request `--temperature`, `--top-p`,
`--top-k`, and `--seed`. Temperature defaults to `1.0`; pass `--temperature 0`
for deterministic argmax generation.

Use `--model 120b` for the larger checkpoint. These commands submit the
repository's local source into the Modal image; they do not require a locally
installed CUDA runtime. GPU quota, the pinned model volume, and the required
FlashInfer/FA4 packages are still required remotely.

## Minimal use

```python
from resin import Engine, SamplingParams
from resin_models_torch.lfm2 import Lfm2TorchRunner

runner = Lfm2TorchRunner()

with Engine(
    runner,
    context_length=32768,
    eos_token_id=runner.eos_token_id,
) as engine:
    request = engine.submit(
        [1, 6, 24131],
        max_new_tokens=8,
        sampling_params=SamplingParams(temperature=1.0, top_p=0.95),
    )
    while engine.has_work():
        for event in engine.step():
            print(event)
    generated_ids = request.take_result()
```

A custom synchronous runner only needs one method:

```python
class Runner:
    def forward(self, plan):
        return [0] * len(plan.sequence_ids)
```

An asynchronous runner supplies `dispatch(plan)` and `poll(completion)` instead.
Managed cache and recurrent state are optional runner-owned declarations that
the Engine binds once during construction through a frozen `RunnerConfig` and
one `bind_resources(binding)` callback. Pass `prefix_cache=True` to `Engine`
to opt into cross-request reuse for paged-KV runners and recurrent runners
that declare an aligned state-checkpoint contract.

## Local HTTP server

The optional server exposes one Engine through OpenAI Chat Completions and
Responses, Anthropic Messages, and a small streaming chat page. Model-specific
prompt rendering and response parsing stay in Python so they can follow model
formats without coupling them to the native scheduler.

```bash
uv sync --extra server
uv run python scripts/lfm2_serve.py
```

Then open <http://127.0.0.1:8000>. The page reports prompt and generated token
counts, elapsed generation time, and output tokens per second. The same process
serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/models`.

## Modal HTTP endpoints

The DeepSeek-V4 and GPT-OSS Modal apps also expose the native server as public,
unauthenticated Web Functions. Populate the pinned checkpoint volumes before
the first request, then deploy the apps:

```bash
# GPT-OSS publishes separate 20B and 120B URLs.
uv run modal run deploy/modal_gpt_oss.py --mode download --model 20b
uv run modal run deploy/modal_gpt_oss.py --mode download --model 120b
uv run modal deploy deploy/modal_gpt_oss.py

# DeepSeek-V4 publishes one TP4 B300 URL.
uv run modal run deploy/modal_deepseek_v4.py --mode download
uv run modal deploy deploy/modal_deepseek_v4.py
```

Each deployed URL serves `/`, `/health`, `/v1/models`,
`/v1/chat/completions`, `/v1/responses`, and `/v1/messages`. The functions
scale to zero after 15 idle minutes and are capped at one GPU container apiece.
The chat demo renders Markdown and TeX math and requests up to 16,384 output
tokens by default. Model loading, CUDA graph capture, and one real prompt
warmup all complete before the endpoint starts accepting traffic.
Because `requires_proxy_auth=False`, anyone with a URL can submit inference
requests and incur GPU usage; remove the deployment when it is no longer
needed.

For a custom model-format adapter, pass `codec=` instead of a renderer:

```python
from resin.server import HttpServer

HttpServer(engine, codec=my_codec, model="my-model").serve()
```

`ConversationCodec.render()` returns prompt and stop token IDs;
`create_parser()` returns a per-request incremental parser that emits text,
reasoning, and tool-call deltas. `RendererCodec` supplies this contract for the
optional `renderers` package.

## Current shape

The native layer currently provides:

- sticky continual batching with prefill priority and memory retraction;
- phase-homogeneous prefill or decode batches, token budgets, and prefill chunks;
- one flat physical cache pool shared by ordered History and Window groups;
- sparse, chronological `CacheGroupPlan` rows and exact append destinations;
- opt-in exact-token LRU prefix caching with immutable page sharing and
  sliding-window and recurrent-state copy-on-write;
- generation-checked `StatePool` slots with shared-read/exclusive-write access;
- retryable pre-accept failure and fail-stop post-accept quarantine.

The native implementation is split in two. `crates/resin-core` is ordinary,
PyO3-free Rust containing typed scheduling, token-range planning, full-context
and sliding-window KV resources, recurrent-slot resources, allocation, and
leases. `crates/resin-py` adapts those types to Python and coordinates runner
acceptance and completion.

History cache grows on demand to each frozen plan. Window cache grows toward
bounded cyclic storage whose plan row is projected into chronological order.
With prefix caching enabled, a block trie indexes exact token prefixes, cached
pages become immutable readers, and Window pages are copied on write when a
request advances its ring. LRU entries use spare capacity in the same physical
pool and are evicted when live allocation needs their pages or state slots.

Request-sized recurrent model state is a separate `StatePool` resource. A mapped runner may
omit both the pool and request handles: Engine sizes the pool from the running
limit, allocates slots at admission, and emits their indices in `BatchPlan`.
Explicit handles remain available for shared-state requests. Prefix reuse is
accepted only when a recurrent mapping declares `checkpoint_granularity`.
At aligned prefill boundaries the cache retains the matching state generation;
a hit supplies an ordered read-checkpoint/write-destination pair, and the
runner copies that checkpoint before updating its private destination. LFM2
uses this contract for convolution state while sharing its attention pages.

Implemented runner modules include a synchronous LFM2 reference and local CUDA
implementations for LFM2, GPT-OSS, and DeepSeek-V4-Flash. The GPT-OSS CUDA path uses one
History group and one Window group over the flat pool. Its deployment harness
supports intra-container tensor parallel sizes 1, 2, 4, and 8, with matching
multi-H200 allocations and reusable distributed decode CUDA graphs. Attention
heads, KV storage, and MXFP4 expert intermediate dimensions are sharded; row
parallel attention and expert outputs are reduced through one NCCL group.

The LFM2 CUDA runner has one route: atomic cache-plus-state binding followed by
asynchronous dispatch/poll. Its continuation harness uses native Engine plans;
there is no parallel manual-plan or synchronous compatibility path.

DeepSeek-V4 uses six logical groups over one fungible physical parent pool:
sliding attention KV, c4 compressed history, the c4 compressor tail, c128
compressed history, the c128 compressor tail, and the c4 indexer tail. Smaller
groups use power-of-two child packing, while the parent count is computed from
post-load free VRAM and a configurable memory-utilization target. Compressor
tails are cache-like rolling token rows rather than whole-request checkpoints,
so aligned prefix snapshots share them exactly like attention pages. The Modal
web deployment enables the native exact-token LRU prefix cache.

Current limits include one in-flight batch per Engine, no prefill/decode
disaggregation, no expert parallel or data-parallel attention, and no
multi-node serving. Tensor parallelism remains a model-backend concern rather
than part of the native scheduler.

## Public surface

Top-level imports are limited to native control-plane types:

```python
Engine
Runner
SyncRunner
AsyncRunner
BatchPlan
BatchPhase
TokenEvent
RequestLease
SamplingParams
SchedulerConfig
RunnerConfig
ResourceBinding
ResourceRunner
CacheGroupLayout
CacheLayout
CacheGroupPlan
StatePool
StateHandle
StateMappingConfig
mint_state_mapping_arena_id
```

Concrete runners are explicit imports from architecture submodules such as
`resin_models_cuda.deepseek_v4`, `resin_models_cuda.gpt_oss`, and
`resin_models_torch.lfm2`.

## Checks

```bash
cargo test --all-targets
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run python scripts/cloc.py
```

GPU validation is model- and machine-specific; local tests use fakes or skip
unavailable accelerator libraries.

## Repository map

- `crates/resin-py/src/lib.rs` — Python API, Engine lifecycle, and transactional orchestration
- `crates/resin-py/src/planning.rs` — conversion from core batch/resource views to `BatchPlan`
- `crates/resin-py/src/runner.rs` — execution discovery and one-time resource binding
- `crates/resin-py/src/cache.rs` — Python adapters for core KV descriptors
- `crates/resin-py/src/state_pool.rs` — Python adapter for the core recurrent-slot pool
- `crates/resin-core/src/scheduler.rs` — cohort selection and queue-age policy
- `crates/resin-core/src/planning.rs` — typed token-range planning
- `crates/resin-core/src/resources/` — KV, recurrent, allocator, and lease logic
- `resin/backend/` — reusable execution and resource gadgets
- `models/` — separately installable model definitions, loaders, and runners
- `scripts/` — examples, benchmarks, and local maintenance tools
- `deploy/` — optional remote validation harnesses
- `tests/` — native and runner-contract coverage

Resin is alpha software. Its surface is intentionally small enough to review
end to end, but the managed plan ABI is still evolving.
