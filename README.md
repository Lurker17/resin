# Resin

_Inspired by SGLang, TokenSpeed, and DeepInfer._

> [!WARNING]
> Almost the entire repository is vibe coded right now. Big chunks of code
> (~10K lines or more) will be added and removed in the coming days, mostly
> removed, as I iterate to make it ready for human programmers.

Resin is an inference engine _engine_. It provides modular, low-level
building blocks for making your own LLM inference engine. At its core is the
`Engine` class: a native request lifecycle, scheduler, and KV cache allocator,
which hands off to a `Runner` – you write the forward pass yourself in Python.
The same lifecycle is available directly to Rust applications through the
PyO3-free `resin-core::engine::EngineCore`.

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
- [code size and simplification](docs/code-size.md)
- [runner/kernel boundary](docs/kernel-boundary.md)
- [authoring a CUDA model runner](docs/model-forward-pass.md)
- [CUDA graph policies and startup](docs/cuda-graphs.md)
- [CUDA model package boundaries](models/cuda/ARCHITECTURE.md)
- [GPT-OSS runner status](docs/gpt-oss-itl.md)
- [persistent kernel artifacts](docs/kernel-cache.md)
- [model recipe and streaming parser boundary](crates/resin-recipe/README.md)
- [recipe differential fuzzing](tests/recipe_fuzz/README.md)

## Install for development

Resin requires Python 3.11+ and Rust.

```bash
uv run scripts/rust_build_and_types.py
```

`uv run` syncs the project dependencies automatically. The build command then
syncs type-library sources, builds the Rust extension with
generated stubs, then normalizes and formats those stubs. It stops if any step
fails.

The control-plane tests do not require a GPU. Importing a concrete model module
loads PyTorch; importing `resin` or a model package namespace does not.

`sync_type_libraries.py` gives ty and its editor extension the Python interfaces of
the CUDA package's Linux-only dependencies. It follows their locked dependency
graph, reuses ordinary installed dependencies, and extracts Python source and
`.pyi` files into ignored `.type-libs/sources/`. Namespace packages are merged,
CUTLASS and bundled TVM import roots are exposed without executing their setup
code, and the small corrections in `type_stubs/` are applied last.
The script verifies the lockfile checksum, installs no packages, and never
imports the extracted code. Rerun it after dependency or stub changes; cached
wheels allow repeat runs offline. To refresh only these sources, run
`uv run scripts/sync_type_libraries.py`.
Only ty uses these sources; Python's runtime import path is unchanged. Type
detail depends on upstream annotations: compiled exports may lack `.pyi` files,
and some decorators erase signatures. Corrections belong in `type_stubs/`.

Install `resin-models-cuda`; it pulls in `resin[cuda]` for the shared
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
uv run scripts/rust_build_and_types.py
uv run modal setup
```

`modal setup` is interactive and only needs to be repeated when the local
Modal credentials change.

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

runner = Lfm2TorchRunner(device="cpu")

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

Managed history and recurrent state are cache groups declared through a frozen
`RunnerConfig`. The Engine binds their one physical arena during construction.
Pass `prefix_cache=True` to reuse aligned prefixes across requests.

`Engine(overlap=True)` enables device submission/completion overlap when the
runner provides it. The initial DeepSeek CUDA path covers stable decode cohorts;
token stops, generated-token logprobs and ownership changes take a synchronous
fallback. Use `overlap=False` for reference execution. See
[Engine overlap](docs/architecture.md#engine-overlap).

## Local HTTP server

The built-in native server exposes one Engine through OpenAI Chat Completions and
Responses, Anthropic Messages, and a small streaming chat page. Model-specific
prompt rendering and incremental response parsing live in the independent
`resin-recipe` Rust crate. Python prepares media tensors and may supply custom
protocol-compatible codecs.

The bridge bounds queued jobs (32), active requests (128), admissions before
an inference step (4), and buffered events per client (64). A full admission
queue returns HTTP 429; slow or disconnected clients are cancelled without
blocking other requests. Actor failure stops the listener, and
`HttpServer.shutdown()` requests shutdown after the current model step.

```bash
# Run one LFM2 prompt in the terminal.
uv run python scripts/lfm2_chat.py

# Serve LFM2 locally over HTTP.
uv run python scripts/lfm2_serve.py
```

Then open <http://127.0.0.1:8000>. The page reports prompt and generated token
counts, elapsed generation time, and output tokens per second. The same process
serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/models`.
Both scripts use PyTorch's automatically selected local device, falling back to
the CPU when no accelerator is available.

## Modal HTTP endpoints

The GPT-OSS Modal app also exposes the native server as public,
unauthenticated Web Functions. Populate the pinned checkpoint volumes before
the first request, then deploy the app:

```bash
# GPT-OSS publishes separate 20B and 120B URLs.
uv run modal run deploy/modal_gpt_oss.py --mode download --model 20b
uv run modal run deploy/modal_gpt_oss.py --mode download --model 120b
uv run modal deploy deploy/modal_gpt_oss.py

# DeepSeek V4.1 Flash uses four B300s with EP4 and attention DP4.
uv run modal run deploy/modal_bench.py --download-only --model deepseek-v41
uv run modal deploy deploy/modal_deepseek_v41.py
```

DeepSeek uses the shared distributed HTTP coordinator, with a 32K context limit,
32 running requests per attention-DP replica, CUDA graphs, and three speculative
draft tokens with natural acceptance. Requests are routed round-robin unless
`routed_dp_rank` in the JSON body or `resin-routed-dp-rank` in the header selects
a replica (0–3). Both MLA families have one KV head and require `attn_tp=1`.
The older `modal_deepseek_v4.py` retains TP-only diagnostics, not this endpoint.

Each deployed URL serves `/`, `/health`, `/v1/models`,
`/v1/chat/completions`, `/v1/responses`, and `/v1/messages`. The functions
scale to zero after 15 idle minutes and are capped at one GPU container apiece.
The chat demo renders Markdown and TeX math and requests up to 16,384 output
tokens by default. Model loading, CUDA graph capture, and one real prompt
warmup all complete before the endpoint starts accepting traffic.
Because `requires_proxy_auth=False`, anyone with a URL can submit inference
requests and incur GPU usage; remove the deployment when it is no longer
needed.

Stock model codecs use the native `resin-recipe` conversation formats and
incremental response parsers. Image/video processors prepare media arrays and
placeholder geometry after the recipe renders the conversation.

For a custom model-format adapter, pass any protocol-compatible `codec=`:

```python
from resin.server import HttpServer

HttpServer(engine, codec=my_codec, model="my-model", stream_interval=4).serve()
```

`ConversationCodec.render()` returns prompt and stop token IDs;
`create_parser()` returns a per-request incremental parser that emits text,
reasoning, and tool-call deltas. An optional `finish("stop" | "length")` flushes
buffered output before the server sends the final response. Custom codecs do
not need to inherit a Resin class or use the native tokenizer/parser.

`stream_interval` is an integer from 1 through `2**32 - 1`, set once per server,
and defaults to 1.
The first semantic output is sent immediately. Later output flushes at a step
boundary once N generated model tokens have accumulated; verified blocks can
produce larger chunks, while completion or a bounded buffer can flush earlier.
Text and its logprobs stay aligned. Inference, stop matching, and cancellation
still run on every step. The setting applies to all requests and HTTP APIs.

`resin.server.media` provides optional image decoding and OpenAI-style message
normalization helpers for model bring-up, demos, and tests. Codecs do not have
to use them. Security-sensitive servers should enforce their own input-size,
MIME, codec, and fetching policy before constructing model media.

## Current shape

The native layer currently provides:

- sticky continual batching with prefill priority and memory retraction;
- phase-homogeneous prefill or decode batches, token budgets, and prefill chunks;
- one flat physical cache pool shared by History, Window, and State groups;
- sparse, chronological `CacheGroupPlan` rows and exact append destinations;
- opt-in exact-token LRU prefix caching with immutable page sharing and
  sliding-window and recurrent-state copy-on-write;
- retryable pre-accept failure and fail-stop post-accept quarantine.

`crates/resin-core` is ordinary, dependency-free Rust containing request
lifecycle, scheduling, token-range planning, grouped cache allocation, and
execution transactions. `crates/resin-py` converts Python inputs and adapts
runner callbacks, plans, errors, handles, and events to that core.

History cache grows on demand to each frozen plan. Window cache grows toward
bounded cyclic storage whose plan row is projected into chronological order.
With prefix caching enabled, a block trie indexes exact token prefixes, cached
pages become immutable readers, and Window pages are copied on write when a
request advances its ring. State groups retain a two-checkpoint ring. At an
aligned prefix hit, the input checkpoint stays immutable while the next page is
writable; the runner copies only when input and output pages differ. LRU entries
use spare capacity in the same pool and are evicted when live requests need it.

Implemented runner modules include a synchronous PyTorch LFM2 reference and
local CUDA implementations for GPT-OSS, DeepSeek-V4-Flash, and
[DeepSeek-V4.1-Flash-1017](models/cuda/resin_models_cuda/deepseek_v41/README.md). The GPT-OSS CUDA
path uses one History group and one Window group over the flat pool. Its
deployment harness supports intra-container tensor parallel sizes 1, 2, 4, and
8, with matching multi-H200 allocations and reusable distributed decode CUDA
graphs. Attention heads, KV storage, and MXFP4 expert intermediate dimensions
are sharded; row-parallel attention and expert outputs are reduced through one
NCCL group.

DeepSeek-V4 uses six logical groups over one fungible physical parent pool:
sliding attention KV, c4 compressed history, the c4 compressor tail, c128
compressed history, the c128 compressor tail, and the c4 indexer tail. Smaller
groups use power-of-two child packing, while the parent count is computed from
post-load free VRAM and a configurable memory-utilization target. Compressor
tails are cache-like rolling token rows rather than whole-request checkpoints,
so aligned prefix snapshots share them exactly like attention pages. The Modal
web deployment enables the native exact-token LRU prefix cache.

V4.1 owns shifted hyper-connections, block32 quantization, Engram, shared source
KV, CED selection, and its changed DSpark stages. Four cache groups hold sliding
KV, shared compressed KV/index keys, compressor state, and Engram token history.
Its initial deployment target is eight B300s with attention DP8, EP8, and
attention TP1. Engram tables are sharded over the GPU domain, and packed FP8/FP4
KV shares Engine-owned slabs. Prepared DeepInfer operators and metadata pass
full-checkpoint eager/graph checks on four B300s; traced eight-B300 throughput
baselines are recorded in the family README. Earlier TP4 debug runs used replicated KV and do
not qualify the supported attention-TP contract. Vision inputs are rejected.
See the family README for memory and optimization limits.

Current limits include no prefill/decode disaggregation. A
[distributed coordinator prototype](docs/distributed.md) covers explicit TP, CP,
attention DP, and EP groups with coordinated submission and completion.
[DeepSeek-V4 Flash](docs/deepseek-v4-parallelism.md) and
[GPT-OSS](docs/gpt-oss-parallelism.md) describe their composed attention TP/DP,
CP, EP, and expert-TP execution paths and validation. Production multi-node
serving remains unimplemented.
Device collectives use explicit model rank groups.
[Generated-token logprobs](docs/logprobs.md) use borrowed NumPy output columns.

## Public surface

Top-level imports are limited to native control-plane types:

```python
Engine
Runner
ManagedRunner
BatchPlan
BatchPhase
TokenEvent
ForwardOutput
TokenLogprobs
RequestLease
SamplingParams
SchedulerConfig
Parallelism
RunnerConfig
ResourceBinding
CacheGroupLayout
CacheGroupFamily
CacheLayout
CacheGroupPlan
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
uv run ty check
uv run python scripts/cloc.py
```

GPU validation is model- and machine-specific; local tests use fakes or skip
unavailable accelerator libraries.

For resident prefill/decode sweeps, fixed speculative acceptance, balanced
expert routing, and throughput/latency charts, see [benchmarking](docs/benchmarks.md).

## Repository map

- `crates/resin-core/src/engine.rs` — request lifecycle and execution transactions
- `crates/resin-py/src/lib.rs` — Python objects and the native Engine adapter
- `crates/resin-py/src/planning.rs` — conversion from core batch/resource views to `BatchPlan`
- `crates/resin-py/src/runner.rs` — execution discovery and one-time resource binding
- `crates/resin-py/src/cache.rs` — Python adapters for core KV descriptors
- `crates/resin-core/src/scheduler.rs` — cohort selection and queue-age policy
- `crates/resin-core/src/planning.rs` — typed token-range planning
- `crates/resin-core/src/resources/` — grouped cache allocation, prefix reuse, and leases
- `resin/backend/` — reusable execution and resource gadgets
- `models/` — separately installable model definitions, loaders, and runners
- `scripts/` — examples, benchmarks, and local maintenance tools
- `deploy/` — optional remote validation harnesses
- `tests/` — native and runner-contract coverage

Resin is alpha software. Its surface is intentionally small enough to review
end to end, but the managed plan ABI is still evolving.
