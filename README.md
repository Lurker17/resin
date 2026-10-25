# Resin

Resin is an inference engine _shell_. It provides modular building blocks for
making your own LLM inference engine.

At its core is the `Engine` class: a Rust scheduler and KV cache allocator,
which hands off to a `Runner` – **you write the model's forward pass, on any
hardware, with any kernels.**

```text
token IDs ──► Rust Engine ──► frozen BatchPlan ──► Python Runner ───► token IDs
                  ▲                                    │
                  └──────── ownership & commit ────────┘
```

Resin also has built-in HTTP serving and metrics above that boundary, using a
per-model _recipe_ for chat formatting, response parsing and media handling.

Model authoring is supported through the library/model-package split and the
runner contract. Model packages supply their weights, kernels, and forward pass
while reusing the engine. The [authoring reference](docs/model-forward-pass.md)
describes that contract. Use the repository's
[add-resin-model skill](skills/add-resin-model/SKILL.md) for architecture
research, runner and speculative-decoding implementation, correctness/evaluation,
performance curves and traces, and optional deployment.
For an existing integration, use the
[optimize-resin-performance skill](skills/optimize-resin-performance/SKILL.md)
to diagnose bottlenecks and validate improvements, drawing on measured results
and lessons from Resin's development history.

Importing `resin` does not import PyTorch or initialize CUDA.

Start with the two-page [architecture and API overview](docs/architecture.md).
The focused contracts are:

- [flat cache pool](docs/cache-pool.md)
- [scheduler](docs/scheduler.md)
- [script and deployment inventory](docs/scripts.md)
- [production code-size report](scripts/cloc.py)
- [runner/kernel boundary](docs/kernel-boundary.md)
- [authoring a CUDA model runner](docs/model-forward-pass.md)
- [CUDA graph policies and startup](docs/cuda-graphs.md)
- [NUMA placement before worker startup](docs/numa.md)
- [engine metrics and Prometheus export](docs/metrics.md)
- [CUDA model package boundaries](models/cuda/ARCHITECTURE.md)
- [GPT-OSS execution and parallelism](docs/gpt-oss-parallelism.md)
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

For one process per GPU,
`resin.process.spawn(worker, args=(config,), devices=devices)` can apply NUMA
placement before the child interpreter starts. Use
`resin.numa.bind(...)` around `Process.start()` with an existing multiprocessing
launcher. See [NUMA startup and placement limits](docs/numa.md).

Install `resin-models-cuda`; it pulls in `resin[cuda]` for the shared
accelerator runtime. Reusable device-neutral helpers live in `resin.backend`; CUDA
helpers are explicit imports from `resin.backend.cuda`.

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

All models share one benchmark CLI and one parameterized serving app. The model
registry in [`deploy/models.py`](deploy/models.py) supplies checkpoint revisions,
hardware, parallelism, and model-specific runner construction.

```bash
# Populate the pinned checkpoint without allocating a GPU.
uv run scripts/bench.py --backend modal --model gpt-oss-20b --download-only

# Sweep prefill and decode, save traces, and plot the measured results.
uv run scripts/bench.py --backend modal --model gpt-oss-20b \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/gpt-oss-20b.jsonl --trace-dir bench/traces --plot
```

For local GPUs, use `--backend local --checkpoint /path/to/checkpoint`; models
with a separate draft checkpoint also accept `--assistant-checkpoint`. The
local backend is the default. Modal submits the repository's source into the
shared CUDA image, so the client machine does not need CUDA installed.

Prefill charts compare complete-context latency with input tokens/s/GPU; decode
charts compare per-user output tokens/s with output tokens/s/GPU. Natural draft
acceptance and model routing are the defaults. `--accepted-drafts K` and
`--balanced-experts` explicitly enable synthetic workloads. See
[benchmarking](docs/benchmarks.md) for the timing protocol and trace scope.

For an already running endpoint, `scripts/check_server.py URL --output REPORT`
checks serving correctness, prefix-cache reuse, context-size TTFT, and optional
image/reasoning/stress cases. It can also evaluate a small local MMLU CSV or
JSONL subset with `--eval-file` (defaults: `--eval-limit 100`,
`--eval-concurrency 32`); see
[server checks](docs/benchmarks.md#http-correctness-and-latency-checks).

## Minimal use

```python
from resin import Engine, SamplingParams
from resin_models_torch.lfm2 import Lfm2TorchRunner

runner = Lfm2TorchRunner(device="cpu")

with Engine(runner, context_length=32768) as engine:
    request = engine.submit(
        [1, 6, 24131],
        sampling_params=SamplingParams(
            max_output_tokens=8,
            stop_token_ids=[runner.eos_token_id],
            temperature=1.0,
            top_p=0.95,
        ),
    )
    while engine.has_work():
        for event in engine.step():
            print(event)
    generated_ids = request.take_result()
```

`SamplingParams` is the frozen request policy for generation limits, stop token
IDs, sampling, and logprob options. Use the recipe's `stop_token_ids`
for chat requests, or supply the model's stop tokens explicitly for raw token
requests. `Engine` has no default EOS token.

A custom synchronous runner only needs one method:

```python
import numpy as np

from resin import BatchPlan, ForwardOutput


class Runner:
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return ForwardOutput(
            token_ids=np.zeros(plan.num_sequences, dtype=np.uint32),
            token_indptr=np.arange(plan.num_sequences + 1, dtype=np.uint32),
        )
```

`ForwardOutput` carries flat token IDs and row offsets, with optional logprob
columns. The same result type covers ordinary and speculative decoding.

Managed history and recurrent state are cache groups declared through a frozen
`RunnerConfig`. The Engine binds their one physical arena during construction.
Pass `prefix_cache=True` to reuse aligned prefixes across requests.

`Engine(enable_overlap=True)` enables device submission/completion overlap when
the runner provides it. The initial DeepSeek CUDA path covers stable decode
cohorts; generated-token logprobs and ownership changes take a synchronous
fallback. Use `enable_overlap=False` for reference execution. See
[Engine overlap](docs/architecture.md#engine-overlap).

## Local HTTP server

The built-in native server exposes one Engine through OpenAI Chat Completions and
Responses, Anthropic Messages, and a small streaming chat page. Model-specific
prompt rendering and incremental response parsing live in the independent
`resin-recipe` Rust crate. Native model recipes implement the same `Recipe`
protocol as application-provided recipes. Stock image/video preprocessing also
runs in Rust; the model runner performs the GPU vision encoder.

The bridge bounds queued jobs with `HttpServer(request_queue_capacity=1024)`.
This shared intake queue absorbs bursts while requests await engine admission;
it does not reserve GPU cache or change the engine's active batch size. Active
HTTP sessions (128 per replica), admissions before an inference step (4 per
replica), and buffered events per client (64) remain bounded. A full intake
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
Both examples explicitly use the CPU Torch runner.

## Modal HTTP endpoints

[`deploy/modal_serve.py`](deploy/modal_serve.py) serves every registered model.
Select `RESIN_MODEL` at launch; each selection has a separate app and endpoint,
with shared download, warmup, process management, and HTTP startup code.
Populate its pinned checkpoint before deploying:

```bash
RESIN_MODEL=gpt-oss-20b uv run modal run deploy/modal_serve.py
RESIN_MODEL=gpt-oss-20b uv run modal deploy deploy/modal_serve.py

RESIN_MODEL=deepseek-v41 uv run modal run deploy/modal_serve.py
RESIN_MODEL=deepseek-v41 uv run modal deploy deploy/modal_serve.py
```

Other selections are `gpt-oss-120b`, `gemma4`, `qwen3_5`, and `inkling`. The
`qwen3_5` runner currently serves the pinned Qwen3.8-27B checkpoint. Add
`--publish-only` to the download command to adopt complete files already in the
checkpoint volume without downloading their contents. Serving and benchmarking
use the same checkpoint inventory and completion marker.

DeepSeek uses the shared distributed HTTP coordinator, with a 1,048,576-token context limit,
32 running requests per attention-DP replica, CUDA graphs, and three speculative
draft tokens with natural acceptance. Requests are routed round-robin unless
`routed_dp_rank` in the JSON body or `resin-routed-dp-rank` in the header selects
a replica (0–3). DeepSeek-V4.1 has one MLA KV head and requires `attn_tp=1`.
GPT-OSS 120B uses the same distributed worker/engine coordinator with attention
TP2 and expert TP2; the 20B endpoint keeps a single local Engine.

Each deployed URL serves `/`, `/health`, `/v1/models`,
`/v1/chat/completions`, and `/v1/responses`; model configuration controls
`/v1/messages` support. The functions
scale to zero after 15 idle minutes and are capped at one GPU container apiece.
The chat demo renders Markdown and TeX math and requests up to 16,384 output
tokens by default. Model loading, CUDA graph capture, and one real prompt
warmup all complete before the endpoint starts accepting traffic.
Because `requires_proxy_auth=False`, anyone with a URL can submit inference
requests and incur GPU usage; remove the deployment when it is no longer
needed.

Select a native recipe by name and supply its tokenizer or multimodal processor.
Messages and content blocks are Rust-owned typed objects:

```python
import resin
from resin.recipe import get_native_recipe
from resin.server import HttpServer

recipe = get_native_recipe("gpt_oss", tokenizer)
prompt = recipe.render(
    [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Hello!")]),
    ]
)
parser = recipe.create_parser(prompt)

HttpServer(engine, recipe=recipe, model="my-model", stream_interval=4).serve()
```

`Recipe.render(messages, *, tools=(), options=None)` returns a `RenderedPrompt`
owning token IDs, optional prepared media, and an internal parser seed.
`prompt.token_ids` is a read-only NumPy `uint32` array, and
`recipe.stop_token_ids` returns a list of integers. Pass the recipe's stops to
`SamplingParams(stop_token_ids=...)` when
submitting a rendered prompt to an Engine; HTTP serving uses them automatically.
The factory returns the common `NativeRecipe` implementation. Supported names
are `gpt_oss`, `deepseek_v41`, `glm5_3_flash`, `gemma4`, `inkling`, `lfm2`, `qwen3_5`, and
`qwen3_8`. Supply formatting defaults directly, for example
`get_native_recipe("gpt_oss", tokenizer, reasoning_effort="high")`.
Native recipes infer model stops by default;
`stop_token_ids=None` selects those defaults, a sequence replaces them, and
`[]` disables token stopping.
`create_parser(prompt)` creates a fresh incremental parser with the same resolved
formatting options. It emits text, reasoning, and tool-call deltas; an optional
`finish("stop" | "length")` flushes buffered output before completion.
Applications can pass their own structural `Recipe` implementation without
inheriting a Resin class or using the native parser. The protocol also exposes
`stop_token_ids` and `tokenizer`, a normalized Resin `Tokenizer` snapshot or
`None` when token vocabulary metadata is unavailable. Native recipes provide
`debug_render_text(...)` for inspecting formatting without preparing media;
use `render(...)` for generation.

For images, put `resin.recipe.Image.from_rgb(array)` or
`resin.recipe.Image.from_bytes(data, media_type="image/png")` in a message and pass
the model's processor to the factory, for example
`get_native_recipe("gemma4", processor)`.
The recipe snapshots processor configuration once; image/video requests then use
Rust decoding, resizing, normalization and patchification. A tokenizer plus an
explicit `media_config` mapping also works. Model-owned preprocessing supplies
token expansions, positions and cache identity; the runner receives native-owned
NumPy arrays and transfers them to the GPU vision encoder. Literal marker
spellings in ordinary text cannot acquire media slots.

Gemma video input contains decoded frames, optionally with frame indices and fps;
video-file decoding remains the application's responsibility. Rust's `image`
crate provides image decoding and resize filters. Geometry, normalization and
patch order follow the model processors; resize pixels intentionally need not
match HF bit-for-bit. No-resize tests compare pixels and positions exactly.
Encoded inputs support PNG, JPEG, WebP, GIF, BMP and TIFF. Other formats,
including AVIF, must be decoded by the application and passed with `Image.from_rgb`.

`stream_interval` is an integer from 1 through `2**32 - 1`, set once per server,
and defaults to 1.
The first semantic output is sent immediately. Later output flushes at a step
boundary once N generated model tokens have accumulated; verified blocks can
produce larger chunks, while completion or a bounded buffer can flush earlier.
Text and its logprobs stay aligned. Inference, stop matching, and cancellation
still run on every step. The setting applies to all requests and HTTP APIs.

Image data URLs are decoded locally; remote image fetching remains an application
concern. Prepared media owns its pixel arrays through request execution and
does not become part of response parser state.

## Current shape

The native layer currently provides:

- sticky continual batching with prefill priority and memory retraction;
- phase-homogeneous prefill or decode batches, token budgets, and prefill chunks;
- one flat physical cache pool shared by History, Window, and State groups;
- sparse, chronological `CacheGroupPlan` rows and exact append destinations;
- opt-in exact-token LRU prefix caching with immutable page sharing and
  sliding-window and recurrent-state copy-on-write;
- retryable pre-accept failure and fail-stop post-accept quarantine.

`crates/resin-core` is Python-independent Rust containing request
lifecycle, scheduling, token-range planning, grouped cache allocation, and
execution transactions. It uses `rkyv` for the distributed
plan wire format. `crates/resin-py` converts Python inputs and adapts runner
callbacks, plans, errors, handles, and events to that core. `crates/resin-recipe`
owns conversation formats and CPU media preparation; `crates/resin-http` owns
the serving protocols and streaming responses.

History cache grows on demand to each frozen plan. Window cache grows toward
bounded cyclic storage whose plan row is projected into chronological order.
With prefix caching enabled, a block trie indexes exact token prefixes, cached
pages become immutable readers, and Window pages are copied on write when a
request advances its ring. State groups retain a two-checkpoint ring. At an
aligned prefix hit, the input checkpoint stays immutable while the next page is
writable; the runner copies only when input and output pages differ. LRU entries
use spare capacity in the same pool and are evicted when live requests need it.

Implemented runner modules include a synchronous PyTorch LFM2 reference and
local CUDA implementations for GPT-OSS and
[DeepSeek-V4.1-Flash-1017](models/cuda/resin_models_cuda/deepseek_v41/README.md). The GPT-OSS CUDA
path uses one History group and one Window group over the flat pool. Its
deployment harness supports intra-container tensor parallel sizes 1, 2, 4, and
8, with matching multi-H200 allocations and reusable distributed decode CUDA
graphs. Attention heads, KV storage, and MXFP4 expert intermediate dimensions
are sharded; row-parallel attention and expert outputs are reduced through one
NCCL group.

DeepSeek-V4.1 owns shifted hyper-connections, block32 quantization, Engram, shared source
KV, CED selection, and its changed DSpark stages. Four cache groups hold sliding
KV, shared compressed KV/index keys, compressor state, and Engram token history.
Smaller groups use power-of-two child packing within the Engine's physical
parent pool. The parent count follows post-load free VRAM and a configurable
memory-utilization target. The HTTP deployment enables exact-token LRU prefix
caching, including the model's rolling state.
Its initial deployment target is eight B300s with attention DP8, EP8, and
attention TP1. Engram tables are sharded over the GPU domain, and packed FP8/FP4
KV shares Engine-owned slabs. Prepared DeepInfer operators and metadata pass
full-checkpoint eager/graph checks on four B300s; traced eight-B300 throughput
baselines are recorded in the family README. Earlier TP4 debug runs used replicated KV and do
not qualify the supported attention-TP contract. Image inputs use the checkpoint's
vision tower and aligner through the shared HTTP media interface, with causal
chunked prefill and image-aware prefix-cache keys.
See the family README for memory and optimization limits.

[GLM-5.3-Flash NVFP4](resin/models/glm5_3_flash/README.md) runs on four B200s with
EP4 and attention DP4, native reasoning/tools and still images, full decode
graphs, and optional NextN/MTP. Source `878d83d4` completes the controlled 8K
K0/K3 curve through batch 64 with 508 exact matched-batch request prefixes, and
the five-point 4K–64K prefill curve with four concurrent requests. Measured MTP3
reaches 107.18 output tokens/s/user at batch one and 2455.95 tokens/s/GPU at batch
64; the separately labeled 75%-acceptance projection gives 143.37 and 3077.33.
These use four authored tasks with filler. Outputs can vary across batch
schedules; retirement lifecycle checks pass. The per-bucket runtime passed
446 B200 and 2158 CPU tests. Serving defaults are context 16384, prefill chunk
8192 and target-only decode; final `c2de4fba` HTTP and compiler/runner audits
remain pending. K3 is opt-in. Use registry key `glm-5.3-flash`. The family README
links measured and projected figures, raw results, 8K traces, source provenance
and qualification limits.

Current limits include no prefill/decode disaggregation. A
[distributed coordinator prototype](docs/distributed.md) covers explicit TP, CP,
attention DP, and EP groups with coordinated submission and completion.
[DeepSeek-V4.1 Flash](models/cuda/resin_models_cuda/deepseek_v41/README.md) and
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
`resin_models_cuda.deepseek_v41`, `resin_models_cuda.gpt_oss`, and
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

The code-size report uses Tokei to count production Rust/Python by component,
excluding comments, blank lines, tests, deployment tools, and generated stubs.
Use the current report instead of a checked-in line-count snapshot.

For resident prefill/decode sweeps, optional fixed speculative acceptance and
balanced routing, Perfetto traces, and throughput/latency charts, see
[benchmarking](docs/benchmarks.md).
`Engine` collects native request/cache metrics and optional runner observations
by default (`enable_metrics=True`). Use `enable_metrics=False` to opt out.
The HTTP server publishes `/metrics`; see [metrics](docs/metrics.md)
for Prometheus queries and forwarding through an OTLP Collector.

## Repository map

- `crates/resin-core/src/engine.rs` — request lifecycle and execution transactions
- `crates/resin-py/src/lib.rs` — Python objects and the native Engine adapter
- `crates/resin-py/src/planning.rs` — conversion from core batch/resource views to `BatchPlan`
- `crates/resin-py/src/runner.rs` — execution discovery and one-time resource binding
- `crates/resin-py/src/cache.rs` — Python adapters for core KV descriptors
- `crates/resin-core/src/scheduler.rs` — cohort selection and queue-age policy
- `crates/resin-core/src/planning.rs` — typed token-range planning
- `crates/resin-core/src/resources/` — grouped cache allocation, prefix reuse, and leases
- `crates/resin-recipe/` — native conversation formats, parsing, and CPU media preparation
- `crates/resin-http/` — HTTP protocols, streaming, and serving lifecycle
- `resin/backend/` — reusable execution and resource gadgets
- `models/` — separately installable model definitions, loaders, and runners
- `scripts/` — examples, benchmarks, and local maintenance tools
- `deploy/` — shared model registry, checkpoint/cache helpers, and parameterized Modal serving/benchmark adapters
- `tests/` — native and runner-contract coverage

Resin is alpha software. Its surface is intentionally small enough to review
end to end, but the managed plan ABI is still evolving.

## Acknowledgements

Resin was based on and inspired by systems from SGLang, TokenSpeed, and DeepInfer.
