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
uv sync --extra dev
uv run maturin develop --generate-stubs
uv run python scripts/normalize_stubs.py
uv run ruff format src/resin/_resin.pyi
```

The control-plane tests do not require a GPU. Accelerator runners import their
dependencies lazily.

## Developing

The Modal harnesses are launched from the repository root. Install the
development dependencies first, then authenticate with Modal once if this is a
new machine:

```bash
uv sync --extra dev
uv run maturin develop --generate-stubs
uv run python scripts/normalize_stubs.py
uv run ruff format src/resin/_resin.pyi
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
  --prompt "Explain why batching improves language-model throughput."

# Sweep 8k prompt / 256 output at batch sizes 1,2,8,16,32.
uv run modal run deploy/modal_gpt_oss.py \
  --mode bench --model 120b --cache-mode prewarm

# Re-run the benchmark without changing saved kernel artifacts.
uv run modal run deploy/modal_gpt_oss.py \
  --mode bench --model 120b --cache-mode verify
```

Use `--model 120b` for the larger checkpoint. These commands submit the
repository's local source into the Modal image; they do not require a locally
installed CUDA runtime. GPU quota, the pinned model volume, and the required
FlashInfer/FA4 packages are still required remotely.

## Minimal use

```python
from resin import Engine
from resin.backends.lfm2_torch import Lfm2TorchRunner

runner = Lfm2TorchRunner()

with Engine(
    runner,
    context_length=32768,
    eos_token_id=runner.eos_token_id,
) as engine:
    request = engine.submit([1, 6, 24131], max_new_tokens=8)
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
the Engine binds once during construction.

## Current shape

The native layer currently provides:

- strict priority with FIFO or round-robin continual batching;
- phase-homogeneous prefill or decode batches, token budgets, and prefill chunks;
- one flat physical cache pool shared by ordered History and Window groups;
- sparse, chronological `CacheGroupPlan` rows and exact append destinations;
- generation-checked `StatePool` slots with shared-read/exclusive-write access;
- retryable pre-accept failure and fail-stop post-accept quarantine.

History cache grows for the declared request lifetime. Window cache is bounded
private cyclic storage whose plan row is projected into chronological order.
Cache retention is exactly History or Window. Recurrent model state is a
separate resource owned through `StatePool`.

Resin has no content index or cross-request cache sharing today, so it performs
no prefix reuse. The current cache surface only describes live allocation and
runner placement.

Implemented runner modules include synchronous LFM2 and GPT-OSS references and
local CUDA implementations for LFM2 and GPT-OSS. The GPT-OSS CUDA path uses one
History group and one Window group over the flat pool. Its deployment harness is
migrated locally but has not been launched, so the repository does not claim a
fresh GPU result for this architecture.

The LFM2 CUDA runner has one route: atomic cache-plus-state binding followed by
asynchronous dispatch/poll. Its continuation harness uses native Engine plans;
there is no parallel manual-plan or synchronous compatibility path.

Current limits include one in-flight batch per Engine, greedy runner-owned
sampling, no tensor/expert parallel control plane, and no distributed serving.

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
SchedulerConfig
CacheGroupLayout
CacheLayout
CacheGroupPlan
StatePool
StateHandle
StateMappingConfig
mint_state_mapping_arena_id
```

Concrete runners are explicit imports from `resin.backends` modules.

## Checks

```bash
cargo test --all-targets
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
```

GPU validation is model- and machine-specific; local tests use fakes or skip
unavailable accelerator libraries.

## Repository map

- `src/lib.rs` — Python API, Engine lifecycle, and transactional orchestration
- `src/planning.rs` — deterministic row scheduling and plan construction
- `src/runner.rs` — execution discovery and one-time resource binding
- `src/scheduler.rs` — cohort selection and queue-age policy
- `src/cache.rs` — public cache layout and grouped plan descriptors
- `src/cache_pool.rs` — flat parent allocator, request ownership, and leases
- `src/state_pool.rs` — generic state ownership, access, and physical mappings
- `src/resin/backends/` — model definitions, loaders, and runners
- `deploy/` — optional remote validation harnesses
- `tests/` — native and runner-contract coverage

Resin is alpha software. Its surface is intentionally small enough to review
end to end, but the managed plan ABI is still evolving.
