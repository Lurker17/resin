# Code size and simplification

The baseline for this audit is `01be552`. Counts use Tokei 12.1.2, exclude
comments and blank lines, and separate production implementation from tests,
deployment tools, examples, and generated stubs. `scripts/cloc.py` now counts
all of `resin`: it previously missed 3,021 lines in model utilities and serving.

## Where the code lives

| Production layer at baseline | Code lines |
| --- | ---: |
| CUDA model package | 15,976 |
| Model configuration, media, and codecs | 2,619 |
| Torch reference model package | 547 |
| Native engine, scheduler, and resources | 4,637 |
| Python bindings in Rust | 2,234 |
| Native HTTP | 1,291 |
| Shared Python backend, package root, and serving | 4,023 |
| **Total** | **31,327** |

Model support accounts for about 61% of production code. Within the CUDA package,
runner/cache/checkpoint glue is 6,396 lines (40%), kernel modules and adapters
are 5,456 (34%), and model definitions and forward computation are 4,124 (26%).
These are whole-file categories: kernel modules also include their launch
wrappers, and model files include tensor operations and some execution glue.

| CUDA family at baseline | Code lines |
| --- | ---: |
| DeepSeek V4 | 6,290 |
| Inkling | 3,208 |
| Qwen 3.5 | 2,468 |
| Gemma 4 | 1,903 |
| GPT-OSS | 1,867 |
| Shared CUDA modules | 240 |

DeepSeek is the largest family: its cuTile module is 2,335 lines and model
implementation is 1,974. Its compressed attention, routing, quantization,
recurrent compressor state, and speculative execution need substantial model
semantics. Gemma's size is concentrated in runner code: 1,512 lines at baseline,
while its CUDA model definition is 230. That made repeated host setup a useful
first reduction without changing numerical kernels.

## First-pass reductions

| Area | Production lines removed |
| --- | ---: |
| Gemma graph attention setup and capture metadata | 117 |
| Native request validation, retirement, and plan construction | 66 |
| HTTP response collection and sampling validation | 27 |
| Shared stop-token encoding across five codecs | 14 |
| **Total** | **224** |

Production implementation decreased from **31,327 to 31,103** lines. Including
all Rust/Python tests, deployment tools, scripts, and examples, source code
decreased from **47,631 to 47,545** lines: **86 fewer overall**, after adding
regression coverage. The generated Python stub remains 560 lines and is excluded
from both implementation totals.

The changes give each shared invariant one implementation: native scheduled-row
checks, cache retirement, non-streaming response completion, tokenized stops,
and Gemma's FlashInfer planning options. Model kernels and numerical launch
closures remain unchanged. Native plan construction also loses one temporary
allocation per batch.

## First-pass validation

All 341 Python tests and 97 Rust tests pass. Formatting, lint, and whitespace
checks pass. The CPU benchmark uses the complete native `EngineCore.step`
with a tiny runner and managed History cache, including scheduling, allocation,
plan creation, events, progress, and requeue; prefill and 128 warmup steps are
excluded. Median microseconds per batch, nine alternating paired process runs
with identical optimized rustc 1.97.1 builds on an Apple M1 Pro:

| Batch / prompt | Before | After |
| --- | ---: | ---: |
| 1 / 4,096 tokens | 13.257 | 13.164 |
| 32 / 4,096 tokens | 396.772 | 393.849 |
| 32 / 16,384 tokens | 1,419.763 | 1,413.807 |

The sample ranges overlap; there is no measurable regression or speedup.
The [Gemma Modal smoke](https://modal.com/apps/ekzhang/main/ap-BSh4ixGQV4mUeHLa0ipBx7)
also passes on RTX PRO 6000 with image input and production speculative decoding:
all 16 generated tokens match the preceding baseline run, with 20 graph shapes
captured and five replays. Peak memory is unchanged. Mean token time is 9.266 ms
versus 9.246 ms; this short smoke is not a rigorous performance comparison.
The refactor preserves numerical launch closures and capture inputs; the local
contract tests exercise both ordinary and speculative attention planning.

```bash
uv run python scripts/cloc.py
cargo test --all-targets
uv run pytest -q
uv run modal run deploy/modal_gemma4.py --mode smoke --max-new-tokens 16
```

## DeepSeek kernel reuse

The next pass replaces three local GEMM quantizers with DeepSeek's TileKernels,
simplifies hash routing to gather six selected logits, shares numerically stable
sqrt-softplus, and removes unused cache-layout branches. Production implementation
decreases by another **272 lines**, from **31,103 to 30,831**. DeepSeek decreases
from **6,290 to 6,018** lines. Including the expanded GPU regression tests and
deployment entry point, all Rust/Python source decreases from **47,545 to 47,520**:
**25 fewer lines overall** in this pass.

Across both passes, this is **496 fewer production lines** and **111 fewer total
source lines** than the audit baseline. See [DeepSeek kernel reuse](deepseek-kernel-reuse.md)
for measured kernel costs, numerical validation, and the replacements retained
or rejected. Counts use the same Tokei method as above.

## CUDA graph feature

The subsequent [CUDA graph policy feature](cuda-graphs.md) adds **420 production
lines** relative to `4ea6384`: 242 in the shared Python backend and 178 across
the five CUDA model integrations. Rust is unchanged. Production now totals
**31,251 lines**, 76 below the original audit baseline. Including CPU/GPU tests
and deployment validation, all Rust/Python source grows by **1,787 lines**, from
47,520 to **49,307**. This is feature and validation work, not another code-size
reduction; the same counters and exclusions are used throughout.

## Further reductions

Repeated checkpoint inventory checks and worker-launch scaffolding in deployment
scripts are good candidates for small shared functions. Model-local duplication
is generally safer to consolidate before introducing shared model base classes.
Any shared attention or speculative-execution abstraction should express an
existing common contract and preserve device operations, capture boundaries,
and memory layout. Further kernel reductions need benchmarked replacements
that preserve each supported model's output and memory requirements.
