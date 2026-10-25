# GLM-5.3-Flash NVFP4

Resin runs the native hybrid KDA/sparse-MLA model on **four B200s with EP4 and
attention DP4**. The shared registry key is `glm-5.3-flash`; the native recipe is
`glm5_3_flash`. The integration includes reasoning, tools, still images, full
decode CUDA graphs, completion overlap, and optional native NextN/MTP decoding.
Attention TP, context parallelism, and expert TP are currently unsupported.

## Checkpoint and architecture

The registry pins
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348)
at `da920bb0b9f4a06727223a349e55468e38352348`. Its model card reports 320B total
and 18B activated parameters.

| Component | Geometry |
| --- | --- |
| Text stack | 45 blocks; hidden width 4096; vocabulary 154880 |
| Attention | 34 KDA blocks and 11 sparse MLA blocks, with sparse MLA every fourth block |
| KDA | 64 heads × 128 dimensions; width-4 causal convolution; recurrent state with per-key decay |
| Sparse MLA | No positional RoPE; 64 query heads; query rank 1536; KV latent rank 512; head width 256 |
| Indexer | 32 heads × 128 dimensions; learned four-token pooling; 2048 selected token positions plus the incomplete causal tail |
| Feed-forward | First three blocks dense; remaining blocks select 8 of 288 routed experts plus one shared expert; clamped SwiGLU |
| Residuals | Four-stream manifold-constrained hyper-connections, with 20 Sinkhorn iterations |
| Vision | 24 blocks; width 1024; 16 heads; spatial patch 14, temporal patch 2, merge 2; output width 4096 |
| Native MTP | One additional sparse-attention NextN block from the same checkpoint |

Each EP rank loads its routed experts. Attention, dense/shared experts,
embeddings, output head, and the BF16 vision tower are replicated across the
attention-DP ranks. Target routed experts and the first three dense blocks use
NVFP4 checkpoint scales; shared and MTP experts retain their BF16 representation.
Global activation calibration is read before expert partitioning. Sparse KV is
stored in BF16; recurrent state and sparse history live in Engine-owned cache
groups, with prefix reuse and copy-on-write ownership handled by the Engine.

## Run through the shared tools

Download the pinned Modal checkpoint without allocating GPUs:

```sh
uv run scripts/bench.py --model glm-5.3-flash --backend modal --download-only
```

Generate target-only 8K prefill/decode curves and separate untimed Perfetto traces:

```sh
uv run scripts/bench.py --model glm-5.3-flash --backend modal \
  --gpu B200 --world-size 4 --attn-dp 4 --attn-tp 1 --cp 1 --ep 4 \
  --draft-tokens 0 --prefill-context-lengths 8192 \
  --decode-context-lengths 8192 --prefill-batch-sizes 1 \
  --decode-batch-sizes 1,2,4,8 \
  --prefill-chunk-size 8192 --prefill-graph breakable --decode-graph full \
  --output artifacts/glm53/target-curves.jsonl \
  --trace-dir artifacts/glm53/target-traces --plot
```

Batch size is per attention-DP replica; throughput is normalized per GPU. The
shared CLI has separate phase grids: its defaults are prefill at 4K/8K/16K/32K/64K
with batch one, and decode at 8K with batches 1/2/4/8/16/32/64. The example narrows
the first run to 8K and batches through eight. For a separate trace-free prefill
comparison, use `--phases prefill --prefill-context-lengths 4096,8192`, omit
`--trace-dir`, and choose a distinct output file. Larger grids require explicit capacity checks; the default grid is
not a qualification claim. The shared CLI uses seeded synthetic prompts. Omit
`--accepted-drafts` to measure actual speculative acceptance; specifying it simulates acceptance and must be
labeled separately. Controlled-prompt results below use retained recipe-rendered
inputs. See the [benchmark protocol](../../../docs/benchmarks.md). A local CUDA
host uses `--backend local --checkpoint /path/to/checkpoint` with the same topology.

For the requested decode presentation, collect a separate K3 run with
`--draft-tokens 3`, retain its raw JSONL, then generate a **projected MTP3 curve
at 75% acceptance**:

```sh
uv run scripts/plot_bench.py artifacts/glm53/mtp3-curves.jsonl \
  --mode decode --draft-tokens 3 --accept-rate 0.75 \
  --output-dir artifacts/glm53/mtp3-projected-75pct
```

This is a plot-only projection using the measured iteration timing and
`τ = 1 + 3 × 0.75 = 3.25` output tokens per iteration. It does not rewrite raw
timing, observed accepted-token counts, measured acceptance, or measured
throughput. Label the figure **“Projected MTP3, 75% acceptance (τ=3.25)”** and
retain measured-output results beside it. This differs from forcing acceptance
inside a benchmark with `--accepted-drafts`. Prefill figures use measured data
without an acceptance projection.

The parameterized serving app uses the same registry:

```sh
RESIN_MODEL=glm-5.3-flash uv run modal deploy deploy/modal_serve.py
uv run scripts/check_server.py "$ENDPOINT" --model nvidia/GLM-5.3-Flash-NVFP4 \
  --output artifacts/glm53/server-check.json --image --stress --dp-size 4 \
  --prefix-cache-repeats 2112 \
  --reasoning-effort low high max --eval-file /path/to/mmlu20.jsonl \
  --eval-limit 20 --eval-max-tokens 2048 --eval-reasoning-effort low \
  --min-accuracy 0.9
```

Defaults are B200×4, EP4/attention DP4, context capacity 16384, batch capacity 8
per replica, prefill chunk 8192, memory utilization 0.65, and zero draft tokens.
KDA state checkpoints and reusable prefixes align to 8192 tokens; an 8448-token
prompt can reuse its first 8192 tokens within the 16384-token window. This new
serving profile awaits the final full-model audit. The server-check example uses
2112 repeated phrases so its cache probe exceeds the 8192-token checkpoint;
the report retains actual server token counts. Larger contexts need explicit capacity and memory checks;
the checkpoint's advertised 1,048,576-token limit is not qualified. B300 and
other parallel layouts need their own kernel-image and correctness validation.

## Graphs and speculative decoding

Full decode graphs stage tokens, positions, page tables, sampling parameters,
and active expert rows. Breakable prefill captures feed-forward regions while
attention uses prepared sequence metadata. Device token/count feedback uses the
shared `DeviceCompletion` interface for overlap with host completion.

Ordinary target decode and MTP verification choose the same projection capacity
for an executing graph batch bucket: `max(64, graph_bucket_rows × 8)`. The
configured concurrency ceiling and draft count do not enlarge a smaller bucket.
Batches through eight retain **M64**, including the B2 validation profile. Eager
fallback uses the shared CPU request counts and batch catalog, without a device
read or additional collective. An absent or undersized catalog uses the next
power-of-two batch bucket.

Padding is confined to pure projections, dense feed-forward operations, routing,
and the final head. Live rows are restored before recurrence, cache updates,
expert exchange or acceptance. Prefill, MTP draft and repair retain their original
operations. Source `878d83d4` completes the 8K sweep through batch 64 with the
full graph catalog, exact matched-batch K0/K3 prefixes and passing lifecycle checks
during retirement. Outputs can vary across batch shapes and schedules, including
in target-only decoding. Cross-batch bitwise identity is not required for this
integration; matched-batch speculative parity and cache/graph ownership remain
separate requirements.

`--draft-tokens 1` through `7` enable native MTP; the full-model comparisons below
use **K3**. A matched controlled-prompt K0/K3/K5 tuning run provisionally selected
K3 for that workload. Each step drafts tokens, verifies the block in one target
forward, and accepts the matching prefix. Rejection replays only accepted KDA updates,
restores convolution history, and repairs the MTP cache. Prefill primes shifted
hidden/token pairs across chunks, including image embeddings. MTP remains an
explicit option rather than the deployment default.

## Recipe and media

`get_native_recipe("glm5_3_flash", tokenizer)` implements the pinned template,
`low`/`high`/default `max` reasoning, `clear_thinking`, XML tool calls, and streaming
parsing directly from `tokenizer.json`. The template always opens `<think>` and
has no `enable_thinking=False` mode. Stop IDs are `[154820, 154827, 154829]`.

Still images use aspect-preserving bicubic resize, an aligned black-padded
canvas, CLIP normalization, and merge-block-major temporal patch packing. The
pinned defaults allow 16–8000 image tokens. `Glm53FlashMedia` carries pixels,
grids, absolute slots, and pixel-dependent cache identity across ranks. The
runner encodes each image once, inserts features into the applicable prefill
chunks, and releases them with the request. Vision uses axial RoPE; language
positions remain one-dimensional. Video and audio are unsupported. Image encoding
is not separately captured by the runner, and its compilation is outside the
text-path compiler audit.

## Qualification and measured performance

Source **`878d83d4`** completes the controlled 8K decode curve through batch 64
and the 4K–64K prefill curve on four B200s with EP4/attention DP4 and chunk 8192.
The underlying per-bucket runtime `da779293` passed 446 B200 regression tests and
2158 CPU tests (188 skipped). The final **`c2de4fba`** target/K3 HTTP and runner/compiler
audits at serving capacity 16384 remain pending; performance and matched-batch
parity results do not substitute for those checks.

The [wide decode report](../../../artifacts/glm53/wide-8k-878d83d4/accepted-batch-variance/README.md)
and [75%-acceptance figure](../../../artifacts/glm53/wide-8k-878d83d4/accepted-batch-variance/decode-8k-projected75.png)
retain measured K0, measured MTP3 and the requested plot-only MTP3 projection.
All seven batch points pass, with **508 exact K0/K3 common-prefix pairs and
20,180 compared positions**. This covers observed prefixes, not unmatched tails.

| 8K decode metric | Measured K0 | Measured MTP3 | Projected MTP3, 75% acceptance (τ=3.25) |
| --- | ---: | ---: | ---: |
| Batch 1, output tokens/s/user | 65.64 | 107.18 | 143.37 |
| Batch 64, output tokens/s/user | 28.56 | 38.37 | 48.08 |
| Batch 64, output tokens/s/GPU | 1827.91 | 2455.95 | 3077.33 |

This uses four authored native-Low tasks with literal `x` filler and identical
copies within each rank; public capacity 8438, configured batch 64, normal graphs
and overlap, 5 warmup and 32 measured completions. Observed acceptance is
47.7–53.1%, not 75%. Timed and traced windows contain no EOS and remain full.
Separate 8K traces at batches 1, 8 and 64 are linked from the report. These
controlled prompts do not represent a varied conversation corpus.

The retirement cohort completes all 256 requests and 10,240 tokens per mode;
execution, ownership/drain, graph, budget and 64→32→16→8 transition checks pass.
The original strict comparison retains 72 differing complete outputs under
different batch histories. A [target-only comparison](../../../artifacts/glm53/wide-8k-878d83d4/analysis-independent/k0-cross-batch-variance.md)
also shows wording changes across fixed batches, without retirement. Ordinary
cross-batch variation is accepted for this integration and is not a release blocker.
The saved strict failure remains historical evidence; it is not relabeled as exact
parity, and it does not weaken the matched-batch or lifecycle checks.

The local [qualification report](../../../artifacts/glm53/qualification-report.md)
retains exact source identities, commands, plots, traces and historical rejected
experiments. The ignored artifact bundle is not distributed with a fresh clone.
Earlier projection-shape parity failures are historical; subsequent comparisons
passed the explicitly bounded scopes below.

| Completed check | Source and result |
| --- | --- |
| B200 family regression suite | `40206b4a`: all 304 passed, including graph replay, cache addressing, fixed projections, KDA/MTP rollback, vision and exact fused pointwise operations |
| Per-bucket runtime regression suites | `da779293`: 446 B200 tests and 2158 CPU tests pass; final serving HTTP/compiler qualification remains pending |
| Short production K0/K3 cohorts | `91d5d30a`: 132 request pairs and 7396 compared tokens, all exact |
| Controlled-prompt 12-point comparison | `91d5d30a`: 180 request pairs and 7380 compared tokens, all exact; K3 was 1.45–1.87× faster than matched K0 |
| Fresh 8K production K0/K3 comparison | `40206b4a`: all 60 common-prefix request pairs exact at batches 1/2/4/8 |
| Complete cross-source 8K outputs | All 120 same-mode `91d5d30a`/`40206b4a` outputs and lengths exact; this is separate from the earlier 12-point scope |
| Controlled prefill chunk comparison | `da779293`: all four 4K/8K prompt × 4K/8K chunk accounting points pass; first-token differences across chunks are retained |
| Wide 8K matched-batch K0/K3 comparison | `878d83d4`: 508 common-prefix pairs / 20,180 positions exact at batches 1/2/4/8/16/32/64; complete retirement lifecycle checks pass |
| Full prefill curve | `878d83d4`: all five 4K/8K/16K/32K/64K points pass accounting, native-input and repeated first-token checks; four concurrent requests per cohort |

In the source `40206b4a` 8K controlled-prompt sweep, K3 reaches
**110.41 output tokens/s/user** at batch one and **733.67 output tokens/s/GPU** at
batch eight; matched K0 reaches 65.39 and 474.95 respectively. This uses four
assistant-authored tasks with literal `x` filler to reach each length, rendered with the native Low recipe and repeated within each rank; public
context 8356, capacity 8, chunk 512, full graphs and overlap, 5 warmup and 32 measured
completions, observed acceptance, and profiling excluded. The same-protocol
comparison to source 91 shows 12.6–15.9% higher K0 throughput and 13.5–15.4% higher K3
throughput. This short-window result does not establish broad model quality or
actual terminal-EOS coverage. Observed acceptance is real model acceptance; these
prompts do not represent varied conversations.

In a separate matched run at public capacity 8438, K3 was 4.9–7.5% faster than K5
across batches 1/2/4/8; all 120 K0/K3 and K0/K5 common request-prefix comparisons
passed. That [width comparison](../../../artifacts/glm53/mtp-width-8k-40206b4a/README.md)
provisionally selects K3 for these retained controlled prompts. It does not
establish the best draft width for other conversations.
No matched external speed benchmark supports a state-of-the-art claim.

The completed [prefill chunk comparison](../../../artifacts/glm53/prefill-chunks-da779293/README.md)
and [figure](../../../artifacts/glm53/prefill-chunks-da779293/comparison/prefill-chunk-ab.png)
use source `da779293`, K0, public capacity 8200 and four concurrent fresh requests,
one per attention-DP rank/GPU. Each point includes one warmup and three measured
complete cohorts; only 8K has two separate trace cohorts. These retain the four
controlled authored prompts and native Low rendering.

| Prompt tokens | Prefill chunk | Mean cohort latency | Input tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 4096 | 4096 | 259.91 ms | 15,759 |
| 8192 | 4096 | 531.34 ms | 15,418 |
| 4096 | 8192 | 260.05 ms | 15,751 |
| 8192 | 8192 | 515.76 ms | 15,883 |

At 8K/chunk 8192, three cohorts process 98,304 aggregate input tokens in
1.547279092 s: 63,533.46 input tokens/s total, divided by four GPUs once to give
15,883.37/GPU. Cached tokens are zero; the 12 first output tokens are excluded
from the numerator. Cohort latency sums the slowest-rank completion interval for
every prefill chunk, excluding admission and benchmark bookkeeping between steps;
it is not HTTP arrival-to-first-token latency.

Chunk 8192 improves 8K throughput by 3.02% in this run; 4K differs by 0.06%.
First tokens repeat within each configuration, but the 8K first token changes
between chunks on ranks 0 and 1; all 4K ranks and 8K ranks 2 and 3 match. This
records numerical chunk sensitivity, not a full-generation quality result.
The second fresh process used a compact startup memory observer; both observer
versions are retained. Timed inference was unchanged, but startup durations
cannot be compared as compilation benchmarks. The separate full curve below uses
capacity 65544; these capacity-8200 measurements are not relabeled as that run.

The completed [full prefill report](../../../artifacts/glm53/prefill-full-878d83d4/report/README.md)
and [figure](../../../artifacts/glm53/prefill-full-878d83d4/report/prefill-4requests-878d83d4.png)
use source `878d83d4`, K0, chunk 8192 and public capacity 65544. Each measured
cohort contains **four simultaneous fresh requests, one per attention-DP rank/GPU**.

| Tokens per request | Mean full-cohort latency | Input tokens/s/GPU |
| ---: | ---: | ---: |
| 4096 | 260.79 ms | 15,705.82 |
| 8192 | 515.93 ms | 15,878.15 |
| 16384 | 1079.83 ms | 15,172.74 |
| 32768 | 2370.41 ms | 13,823.78 |
| 65536 | 5387.98 ms | 12,163.36 |

Each point has one warmup and three complete measured cohorts. The aggregate
input numerator is `3 × 4 × prompt_length`; divide by total cohort time and four
GPUs exactly once. Cache hits are zero and output tokens are excluded. All 20
rank/input records pass exact native-input and repeated first-token checks, with
no eager fallbacks. Only 8K has separate traces, covering two complete target
forwards on each rank. The larger fixed capacity affects sparse-attention work;
this is not a matched-capacity comparison against the earlier chunk A/B. Like
that A/B, these are controlled authored prompts and engine cohort timings, not
HTTP TTFT or a quality evaluation.

The earlier chunk 512 diagnostic remains historical: source `40206b4a` 8K/B1
mean prefill latency was 2.5091 s (3264.9 input tokens/s/GPU), versus 1.9765 s
(4144.6) for source 91. Its traces showed less GPU work but more launch/scheduling
gaps. Those results and 8K-only traces remain in the
[8K report](../../../artifacts/glm53/fusion-8k-40206b4a/README.md).

The original target HTTP run on source 91 scored Low 20/20 and Max 19/20, with Max 16
capped at 2048 output tokens. That report and its then-declared failed gate stay
unchanged. At the user's request, the prospective Max budget is now 8192, while
Low stays 2048; neither historical score transfers to the new run. The final
quality harness uses context 16384, batch 2 and the default serving/audit chunk
8192. Performance runs record their explicit chunk settings.

Final-source `c2de4fba` target and native K3 HTTP evaluation are pending, using the original
20 questions, Low/2048 and Max/8192 output budgets, a 90% accuracy floor for each
effort, and no missing, truncated or errored responses. Behavior, tools, media,
concurrency and stress remain separate checks. The old 2048-budget report is
preserved; the revised budget does not change its historical score.

The pending runner audit covers actual 8192-token prefix hits on an 8448-token
prompt with exact 64-token warm/cached/fresh continuations, separate genuine-EOS cohorts, paired 8K raw-64
outputs, and the 16382+2 public-context boundary. All-rank compiler observations
compare loaded variants, compiler counters and artifact hashes after those
requests and 8K traces against bounded B1/B2 warmup. That audit ends before HTTP;
it does not qualify image-encoder compilation or every vendor/driver JIT path.

The completed wide-decode and full-prefill curves use the explicit capacities
above; they do not qualify the checkpoint's full advertised context range.
Decode figures show the labeled 75%-acceptance MTP3 projection alongside unchanged
measured results. Prefill plots show measured input throughput and complete-cohort
latency. No matched external speed benchmark establishes state-of-the-art performance.

MMLU here is a fixed 20-question integration smoke subset, not a full MMLU score.
CPU, primitive GPU, production-token, HTTP and compiler checks have distinct
scopes. Serving defaults are context 16384, prefill chunk 8192 and target-only
decoding; full-model qualification of that profile remains pending.

## Sources and ownership

- [NVIDIA checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348), MIT; Z.AI's notice is retained in [glm-MIT.txt](../../../crates/resin-recipe/licenses/glm-MIT.txt).
- [Transformers GLM5-Next](https://github.com/huggingface/transformers/tree/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next), revision `50e35cc0ca18ec19b80653690de553e6decf74f5`, Apache-2.0: architecture, vision and processor reference.
- [SGLang model and quantization reference](https://github.com/sgl-project/sglang/tree/5971f47bc8eacd8989cb6dd67e46b4fc6dbec2ad/python/sglang/srt), revision `5971f47bc8eacd8989cb6dd67e46b4fc6dbec2ad`, Apache-2.0; [serving mHC reference](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/kernels/ops/layernorm/mhc.py), revision `baa9befa381511cd4d56f83980951e5a45fffc6d`.

[config.py](config.py) owns backend-neutral geometry; [resin-recipe](../../../crates/resin-recipe/README.md)
owns formatting and media preprocessing. The [CUDA family package](../../../models/cuda/resin_models_cuda/glm5_3_flash/)
owns loading, attention, vision, MTP, cache views, and Runner integration. It reuses
the shared NVFP4 expert path, FLA KDA, FlashMLA, and Engine graph/cache protocols.
Adapted-code attributions are recorded in [NOTICE](../../../NOTICE).
