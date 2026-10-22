# Benchmark sweeps

The sweep loads one model per GPU, keeps it resident across cases, and measures
synchronous engine steps after warmup. It writes raw measurements to JSONL;
plotting runs locally without importing PyTorch. Startup, weight loading, and
graph capture are outside the measured windows and recorded separately.
The requested result file is written after normal sweep completion. Each completed
point is also flushed as a `[bench-result] {json}` log line, so earlier points can
be recovered from captured logs if a later case fails.
Before capture, the launcher logs each rank's cache budget and checks every
cohort's peak context against the native `CacheLayout.estimated_parent_count`
estimator. Decode includes warmup, measured generation, and draft write
lookahead. A case that cannot remain fully resident fails before prefill rather
than measuring a smaller admitted batch.
The chart axes and speculative projection follow
[DeepInfer's sweep plotter](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/plot_pareto.py).

Download the pinned checkpoint before allocating GPUs:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v4 --download-only
```

Run prefill and decode sweeps, then produce both charts:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v4 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v4.jsonl
uv run scripts/plot_bench.py bench/deepseek-v4.jsonl
```

To reuse one long prefix within each attention replica, set `--shared-prefix-tokens`.
`--context-lengths` still specifies the total context; the remainder is private:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm --context-lengths 131072 \
  --shared-prefix-tokens 126976 --batch-sizes 1,4,16,64,256 \
  --phases decode --prefill-chunk-size 1024 --gpu-memory-utilization 0.5 \
  --output bench/deepseek-v41-shared.jsonl
```

The shared prefix is seeded once per case outside timing. Each request receives
a distinct private suffix, and its reported cached length must exactly match the
requested prefix. Unsupported cache/checkpoint reuse fails explicitly. Records
include shared/private token counts and actual cached-length bounds; prefill
throughput counts newly processed inputs. Independent and shared-prefix points
must use separate chart files. The default is zero shared tokens.
The lower cache budget in this long-context example leaves workspace for the
captured decode graphs and uncaptured prefill operations. Cache preflight still
requires the entire requested cohort to fit; unused arena capacity does not
improve throughput.

The default DeepSeek workload uses four draft tokens, forces three accepted
drafts per verification step, and balances expert routing. These are controlled
performance measurements, rather than measurements of natural acceptance or
routing. Use `--natural-acceptance` to run the actual verifier and
`--no-balanced-experts` to preserve model routing. The JSONL records both the
configured acceptance policy and the observed acceptance rate.

V4.1 uses the pinned `DeepSeek-V4.1-Flash-1017` checkpoint and its own native text
architecture and recipe. Download and sweep it with:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v41 --download-only
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v41.jsonl
uv run scripts/plot_bench.py bench/deepseek-v41.jsonl
```

`--expert-backend deepgemm` selects the upstream fused MoE operation, including
sparse EP exchange and shared experts. Its initial mesh is ADP=EP=world size,
attention TP1, CP1, and expert TP1. The default `flashinfer` backend retains the
existing prepared expert path. Backend identity is recorded and checked when
combining chart points. The two-GPU B300 kernel/adapter probe passes numerical,
idle-rank, changing-route, and CUDA graph checks, including 1,280 physical rows
per rank (batch 256 with four draft positions). The separate Engine smoke needs
only two B300s and generates deterministic synthetic weights locally:

```bash
uv run modal run deploy/modal_deepinfer_kernels.py --mode engine-smoke \
  --output artifacts/benchmarks/deepseek-v41-synthetic-engine-validation.json
```

It runs the production V4.1 runner, pages, EP/ADP coordinator, scoring, and graph
paths at K0 and K4. Its unrelated draft weights exercise rejection and rollback;
the real-checkpoint check below additionally exercises accepted drafts. This
synthetic smoke does not test model quality or Engram tables.

Both DeepSeek launchers share the same checkpoint volume and completion marker.
For an older generic download whose files are already complete, publish only
the small inventory marker with
`uv run modal run deploy/modal_bench.py --model deepseek-v41 --publish-only`.
This CPU operation stats the indexed files and publishes metadata atomically;
it does not download or read the weight bodies. New downloads publish the marker
automatically. GPU sweeps require that complete inventory. The legacy
`modal_deepseek_v4.py` run/serve/benchmark paths use attention TP and now reject
these one-KV-head MLA models; their checkpoint download remains usable.

Attention TP means KV-head sharding, so both DeepSeek MLA families require
`attn_tp=1`. Earlier four-GPU TP4 debug reports sliced query heads while
replicating KV; their results do not qualify this contract. The initial V4.1
target used eight B300s with attention DP8/EP8 and replicated Engram tables.
The current runner shards Engram rows across the device domain and stores packed
FP8 sliding KV and FP4 compressed KV. On eight ranks, Engram table storage drops
from 188.83 to 23.60 GiB per GPU. The full checkpoint passes eager/captured K0
checks on both ADP2/EP2 and ADP8/EP8, and K4 on ADP8/EP8. See the
[family memory and qualification details](../models/cuda/resin_models_cuda/deepseek_v41/README.md)
for exact allocations and the distinction between the current path and older
replicated-table/BF16-cache measurements.

Run the target comparison with normal routing and no speculation:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm \
  --draft-tokens 0 --natural-acceptance --no-balanced-experts --validate-model \
  --prefill-chunk-size 8 --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepseek-v41-adp8ep8-target-validation.json
```

Run a separate natural four-draft DSpark comparison with:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm \
  --draft-tokens 4 --natural-acceptance --no-balanced-experts --validate-model \
  --prefill-chunk-size 8 --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepseek-v41-adp8ep8-dspark-validation.json
```

Validation writes a JSON report rather than chart measurements. The implementation uses fixed-width greedy
proposals, loads the confidence head without applying its length policy, and
supports up to the released five-token block. Benchmark defaults remain four
drafts. On 2026-10-22, both commands passed on all eight B300 drivers: K0 replayed
seven full decode graphs, while natural K4 replayed three and accepted all 12
proposed drafts on the short counting prompt. Eager/captured generated IDs,
selected-token scores, and prompt scores matched exactly; the first eight
K4 IDs also matched K0. These are focused execution checks, not a model-quality
evaluation or a natural-acceptance estimate. Wider meshes, longer contexts, and throughput tuning need separate qualification;
see the [family limits](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

Available model names are `deepseek-v4`, `deepseek-v41`, `gpt-oss-20b`, and
`gpt-oss-120b`. The GPT-OSS benchmark path currently uses the Hopper expert kernel
and has no draft model:

```bash
uv run modal run deploy/modal_bench.py \
  --model gpt-oss-20b --gpu 'H100!' --world-size 1 --attn-dp 1 --attn-tp 1 --ep 1 \
  --draft-tokens 0 --prefill-chunk-size 1024 --output bench/gpt-oss-20b.jsonl
uv run scripts/plot_bench.py bench/gpt-oss-20b.jsonl --draft-tokens 0
```

Use `--gpu 'H100!'` for an H100 comparison. Modal can automatically upgrade
plain `H100` requests to H200; the exclamation mark
[disables that upgrade](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).
Records always retain the actual device name.

Use `--phases decode` or `--phases prefill` for one phase and the matching
`--mode decode` or `--mode prefill` when plotting. Parallelism is configured with
`--world-size`, `--attn-dp`, `--attn-tp`, `--cp`, and `--ep`. The launcher defaults
to eight GPUs, attention DP8, attention TP1, CP1, and EP8 for DeepSeek; override
all relevant fields when selecting another model. Batch sizes count
requests **per attention-DP replica**; the global batch is `batch_size * attn_dp`.
`--prefill-chunk-size`, `--prefill-graph`, and `--decode-graph` select execution
policies. Defaults are a 4,096-token prefill chunk, breakable prefill graphs,
full decode graphs, five warmup steps, and twenty measured steps. Prefill may
measure additional steps to finish its request cohort.

## Reading the charts

The plotter saves `decode.png`, `decode.svg`, `prefill.png`, and `prefill.svg`
under the JSONL directory's `plots/` folder. Use `--output-dir` to change it.
Different context lengths form separate curves.

To compare meshes for the same model, pass their result files together with
`--compare-topologies`. Each topology becomes a separate curve; model revision,
source, kernels, graph and workload policies must still match. Decode comparisons
allow different prefill capacities between meshes because prefill runs outside
the timed decode window:

```bash
uv run scripts/plot_bench.py bench/gpt-oss-120b-{tp1,tp2,tp4,ep2,ep4}.jsonl \
  --compare-topologies --draft-tokens 0 --mode decode \
  --output-dir bench/gpt-oss-120b-comparison
```

The decode chart has per-user output tokens/s on the horizontal axis and output
tokens/s/GPU on the vertical axis. Point labels are requests per attention-DP
replica. Its acceptance setting is an explicit projection of measured step
latency:

```text
τ = 1 + draft_tokens * accept_rate
per-user tokens/s = measured_steps / window_seconds * τ
output tokens/s/GPU = per-user tokens/s * global_batch_size / gpu_count
```

Plot defaults are `--draft-tokens 4 --accept-rate 0.75`, hence four committed
tokens per step. The draft count must match the measured verification width.
Changing `--accept-rate` rescales throughput using the same measured latency;
it does not rerun verification or change the recorded observed acceptance.
The chart footer shows both projected and observed acceptance. Actual measured
output throughput remains available as
`committed_tokens / window_seconds / gpu_count` in the raw data.

The prefill chart uses mean milliseconds per measured prefill step horizontally
and actual input tokens/s/GPU vertically. Point labels show actual unique input
tokens per global step, rather than the configured token budget. These counts
exclude replication across attention TP/CP ranks.

## Raw data and comparisons

Each JSONL row has `schema_version: 1`, a phase, context length, local/global
batch sizes, GPU count, attention-DP size, every measured step duration, and the
sum of those durations as `window_seconds`. `input_tokens` counts actual unique
prefill inputs; `prefill_tokens` is their measured per-step mean.
`committed_tokens` always counts generated outputs. `draft_tokens`,
`accepted_drafts`, `measured_accept_rate`, and `balanced_experts` describe the
workload. A null `accepted_drafts` means natural verification.

Balanced routing covers physical router rows, including any padding used by a
captured prefill bucket. Successive assignments cycle through EP owners first,
then through experts within each owner, so even a small batch exercises balanced
communication. Attention-DP replicas receive offset portions of that assignment;
other attention ranks apply the same ordinal rule to their router rows. The real router GEMM still runs,
and selected router scores still determine weights.
`graph_counts` sums replay and fallback counter changes over unique attention-DP
drivers during measurement, excluding warmup. Distributed timings use the slowest replica driver per step;
token counts are summed over unique drivers. `window_seconds` sums timed
`Engine.step` calls, excluding admission and benchmark metric reductions.
`context_length` is the requested prompt length; the
`measured_context_start_min/max` and `measured_context_end_min/max` fields record
the actual contexts after warmup and measurement as decode advances.

Metadata records the model, checkpoint revision, Git commit and source digest,
hardware, parallelism, graph policies, checkpoint quantization, and prefill
chunk size. The plotter rejects incompatible configurations within a phase
instead of connecting them as one curve. Keep separate result files for
different models, meshes, quantization, graph policies, or acceptance policies.
The source digest also prevents combining different uncommitted code under the
same Git commit. Repeated identical sweep points use their latest record;
validation still checks all earlier rows.

The reusable measurement API is `resin.bench.run_sweep`; the Modal launcher is
`deploy/modal_bench.py`. The harness observes the real runner and scheduler;
it does not substitute a synthetic latency model. Plot tests use explicitly
synthetic fixtures only.
