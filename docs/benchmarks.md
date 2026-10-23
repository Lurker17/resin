# Benchmark sweeps

The sweep loads one model per GPU, keeps it resident across cases, and measures
engine completions after warmup. It writes raw measurements to JSONL;
plotting runs locally without importing PyTorch. Startup, weight loading, and
graph capture are outside the measured windows and recorded separately.
Overlap is enabled by default for capable runners; pass `--no-overlap` to time
the synchronized reference path. Overlapped decode measures continuous completion
intervals, including CPU work between completions, without per-step CUDA
synchronization or metric reductions. Warmup reductions are deferred too, so
the first measured completion cannot advance in an untimed reduction. Each point
records actual enqueue/continuation counts; static graph phase events are omitted
when a following replay could overwrite them. Perfetto traces contain two live
completions after timing, with no benchmark metric collectives between them.
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
from 188.83 to 23.60 GiB per GPU. Earlier full-checkpoint reports passed
eager/captured K0 on ADP2/EP2 and ADP8/EP8, and K4 on ADP8/EP8; those reports
predate the prepared DeepInfer forward operators. See the
[family memory and qualification details](../models/cuda/resin_models_cuda/deepseek_v41/README.md)
for exact allocations and the distinction between the current path and older
replicated-table/BF16-cache measurements.

Run the current prepared forward comparison with natural routing and K3:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 4 --attn-dp 4 --attn-tp 1 --ep 4 \
  --expert-backend deepgemm \
  --draft-tokens 3 --natural-acceptance --no-balanced-experts --validate-model \
  --prefill-chunk-size 8 --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepinfer-v41-forward-ep4-k3-validation.json
```

Validation writes a JSON report rather than chart measurements. At clean commit
`7087a31`, this [four-B300 run](https://modal.com/apps/ekzhang/main/ap-RGV18BLKlCCRqSzjNVwMBm)
passes with the released 384 experts on all four drivers. Each driver runs four
prefill and three decode steps, including three full decode graph replays. Eager
and captured greedy IDs, prompt logprobs and generated-token logprobs match
exactly (maximum absolute difference zero). K3 naturally accepts all nine drafts
on this short counting prompt. The report is
`artifacts/benchmarks/deepinfer-v41-forward-ep4-k3-validation.json`; this focused
execution check does not establish model quality or representative acceptance.

The [FP8 output-boundary probe](https://modal.com/apps/ekzhang/main/ap-AeDmb4KbJfjlNfVVL0wA0M)
uses the released H64/G8/R1024 geometry at 1/5/1,024/1,280 rows. Its direct FP8
O-A values, packed scales and final O-B outputs equal BF16-plus-cast bitwise,
with five exact graph replays. The separate
[FP32 route EP2 probe](https://modal.com/apps/ekzhang/main/ap-72WXKqszkwgN9zSWQIn3gv)
at `14af36b` uses synthetic 16-expert top-6 routing, widths 5,120/2,304 and
1,280 rows/rank; it qualifies route precision and captured FFN execution, rather
than the full 384-expert checkpoint. Artifacts live under
`artifacts/benchmarks/2026-10-22-v41-forward-port-microprobes/`, alongside
`artifacts/benchmarks/deepinfer-v41-output-probe.json`.

The [prepared-forward EP8 baseline](https://modal.com/apps/ekzhang/main/ap-lfw5WzD1XFQrU1UZ06dAXN)
at clean `de438e4` completes six B300 points at 128K context: 126,976 shared tokens
and a 4,096-token private suffix, K3 with two forced draft acceptances, balanced
experts, five warmup steps and 20 measured steps. Device times use CUDA phase
events; Engine times include native planning and distributed coordination.
Throughput below projects 75% draft acceptance (3.25 tokens/step); measured
acceptance was fixed at two of three drafts (3 tokens/step).

| Batch/GPU | Engine ms | Device ms | Projected tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 8 | 60.920 | 21.640 | 426.79 |
| 16 | 65.611 | 22.727 | 792.55 |
| 32 | 70.023 | 23.920 | 1,485.24 |
| 64 | 77.823 | 26.926 | 2,672.71 |
| 128 | 99.315 | 33.906 | 4,188.70 |
| 256 | 140.853 | 46.469 | 5,906.86 |

At batch8/GPU, the comparable four-GPU baseline takes 41.94 ms in the Engine and
26.75 ms on device. Eight GPUs reduce device time but increase the interval
outside those device phases to 39.28 ms; at batch256/GPU that interval is
94.38 ms. This difference includes host work and synchronization waits, so it
does not measure isolated CPU execution. Raw records, source provenance, plots
and all-rank CPU/CUDA Perfetto traces are archived under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-deepinfer/`; four-GPU
records use the sibling `2026-10-22-deepseek-v41-b300-ep4-deepinfer/` directory.
The B200 attempt failed during Quack RMSNorm graph capture with
`cudaErrorNoKernelImageForDevice`, producing no throughput result; its log and
failure metadata are under the sibling `2026-10-22-deepseek-v41-b200-ep8-deepinfer/`.

These measurements precede the current CPU planning, native overlap and prepared
metadata changes. Current source requires fresh full-checkpoint validation and
throughput measurement.
The implementation uses fixed-width greedy proposals, loads the confidence head
without applying its length policy, and supports up to the released five-token
block. Benchmark defaults remain four drafts. Wider meshes and longer contexts
need separate qualification; see the
[family limits](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

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

## Decode traces

Add `--trace-dir bench/traces` to save Perfetto-compatible Chrome JSON traces.
After each decode timing window, every attention-DP driver profiles one extra
full-cohort step with CPU and CUDA activities, including individual CUDA graph
nodes. This step is excluded from measured durations, acceptance, counters and
context bounds. Prefill and checkpoint-validation runs are not traced.

The launcher downloads compressed raw rank traces and a merged
`decode-ctx<length>-b<batch>-all-ranks.json.gz` for each case. The merger aligns
PyTorch timestamp bases and separates rank process/flow IDs. Open the merged
file in [Perfetto](https://ui.perfetto.dev/). Source and workload metadata stay
in the trace. Raw files can also be merged again locally:

```bash
uv run scripts/merge_bench_traces.py bench/traces
```

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

## Recorded decode sweeps

The 2026-10-22 GPT-OSS-120B comparison uses actual H100 80GB HBM3 GPUs,
checkpoint revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, and clean runtime
commit `b2f7c3c`. All five meshes use independent 8,192-token prompts, balanced
experts, no drafts, full decode graphs, a 32-token prefill chunk, and 0.95 memory
utilization. Each point retains its complete 20-warmup/100-measurement window;
measured contexts advance from 8,212 to 8,312 tokens. All 25 points retain the
full requested cohort and report zero measured decode fallbacks.

TP meshes use attention TP equal to world size, attention DP1, and EP1. EP meshes
use attention DP and EP equal to world size, with attention TP1. CP is 1 throughout.
TP1 sweeps batches 1/2/4/8/14; the other meshes sweep 1/4/16/32/64. Point labels
count requests per attention-DP replica, so EP4 batch 64 means 256 global requests.

| Mesh | Batch at best measured throughput | Output tokens/s/GPU | Per-user tokens/s |
| --- | ---: | ---: | ---: |
| TP1 | 14 | 574.750 | 41.054 |
| TP2 | 64 | 488.915 | 15.279 |
| TP4 | 64 | 286.481 | 17.905 |
| EP2 | 64 | 988.479 | 15.445 |
| EP4 | 64 | 1,013.632 | 15.838 |

The largest per-point timing CV is 4.05%; the largest half-window drift is
4.93%. These are complete fixed-window means. Raw JSONL files and PNG/SVG charts
are saved locally under `artifacts/benchmarks/2026-10-22-gpt-oss-120b-h100-*`;
the combined chart is in the `comparison` directory and the summary is
`2026-10-22-gpt-oss-120b-h100-summary.json`.

The separate V4.1 EP8/ADP8 sweep uses eight B300s at runtime commit `41e7814`,
131,072-token contexts with a physically shared 126,976-token prefix and a
4,096-token private suffix, and fixed three-of-four draft acceptance. Its
five-warmup/20-measurement windows reach 1,430.895 output tokens/s/GPU at batch
256 per replica, or 5.589 tokens/s/user. Its source and protocol are retained
separately in `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-128k` and the
[family qualification record](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

## Shared sampler checks

On a CUDA environment with deployment dependencies, run:

```bash
uv run pytest tests/test_sampling_cuda.py -q
```

These checks cover request-local random streams across batch permutations and
padding, high seed and offset bits, conditional top-k/top-p distributions, and
mutable graph inputs. The graph catalogue captures batches 1 through 128 in a
shared pool, and the vocabulary smoke uses 201,088 classes. Environments without
CUDA skip these tests; the CPU sampler contract tests remain in the normal suite.

Filtering uses FlashInfer with deterministic nucleus normalization. A shared
categorical kernel owns the random draw: the full request seed is its Philox key,
while sampling offset and vocabulary index occupy separate counter words. This
keeps the stream independent of physical batch position, including when a graph
replays with new request state.
