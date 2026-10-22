# Benchmark sweeps

The sweep loads one model per GPU, keeps it resident across cases, and measures
synchronous engine steps after warmup. It writes raw measurements to JSONL;
plotting runs locally without importing PyTorch. Startup, weight loading, and
graph capture are outside the measured windows and recorded separately.

Download the pinned checkpoint before allocating GPUs:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v4 --download-only
```

Run prefill and decode sweeps, then produce both charts:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v4 --gpu B300 --world-size 4 --attn-tp 4 \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v4.jsonl
uv run scripts/plot_bench.py bench/deepseek-v4.jsonl
```

The default DeepSeek workload uses four draft tokens, forces three accepted
drafts per verification step, and balances expert routing. These are controlled
performance measurements, rather than measurements of natural acceptance or
routing. Use `--natural-acceptance` to run the actual verifier and
`--no-balanced-experts` to preserve model routing. The JSONL records both the
configured acceptance policy and the observed acceptance rate.

Available model names are `deepseek-v4`, `gpt-oss-20b`, and
`gpt-oss-120b`. The GPT-OSS benchmark path currently uses the Hopper expert kernel
and has no draft model:

```bash
uv run modal run deploy/modal_bench.py \
  --model gpt-oss-20b --gpu H100 --world-size 1 --attn-tp 1 \
  --draft-tokens 0 --output bench/gpt-oss-20b.jsonl
uv run scripts/plot_bench.py bench/gpt-oss-20b.jsonl --draft-tokens 0
```

Use `--phases decode` or `--phases prefill` for one phase and the matching
`--mode decode` or `--mode prefill` when plotting. Parallelism is configured with
`--world-size`, `--attn-dp`, `--attn-tp`, `--cp`, and `--ep`. Batch sizes count
requests **per attention-DP replica**; the global batch is `batch_size * attn_dp`.
`--prefill-chunk-size`, `--prefill-graph`, and `--decode-graph` select execution
policies. Defaults are a 4,096-token prefill chunk, breakable prefill graphs,
full decode graphs, five warmup steps, and twenty measured steps. Prefill may
measure additional steps to finish its request cohort.

## Reading the charts

The plotter saves `decode.png`, `decode.svg`, `prefill.png`, and `prefill.svg`
under the JSONL directory's `plots/` folder. Use `--output-dir` to change it.
Different context lengths form separate curves.

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
captured prefill bucket. Attention-DP replicas receive different portions of the
round-robin assignment; attention TP/CP copies keep the same assignment.
`graph_counts` records replay and fallback counter changes during measurement,
excluding warmup. Distributed timings use the slowest replica driver per step;
token counts are summed over unique drivers. `window_seconds` sums timed
`Engine.step` calls, excluding admission and benchmark metric reductions.

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
