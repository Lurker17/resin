# GPT-OSS runner status

`resin_models_cuda.gpt_oss.cuda.GptOssCudaRunner` runs the released GPT-OSS 20B
and 120B checkpoints on Hopper GPUs with BF16 activations and MXFP4 experts.
The model implementation and checkpoint interpretation live in
`resin_models_cuda.gpt_oss`; shared backend helpers cover page lowering,
staging, sampling, and graph lifetime. Importing `resin` alone does not import
Torch or initialize CUDA.

## Cache and execution

The runner declares one History group and one Window group over a common
physical slab pool. Both use 128-token blocks. Sliding attention reads a
128-token window; physical retention is `8192 + 128 - 1` tokens to preserve the
earliest query's history throughout an 8192-token prefill chunk. The layout
explicitly declares `read_window_tokens=128`, and the native Engine enforces the
safe write span.

The Engine binds one physical arena and passes chronological sparse group rows
and exact append destinations. The runner lowers these rows for FlashAttention-4;
MXFP4 experts use the selected FlashInfer MoE adapter. Sampling honors each
request's policy, seed, and absolute sequence position. `prefix_cache=True`
enables aligned, exact-token sharing of both groups with copy-on-write as the
Window advances. The Modal HTTP endpoint enables this option.

Decode graph shapes are `(batch_size, full_page_table_width)` pairs. The runner
selects a configured shape that fits the current batch; prefill and other
shapes execute eagerly. Graphs stage the current native plan, and do not
schedule ahead or reserve a future batch. `cuda_graph_stats["decode"]` reports
captures, replays, and fallbacks.

Tensor parallelism shards attention heads, KV storage, and expert intermediate
dimensions. The Modal harness supports sizes 1, 2, 4, and 8 with matching H200
allocations. Attention geometry follows the checkpoint and the tensor-parallel
layout rather than one fixed global head count.

`cache_parent_capacity` is a total physical-parent budget shared by both
groups. By default the runner sizes it from post-load memory and
`gpu_memory_utilization`; page zero is a null row. The runner materializes its
token result before synchronous `forward()` returns. Once device execution has
begun, failures quarantine resources even if synchronization succeeds, because
in-place state changes cannot be undone by waiting.

## Validation

The Modal harness uploads local source and builds a Linux wheel. Its `run` mode
checks a real prompt, while `bench` can exercise long prefill, batching, and
graph replay. For example:

```bash
uv run modal run deploy/modal_gpt_oss.py \
  --mode run --model 20b --max-new-tokens 32 --temperature 0
```

Current-tree results and their limits are recorded in the
[architecture implementation report](architecture-review.md). Healthy-path
GPU runs complement native ownership tests and host-side failure injection;
they do not establish equivalence against an independent reference model or
coverage of arbitrary device faults.
