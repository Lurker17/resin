# Authoring a CUDA model runner

A model runner receives a frozen `BatchPlan`, stages its token and page-table
arrays, executes the model, and returns sampled token IDs. The Engine already
owns admission, batching, cache allocation, prefix reuse, and request lifetime;
do not recreate those policies in the forward pass.

## Shape rules

- Write persistent kernels whose work is bounded by runtime metadata. Attention
  must not specialize or do useful work up to the configured maximum sequence
  length, and MoE kernels must not specialize on the current token count.
- Do not make batch size 1 a separate architecture. A small number of measured
  batch regimes is acceptable when the implementation genuinely changes, such
  as switching tiny projections to a TinyGEMM path.
- Padding may establish a safe tile boundary, but it must not turn logical
  sequence length into work. Pass logical lengths to the kernel and mask or
  terminate persistent work at those lengths.
- Treat new JIT artifacts for novel prompt lengths, token counts, or ordinary
  batch sizes as a bug. Exercise several unseen shapes while recording compiler
  logs and artifact-cache entry counts.
- Keep `BatchPlan` and cache CSR fields as NumPy arrays. Stage them through
  reusable pinned buffers with nonblocking copies instead of converting large
  prefills or page tables to Python lists.
- Use `cuda.tile.assume_divisible_by` for true runtime divisibility facts. It is
  an unchecked compiler hint, not a runtime assertion; a false claim is
  undefined behavior.

## Kernel selection and fusion

- Prefer FlashInfer and Quack kernels when they implement the operation and
  layout directly. Prefer FlashInfer collectives; its all-reduce selects the
  one-shot path for small tensors automatically.
- Use FlashAttention/FlashInfer attention plans rather than rebuilding generic
  attention from PyTorch operations.
- Write custom cuTile or TileLang when fusing a long architecture-specific
  chain materially removes launches or memory traffic. MHC normalization and
  DeepSeek-V4 compressor/state updates are representative examples.
- Keep the launch sequence short. Compare a Perfetto trace with the reference
  implementation and account for every material dispatch, synchronization, and
  collective. Avoid host reads, `.item()`, implicit stream waits, and temporary
  allocations in the steady-state path.
- When streams overlap work, record the exact data dependency with events and
  test it under replay. Do not rely on incidental default-stream ordering.
- Preserve numerical intent when fusing: accumulation dtype, quantization scale
  layout, causal masking, top-k sentinel handling, and collective placement are
  part of model correctness.

## CUDA graphs

Every decode path must be CUDA graphable. Allocate graph-owned workspaces and
stage mutable values into fixed buffers; do not compile kernels, allocate CUDA
memory, create plans, or initialize libraries during replay. Capture graphs at
startup and fail startup if capture or replay parity fails.

Graph capacity is not logical work size. A maximum-context graph must still
read the current positions and execute only the active attention/indexing work.
Use graph variants only for a small number of measured execution regimes, not
as a lookup table for sequence lengths.

Test eager-versus-graph output parity, repeated replay, the largest supported
shape, and transitions across any graph regime boundary. A replay failure
poisons the runner; never silently fall back after the GPU may have accepted
work.

## Paged state and prefix reuse

Declare one `CacheLayout` for full attention, sliding-window attention, and
recurrent checkpoints. The Engine allocates the pages and passes their physical
IDs in `BatchPlan.cache_groups`; the model only maps those IDs to its device
arena. Do not keep a second request-to-page allocator in Python.

Test prefix caching with real continuation requests. A hit must prefill the
uncached suffix as one chunk, restore recurrent and sliding-window state at the
same checkpoint, and copy on write before mutating shared pages. Cover:

- zero-hit and exact-hit requests;
- hits immediately before and after every cache block/checkpoint boundary;
- a long generated response followed by another user turn;
- shared prefixes decoded concurrently, then diverging;
- LRU eviction under cache pressure and physical page reuse.

Compare cached continuation output with a fresh full prefill under greedy
sampling. Run the comparison repeatedly; coherent first output is not enough to
detect stale recurrent state or a bad ring offset.

## Speculative decode

Speculative decoding belongs inside the runner. Set
`cache_write_lookahead_tokens` to the largest verification block, write drafts
only into leased destinations, commit exactly the accepted prefix, and clear or
logically hide rejected recurrent/SWA state before the next step.

Track acceptance rate and time draft, target verification, commit, and sampling
separately. Always report output-token throughput as well as block latency: a
fast kernel can still produce slow generation when acceptance collapses. Test
greedy and stochastic sampling with fixed seeds, EOS inside a draft block, and
page/checkpoint crossings inside the verified block.

## Minimum validation

Before publishing a runner:

1. Compare logits or greedy tokens with the checkpoint reference on short and
   long prefills, decode, and continuation.
2. Run novel prompt lengths and batch sizes twice; the second run must not hide
   shape-specialized compilation required by the first.
3. Verify CUDA graph capture/replay and prefix-cache parity across page and
   recurrent-state boundaries.
4. Stress multiple requests, cancellation, EOS, cache pressure, and page reuse.
5. Record TTFT, output tok/s, speculative acceptance, phase timings, and a
   Perfetto trace. Compare kernel dispatches with the fastest trusted runner.
6. Warm every unavoidable library/kernel regime during startup, then confirm a
   warm server has no lazy compilation stalls.
