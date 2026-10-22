# Token logprobs

Request scores with `SamplingParams(logprobs=True)` or alternatives with
`SamplingParams(top_logprobs=5)`. Each returned `TokenEvent.logprobs` contains the
selected token's `logprob` and a list of `(token_id, logprob)` alternatives.
Scores describe the raw model distribution, before temperature, top-k, or top-p
sampling; requesting them does not change token selection.

Request input scores independently with `SamplingParams(prompt_logprobs=True)`.
`RequestLease.prompt_logprobs` contains one entry per prompt token. The first is
`None` because it has no predecessor; other entries contain raw-model scores
after the predecessor token's prefill chunk commits. Unprocessed entries remain
`None`. Set
`max_new_tokens=0` to score only the input and complete without output events.
All stock runners reuse the same prefill hidden states and project scores in
bounded chunks, with no second transformer pass. Cross-chunk targets are part
of the execution plan, so the final predecessor in a chunk scores the first
token of the next chunk.

Scoring requests bypass prefix restoration: KV cache entries do not retain
the predecessor distributions needed to score a cached prefix. They still
publish reusable KV prefixes for ordinary requests. Generated-token scoring,
including speculative acceptance, keeps its existing behavior.

Model runners use `resin.backend.sampling.format_output(logits, tokens, plan)`
for ordinary and speculative output. Verification positions must align with the
returned accepted tokens. Unrequested and continuation-prefill rows carry no
scores. Generation scores use the retained output logits after execution;
model runners keep their existing graph eligibility and eager fallback rules.
`BatchPlan.needs_logprobs` checks the score flag natively, so unscored graph
replay does not create Python sampling-policy objects merely to decide whether
scores are needed.

## Array boundary

Runners return a `ForwardOutput` with contiguous one-dimensional NumPy arrays:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `token_ids` | uint32 | Packed output tokens |
| `token_indptr` | uint32 | Token row offsets, length batch + 1 |
| `logprobs` | float32 | Packed selected-token scores |
| `logprob_indptr` | uint32 | Score row offsets; defaults to token_indptr |
| `top_indptr` | uint32 | Alternative offsets, length scored tokens + 1 |
| `top_token_ids` | uint32 | Packed alternative token IDs |
| `top_logprobs` | float32 | Packed alternative scores |
| `prompt_logprobs` | float32 | Packed requested predecessor scores |
| `prompt_logprob_indptr` | uint32 | Prompt-score row offsets, length batch + 1 |

The three `top_*` columns are supplied together or omitted. An empty score row
opts out; every other score row must cover its complete raw token row, including
positions beyond EOS or the remaining output budget. All offsets start at zero,
are monotone, and end at their corresponding column length.

Construction retains the exact arrays without dtype conversion or copying. The
Rust adapter acquires read-only NumPy borrows once and retains their owners until
the synchronous `step()` returns. Runners must not modify or recycle these buffers
during that interval, including from `release()`. Tensor-backed NumPy views keep
their CPU tensors alive through NumPy's base owner. The adapter rejects incorrect
dtypes and noncontiguous or unaligned storage instead of silently coercing them.

This removes Python scalar/tuple boxing and the additional Python-to-Rust score
matrix copy. GPU-to-host transfers still occur; only scores retained in committed
events acquire Rust-owned storage. Token IDs join the core's existing owned token
row contract. Plain list-of-token outputs remain supported when scores are off.

Native Rust runners can expose the same borrowed columns through
`Runner::logprobs() -> Option<LogprobView<'_>>`. The core validates every raw score
before committing any request, including scores that EOS would otherwise hide.
Invalid outputs follow the existing execution quarantine path.

Prompt-score columns are supplied together, independently of generated-token
scores. `BatchPlan.prompt_score_indices` selects flattened prefill predecessor
rows, `prompt_score_token_ids` supplies their next prompt tokens, and
`prompt_score_indptr` partitions them by sequence. Runners return every requested
score, including on intermediate prefill chunks with no sampled output. Native
runners expose these columns through `Runner::prompt_logprobs()`; invalid or
missing scores quarantine the entire execution before progress becomes visible.

## HTTP

Chat Completions accepts `logprobs: true` and `top_logprobs: 0..20`. Responses
accepts `include: ["message.output_text.logprobs"]` and `top_logprobs`; a positive
alternative count also enables selected-token scores. These flags reach the
same native `SamplingParams` and NumPy score columns used by direct Engine calls.

Every scored generated token appears in `resin.generation_logprobs`, with
`token_id`, zero-based `token_index`, display `token`, `bytes`, `logprob`, optional
semantic `channel`, and `top_logprobs`. Alternatives include their IDs, display
text, bytes, and scores. Byte-level and byte-fallback tokenizers retain the actual
bytes, including incomplete UTF-8; concatenating display strings can introduce
replacement characters, so use `bytes` to reconstruct text. Unsupported decoder
formats and custom codecs without a serialized tokenizer report `bytes: null`.
JSON represents a valid negative-infinite logprob as `null`.

The complete raw list includes reasoning, tool markup, headers, and model
terminators. Canonical Chat `choices[0].logprobs.content` and Responses
`output_text.logprobs` contain only tokens proven to belong wholly to visible
content. Parsers that trim, buffer, or transform a token can leave its channel
unknown; custom parsers have the same conservative fallback. User stop strings
also disable canonical content association because a held prefix can consume
part of a token. The raw list remains complete through the token that matches
the stop, with no records for the remaining verified-block suffix.

Chat streams publish the raw records on chunks under
`resin.generation_logprobs`; their canonical content records accompany the same
chunk. Responses streams publish `resin.generation_logprobs.delta` events with a
`logprobs` array, and include canonical content records on the completed text
part and final response. Concatenating raw score arrays yields the same records
as the non-streaming response, independently of semantic parser chunk boundaries.

Both APIs accept `prompt_logprobs: true`, independently of generated scores.
The result's `resin.prompt_logprobs` follows the complete rendered prompt,
including model scaffold and control tokens. Its first entry is `null`; later
entries carry token IDs, text, bytes, and scores. Set `max_completion_tokens: 0`
(Chat) or `max_output_tokens: 0` (Responses) for input scoring without generation.
Chat streams send one `resin.prompt_logprobs` field; Responses streams send one
`resin.prompt_logprobs` event with a `logprobs` array before completion.

## GPU validation limits

On the pinned GPT-OSS-20B checkpoint, an independent raw output-head
`log_softmax` oracle matches unchunked input scores within 1.91e-6 with
temperature/top-k/top-p enabled. Input-only requests return no generated events,
retain the initial `None`, and complete on one H100 or with composed EP/attention
DP/attention TP/CP and prefill graph replay. The composed full-model run retains
nine failed strict numerical comparisons despite matching greedy probes;
[the model validation record](gpt-oss-parallelism.md#sparse-transport-qualification)
separates those failures from the exact same-shard sparse transport checks.

Cross-batch numerical parity remains unqualified: the same
[single-H100 checkpoint](https://modal.com/apps/ekzhang/main/ap-1Roe4kUsfpXJXV9f3dMY8x)
scores a 180-token prompt differently in one chunk versus 32-token chunks
(maximum 5.15 nats, relative L2 0.0892). This reproduces without EP or TP.
[Actual first-layer FA4 inputs](https://modal.com/apps/ekzhang/main/ap-FasvATBT14nPXXMCnH0stM)
pass an explicit FP32 causal/window attention oracle including the learned sink
for all seven shapes; the cause of the full-model sensitivity is not established.
The durable reports in `resin-gpt-oss-profiles` are
`prompt-scoring-isolation-20261022T032541Z.json` and
`prompt-scoring-isolation-20261022T033120Z.json`; the latter also has matching
`.pt` attention inputs/outputs for bounded follow-up.
