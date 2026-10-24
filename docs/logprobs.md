# Token logprobs

Request scores with `SamplingParams(logprobs=True)` or alternatives with
`SamplingParams(top_logprobs=5)`. Each returned `TokenEvent.logprobs` contains the
selected token's `logprob` and a list of `(token_id, logprob)` alternatives.
Scores describe the raw model distribution, before temperature, top-k, or top-p
sampling; requesting them does not change token selection.

Request input scores independently with `SamplingParams(prompt_logprobs=True)`.
Use `prompt_top_logprobs` for top alternatives at each prompt position, or pass
`prompt_target_logprobs` to `Engine.submit` for specific token IDs at each position:

```python
lease = engine.submit(
    [10, 20, 30],
    sampling_params=SamplingParams(max_output_tokens=0, prompt_top_logprobs=3),
    prompt_target_logprobs=[[], [42, 43], [7, 7]],
)
```

This asks the distribution after token `10` to score targets `42` and `43`, and
the distribution after token `20` to score target `7` twice. The target matrix
must have one row per prompt token, with an empty first row. Target order and
duplicates are preserved. Targets are prompt-only; there is no generated-token
target parameter.

`RequestLease.prompt_logprobs` contains one entry per prompt token. The first is
`None` because it has no predecessor; other entries contain raw-model scores
after the predecessor token's prefill chunk commits. Unprocessed entries remain
`None`. Each scored entry has the actual prompt token's `logprob`, requested
`top_logprobs`, and requested `target_logprobs`; both alternative lists contain
`(token_id, logprob)` pairs. Any of the three prompt options enables input
scoring. `top_logprobs` and `prompt_top_logprobs` are independent. Set
`SamplingParams(max_output_tokens=0, prompt_logprobs=True)` to score only the input
and complete without output events.
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
generated-token scores. Generation scores use the retained output logits after
execution; model runners keep their existing graph eligibility and eager fallback rules.
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
| `prompt_top_indptr` | uint32 | Top-alternative offsets, length prompt scores + 1 |
| `prompt_top_token_ids` | uint32 | Packed prompt top-alternative IDs |
| `prompt_top_logprobs` | float32 | Packed prompt top-alternative scores |
| `prompt_target_indptr` | uint32 | Target offsets, length prompt scores + 1 |
| `prompt_target_token_ids` | uint32 | Packed requested prompt target IDs |
| `prompt_target_logprobs` | float32 | Packed prompt target scores |

The three `top_*` columns are supplied together or omitted. An empty generated-score
row opts out; every other generated-score row must cover its complete raw token
row, including positions beyond EOS or the remaining output budget. All offsets start at zero,
are monotone, and end at their corresponding column length.

Construction retains the exact arrays without dtype conversion or copying. The
Rust adapter acquires read-only NumPy borrows once and retains their owners until
the synchronous `step()` returns. Runners must not modify or recycle these buffers
during that interval, including from `release()`. Tensor-backed NumPy views keep
their CPU tensors alive through NumPy's base owner. The adapter rejects incorrect
dtypes and noncontiguous or unaligned storage instead of silently coercing them.

This removes Python scalar/tuple boxing and intermediate Python-to-Rust token
and score copies. GPU-to-host transfers still occur; only committed history and
events acquire Rust-owned storage. Token-only results use the same flat
`ForwardOutput` contract; plain lists are rejected.

Native Rust runners can expose the same borrowed columns through
`Runner::logprobs() -> Option<LogprobView<'_>>`. The core validates every raw score
before committing any request, including scores that EOS would otherwise hide.
Invalid outputs follow the existing execution quarantine path.

Prompt-score columns are supplied together, independently of generated-token
scores. `BatchPlan.prompt_score_indices` selects flattened prefill predecessor
rows, `prompt_score_token_ids` supplies their next prompt tokens, and
`prompt_score_indptr` partitions them by sequence. `prompt_score_top_counts`
supplies each predecessor's requested top count. `prompt_target_indptr` and
`prompt_target_token_ids` supply its requested targets in the same order as the
score rows. Each optional top/target column triplet is supplied together.
Runners return every requested score, including on intermediate prefill chunks
with no sampled output. The shared scorer normalizes each raw distribution once
for selected, top, and target scores, and checks target vocabulary bounds before
gathering. Native runners expose these columns through `Runner::prompt_logprobs()`;
invalid or missing scores quarantine the entire execution before progress becomes visible.

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
formats and custom recipes without a serialized tokenizer report `bytes: null`.
JSON represents a valid negative-infinite logprob as `null`.

The complete raw list includes reasoning, tool markup, headers, and model
terminators. Canonical Chat `choices[0].logprobs.content` and Responses
`output_text.logprobs` contain only tokens proven to belong wholly to visible
content. All six stock parsers carry source-byte spans through grammar buffers,
UTF-8 decoding, trimming, and stop matching. Held text waits until a complete
source token can be associated with its score. If trimming or a stop string cuts
inside a token, its surviving text still appears, but that token's score remains
raw-only: a whole-token probability cannot describe a fraction of that token.
Transformed tool arguments have no fabricated text attribution. The raw list
remains complete through the token that matches the stop, with no records for
the remaining verified-block suffix.

Chat streams publish the raw records on chunks under
`resin.generation_logprobs`; their canonical content records accompany the text
they describe. Responses streams publish `resin.generation_logprobs.delta` events
with a `logprobs` array. Canonical records accompany each
`response.output_text.delta`, and accumulate on the completed text part and final
response. Concatenating raw score arrays yields the same records as the
non-streaming response, independently of semantic parser chunk boundaries.

Every recipe parser returns provenance from its ordinary `feed` and `finish`
methods, for scored and unscored requests. Custom parsers use the same
`ResponseParser`/`FinalizingResponseParser` protocols from `resin.protocols`.
They return native `resin.recipe.ParserEvent` objects: `TextDelta`,
`ReasoningDelta`, `ToolCallStart`, or `ToolCallArgumentsDelta`. Fields are read
as attributes; parser events do not use dictionaries. Each `token_spans` entry
is a five-integer tuple:

```python
resin.recipe.TextDelta("ab", token_spans=[(0, 0, 1, 0, 1), (1, 0, 1, 1, 2)])
```

The fields are `(token_index, source_start, source_end, text_start, text_end)`.
The example maps the full bytes of generated tokens 0 and 1 onto `a` and `b`.
Token indices count all generated tokens, including headers and controls.
Source ranges address the original vocabulary token's bytes; text ranges address
the event delta's UTF-8 bytes. Both are half-open, and every field fits uint32.
Several byte tokens can share one whole Unicode character's text range. Preserve
these compact spans in source-token order through buffering; the native representation uses five u32
values rather than six machine-sized offsets. Full-token byte lengths come from
the tokenizer vocabulary instead of being repeated in each span.

Empty spans, the constructor default, mean synthetic text or unknown provenance.
They still allow raw scores, with no guessed canonical association. Custom Python
tokenizers without a serialized backend can use this fallback; native parser
creation rejects unsupported serialized decoder chains. Malformed span values,
unseen or out-of-order source indices, empty/reversed
ranges, and invalid text bounds fail the request even when scores are off,
without poisoning the engine. Scored requests also check source byte extents
against the raw vocabulary bytes when those bytes are available.

Both APIs accept `prompt_logprobs: true`, independently of generated scores.
Prompt top alternatives and target scoring are currently engine-only.
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
