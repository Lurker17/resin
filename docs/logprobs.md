# Generated token logprobs

Request scores with `SamplingParams(logprobs=True)` or alternatives with
`SamplingParams(top_logprobs=5)`. Each returned `TokenEvent.logprobs` contains the
selected token's `logprob` and a list of `(token_id, logprob)` alternatives.
Scores describe the raw model distribution, before temperature, top-k, or top-p
sampling; requesting them does not change token selection. Prompt scoring is not
implemented in this prototype.

Model runners use `resin.backend.sampling.format_output(logits, tokens, plan)`
for ordinary and speculative output. Verification positions must align with the
returned accepted tokens. Unrequested and continuation-prefill rows carry no
scores. CUDA runners currently use eager execution for scored requests; captured
graphs retain their existing token-only output contract. `BatchPlan.needs_logprobs`
checks the score flag natively, so unscored graph replay does not create Python
sampling-policy objects merely to decide whether scores are needed.

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

This API is currently available through `Engine` and its runner protocol. HTTP
response serialization and prompt-token scoring are separate work.
