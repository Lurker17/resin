# resin-recipe

Model-format code above the token-based engine. This crate has no Python, HTTP,
or GPU dependency. Its first implementation parses Inkling's typed response
blocks into reasoning, text, and tool events.

`ResponseParser` consumes new token IDs with `feed` and flushes buffered output
with `finish(Stop | Length)`. A `Decoder` supplies new decoded text while the
model parser owns control-token transitions. `TokenizerDecoder` uses a shared
Hugging Face tokenizer and rolling decode state, retaining incomplete UTF-8 and
the decoder's required context. Only explicit control-token IDs affect the
grammar; ordinary text spelling a marker remains content.

Tool JSON is buffered and validated when its block closes. An unfinished tool
is discarded at finalization; finalization never invents a completed call.
Text still flushes at the token limit, including replacement characters for
truncated UTF-8. Repeated `finish` is harmless, and `feed` after it is an error.

`resin-py` exposes these parsers through the existing Python `ResponseParser`
protocol, with optional `FinalizingResponseParser.finish`. The HTTP actor takes
the native parser handle at admission and consumes its Rust events directly.
Python callers receive the same event dictionaries as other model codecs.

Prepare `resin.recipe.Tokenizer` once per codec. Fast Hugging Face tokenizers
are snapshotted into Rust and shared across requests. Custom Python tokenizers
can supply `decode`; that compatibility path retains cumulative decoding.
The native path uses backend tokenizer semantics, without Transformers'
additional `clean_up_tokenization_spaces` postprocessing.

Prompt rendering and media preparation still belong to the existing model
codecs in this first slice. Other model grammars can implement the same parser
trait without depending on the inference scheduler or HTTP wire formats.

Run `cargo test -p resin-recipe` for Rust invariants and
`uv run pytest tests/test_recipe.py -q` for Python/tokenizer integration.
