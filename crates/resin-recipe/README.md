# resin-recipe

Stock model conversation formats, independent of Python, HTTP and GPU execution.
`ModelFormat` selects a matching renderer, incremental parser and stop markers
for GPT-OSS, DeepSeek V4/V4.1, Inkling, Qwen3.5/3.8, Gemma4 or LFM2.5.

Model code describes format rules. Shared `Parser<G>` owns token batching,
decoder boundaries, EOS handling and finalization; each grammar handles only
text and control transitions. A small token declaration supplies spellings,
aliases, lookup and validation once. Renderers share request/tool conventions
and span operations while retaining their model-specific ordering and framing.

A renderer produces a `Prompt`: ordinary text plus explicit control-token spans.
Adjacent text coalesces before encoding, preserving BPE boundaries. A prepared
`TokenizerEncoder` shares the tokenizer and encodes ordinary marker spellings as
text. Native image/video processor expansions replace only explicit media
slots, then use the same encoding boundary. Model-owned modules under `media/`
prepare pixels and model geometry using the Rust `image` crate. The Python
binding exposes owned NumPy payload arrays and cache identities; runners execute
the GPU vision encoder. Resize pixels can differ from HF filters; strict
no-resize oracles check normalization, padding and patch order.

`ResponseParser::feed` consumes only new token IDs. `finish(Stop | Length)` flushes
buffered text, including truncated UTF-8; finalization is idempotent. A shared
`TokenizerDecoder` retains the tokenizer's rolling decode context. Control IDs,
not printable text, drive model grammar. Tool behavior follows the format:
Inkling/Harmony/Qwen/Gemma/LFM commit validated calls at their closing marker;
DeepSeek streams DSML calls and argument deltas as they arrive. Finalization does
not invent complete arguments for an unfinished call.

Python callers import concrete native classes such as
`from resin.recipe import GptOssRecipe, Gemma4Recipe` and instantiate
`GptOssRecipe(tokenizer)` or `Gemma4Recipe(processor)`. A media recipe snapshots
processor configuration once; callers may instead supply a tokenizer and
`media_config` mapping. Video inputs are decoded frames. `resin.recipe.Recipe`
is a structural protocol. `render(messages, *, tools=(), options=None)` accepts typed `Message`
content and returns `RenderedPrompt`; its token IDs, model stops and prepared
media are available as attributes. `create_parser(prompt)` binds a fresh parser
to the actual rendered prefix and options. A native prompt belongs to the
recipe that rendered it.

`Message` holds a `Role` and ordered `Text`, `Thinking`, `Image`, `Video`, `Audio`,
`ToolCall` or `ToolResult` blocks. Recipes validate model-specific roles, block
orders and media support; the role enum can grow. Tool declarations and
`RenderOptions` are separate rendering arguments. Qwen3.5 and Qwen3.8 select
their distinct formats through explicit constructors.

Pass `prompt.token_ids`, `prompt.stop_token_ids`, and `prompt.media` to the
Engine's corresponding submission arguments. Each `create_parser(prompt)`
call creates independent response state without retaining pixels. Custom
Python recipes may construct `RenderedPrompt` themselves and supply their own
parser; native recipes require a prompt rendered by the same recipe instance.

These parsers use one common native handle; the HTTP actor consumes Rust events
directly. Python callers receive immutable `ParserEvent` variants: `TextDelta`,
`ReasoningDelta`, `ToolCallStart` and `ToolCallArgumentsDelta`. Their payload and
`token_spans` provenance are attributes. Applications can supply a custom recipe
with the same `render` and `create_parser` methods and parser `feed` (plus optional
`finish`), returning these native events. HTTP wire dictionaries stay in the HTTP
adapter, outside the native recipe API.

Text and reasoning events carry `delta`; tool starts carry `index`, `id`, and
`name`; tool argument deltas carry `index` and `delta`. Every variant exposes a
`type` discriminator and immutable `token_spans` for source-token attribution.
See [logprob attribution](../../docs/logprobs.md#http).

Fast Hugging Face tokenizers are snapshotted once and shared across requests.
Custom Python tokenizers can supply `encode`, `convert_tokens_to_ids` and
`decode`, with optional `encode_ordinary` for literal text. This compatibility
path retains Python callbacks and cumulative
decoding. Its special-token behavior is the custom tokenizer's responsibility.
The native path follows backend tokenizer semantics rather than Transformers'
additional `clean_up_tokenization_spaces` postprocessing.

References are pinned checkpoint templates/encoders for prompt parity. Harmony
and DeepSeek Recipe are primary parser oracles; TML supplies secondary checks
where available. All reference libraries and templates are development-only.
DeepSeek V4.1 has its own recipe for the revised DSML grammar and image slots. LFM's
nested Python-literal strings use a fixed Python 3.13 / Unicode 15.1 printable
repertoire for deterministic rendering across deployment Python versions.

Run `cargo test -p resin-recipe` for native invariants,
`uv run pytest tests/test_recipe*.py -q` for integration, and
`uv run pytest tests/recipe_fuzz -q` for shared differential fuzzing.
See the [fuzz README](../../tests/recipe_fuzz/README.md) for budgets and oracle
limitations and the [Python API example](../../README.md#modal-http-endpoints)
for native recipe construction.
