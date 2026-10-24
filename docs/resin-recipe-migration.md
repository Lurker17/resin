# Stock model recipe migration

The engine stays token-based. `resin-recipe` owns stock conversation formatting,
token encoding, model-output parsing, and semantic response events. Python model
adapters prepare image/video tensors; the HTTP crate owns wire protocols.
Applications can pass any structural `resin.recipe.Recipe` to `HttpServer(recipe=...)`.
The public recipe interface accepts typed messages, tools and render options.

## Python API

```python
import resin
from resin.recipe import GptOssRecipe

recipe = GptOssRecipe(tokenizer)
prompt = recipe.render(
    [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Hello")])],
    options=resin.recipe.RenderOptions(reasoning_effort="medium"),
)
parser = recipe.create_parser(prompt)
request = engine.submit(
    prompt.token_ids,
    max_new_tokens=128,
    stop_token_ids=prompt.stop_token_ids,
)
```

Use an explicit native constructor: `GptOssRecipe`, `DeepSeekV4Recipe`,
`DeepSeekV41Recipe`, `Qwen35Recipe`, `Qwen38Recipe`, `Gemma4Recipe`,
`InklingRecipe`, or `Lfm2Recipe`. Qwen versions select different formats explicitly.
`resin.recipe.Recipe` is a structural protocol, not a model factory.

`Message` contains a `Role` and typed `Text`, `Thinking`, `Image`, `Video`, `Audio`,
`ToolCall` or `ToolResult` blocks. Tool declarations and `RenderOptions` are
separate rendering arguments. Supported media depends on the model recipe.
Image/video recipes accept a processor; V4.1 prepares its checkpoint's pixels
internally. `prompt.media` carries the prepared model payload for
`engine.submit(media=prompt.media, ...)`.

`RenderedPrompt` exposes `token_ids`, `stop_token_ids` and `media`. Native prompts
also retain the parser context. Pass the actual prompt back to the same recipe;
`create_parser(prompt)` creates independent state for each response. A custom
Python recipe may construct `RenderedPrompt` and supply its own parser.
HTTP request dictionaries are converted at the serving boundary; they are not
accepted as native recipe inputs.

Parsers return immutable native `ParserEvent` objects. Use the concrete variants
to read text, reasoning and tool-call payloads:

```python
for event in parser.feed(generated_token_ids):
    if isinstance(event, resin.recipe.TextDelta):
        print(event.delta, end="", flush=True)
```

`ReasoningDelta` also carries `delta`; `ToolCallStart` carries `index`, `id` and
`name`; `ToolCallArgumentsDelta` carries `index` and `delta`. All variants expose
`type` and immutable `token_spans` for source-token attribution. Custom Python
parsers construct the same native variants. Native parsers feed the Rust HTTP
actor directly, without converting events through Python or JSON.

## Contracts

- A `Prompt` records ordinary text separately from explicit control tokens.
  Adjacent text coalesces before encoding so BPE boundaries remain correct.
- One prepared tokenizer is shared by requests. Ordinary text cannot insert a
  control token by spelling its printable representation.
- A native model recipe renders the request and initializes its corresponding
  parser from the rendered prefix. Parsers consume new token IDs and finalize
  on stop or length; streaming and complete responses share these events.
- The Python binding uses one native parser handle for all formats, and the
  HTTP actor consumes its events directly. Custom Python parsers retain their
  `feed` and optional `finish` contract.
- Media preprocessing may use model processors, but stock processors must not
  execute chat templates or own model-format interpretation.

## Required model coverage

| Family | Rendering reference | Output grammar |
| --- | --- | --- |
| GPT-OSS 20B/120B | Harmony / pinned tokenizer | channels and JSON tools |
| DeepSeek V4 Flash-0731/Pro-0813 | pinned checkpoint encoder | reasoning and DSML |
| DeepSeek V4.1 Flash-1017 | released DeepSeek Recipe / checkpoint processor | revised DSML and image slots |
| Inkling-Small | pinned HF template / TML v0 | typed content blocks |
| Qwen3.5/3.8 | pinned HF template / TML | reasoning and XML tools |
| Gemma4 | pinned HF template | model turns and tools |
| LFM2.5 | pinned HF template | reasoning and safe Python-literal tools |

## Migration and verification

1. Implement shared prompt encoding and the common native recipe/parser binding.
2. Port all six format families and preserve tools, reasoning, prefixes, stops,
   and the existing image/video geometry and cache identities.
3. Replace stock codecs, local scripts, deployment warmups, reports, and HTTP
   construction. Remove production Prime and checkpoint encoder execution.
4. Extend the shared differential harness with official reference oracles and
   TML as an additional oracle where formats overlap. Keep default fuzz work
   around one second per model, with opt-in longer runs and minimized failures.
5. Verify native streaming/complete HTTP behavior for every family, custom
   recipe compatibility, real pinned tokenizers/templates, multimodal admission,
   type checking, formatting, Rust checks, and the full CPU suite.

Completion requires evidence for every row and every stock entrypoint. A native
parser alone, a text-only migration, or green tests with reference formatting
still used at runtime does not complete this migration.

## Verification recorded at migration

All six stock families use the native renderer/parser boundary. Stock deployment
warmups, reports, HTTP recipes and local LFM scripts no longer interpret reference
templates or execute checkpoint encoders. Custom structural recipes retain their
Python protocol, including parsers without finalization.

The shared suite runs 40 deterministic examples per style/model, including
reference prompt rendering, chunked responses, partial tools and arbitrary byte
decoding. Fuzz execution is approximately 0.2–0.9 seconds/model, excluding imports
and one-time oracle setup. Primary values remain exact; TML's documented
one-ULP secondary float exception is isolated in the test comparison.

Native HTTP tests cover each family's reasoning, text and tool events with both
streaming and complete responses, stop/length finalization, ordinary marker
provenance and user stop encoding. Cached pinned processors match all output
tensors for Inkling/Qwen/Gemma image prompts and Gemma video, including explicit
media expansion without retokenizing ordinary text. Native fixtures and actual
pinned tokenizers verify format/token parity; the few deliberate HF differences
preserve literal user marker text as ordinary tokens.

The complete local CPU suite passed 812 Python tests (67 skips) and 150 Rust
tests. The shared build/type-refresh script, ty, Ruff and workspace Clippy passed.

The cleanup shares the parser driver, token declarations and rendering
conventions across formats. Production Rust in resin-recipe fell from 4,595 to
3,917 lines (Tokei, excluding comments, blanks and tests), with reference fixtures
and semantic coverage retained.
