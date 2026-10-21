Run `uv run pytest tests/recipe_fuzz -q` for the small default corpus: 50
reproducible examples per style, targeting roughly one second of test execution
per model (excluding interpreter/import/setup time). Increase
coverage with `RESIN_FUZZ_EXAMPLES=10000 uv run pytest tests/recipe_fuzz -q`.
Hypothesis shrinks failures into reproducible counterexamples; commit minimized
regressions as ordinary tests when fixing a bug.

Two styles share one semantic normalizer and streaming driver: structured valid
responses compared with an independent reference parser, and arbitrary bytes
compared with the tokenizer's complete decode. Each case exercises whole input,
one-token chunks, empty feeds, random boundaries, and idempotent finalization.
Structured cases also truncate the final tool before its closing control token.

The first format adapter is Inkling against TML v0. Its oracle domain consists of
nonempty text/thinking blocks and named tools with `args` objects. Empty blocks,
the `arguments` alias, and malformed tool grammar have different reference
semantics and remain explicit unit-test cases. Ordinary marker spellings are
encoded as ordinary text. Semantic comparison preserves part order, combines
adjacent text deltas, compares parsed JSON values, and ignores generated tool IDs
and JSON whitespace. Reference errors or unknown output kinds fail the test.

Add a format adapter to `ADAPTERS` when its Resin codec exists; reuse this harness
with that format's actual reference library rather than comparing it to itself.
Each adapter exposes named `oracles`; all are checked independently. For future
formats this can include both the official Harmony/DeepSeek library and TML as a
secondary reference, with no majority vote or suppressed disagreements.
