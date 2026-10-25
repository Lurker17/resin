These unchanged upstream files are development-only prompt-rendering oracles.
Keeping the small encoders here makes the default fuzz tests work offline.
They are not imported by Resin or included in its runtime package.

- `lfm2.jinja`: `chat_template.jinja` from `LiquidAI/LFM2.5-230M`, revision
  `13a53837c4906b4f7405932532ba85d182bb013b`.
  SHA256 `c5b67247c9f736f6c40eaa91ae1ef27dc67f5b041eedaadb97a1bea5030fb0ff`.
  The upstream LFM Open License is `lfm2-LICENSE.txt`.
- `../templates/gemma4.jinja`: `chat_template.jinja` from
  `nvidia/Gemma-4-31B-IT-NVFP4`, revision
  `4135a98a9b728a548947683219633b25682223ac`.
  SHA256 `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`.
  Its [pinned model card](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4/blob/4135a98a9b728a548947683219633b25682223ac/README.md)
  declares Apache License 2.0; the unmodified license text is
  `gemma4-LICENSE.txt`, from <https://www.apache.org/licenses/LICENSE-2.0.txt>.

Gemma currently has no public reference output parser. Its adapter checks
generated values against the official template's `format_argument` macro and
then verifies the semantic round trip through the native parser. This is
distinct from the independent reference-parser checks used for other models.

DeepSeek's released `deepseek-recipe` package is the primary response-parser
oracle. V4.1 uses the package's `DeepseekV41Encoding` and V4.1 stream parser
as its fast oracle. Native fixtures also pin the 1017 numeric-effort template,
middle-system headers and tool-only history semantics against
`DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py`.
That checkpoint skips tool-result-only blocks when finding the last user turn;
the package's conversation conversion differs there, so this case has an
explicit stock-format fixture. Numeric budgets 1–100 are supported by the
native recipe and `extra.reasoning_effort`; HTTP aliases are low/high/max.
