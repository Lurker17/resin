These unchanged upstream files are development-only prompt-rendering oracles.
Keeping the small encoders here makes the default fuzz tests work offline.
They are not imported by Resin or included in its runtime package.

- `deepseek_v4_encoding.py.txt`: `encoding/encoding_dsv4.py` from
  `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Identical to the Pro-0813 file
  at revision `72e1d3230f6c080a530b0a1d46f8eb4602340597`.
  SHA256 `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`.
  The upstream MIT license is `deepseek-LICENSE.txt`.
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
oracle. Its newer V4 history rendering differs from the two stock checkpoints,
so it cannot substitute for their pinned prompt encoder.
