These are deployment scripts for local development with Modal GPUs.

Resin is an inference engine and requires accelerators to run, but most
development happens on the author's personal laptop. These are some scripts to
make development easier.

Also, you can publish some basic generation endpoints:

```bash
uv run modal deploy deploy/modal_gpt_oss.py
uv run modal deploy deploy/modal_deepseek_v4.py
uv run modal deploy deploy/modal_qwen3_5.py
```

Validate DeepSeek's sliding-window metadata on one B300 without loading a
checkpoint:

```bash
uv run modal run deploy/modal_deepseek_v4.py --mode validate-window
```

The full model profile path also compares a cached continuation with cold
prefill after an 8192-token prompt, crossing the default 4096-token chunks:

```bash
uv run modal run deploy/modal_deepseek_v4.py --mode profile \
  --input-tokens 8192 --max-new-tokens 32 --temperature 0 --speculative-tokens 5
```

Use `--speculative-tokens 0` to exercise ordinary decode. These full-model
checks require the pinned checkpoint volume and four B300s.
