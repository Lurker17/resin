These are deployment scripts for local development with Modal GPUs.

Resin is an inference engine and requires accelerators to run, but most
development happens on the author's personal laptop. These are some scripts to
make development easier.

Also, you can publish some basic generation endpoints:

```bash
uv run modal deploy deploy/modal_gpt_oss.py
uv run modal deploy deploy/modal_deepseek_v41.py
uv run modal deploy deploy/modal_qwen3_5.py
```

Download the pinned DeepSeek-V4.1 checkpoint before serving:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v41 --download-only
```

Validate eager and captured inference with natural speculative acceptance:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v41 --gpu B300 \
  --world-size 4 --attn-dp 4 --ep 4 --expert-backend deepgemm \
  --draft-tokens 3 --natural-acceptance --no-balanced-experts --validate-model
```

Use `--draft-tokens 0` to exercise ordinary decode. These full-model
checks require the pinned checkpoint volume and four B300s.
