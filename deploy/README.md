These are deployment scripts for local development with Modal GPUs.

Resin is an inference engine and requires accelerators to run, but most
development happens on the author's personal laptop. These are some scripts to
make development easier.

Also, you can publish some basic generation endpoints:

```bash
uv run modal deploy deploy/modal_gpt_oss.py
uv run modal deploy deploy/modal_deepseek_v4.py
```
