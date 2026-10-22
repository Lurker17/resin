"""Pinned, data-only checkpoint access for DeepSeek-V4.1 Flash."""

import json
from pathlib import Path

from resin.models.deepseek_v41.config import DeepSeekV41Config
from resin_models_cuda.deepseek_v4.checkpoint import resolve_model_path

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def load_config(model_path: str | Path) -> DeepSeekV41Config:
    root = resolve_model_path(model_path)
    return DeepSeekV41Config.from_hugging_face(
        json.loads((root / "config.json").read_text())
    )
