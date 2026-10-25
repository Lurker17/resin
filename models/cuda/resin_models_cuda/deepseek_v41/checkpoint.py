"""Pinned, data-only checkpoint access for DeepSeek-V4.1 Flash."""

import json
from pathlib import Path

from resin.models.deepseek_v41.config import DeepSeekV41Config

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a complete local checkpoint without executing checkpoint code."""

    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"DeepSeek-V4.1 checkpoint directory does not exist: {path}"
        )
    required = (
        "config.json",
        "model.safetensors.index.json",
        "inference/config.json",
    )
    if missing := [name for name in required if not (path / name).is_file()]:
        raise FileNotFoundError(f"DeepSeek-V4.1 checkpoint is missing files: {missing}")
    return path


def load_config(model_path: str | Path) -> DeepSeekV41Config:
    root = resolve_model_path(model_path)
    return DeepSeekV41Config.from_hugging_face(
        json.loads((root / "config.json").read_text())
    )
