"""Resolve and validate DeepSeek-V4 checkpoints as data-only inputs."""

from __future__ import annotations

import json
from pathlib import Path

from resin_models_cuda.deepseek_v4.config import DeepSeekV4Config

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a complete local checkpoint without executing checkpoint code."""

    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"DeepSeek-V4 checkpoint directory does not exist: {path}"
        )
    required = (
        "config.json",
        "model.safetensors.index.json",
        "inference/config.json",
    )
    if missing := [name for name in required if not (path / name).is_file()]:
        raise FileNotFoundError(f"DeepSeek-V4 checkpoint is missing files: {missing}")
    return path


def load_config(model_path: str | Path) -> DeepSeekV4Config:
    root = resolve_model_path(model_path)
    try:
        values = json.loads((root / "config.json").read_text())
    except json.JSONDecodeError as error:
        raise ValueError("DeepSeek-V4 config.json is invalid") from error
    if not isinstance(values, dict):
        raise ValueError("DeepSeek-V4 config.json must contain an object")
    return DeepSeekV4Config.from_hugging_face(values)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "load_config",
    "resolve_model_path",
]
