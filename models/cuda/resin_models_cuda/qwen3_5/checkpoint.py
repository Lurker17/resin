"""Resolve Qwen3.5-family checkpoints without loading their vision tower."""

from __future__ import annotations

from pathlib import Path

from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin_models_cuda.qwen3_5.config import Qwen35Config

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

TOKENIZER_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*MODEL_FILES, *TOKENIZER_FILES],
        )
    )


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    """Resolve a local directory or download a pinned Hub checkpoint."""

    selected = (
        DEFAULT_REVISION
        if revision is None and str(model) == DEFAULT_MODEL
        else revision
    )
    return resolve_model(model, selected, _download_model)


def load_config(model_path: str | Path) -> tuple[Qwen35Config, dict]:
    raw_config = read_config(Path(model_path))
    return Qwen35Config.from_hugging_face(raw_config), raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "load_config",
    "resolve_model_path",
]
