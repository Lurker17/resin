"""Inkling-Small checkpoint identities and local validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from resin.models.inkling.config import InklingConfig, InklingDSparkConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class InklingModelSpec:
    target_repo: str
    target_revision: str
    draft_repo: str
    draft_revision: str


MODEL_SPEC = InklingModelSpec(
    target_repo="thinkingmachines/Inkling-Small-NVFP4",
    target_revision="b6a99534467840620d411e4cd4ad5819b2610d9c",
    draft_repo="RadixArk/Inkling-Small-DSpark",
    draft_revision="736501c3901cfc6bbb53ba382781eb0e5d9ad66a",
)


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        values = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} config.json is invalid") from error
    if not isinstance(values, dict):
        raise ValueError(f"{description} config.json must contain an object")
    return values


def resolve_target_path(model: str | Path) -> Path:
    root = Path(model).expanduser().resolve()
    required = ("config.json", "model.safetensors.index.json")
    if not root.is_dir():
        raise FileNotFoundError(f"Inkling target directory does not exist: {root}")
    if missing := [name for name in required if not (root / name).is_file()]:
        raise FileNotFoundError(f"Inkling target is missing files: {missing}")
    return root


def resolve_draft_path(model: str | Path) -> Path:
    root = Path(model).expanduser().resolve()
    required = ("config.json", "model.safetensors")
    if not root.is_dir():
        raise FileNotFoundError(f"Inkling draft directory does not exist: {root}")
    if missing := [name for name in required if not (root / name).is_file()]:
        raise FileNotFoundError(f"Inkling draft is missing files: {missing}")
    return root


def load_target_config(model: str | Path) -> InklingConfig:
    root = resolve_target_path(model)
    return InklingConfig.from_hugging_face(
        _read_json(root / "config.json", "Inkling target")
    )


def load_draft_config(model: str | Path) -> InklingDSparkConfig:
    root = resolve_draft_path(model)
    return InklingDSparkConfig.from_hugging_face(
        _read_json(root / "config.json", "Inkling DSpark")
    )


__all__ = [
    "MODEL_SPEC",
    "InklingModelSpec",
    "load_draft_config",
    "load_target_config",
    "resolve_draft_path",
    "resolve_target_path",
]
