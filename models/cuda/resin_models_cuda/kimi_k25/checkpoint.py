"""Resolve Kimi K2.5-family ModelOpt NVFP4 targets and EAGLE3 MLA drafts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from resin_models_cuda.kimi_k25.config import (
    EagleVersion,
    KimiK25Config,
    KimiK25DraftConfig,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class KimiModelSpec:
    """Pinned target and draft identities for one Kimi release."""

    key: str
    target_repo: str
    target_revision: str
    draft_repo: str
    draft_revision: str
    eagle_version: EagleVersion


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        KimiModelSpec(
            key="k2.5",
            target_repo="nvidia/Kimi-K2.5-NVFP4",
            target_revision="0fd0a5e6879298d3476e3b61852a79792a35ae3d",
            draft_repo="lightseekorg/kimi-k2.5-eagle3-mla",
            draft_revision="9c0c42dfe8460a1d90ff9825de910fb51b0d293c",
            eagle_version=EagleVersion.EAGLE3,
        ),
        KimiModelSpec(
            key="k2.6",
            target_repo="nvidia/Kimi-K2.6-NVFP4",
            target_revision="2fd3a800dedd098b8327eb49e93ebc75f85da19f",
            draft_repo="lightseekorg/kimi-k2.6-eagle3.1-mla",
            draft_revision="35194ee8feb2826812f716eb42a924f99a5404f3",
            eagle_version=EagleVersion.EAGLE3_1,
        ),
        KimiModelSpec(
            key="k2.7-code",
            target_repo="nvidia/Kimi-K2.7-Code-NVFP4",
            target_revision="9f28d60908c98034bb288e755566764eac77cc0d",
            draft_repo="lightseekorg/kimi-k2.7-coder-eagle3.1-mla",
            draft_revision="20afca3aa32d44c32c0226fb5e4915dfa905af28",
            eagle_version=EagleVersion.EAGLE3_1,
        ),
    )
}
DEFAULT_MODEL_KEY = "k2.6"


def model_spec(key: str = DEFAULT_MODEL_KEY) -> KimiModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as error:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Kimi model must be one of: {choices}") from error


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        values = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} config.json is invalid") from error
    if not isinstance(values, dict):
        raise ValueError(f"{description} config.json must contain an object")
    return values


def resolve_target_path(model: str | Path) -> Path:
    """Validate a complete local NVIDIA NVFP4 target checkpoint."""

    root = Path(model).expanduser().resolve()
    required = ("config.json", "model.safetensors.index.json")
    if not root.is_dir():
        raise FileNotFoundError(f"Kimi target directory does not exist: {root}")
    if missing := [name for name in required if not (root / name).is_file()]:
        raise FileNotFoundError(f"Kimi target is missing files: {missing}")
    return root


def resolve_draft_path(model: str | Path) -> Path:
    """Validate a complete local EAGLE3 MLA draft checkpoint."""

    root = Path(model).expanduser().resolve()
    required = ("config.json", "model.safetensors")
    if not root.is_dir():
        raise FileNotFoundError(f"Kimi draft directory does not exist: {root}")
    if missing := [name for name in required if not (root / name).is_file()]:
        raise FileNotFoundError(f"Kimi draft is missing files: {missing}")
    return root


def load_target_config(model: str | Path) -> KimiK25Config:
    root = resolve_target_path(model)
    return KimiK25Config.from_hugging_face(
        _read_json(root / "config.json", "Kimi target")
    )


def load_draft_config(model: str | Path) -> KimiK25DraftConfig:
    root = resolve_draft_path(model)
    return KimiK25DraftConfig.from_hugging_face(
        _read_json(root / "config.json", "Kimi draft")
    )


__all__ = [
    "DEFAULT_MODEL_KEY",
    "MODEL_SPECS",
    "KimiModelSpec",
    "load_draft_config",
    "load_target_config",
    "model_spec",
    "resolve_draft_path",
    "resolve_target_path",
]
