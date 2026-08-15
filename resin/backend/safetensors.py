"""Validated indexed-safetensor access shared by checkpoint loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self


def indexed_tensor_files(root: Path, *, require_index: bool = False) -> dict[str, Path]:
    """Resolve every tensor name to one validated local safetensor file."""

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        from safetensors import safe_open

        if require_index:
            raise FileNotFoundError(f"missing checkpoint index: {index_path}")
        model_file = root / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"missing checkpoint weights: {model_file}")
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            return {key: model_file for key in handle.keys()}

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {index_path}: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path.name} must contain a non-empty weight_map")

    tensor_files = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint index contains an invalid tensor name")
        if not isinstance(filename, str):
            raise ValueError("checkpoint index contains an invalid shard name")
        relative = Path(filename)
        if (
            not filename
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"invalid checkpoint shard name: {filename!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {path}")
        tensor_files[key] = path
    return tensor_files


class SafetensorReader:
    """Keep at most one memory-mapped checkpoint shard open."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._context: Any | None = None
        self._handle: Any | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._path = None
        self._context = None
        self._handle = None

    def read(self, path: Path, key: str):
        if path != self._path:
            from safetensors import safe_open

            self.close()
            self._context = safe_open(str(path), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._path = path
        return self._handle.get_tensor(key)


__all__ = ["SafetensorReader", "indexed_tensor_files"]
