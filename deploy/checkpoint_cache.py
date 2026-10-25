"""Data-only checkpoint publication shared by all serving and benchmark jobs."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

MARKER_NAME = ".resin-checkpoint.json"


DEFAULT_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference/config.json",
    "model.safetensors.index.json",
)


def checkpoint_inventory(
    root: Path,
    *,
    required_files: tuple[str, ...] = DEFAULT_REQUIRED_FILES,
) -> dict[str, int]:
    """Validate the indexed shards (or a single weight file) and stat required files."""
    index = root / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("checkpoint has an invalid safetensor index") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("checkpoint safetensor index has no weights")
        if not all(
            isinstance(name, str)
            and Path(name).name == name
            and name.endswith(".safetensors")
            for name in weight_map.values()
        ):
            raise RuntimeError("checkpoint safetensor index has invalid shard paths")
        weights = sorted(set(weight_map.values()))
        required_files = (*required_files, index.name)
    else:
        weights = [path.name for path in root.glob("model*.safetensors")]
        if len(weights) != 1:
            raise RuntimeError(
                "checkpoint needs one weight file or a complete safetensor index"
            )
    files = [root / name for name in dict.fromkeys((*required_files, *weights))]
    if missing := [str(path.relative_to(root)) for path in files if not path.is_file()]:
        raise RuntimeError(f"checkpoint is missing files: {missing}")
    inventory = {str(path.relative_to(root)): path.stat().st_size for path in files}
    if empty := [name for name, size in inventory.items() if not size]:
        raise RuntimeError(f"checkpoint contains empty files: {empty}")
    return inventory


def _marker(repo_id, revision, inventory):
    return {"schema": 1, "repo_id": repo_id, "revision": revision, "files": inventory}


def checkpoint_is_complete(
    root: Path,
    repo_id: str,
    revision: str,
    *,
    required_files: tuple[str, ...] = DEFAULT_REQUIRED_FILES,
) -> bool:
    try:
        marker = json.loads((root / MARKER_NAME).read_text())
        return marker == _marker(
            repo_id, revision, checkpoint_inventory(root, required_files=required_files)
        )
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False


def publish_checkpoint(
    root: Path,
    repo_id: str,
    revision: str,
    *,
    required_files: tuple[str, ...] = DEFAULT_REQUIRED_FILES,
) -> dict[str, int]:
    """Publish an atomic metadata marker after validation; never read weight bodies."""
    inventory = checkpoint_inventory(root, required_files=required_files)
    with NamedTemporaryFile(
        mode="w", dir=root, prefix=MARKER_NAME, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(_marker(repo_id, revision, inventory), stream, sort_keys=True)
    try:
        temporary.replace(root / MARKER_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return inventory
