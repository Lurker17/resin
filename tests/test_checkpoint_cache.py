"""Checkpoint metadata publication reads file sizes, never model weight bodies."""

import json

import pytest

from deploy.checkpoint_cache import (
    MARKER_NAME,
    checkpoint_is_complete,
    publish_checkpoint,
)


def checkpoint(root):
    (root / "inference").mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "inference/config.json",
    ):
        (root / name).write_text("{}")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-1.safetensors",
                    "second": "model-2.safetensors",
                }
            }
        )
    )
    for number in (1, 2):
        (root / f"model-{number}.safetensors").write_bytes(b"fixture" * number)


def test_marker_preserves_schema_and_detects_changed_inventory(tmp_path):
    checkpoint(tmp_path)
    assert not checkpoint_is_complete(tmp_path, "repo", "revision")
    inventory = publish_checkpoint(tmp_path, "repo", "revision")
    assert json.loads((tmp_path / MARKER_NAME).read_text()) == {
        "schema": 1,
        "repo_id": "repo",
        "revision": "revision",
        "files": inventory,
    }
    assert checkpoint_is_complete(tmp_path, "repo", "revision")
    assert not checkpoint_is_complete(tmp_path, "other", "revision")
    assert not checkpoint_is_complete(tmp_path, "repo", "other")
    (tmp_path / "model-1.safetensors").write_bytes(b"changed size")
    assert not checkpoint_is_complete(tmp_path, "repo", "revision")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("filename", ["../outside.safetensors", [], None])
def test_bad_index_shard_paths_cannot_be_published(tmp_path, filename):
    checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": filename}})
    )
    with pytest.raises(RuntimeError, match="invalid shard paths"):
        publish_checkpoint(tmp_path, "repo", "revision")
    assert not (tmp_path / MARKER_NAME).exists()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_shard_cannot_be_published(tmp_path, missing):
    checkpoint(tmp_path)
    shard = tmp_path / "model-1.safetensors"
    if missing:
        shard.unlink()
    else:
        shard.write_bytes(b"")
    with pytest.raises(RuntimeError, match="missing files|empty files"):
        publish_checkpoint(tmp_path, "repo", "revision")
    assert not (tmp_path / MARKER_NAME).exists()


def test_single_file_checkpoint_does_not_require_an_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    options = {"required_files": ("config.json",)}
    files = publish_checkpoint(tmp_path, "assistant", "pin", **options)
    assert set(files) == {"config.json", "model.safetensors"}
    assert checkpoint_is_complete(tmp_path, "assistant", "pin", **options)
    (tmp_path / "model-2.safetensors").write_bytes(b"more weights")
    assert not checkpoint_is_complete(tmp_path, "assistant", "pin", **options)
