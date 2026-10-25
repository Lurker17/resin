"""Checkpoint metadata publication reads file sizes, never model weight bodies."""

import json
from types import SimpleNamespace

import pytest

from deploy.checkpoint_cache import (
    MARKER_NAME,
    checkpoint_inventory,
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


def test_metadata_only_publication_is_ready_for_v41_serving(monkeypatch, tmp_path):
    from deploy import modal_bench as generic
    from deploy import modal_deepseek_v41 as serving

    target = tmp_path / "flash-1017" / serving.REVISION
    checkpoint(target)
    operations = []
    volume = SimpleNamespace(
        reload=lambda: operations.append("reload"),
        commit=lambda: operations.append("commit"),
    )
    monkeypatch.setattr(
        generic, "_checkpoint", lambda name: (target, serving.MODEL, serving.REVISION)
    )
    monkeypatch.setitem(
        generic.MODEL_VOLUMES, generic.CHECKPOINTS["deepseek-v41"][2], volume
    )

    def no_redownload(*args, **kwargs):
        raise AssertionError("metadata-only publication must not download weights")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_redownload)
    with pytest.warns(UserWarning, match="executing locally"):
        report = generic.download.local("deepseek-v41", True)
    assert report["metadata_only"] is True
    assert report["files"] == len(checkpoint_inventory(target))
    assert operations == ["reload", "commit"]
    assert checkpoint_is_complete(target, serving.MODEL, serving.REVISION)


def test_generic_deepseek_sweep_rejects_config_only_before_model_loading(
    monkeypatch, tmp_path
):
    from deploy import modal_bench as generic

    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(generic, "_checkpoint", lambda name: (tmp_path, "repo", "pin"))
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(FileNotFoundError, match="publish-only"):
            generic.sweep.local("deepseek-v41", [], {"world_size": 1}, {})
