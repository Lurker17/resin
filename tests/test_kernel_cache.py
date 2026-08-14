from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from resin.kernel_cache import ArtifactCache, wrap_flashinfer_build_and_load


class TestArtifactCache:
    def test_same_size_corruption_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "sm90-cuda-test")
            artifact = cache.root / "flashinfer" / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"abc")

            record = cache.publish(artifact, identity="flashinfer:module")
            assert record.schema == 2
            assert record.size == 3
            assert (
                record.blake3
                == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
            )
            assert cache.validate(artifact, identity="flashinfer:module")

            artifact.write_bytes(b"xyz")
            assert not cache.validate(artifact, identity="flashinfer:module")

    def test_missing_and_invalid_records_fall_back_to_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            calls: list[str] = []

            def build() -> str:
                calls.append("build")
                artifact.write_bytes(b"compiled")
                return "built"

            def load(path: Path) -> str:
                calls.append(f"load:{path.name}")
                return "loaded"

            assert (
                cache.load_or_build(artifact, identity="kernel", load=load, build=build)
                == "built"
            )
            assert (
                cache.load_or_build(artifact, identity="kernel", load=load, build=build)
                == "loaded"
            )

            cache.record_path(artifact).write_text("not json", encoding="utf-8")
            assert (
                cache.load_or_build(artifact, identity="kernel", load=load, build=build)
                == "built"
            )
            assert calls == ["build", "load:module.so", "build"]
            assert cache.hits == 1
            assert cache.misses == 2
            assert cache.validate(artifact, identity="kernel")

    def test_load_or_build_keeps_relative_artifact_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "nested" / "module.so"
            artifact.parent.mkdir(parents=True)

            assert (
                cache.load_or_build(
                    "nested/module.so",
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: (artifact.write_bytes(b"compiled"), "built")[1],
                )
                == "built"
            )
            assert cache.validate(artifact, identity="kernel")

    def test_record_rejects_wrong_abi_identity_and_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"compiled")
            cache.publish(artifact, identity="kernel")
            record_path = cache.record_path(artifact)
            original = json.loads(record_path.read_text(encoding="utf-8"))

            for key, value in (
                ("abi", "other-abi"),
                ("identity", "other-kernel"),
                ("artifact", "elsewhere/module.so"),
            ):
                changed = {**original, key: value}
                record_path.write_text(json.dumps(changed), encoding="utf-8")
                assert not cache.validate(artifact, identity="kernel")

    def test_atomic_publish_preserves_old_record_if_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"old")
            cache.publish(artifact, identity="kernel")
            record_path = cache.record_path(artifact)
            old_record = record_path.read_bytes()
            artifact.write_bytes(b"new")

            with (
                patch("resin.kernel_cache.os.replace", side_effect=OSError("failed")),
                pytest.raises(OSError, match="failed"),
            ):
                cache.publish(artifact, identity="kernel")

            assert record_path.read_bytes() == old_record
            assert list(record_path.parent.glob(".*.tmp")) == []

    def test_unsupported_directory_fsync_does_not_turn_publish_into_a_miss(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"compiled")

            real_fsync = os.fsync
            fsync_calls = 0

            def reject_directory(descriptor: int) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 2:
                    raise OSError(errno.EINVAL, "directory fsync unsupported")
                real_fsync(descriptor)

            with patch("resin.kernel_cache.os.fsync", side_effect=reject_directory):
                cache.publish(artifact, identity="kernel")

            assert cache.validate(artifact, identity="kernel")

    def test_artifact_must_stay_inside_abi_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            outside = Path(temporary) / "outside.so"
            outside.write_bytes(b"compiled")
            with pytest.raises(ValueError, match="escaped"):
                cache.publish(outside, identity="kernel")

    def test_publish_from_copies_local_artifact_and_records_copied_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            source = root / "scratch" / "module.so"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"locally compiled")

            record = cache.publish_from(
                source,
                "flashinfer/module.so",
                identity="flashinfer:module",
            )
            artifact = cache.root / "flashinfer" / "module.so"

            assert artifact.read_bytes() == b"locally compiled"
            assert record.artifact == "flashinfer/module.so"
            assert record.size == len(b"locally compiled")
            assert cache.validate(artifact, identity="flashinfer:module")
            assert list(artifact.parent.glob(".*.tmp")) == []

    def test_publish_from_copy_replace_failure_preserves_previous_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"old compiled")
            cache.publish(artifact, identity="kernel")
            old_record = cache.record_path(artifact).read_bytes()
            source = root / "scratch.so"
            source.write_bytes(b"new compiled")

            with (
                patch("resin.kernel_cache.os.replace", side_effect=OSError("failed")),
                pytest.raises(OSError, match="failed"),
            ):
                cache.publish_from(source, artifact, identity="kernel")

            assert artifact.read_bytes() == b"old compiled"
            assert cache.record_path(artifact).read_bytes() == old_record
            assert cache.validate(artifact, identity="kernel")
            assert list(artifact.parent.glob(".*.tmp")) == []

    def test_publish_from_same_path_does_not_copy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"compiled")

            with patch("resin.kernel_cache._atomic_copy_artifact") as copy:
                cache.publish_from(artifact, artifact, identity="kernel")

            copy.assert_not_called()
            assert cache.validate(artifact, identity="kernel")

    def test_load_or_build_from_propagates_failures_without_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            artifact = cache.root / "module.so"
            source = root / "scratch.so"

            with pytest.raises(RuntimeError, match="compiler failed"):
                cache.load_or_build_from(
                    artifact,
                    source=source,
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: (_ for _ in ()).throw(
                        RuntimeError("compiler failed")
                    ),
                )

            assert not artifact.exists()
            assert not cache.record_path(artifact).exists()
            assert cache.misses == 1

            with pytest.raises(FileNotFoundError):
                cache.load_or_build_from(
                    artifact,
                    source=source,
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: "built without output",
                )

            assert not artifact.exists()
            assert not cache.record_path(artifact).exists()
            assert cache.hits == 0
            assert cache.misses == 2


class _FakeSpec:
    def __init__(self, artifact: Path, events: list[str], *, is_aot: bool = False):
        self.jit_library_path = artifact
        self.name = "module"
        self.is_aot = is_aot
        self._events = events
        self.loaded_paths: list[Path] = []

    def load(self, path: Path) -> str:
        self.loaded_paths.append(path)
        self._events.append(f"load:{path.name}")
        return "loaded"


class TestFlashInferAdapter:
    def test_wrapper_bypasses_builder_only_for_content_validated_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            events: list[str] = []

            def original(spec: _FakeSpec) -> str:
                events.append("build")
                spec.jit_library_path.write_bytes(b"compiled")
                return "built"

            wrapped = wrap_flashinfer_build_and_load(original, cache)
            spec = _FakeSpec(artifact, events)
            assert wrapped(spec) == "built"
            assert wrapped(spec) == "loaded"

            artifact.write_bytes(b"corrupt!")  # Same byte count as "compiled".
            assert wrapped(spec) == "built"
            assert events == ["build", "load:module.so", "build"]

    def test_aot_spec_always_uses_flashinfer_original(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            events: list[str] = []

            def original(spec: _FakeSpec) -> str:
                events.append("aot")
                return "original"

            artifact = cache.root / "aot.so"
            spec = _FakeSpec(artifact, events, is_aot=True)
            wrapped = wrap_flashinfer_build_and_load(original, cache)
            assert wrapped(spec) == "original"
            assert events == ["aot"]
            assert not cache.record_path(artifact).exists()

    def test_wrapper_builds_locally_then_loads_from_durable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            local_artifact = root / "scratch" / "module.so"
            local_artifact.parent.mkdir(parents=True)
            durable_artifact = cache.root / "flashinfer" / "module.so"
            events: list[str] = []

            def original(spec: _FakeSpec) -> str:
                events.append("build-local")
                spec.jit_library_path.write_bytes(b"compiled locally")
                return "built"

            resolver = Mock(return_value="flashinfer/module.so")
            wrapped = wrap_flashinfer_build_and_load(
                original,
                cache,
                durable_artifact_resolver=resolver,
            )
            spec = _FakeSpec(local_artifact, events)

            assert wrapped(spec) == "built"
            assert durable_artifact.read_bytes() == b"compiled locally"
            assert cache.validate(durable_artifact, identity="flashinfer:module")

            # The build output is no longer consulted after durable publication.
            local_artifact.write_bytes(b"changed scratch")
            assert wrapped(spec) == "loaded"
            assert spec.loaded_paths == [durable_artifact]
            assert events == ["build-local", "load:module.so"]
            assert cache.hits == 1
            assert cache.misses == 1
            assert resolver.call_count == 2

    def test_durable_corruption_rebuilds_from_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            local_artifact = root / "scratch" / "module.so"
            local_artifact.parent.mkdir(parents=True)
            durable_artifact = cache.root / "module.so"
            events: list[str] = []
            build_count = 0

            def original(spec: _FakeSpec) -> str:
                nonlocal build_count
                build_count += 1
                events.append(f"build:{build_count}")
                spec.jit_library_path.write_bytes(f"compiled-{build_count}".encode())
                return "built"

            wrapped = wrap_flashinfer_build_and_load(
                original,
                cache,
                durable_artifact_resolver=lambda _spec: "module.so",
            )
            spec = _FakeSpec(local_artifact, events)

            assert wrapped(spec) == "built"
            assert wrapped(spec) == "loaded"
            durable_artifact.write_bytes(b"corruptd-1")  # Same size as compiled-1.
            assert wrapped(spec) == "built"

            assert durable_artifact.read_bytes() == b"compiled-2"
            assert cache.validate(durable_artifact, identity="flashinfer:module")
            assert events == ["build:1", "load:module.so", "build:2"]
            assert cache.hits == 1
            assert cache.misses == 2

    def test_wrapper_build_failure_does_not_publish_durable_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            local_artifact = root / "scratch" / "module.so"
            durable_artifact = cache.root / "module.so"
            events: list[str] = []

            def original(_spec: _FakeSpec) -> str:
                events.append("failed-build")
                raise RuntimeError("compile failed")

            wrapped = wrap_flashinfer_build_and_load(
                original,
                cache,
                durable_artifact_resolver=lambda _spec: "module.so",
            )

            with pytest.raises(RuntimeError, match="compile failed"):
                wrapped(_FakeSpec(local_artifact, events))

            assert events == ["failed-build"]
            assert not durable_artifact.exists()
            assert not cache.record_path(durable_artifact).exists()
            assert cache.hits == 0
            assert cache.misses == 1
