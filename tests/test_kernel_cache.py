from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from resin.kernel_cache import ArtifactCache, wrap_flashinfer_build_and_load


class ArtifactCacheTests(unittest.TestCase):
    def test_same_size_corruption_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "sm90-cuda-test")
            artifact = cache.root / "flashinfer" / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"abc")

            record = cache.publish(artifact, identity="flashinfer:module")
            self.assertEqual(record.schema, 2)
            self.assertEqual(record.size, 3)
            self.assertEqual(
                record.blake3,
                "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85",
            )
            self.assertTrue(cache.validate(artifact, identity="flashinfer:module"))

            artifact.write_bytes(b"xyz")
            self.assertFalse(cache.validate(artifact, identity="flashinfer:module"))

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

            self.assertEqual(
                cache.load_or_build(
                    artifact, identity="kernel", load=load, build=build
                ),
                "built",
            )
            self.assertEqual(
                cache.load_or_build(
                    artifact, identity="kernel", load=load, build=build
                ),
                "loaded",
            )

            cache.record_path(artifact).write_text("not json", encoding="utf-8")
            self.assertEqual(
                cache.load_or_build(
                    artifact, identity="kernel", load=load, build=build
                ),
                "built",
            )
            self.assertEqual(calls, ["build", "load:module.so", "build"])
            self.assertEqual(cache.hits, 1)
            self.assertEqual(cache.misses, 2)
            self.assertTrue(cache.validate(artifact, identity="kernel"))

    def test_load_or_build_keeps_relative_artifact_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "nested" / "module.so"
            artifact.parent.mkdir(parents=True)

            self.assertEqual(
                cache.load_or_build(
                    "nested/module.so",
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: (artifact.write_bytes(b"compiled"), "built")[1],
                ),
                "built",
            )
            self.assertTrue(cache.validate(artifact, identity="kernel"))

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
                with self.subTest(key=key):
                    changed = {**original, key: value}
                    record_path.write_text(json.dumps(changed), encoding="utf-8")
                    self.assertFalse(cache.validate(artifact, identity="kernel"))

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
                self.assertRaisesRegex(OSError, "failed"),
            ):
                cache.publish(artifact, identity="kernel")

            self.assertEqual(record_path.read_bytes(), old_record)
            self.assertEqual(list(record_path.parent.glob(".*.tmp")), [])

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

            self.assertTrue(cache.validate(artifact, identity="kernel"))

    def test_artifact_must_stay_inside_abi_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            outside = Path(temporary) / "outside.so"
            outside.write_bytes(b"compiled")
            with self.assertRaisesRegex(ValueError, "escaped"):
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

            self.assertEqual(artifact.read_bytes(), b"locally compiled")
            self.assertEqual(record.artifact, "flashinfer/module.so")
            self.assertEqual(record.size, len(b"locally compiled"))
            self.assertTrue(cache.validate(artifact, identity="flashinfer:module"))
            self.assertEqual(list(artifact.parent.glob(".*.tmp")), [])

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
                self.assertRaisesRegex(OSError, "failed"),
            ):
                cache.publish_from(source, artifact, identity="kernel")

            self.assertEqual(artifact.read_bytes(), b"old compiled")
            self.assertEqual(cache.record_path(artifact).read_bytes(), old_record)
            self.assertTrue(cache.validate(artifact, identity="kernel"))
            self.assertEqual(list(artifact.parent.glob(".*.tmp")), [])

    def test_publish_from_same_path_does_not_copy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = ArtifactCache(temporary, "test-abi")
            artifact = cache.root / "module.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"compiled")

            with patch("resin.kernel_cache._atomic_copy_artifact") as copy:
                cache.publish_from(artifact, artifact, identity="kernel")

            copy.assert_not_called()
            self.assertTrue(cache.validate(artifact, identity="kernel"))

    def test_load_or_build_from_propagates_failures_without_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = ArtifactCache(root / "durable", "test-abi")
            artifact = cache.root / "module.so"
            source = root / "scratch.so"

            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                cache.load_or_build_from(
                    artifact,
                    source=source,
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: (_ for _ in ()).throw(
                        RuntimeError("compiler failed")
                    ),
                )

            self.assertFalse(artifact.exists())
            self.assertFalse(cache.record_path(artifact).exists())
            self.assertEqual(cache.misses, 1)

            with self.assertRaises(FileNotFoundError):
                cache.load_or_build_from(
                    artifact,
                    source=source,
                    identity="kernel",
                    load=lambda _path: "loaded",
                    build=lambda: "built without output",
                )

            self.assertFalse(artifact.exists())
            self.assertFalse(cache.record_path(artifact).exists())
            self.assertEqual(cache.hits, 0)
            self.assertEqual(cache.misses, 2)


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


class FlashInferAdapterTests(unittest.TestCase):
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
            self.assertEqual(wrapped(spec), "built")
            self.assertEqual(wrapped(spec), "loaded")

            artifact.write_bytes(b"corrupt!")  # Same byte count as "compiled".
            self.assertEqual(wrapped(spec), "built")
            self.assertEqual(events, ["build", "load:module.so", "build"])

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
            self.assertEqual(wrapped(spec), "original")
            self.assertEqual(events, ["aot"])
            self.assertFalse(cache.record_path(artifact).exists())

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

            self.assertEqual(wrapped(spec), "built")
            self.assertEqual(durable_artifact.read_bytes(), b"compiled locally")
            self.assertTrue(
                cache.validate(durable_artifact, identity="flashinfer:module")
            )

            # The build output is no longer consulted after durable publication.
            local_artifact.write_bytes(b"changed scratch")
            self.assertEqual(wrapped(spec), "loaded")
            self.assertEqual(spec.loaded_paths, [durable_artifact])
            self.assertEqual(events, ["build-local", "load:module.so"])
            self.assertEqual(cache.hits, 1)
            self.assertEqual(cache.misses, 1)
            self.assertEqual(resolver.call_count, 2)

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

            self.assertEqual(wrapped(spec), "built")
            self.assertEqual(wrapped(spec), "loaded")
            durable_artifact.write_bytes(b"corruptd-1")  # Same size as compiled-1.
            self.assertEqual(wrapped(spec), "built")

            self.assertEqual(durable_artifact.read_bytes(), b"compiled-2")
            self.assertTrue(
                cache.validate(durable_artifact, identity="flashinfer:module")
            )
            self.assertEqual(events, ["build:1", "load:module.so", "build:2"])
            self.assertEqual(cache.hits, 1)
            self.assertEqual(cache.misses, 2)

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

            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                wrapped(_FakeSpec(local_artifact, events))

            self.assertEqual(events, ["failed-build"])
            self.assertFalse(durable_artifact.exists())
            self.assertFalse(cache.record_path(durable_artifact).exists())
            self.assertEqual(cache.hits, 0)
            self.assertEqual(cache.misses, 1)


if __name__ == "__main__":
    unittest.main()
