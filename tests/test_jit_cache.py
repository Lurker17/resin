from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.jit_cache import (
    _LOCAL_BUILD_PATH_ENV_KEYS,
    _PATH_ENV_KEYS,
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    changed_cache_files,
    kernel_cache_summary,
)


class KernelCacheContractTests(unittest.TestCase):
    def test_every_runtime_cache_is_routed_below_the_volume_mount(self) -> None:
        for key in _PATH_ENV_KEYS:
            with self.subTest(key=key):
                self.assertTrue(
                    Path(KERNEL_CACHE_ENV[key]).is_relative_to(KERNEL_CACHE_MOUNT)
                )

    def test_flashinfer_build_workspace_is_local_not_durable(self) -> None:
        for key in _LOCAL_BUILD_PATH_ENV_KEYS:
            with self.subTest(key=key):
                path = Path(KERNEL_CACHE_ENV[key])
                self.assertTrue(path.is_absolute())
                self.assertFalse(path.is_relative_to(KERNEL_CACHE_MOUNT))

    def test_required_opt_in_caches_are_enabled(self) -> None:
        self.assertEqual(KERNEL_CACHE_ENV["USE_PYTORCH_KERNEL_CACHE"], "1")
        self.assertEqual(
            KERNEL_CACHE_ENV["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"], "1"
        )
        self.assertEqual(KERNEL_CACHE_ENV["QUACK_CACHE_ENABLED"], "1")

    def test_manifest_helpers_detect_only_new_or_resized_files(self) -> None:
        before = {"fa4/a.o": 10, "flashinfer/module.so": 20}
        after = {**before, "triton/kernel.cubin": 30}
        self.assertEqual(changed_cache_files(before, after), ["triton/kernel.cubin"])
        self.assertEqual(
            kernel_cache_summary(after),
            {
                "fa4": {"files": 1, "bytes": 10},
                "flashinfer": {"files": 1, "bytes": 20},
                "triton": {"files": 1, "bytes": 30},
            },
        )

    def test_environment_override_cannot_escape_mount(self) -> None:
        # Import locally so the test never needs Modal or a mounted volume.
        from deploy.jit_cache import ensure_kernel_cache_dirs

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {"CUDA_CACHE_PATH": str(Path(temporary) / "outside")},
            ),
            self.assertRaisesRegex(RuntimeError, "escaped"),
        ):
            ensure_kernel_cache_dirs()

    def test_deployment_uses_one_abi_scoped_namespace(self) -> None:
        from deploy.jit_cache import (
            KERNEL_CACHE_ABI,
            KERNEL_CACHE_MARKER_SCHEMA,
            KERNEL_CACHE_ROOT,
        )

        self.assertEqual(
            Path(KERNEL_CACHE_ROOT), Path(KERNEL_CACHE_MOUNT) / KERNEL_CACHE_ABI
        )
        self.assertTrue(KERNEL_CACHE_ABI.startswith("v6-blake3-"))
        self.assertEqual(KERNEL_CACHE_MARKER_SCHEMA, 2)

    def test_artifact_fingerprints_ignore_mtime_changes(self) -> None:
        from deploy.jit_cache import kernel_artifact_fingerprints

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("deploy.jit_cache.KERNEL_CACHE_ROOT", temporary),
        ):
            artifact = Path(temporary) / "flash-attention-4" / "fwd" / "kernel.o"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"compiled")
            before = kernel_artifact_fingerprints()
            self.assertEqual(
                before["flash-attention-4/fwd/kernel.o"],
                "8:0a1c9344b835c6d495d9c0373e619936990f97b18cb8ba615b5344adf724a5e5",
            )
            stat = artifact.stat()
            os.utime(
                artifact,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            self.assertEqual(kernel_artifact_fingerprints(), before)

    def test_autotune_json_is_content_fingerprinted_without_mtimes(self) -> None:
        from deploy.jit_cache import kernel_artifact_fingerprints

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("deploy.jit_cache.KERNEL_CACHE_ROOT", temporary),
        ):
            artifact = Path(temporary) / "flashinfer-autotune" / "gpt-oss.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b'{"tactic":1}')
            before = kernel_artifact_fingerprints()
            self.assertIn("flashinfer-autotune/gpt-oss.json", before)

            stat = artifact.stat()
            os.utime(
                artifact,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            self.assertEqual(kernel_artifact_fingerprints(), before)

            artifact.write_bytes(b'{"tactic":2}')
            self.assertNotEqual(kernel_artifact_fingerprints(), before)


if __name__ == "__main__":
    unittest.main()
