from __future__ import annotations

import os
from pathlib import Path

import pytest

from deploy.jit_cache import (
    _LOCAL_BUILD_PATH_ENV_KEYS,
    _PATH_ENV_KEYS,
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    changed_cache_files,
    kernel_cache_summary,
)


class TestKernelCacheContract:
    def test_every_runtime_cache_is_routed_below_the_volume_mount(self) -> None:
        for key in _PATH_ENV_KEYS:
            assert Path(KERNEL_CACHE_ENV[key]).is_relative_to(KERNEL_CACHE_MOUNT)

    def test_flashinfer_build_workspace_is_local_not_durable(self) -> None:
        for key in _LOCAL_BUILD_PATH_ENV_KEYS:
            path = Path(KERNEL_CACHE_ENV[key])
            assert path.is_absolute()
            assert not path.is_relative_to(KERNEL_CACHE_MOUNT)

    def test_required_opt_in_caches_are_enabled(self) -> None:
        assert KERNEL_CACHE_ENV["USE_PYTORCH_KERNEL_CACHE"] == "1"
        assert KERNEL_CACHE_ENV["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "1"
        assert KERNEL_CACHE_ENV["QUACK_CACHE_ENABLED"] == "1"

    def test_manifest_helpers_detect_only_new_or_resized_files(self) -> None:
        before = {"fa4/a.o": 10, "flashinfer/module.so": 20}
        after = {**before, "triton/kernel.cubin": 30}
        assert changed_cache_files(before, after) == ["triton/kernel.cubin"]
        assert kernel_cache_summary(after) == {
            "fa4": {"files": 1, "bytes": 10},
            "flashinfer": {"files": 1, "bytes": 20},
            "triton": {"files": 1, "bytes": 30},
        }

    def test_environment_override_cannot_escape_mount(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Import locally so the test never needs Modal or a mounted volume.
        from deploy.jit_cache import ensure_kernel_cache_dirs

        monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "outside"))
        with pytest.raises(RuntimeError, match="escaped"):
            ensure_kernel_cache_dirs()

    def test_deployment_uses_one_abi_scoped_namespace(self) -> None:
        from deploy.jit_cache import (
            KERNEL_CACHE_ABI,
            KERNEL_CACHE_MARKER_SCHEMA,
            KERNEL_CACHE_ROOT,
        )

        assert Path(KERNEL_CACHE_ROOT) == Path(KERNEL_CACHE_MOUNT) / KERNEL_CACHE_ABI
        assert KERNEL_CACHE_ABI.startswith("v10-blake3-")
        assert KERNEL_CACHE_MARKER_SCHEMA == 2

    def test_artifact_fingerprints_ignore_mtime_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deploy.jit_cache import kernel_artifact_fingerprints

        monkeypatch.setattr("deploy.jit_cache.KERNEL_CACHE_ROOT", tmp_path)
        artifact = tmp_path / "flash-attention-4" / "fwd" / "kernel.o"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"compiled")
        before = kernel_artifact_fingerprints()
        assert (
            before["flash-attention-4/fwd/kernel.o"]
            == "8:0a1c9344b835c6d495d9c0373e619936990f97b18cb8ba615b5344adf724a5e5"
        )
        stat = artifact.stat()
        os.utime(
            artifact,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        assert kernel_artifact_fingerprints() == before

    def test_autotune_json_is_content_fingerprinted_without_mtimes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deploy.jit_cache import kernel_artifact_fingerprints

        monkeypatch.setattr("deploy.jit_cache.KERNEL_CACHE_ROOT", tmp_path)
        artifact = tmp_path / "flashinfer-autotune" / "gpt-oss.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'{"tactic":1}')
        before = kernel_artifact_fingerprints()
        assert "flashinfer-autotune/gpt-oss.json" in before

        stat = artifact.stat()
        os.utime(
            artifact,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        assert kernel_artifact_fingerprints() == before

        artifact.write_bytes(b'{"tactic":2}')
        assert kernel_artifact_fingerprints() != before
