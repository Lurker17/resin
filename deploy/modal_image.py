"""Build the shared Modal image used by Resin GPU deployments."""

from __future__ import annotations

from pathlib import Path

import modal

_DEPENDENCY_ROOT = "/opt/resin-deps"
_SOURCE_ROOT = "/opt/resin"
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def build_modal_image(
    root: Path,
    *,
    env: dict[str, str] | None = None,
) -> modal.Image:
    """Build the common CUDA image and install one deployment's source tree.

    Modal's built-in ``Image.uv_sync`` currently rejects uv workspaces. Copying
    only the workspace manifests keeps dependency installation cacheable while
    retaining ``pyproject.toml`` and ``uv.lock`` as the single source of truth.
    """

    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.3.1-devel-ubuntu22.04", add_python="3.12"
        )
        .apt_install("build-essential", "curl", "git", "ninja-build")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --profile minimal"
        )
        .env(
            {
                "PATH": f"/root/.cargo/bin:{_SYSTEM_PATH}",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "MAX_JOBS": "16",
                "CMAKE_BUILD_PARALLEL_LEVEL": "16",
            }
        )
        .pip_install("uv==0.12.5")
        .add_local_file(
            root / "pyproject.toml",
            f"{_DEPENDENCY_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file(root / "uv.lock", f"{_DEPENDENCY_ROOT}/uv.lock", copy=True)
        .add_local_file(
            root / "models/cuda/pyproject.toml",
            f"{_DEPENDENCY_ROOT}/models/cuda/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            root / "models/torch/pyproject.toml",
            f"{_DEPENDENCY_ROOT}/models/torch/pyproject.toml",
            copy=True,
        )
        .run_commands(
            "uv sync "
            f"--project {_DEPENDENCY_ROOT} "
            "--frozen --no-install-workspace --no-default-groups "
            "--group deployment --compile-bytecode"
        )
        .env(
            {"PATH": (f"{_DEPENDENCY_ROOT}/.venv/bin:/root/.cargo/bin:{_SYSTEM_PATH}")}
        )
        # These accelerator-specific supplement wheels live outside PyPI.
        .uv_pip_install(
            "flashinfer-cubin==0.6.15.post1",
            extra_index_url="https://flashinfer.ai/whl",
        )
        .uv_pip_install(
            "flashinfer-jit-cache==0.6.15.post1+cu130",
            extra_index_url="https://flashinfer.ai/whl/cu130",
        )
        # Keep dependency layers independent of local source edits.
        .add_local_file(
            root / "deploy/__init__.py", "/root/deploy/__init__.py", copy=True
        )
        .add_local_file(root / "deploy/http.py", "/root/deploy/http.py", copy=True)
        .add_local_file(
            root / "deploy/jit_cache.py", "/root/deploy/jit_cache.py", copy=True
        )
        .add_local_file(
            root / "deploy/modal_image.py", "/root/deploy/modal_image.py", copy=True
        )
        .add_local_file(root / "Cargo.toml", f"{_SOURCE_ROOT}/Cargo.toml", copy=True)
        .add_local_file(root / "Cargo.lock", f"{_SOURCE_ROOT}/Cargo.lock", copy=True)
        .add_local_file(
            root / "pyproject.toml", f"{_SOURCE_ROOT}/pyproject.toml", copy=True
        )
        .add_local_file(root / "README.md", f"{_SOURCE_ROOT}/README.md", copy=True)
        .add_local_dir(root / "docs", f"{_SOURCE_ROOT}/docs", copy=True)
        .add_local_dir(root / "crates", f"{_SOURCE_ROOT}/crates", copy=True)
        .add_local_dir(root / "resin", f"{_SOURCE_ROOT}/resin", copy=True)
        .add_local_file(
            root / "models/torch/pyproject.toml",
            f"{_SOURCE_ROOT}/models/torch/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            root / "models/torch/resin_models_torch/__init__.py",
            f"{_SOURCE_ROOT}/models/torch/resin_models_torch/__init__.py",
            copy=True,
        )
        .add_local_dir(
            root / "models/cuda",
            f"{_SOURCE_ROOT}/models/cuda",
            copy=True,
            ignore=["tests", "resin_models_cuda/lfm2", "**/__pycache__"],
        )
        .uv_pip_install(_SOURCE_ROOT, extra_options="--no-deps")
        .uv_pip_install(f"{_SOURCE_ROOT}/models/torch", extra_options="--no-deps")
        .uv_pip_install(f"{_SOURCE_ROOT}/models/cuda", extra_options="--no-deps")
        .env(env or {})
    )


__all__ = ["build_modal_image"]
