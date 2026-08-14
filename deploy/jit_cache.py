"""Shared persistent-kernel-cache contract for Resin's Modal harnesses.

Only compiled code and kernel-source packages belong here. Model checkpoints use
their own image layer or model volume so a kernel-cache inventory stays small and
auditable.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from blake3 import blake3

KERNEL_CACHE_VOLUME_NAME = "resin-kernel-cache-v2"
KERNEL_CACHE_VOLUME_VERSION = 2
KERNEL_CACHE_MARKER_SCHEMA = 2
KERNEL_CACHE_MOUNT = "/cache/resin-kernels"
KERNEL_CACHE_ABI = (
    "v10-blake3-localpublish1-h100-sm90a-py312-cuda13.3.1-torch2.13.0-cu130-"
    "triton3.7.1-"
    "flashinfer0.6.15p1-fijit0.6.15p1cu130-fa4b26-cutlassdsl4.6.2-"
    "tvmffi0.1.13p2-quack0.6.4-cutile1.4.0-qkvpack-tinygemm-top4-fiar"
)
KERNEL_CACHE_ROOT = f"{KERNEL_CACHE_MOUNT}/{KERNEL_CACHE_ABI}"
FLASHINFER_BUILD_WORKSPACE = "/tmp/resin-flashinfer-build"
_CACHE_PREPARED = False
_KERNEL_ARTIFACT_CACHE = None


def _cache_dir(name: str) -> str:
    return f"{KERNEL_CACHE_ROOT}/{name}"


# Each setting is read by a different runtime. Keep the paths disjoint so the
# remote report can show exactly which compiler produced which artifacts.
KERNEL_CACHE_ENV = {
    # NVIDIA's driver cache for PTX -> cubin compilation.
    "CUDA_CACHE_PATH": _cache_dir("cuda-driver"),
    "CUDA_CACHE_MAXSIZE": str(4 * 1024**3),
    "CUDA_CACHE_DISABLE": "0",
    "TORCH_CUDA_ARCH_LIST": "9.0a",
    # PyTorch eager Jiterator and C++/CUDA extensions.
    "USE_PYTORCH_KERNEL_CACHE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": _cache_dir("pytorch-jiterator"),
    "TORCH_EXTENSIONS_DIR": _cache_dir("torch-extensions"),
    # torch.compile / Inductor and its Triton kernels.
    "TORCHINDUCTOR_CACHE_DIR": _cache_dir("torchinductor"),
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "TRITON_CACHE_DIR": _cache_dir("triton"),
    # FlashInfer builds in local scratch. Resin publishes only a completed,
    # content-validated shared library into the durable ``flashinfer``
    # namespace; Ninja never runs against the mounted cache filesystem.
    "FLASHINFER_WORKSPACE_BASE": FLASHINFER_BUILD_WORKSPACE,
    "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
    # FA4's CuTe-DSL object cache is opt-in in the pinned beta wheel.
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": _cache_dir("flash-attention-4"),
    "FLASH_ATTENTION_ARCH": "sm_90a",
    # Underlying CUTLASS DSL MLIR cache and QuACK's compiled object cache.
    "CUTE_DSL_CACHE_DIR": _cache_dir("cute-dsl"),
    "CUTE_DSL_ARCH": "sm_90a",
    "QUACK_CACHE_ENABLED": "1",
    "QUACK_CACHE_DIR": _cache_dir("quack"),
    # Hugging Face kernels source packages (not model weights).
    "KERNELS_CACHE": _cache_dir("hf-kernels"),
}

_PATH_ENV_KEYS = (
    "CUDA_CACHE_PATH",
    "PYTORCH_KERNEL_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "QUACK_CACHE_DIR",
    "KERNELS_CACHE",
)
_LOCAL_BUILD_PATH_ENV_KEYS = ("FLASHINFER_WORKSPACE_BASE",)


def ensure_kernel_cache_dirs() -> None:
    """Create every compiler-specific namespace before importing GPU libraries."""

    paths = []
    for key in _PATH_ENV_KEYS:
        path = Path(os.environ.get(key, KERNEL_CACHE_ENV[key]))
        if not path.is_relative_to(KERNEL_CACHE_MOUNT):
            raise RuntimeError(f"{key} escaped the kernel-cache mount: {path}")
        paths.append(path)

    root = Path(KERNEL_CACHE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    for key in _LOCAL_BUILD_PATH_ENV_KEYS:
        path = Path(os.environ.get(key, KERNEL_CACHE_ENV[key]))
        if not path.is_absolute() or path.is_relative_to(KERNEL_CACHE_MOUNT):
            raise RuntimeError(
                f"{key} must be an absolute local build workspace outside "
                f"the durable cache mount: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)

    # FlashInfer's trace-apply solutions currently bypass its workspace-base
    # setting. Redirect that one hard-coded cache without moving HOME (which
    # would also move model downloads into the kernel volume).
    trace_target = root / "flashinfer-trace-apply"
    trace_target.mkdir(parents=True, exist_ok=True)
    trace_source = Path.home() / ".cache/flashinfer/trace_apply/solutions"
    trace_source.parent.mkdir(parents=True, exist_ok=True)
    if trace_source.is_symlink():
        if trace_source.resolve() != trace_target.resolve():
            raise RuntimeError(f"unexpected FlashInfer trace cache: {trace_source}")
    elif trace_source.exists():
        raise RuntimeError(
            f"FlashInfer trace cache exists outside the kernel volume: {trace_source}"
        )
    else:
        trace_source.symlink_to(trace_target, target_is_directory=True)


def kernel_cache_manifest() -> dict[str, int]:
    """Return relative file names and sizes without reading compiled binaries."""

    root = Path(KERNEL_CACHE_ROOT)
    if not root.exists():
        return {}
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if relative_path.parts and relative_path.parts[0] == "_resin":
            continue
        manifest[relative_path.as_posix()] = path.stat().st_size
    return manifest


def kernel_cache_summary(manifest: dict[str, int]) -> dict[str, dict[str, int]]:
    """Aggregate a manifest by compiler namespace for compact Modal output."""

    summary: dict[str, dict[str, int]] = {}
    for relative_path, size in manifest.items():
        namespace = relative_path.split("/", 1)[0]
        item = summary.setdefault(namespace, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += size
    return summary


def changed_cache_files(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """List new, removed, or resized files; mtimes are intentionally ignored."""

    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


def cache_files_with_suffix(
    manifest: dict[str, int], namespace: str, suffix: str
) -> list[str]:
    prefix = f"{namespace}/"
    return sorted(
        path for path in manifest if path.startswith(prefix) and path.endswith(suffix)
    )


def _is_stable_cache_artifact(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.parts and path.parts[0] == "_resin":
        return False
    if path.name in {".ninja_deps", ".ninja_log", "lock"}:
        return False
    if path.name.endswith((".lock", ".tmp")):
        return False
    namespace = path.parts[0]
    if namespace == "flashinfer":
        return path.suffix in {".so", ".o", ".cubin", ".ptx"} or path.name.endswith(
            ".resin-cache.json"
        )
    if namespace == "flashinfer-autotune":
        return path.suffix == ".json"
    if namespace == "flash-attention-4":
        return path.suffix == ".o"
    if namespace == "cute-dsl":
        return path.suffix in {".mlir", ".o", ".cubin", ".ptx"}
    if namespace == "quack":
        return path.suffix == ".o"
    return True


def kernel_artifact_fingerprints() -> dict[str, str]:
    """Hash stable artifacts without consulting filesystem timestamps."""

    root = Path(KERNEL_CACHE_ROOT)
    if not root.exists():
        return {}
    fingerprints = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if not _is_stable_cache_artifact(relative_path):
            continue
        fingerprints[relative_path] = f"{path.stat().st_size}:{_blake3_file(path)}"
    return fingerprints


def _blake3_file(path: Path) -> str:
    digest = blake3()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _container_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or f"{socket.gethostname()}:{os.getpid()}"


def _marker_path(workload_name: str) -> Path:
    if not workload_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in workload_name
    ):
        raise ValueError(
            "workload_name must contain only lowercase letters, digits, - or _"
        )
    return Path(KERNEL_CACHE_ROOT) / "_resin" / f"{workload_name}.json"


def _run_workload(workload: Callable[[], dict[str, object]]):
    start = time.perf_counter()
    result = workload()
    return result, (time.perf_counter() - start) * 1000


def _cache_report(
    *,
    mode: str,
    before: dict[str, int],
    after: dict[str, int],
    elapsed_ms: float,
) -> dict[str, Any]:
    changed = changed_cache_files(before, after)
    artifact_cache = _KERNEL_ARTIFACT_CACHE
    return {
        "mode": mode,
        "volume": KERNEL_CACHE_VOLUME_NAME,
        "mount": KERNEL_CACHE_MOUNT,
        "root": KERNEL_CACHE_ROOT,
        "container": _container_id(),
        "workload_elapsed_ms": elapsed_ms,
        "files_before": len(before),
        "files_after": len(after),
        "namespaces_before": kernel_cache_summary(before),
        "namespaces_after": kernel_cache_summary(after),
        "new_or_resized_count": len(changed),
        "new_or_resized_files": changed[:100],
        "flashinfer_hash_cache_hits": artifact_cache.hits if artifact_cache else 0,
        "flashinfer_hash_cache_misses": artifact_cache.misses if artifact_cache else 0,
    }


def _reload_and_prepare(volume) -> None:
    # The image installs cache environment variables before importing GPU
    # packages; this prepares the mounted directories before any runtime JIT.
    global _CACHE_PREPARED
    if not _CACHE_PREPARED:
        volume.reload()
        ensure_kernel_cache_dirs()
        _CACHE_PREPARED = True


def install_flashinfer_hash_cache() -> None:
    """Use a content-validated JIT library without consulting Ninja on hits.

    Modal Volume v2 preserves bytes but not nanosecond mtimes, while Ninja uses
    mtimes in its dependency database. FlashInfer 0.6.15 also invokes Ninja
    unconditionally before loading a runtime-built library. This wrapper stores
    a BLAKE3 sidecar after a normal miss and loads the library directly only
    when the exact ABI, module name, size, and content hash still match.
    """

    from flashinfer.jit import core

    from resin.kernel_cache import ArtifactCache, wrap_flashinfer_build_and_load

    global _KERNEL_ARTIFACT_CACHE
    if _KERNEL_ARTIFACT_CACHE is None:
        _KERNEL_ARTIFACT_CACHE = ArtifactCache(
            KERNEL_CACHE_MOUNT,
            KERNEL_CACHE_ABI,
        )
        if _KERNEL_ARTIFACT_CACHE.root != Path(KERNEL_CACHE_ROOT).resolve(strict=False):
            raise RuntimeError(
                "artifact cache root disagrees with deployment namespace"
            )

    core.JitSpec.build_and_load = wrap_flashinfer_build_and_load(
        core.JitSpec.build_and_load,
        _KERNEL_ARTIFACT_CACHE,
        durable_artifact_resolver=_flashinfer_durable_artifact,
    )


@contextmanager
def flashinfer_autotune_context(
    path: Path,
    *,
    tune_missing: bool,
    tuning_buckets: tuple[int, ...],
) -> Iterator[bool | None]:
    """Load saved tactics and scope FlashInfer autotuning after cache setup."""

    # This import must remain after ``install_flashinfer_hash_cache`` so a
    # runtime JIT cannot observe FlashInfer's unwrapped build entrypoint.
    import flashinfer
    from flashinfer.autotuner import AutoTuner

    cache_loaded = None
    if path.exists():
        cache_loaded = bool(AutoTuner.get().load_configs(str(path)))
        if not cache_loaded:
            raise RuntimeError(f"FlashInfer rejected its saved tactics: {path}")
    elif not tune_missing:
        raise RuntimeError("saved FlashInfer tactics are absent; prewarm first")
    with flashinfer.autotune(
        tune_missing,
        cache=str(path) if tune_missing else None,
        tuning_buckets=tuning_buckets,
        round_up=True,
    ):
        yield cache_loaded


def _flashinfer_durable_artifact(spec) -> Path:
    """Map one scratch-built FlashInfer library into persistent storage."""

    workspace = Path(
        os.environ.get(
            "FLASHINFER_WORKSPACE_BASE",
            KERNEL_CACHE_ENV["FLASHINFER_WORKSPACE_BASE"],
        )
    ).resolve(strict=False)
    source = Path(spec.jit_library_path).resolve(strict=False)
    try:
        relative = source.relative_to(workspace)
    except ValueError as error:
        raise RuntimeError(
            f"FlashInfer JIT artifact escaped its local workspace: {source}"
        ) from error
    return Path("flashinfer") / relative


def run_cached_workload(
    volume, workload: Callable[[], dict[str, object]]
) -> dict[str, object]:
    """Run a production-style workload, persisting any unseen specializations."""

    _reload_and_prepare(volume)
    before = kernel_cache_manifest()
    try:
        result, elapsed_ms = _run_workload(workload)
    finally:
        # Preserve successfully compiled kernels even if a later correctness
        # assertion in the workload raises.
        volume.commit()
    after = kernel_cache_manifest()
    result["kernel_cache"] = _cache_report(
        mode="run", before=before, after=after, elapsed_ms=elapsed_ms
    )
    return result


def run_cache_prewarm(
    volume,
    workload_name: str,
    workload: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Compile a manifest once and commit it with an auditable marker."""

    _reload_and_prepare(volume)
    before = kernel_cache_manifest()
    try:
        result, elapsed_ms = _run_workload(workload)
        after = kernel_cache_manifest()
        artifacts = kernel_artifact_fingerprints()
        marker = _marker_path(workload_name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema": KERNEL_CACHE_MARKER_SCHEMA,
                    "digest": "blake3",
                    "workload": workload_name,
                    "container": _container_id(),
                    "elapsed_ms": elapsed_ms,
                    "manifest": after,
                    "artifacts": artifacts,
                },
                sort_keys=True,
            )
        )
    finally:
        volume.commit()
    result["kernel_cache"] = _cache_report(
        mode="prewarm", before=before, after=after, elapsed_ms=elapsed_ms
    )
    result["kernel_cache"]["marker"] = marker.relative_to(KERNEL_CACHE_ROOT).as_posix()
    result["kernel_cache"]["stable_artifact_count"] = len(artifacts)
    return result


def run_cache_reuse_probe(
    volume,
    workload_name: str,
    workload: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Prove the same workload reuses artifacts in a fresh Modal container."""

    _reload_and_prepare(volume)
    marker_path = _marker_path(workload_name)
    if not marker_path.exists():
        raise RuntimeError(
            f"no prewarm marker for {workload_name!r}; run the prewarm function first"
        )
    marker = json.loads(marker_path.read_text())
    if not isinstance(marker, dict) or (
        marker.get("schema") != KERNEL_CACHE_MARKER_SCHEMA
        or marker.get("digest") != "blake3"
    ):
        raise RuntimeError(
            f"incompatible prewarm marker for {workload_name!r}; run prewarm again"
        )
    before = kernel_cache_manifest()
    artifact_contents = kernel_artifact_fingerprints()
    missing_artifacts = sorted(
        path
        for path, fingerprint in marker["artifacts"].items()
        if artifact_contents.get(path) != fingerprint
    )
    if missing_artifacts:
        raise AssertionError(
            "prewarmed stable artifacts are missing or changed: "
            f"{missing_artifacts[:20]}"
        )
    artifacts_before = kernel_artifact_fingerprints()

    try:
        result, elapsed_ms = _run_workload(workload)
    finally:
        volume.commit()
    after = kernel_cache_manifest()
    artifacts_after = kernel_artifact_fingerprints()
    artifact_changes = sorted(
        path
        for path in artifacts_before.keys() | artifacts_after.keys()
        if artifacts_before.get(path) != artifacts_after.get(path)
    )
    if artifact_changes:
        raise AssertionError(
            f"the reuse probe changed stable cache artifacts: {artifact_changes[:20]}"
        )

    result["kernel_cache"] = _cache_report(
        mode="reuse", before=before, after=after, elapsed_ms=elapsed_ms
    )
    result["kernel_cache"].update(
        {
            "cache_hit": True,
            "stable_artifact_count": len(artifacts_after),
            "stable_artifact_changes": artifact_changes,
            "prewarm_container": marker["container"],
            "prewarm_workload_elapsed_ms": marker["elapsed_ms"],
            "cross_container_speedup": marker["elapsed_ms"] / elapsed_ms,
        }
    )
    if result["kernel_cache"]["container"] == marker["container"]:
        raise AssertionError("reuse probe ran in the prewarm container")
    return result
