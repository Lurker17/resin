from __future__ import annotations

import hashlib
import io
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from scripts import sync_type_libraries
from scripts.sync_type_libraries import sync

COMMIT = "1fc9c66975ee322d628b3288534ca0e9ce12a64d"
GIT_SOURCE = f"https://github.com/deepseek-ai/DeepGEMM?rev={COMMIT}#{COMMIT}"


def _wheel(
    root: Path,
    entries: dict[str, str],
    name="flashinfer-python",
    version="1.0",
    tag="py3-none-any",
) -> Path:
    wheel = root / f"{name.replace('-', '_')}-{version}-{tag}.whl"
    with ZipFile(wheel, "w") as archive:
        for path, contents in entries.items():
            archive.writestr(path, contents)
    return wheel


def _lock(
    root: Path,
    wheels: dict[str, Path | None],
    *,
    roots=("flashinfer-python",),
    dependencies=None,
    checksum=None,
):
    parts = [
        '[[package]]\nname = "workspace"\nversion = "1.0"\nsource = { virtual = "." }\ndependencies = ['
    ]
    parts.extend(
        f'{{ name = "{name}", marker = "sys_platform == \'linux\'" }},'
        for name in roots
    )
    parts.append("]\n")
    for name, wheel in wheels.items():
        parts.append(f'[[package]]\nname = "{name}"\nversion = "1.0"\n')
        refs = (dependencies or {}).get(name, ())
        parts.append(
            "dependencies = ["
            + ",".join(f'{{ name = "{ref}" }}' for ref in refs)
            + "]\n"
        )
        if wheel is not None:
            digest = checksum or hashlib.sha256(wheel.read_bytes()).hexdigest()
            parts.append(
                f'wheels = [{{ url = "{wheel.as_uri()}", hash = "sha256:{digest}", size = {wheel.stat().st_size} }}]\n'
            )
    (root / "uv.lock").write_text("\n".join(parts))


def _overlay(root: Path, contents: str) -> None:
    overlay = root / "type_stubs" / "flashinfer" / "api_logging.pyi"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(contents)


def _existing_library(root: Path) -> Path:
    package = root / ".type-libs" / "sources" / "flashinfer"
    package.mkdir(parents=True)
    (package / "previous.py").write_text("previous = True\n")
    return package


def _git_lock(root: Path, source: str = GIT_SOURCE) -> None:
    _lock(root, {"deep-gemm": None}, roots=("deep-gemm",))
    with (root / "uv.lock").open("a") as lock:
        lock.write(f'\nsource = {{ git = "{source}" }}\n')


def _snapshot(entries: dict[str, str], *, root=f"DeepGEMM-{COMMIT}") -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for path, contents in entries.items():
            archive.writestr(f"{root}/{path}", contents)
    return buffer.getvalue()


def test_git_snapshot_exposes_only_packages_without_running_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _git_lock(tmp_path)
    snapshot = _snapshot(
        {
            "deep_gemm/__init__.py": "raise AssertionError('must not import CUDA')\n",
            "deep_gemm/mega/mega_moe.py": "def mega_moe(value: int) -> int: return value\n",
            "deep_gemm/_C.pyi": "def bf16_mega_gate() -> None: ...\n",
            "deep_gemm/py.typed": "",
            "deep_gemm/_C.so": "native binary",
            "src/another/__init__.py": "value = 1\n",
            "tests/__init__.py": "raise AssertionError('test helper')\n",
            "scripts/__init__.py": "raise AssertionError('script helper')\n",
            "examples/__init__.py": "raise AssertionError('example helper')\n",
            "setup.py": "raise AssertionError('must not run setup')\n",
            "build_backend.py": "raise AssertionError('must not run build')\n",
        }
    )
    urls = []

    def download(url: str) -> bytes:
        urls.append(url)
        return snapshot

    monkeypatch.setattr(sync_type_libraries, "download", download)
    target = sync(tmp_path)
    assert urls == [f"https://codeload.github.com/deepseek-ai/DeepGEMM/zip/{COMMIT}"]
    assert {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    } == {
        "deep_gemm/__init__.py",
        "deep_gemm/mega/mega_moe.py",
        "deep_gemm/_C.pyi",
        "deep_gemm/py.typed",
        "another/__init__.py",
    }
    (target / "deep_gemm/stale.py").write_text("stale = True\n")

    def offline(url: str) -> bytes:
        raise AssertionError(f"offline sync must use its cached snapshot: {url}")

    monkeypatch.setattr(sync_type_libraries, "download", offline)
    sync(tmp_path)
    assert not (target / "deep_gemm/stale.py").exists()
    assert (target / "deep_gemm/_C.pyi").exists()


@pytest.mark.parametrize(
    "source",
    [
        "https://github.com/deepseek-ai/DeepGEMM?rev=main#main",
        f"https://github.com/deepseek-ai/DeepGEMM?rev=main#{COMMIT}",
        f"https://github.com/deepseek-ai/DeepGEMM?rev={'0' * 40}#{COMMIT}",
        f"https://gitlab.com/deepseek-ai/DeepGEMM?rev={COMMIT}#{COMMIT}",
    ],
)
def test_git_requires_supported_immutable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
):
    package = _existing_library(tmp_path)
    _git_lock(tmp_path, source)

    def unexpected(url: str) -> bytes:
        raise AssertionError(f"invalid source must not be fetched: {url}")

    monkeypatch.setattr(sync_type_libraries, "download", unexpected)
    with pytest.raises(ValueError, match="(?i)Git source"):
        sync(tmp_path)
    assert (package / "previous.py").exists()


@pytest.mark.parametrize("unsafe_path", ["../../escape.py", "/escape.py"])
def test_git_unsafe_snapshot_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_path: str
):
    package = _existing_library(tmp_path)
    _git_lock(tmp_path)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(f"DeepGEMM-{COMMIT}/deep_gemm/__init__.py", "new = True\n")
        archive.writestr(unsafe_path, "escaped = True\n")
    monkeypatch.setattr(sync_type_libraries, "download", lambda url: buffer.getvalue())
    with pytest.raises(ValueError, match="unsafe Git snapshot"):
        sync(tmp_path)
    assert (package / "previous.py").exists()
    assert not (package.parent / "deep_gemm").exists()
    assert not list(tmp_path.rglob("escape.py"))


def test_git_rejects_symlinks_and_wrong_commit(tmp_path: Path):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        link = ZipInfo(f"DeepGEMM-{COMMIT}/deep_gemm/__init__.py")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../../outside.py")
    with pytest.raises(ValueError, match="symlink"):
        sync_type_libraries.extract_git(buffer.getvalue(), COMMIT, tmp_path)
    with pytest.raises(ValueError, match="pinned commit"):
        sync_type_libraries.extract_git(
            _snapshot({"deep_gemm/__init__.py": ""}, root="DeepGEMM-main"),
            COMMIT,
            tmp_path,
        )


def test_corrupted_git_cache_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _git_lock(tmp_path)
    snapshot = _snapshot({"deep_gemm/__init__.py": "old = True\n"})
    monkeypatch.setattr(sync_type_libraries, "download", lambda url: snapshot)
    package = sync(tmp_path) / "deep_gemm"
    cached = next((tmp_path / ".type-libs/.wheels").glob("git-*.zip"))
    cached.write_bytes(snapshot + b"corruption")
    with pytest.raises(ValueError, match="Git snapshot checksum mismatch"):
        sync(tmp_path)
    assert (package / "__init__.py").read_text() == "old = True\n"


def test_git_commit_change_replaces_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _git_lock(tmp_path)
    snapshots = {
        COMMIT: _snapshot({"deep_gemm/__init__.py": "old = True\n"}),
        "a" * 40: _snapshot(
            {"deep_gemm/__init__.py": "new = True\n", "deep_gemm/new.py": ""},
            root=f"DeepGEMM-{'a' * 40}",
        ),
    }
    monkeypatch.setattr(
        sync_type_libraries, "download", lambda url: snapshots[url.rsplit("/", 1)[1]]
    )
    package = sync(tmp_path) / "deep_gemm"
    _git_lock(tmp_path, GIT_SOURCE.replace(COMMIT, "a" * 40))
    sync(tmp_path)
    assert (package / "__init__.py").read_text() == "new = True\n"
    assert (package / "new.py").exists()


def test_sync_exposes_sources_and_overlay_without_importing(tmp_path: Path):
    wheel = _wheel(
        tmp_path,
        {
            "flashinfer/__init__.py": "raise AssertionError('must not import CUDA')\n",
            "build_backend.py": "raise AssertionError('build helper')\n",
            "setup.py": "raise AssertionError('setup helper')\n",
            "flashinfer/attention.py": "def attention(x: int) -> int: return x\n",
            "flashinfer/nested/types.pyi": "value: int\n",
            "flashinfer/py.typed": "",
            "flashinfer/kernel.so": "native binary",
            "flashinfer/data/kernel.cu": "CUDA source",
            "flashinfer_python-1.0.dist-info/METADATA": "Name: flashinfer-python\n",
        },
        tag="cp313-cp313-manylinux_2_28_aarch64",
    )
    _lock(tmp_path, {"flashinfer-python": wheel})
    _overlay(tmp_path, "def flashinfer_api() -> None: ...\n")
    target = sync(tmp_path)
    assert target == tmp_path / ".type-libs" / "sources"
    assert {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    } == {
        "flashinfer/__init__.py",
        "flashinfer/attention.py",
        "flashinfer/nested/types.pyi",
        "flashinfer/py.typed",
        "flashinfer/api_logging.pyi",
    }
    assert (
        target / "flashinfer/api_logging.pyi"
    ).read_text() == "def flashinfer_api() -> None: ...\n"


def test_bad_checksum_preserves_existing_sources(tmp_path: Path):
    package = _existing_library(tmp_path)
    wheel = _wheel(tmp_path, {"flashinfer/replacement.py": "new = True\n"})
    _lock(tmp_path, {"flashinfer-python": wheel}, checksum="0" * 64)
    with pytest.raises(ValueError, match="(?i)sha256|checksum|hash"):
        sync(tmp_path)
    assert (package / "previous.py").read_text() == "previous = True\n"
    assert not (package / "replacement.py").exists()


@pytest.mark.parametrize(
    "unsafe_name", ["flashinfer/../../escape.py", "/flashinfer/escape.py"]
)
def test_unsafe_archive_preserves_existing_sources(tmp_path: Path, unsafe_name: str):
    package = _existing_library(tmp_path)
    wheel = _wheel(
        tmp_path,
        {"flashinfer/replacement.py": "new = True\n", unsafe_name: "escaped = True\n"},
    )
    _lock(tmp_path, {"flashinfer-python": wheel})
    with pytest.raises(ValueError, match="(?i)unsafe|path"):
        sync(tmp_path)
    assert (package / "previous.py").read_text() == "previous = True\n"
    assert not (package / "replacement.py").exists()
    assert not list(tmp_path.rglob("escape.py"))


def test_cached_sync_refreshes_overlays_and_removes_stale_sources(tmp_path: Path):
    wheel = _wheel(tmp_path, {"flashinfer/old_api.py": "old = True\n"})
    _lock(tmp_path, {"flashinfer-python": wheel})
    _overlay(tmp_path, "old_annotation: int\n")
    package = sync(tmp_path) / "flashinfer"
    wheel.unlink()  # Subsequent sync must use the verified cached wheel.
    (package / "stale.py").write_text("stale = True\n")
    _overlay(tmp_path, "updated_annotation: str\n")
    sync(tmp_path)
    assert (package / "api_logging.pyi").read_text() == "updated_annotation: str\n"
    assert (package / "old_api.py").exists() and not (package / "stale.py").exists()
    wheel = _wheel(tmp_path, {"flashinfer/new_api.py": "new = True\n"}, version="2.0")
    _lock(tmp_path, {"flashinfer-python": wheel})
    sync(tmp_path)
    assert (package / "new_api.py").exists() and not (package / "old_api.py").exists()
    assert (package / "api_logging.pyi").read_text() == "updated_annotation: str\n"


def test_missing_wheel_preserves_existing_sources(tmp_path: Path):
    package = _existing_library(tmp_path)
    _lock(tmp_path, {"flashinfer-python": None})
    with pytest.raises(ValueError, match="(?i)wheel"):
        sync(tmp_path)
    assert (package / "previous.py").read_text() == "previous = True\n"


def test_namespace_merge_and_required_closure_include_installed_roots(tmp_path: Path):
    # pytest is installed, but explicit Linux roots must still be hydrated.
    wheels = {
        "pytest": _wheel(tmp_path, {"fla/layers/layer.py": "value = 1\n"}, "pytest"),
        "gpu-test-core": _wheel(
            tmp_path, {"fla/ops/kernel.py": "value = 2\n"}, "gpu-test-core"
        ),
        "gpu-test-meta": _wheel(tmp_path, {}, "gpu-test-meta"),
        "torch": None,  # Installed transitive dependencies need no wheel download.
    }
    _lock(
        tmp_path,
        wheels,
        roots=("pytest",),
        dependencies={"pytest": ("gpu-test-core", "gpu-test-meta", "torch")},
    )
    target = sync(tmp_path)
    assert (target / "fla/layers/layer.py").read_text() == "value = 1\n"
    assert (target / "fla/ops/kernel.py").read_text() == "value = 2\n"
    assert not (target / "fla/__init__.py").exists()
    assert not (target / "torch").exists()


def test_upstream_package_layouts_are_normalized(tmp_path: Path):
    entries = {
        "example-1.0.data/purelib/cuda/tile/__init__.py": "pure = True\n",
        "example-1.0.data/platlib/native/types.pyi": "value: int\n",
        "nvidia_cutlass_dsl/dsl_packages/cutlass/cute/__init__.py": "cute = True\n",
        "tilelang/3rdparty/tvm/python/tvm/__init__.py": "tvm = True\n",
    }
    _lock(tmp_path, {"flashinfer-python": _wheel(tmp_path, entries)})
    target = sync(tmp_path)
    for path, contents in {
        "cuda/tile/__init__.py": "pure = True\n",
        "native/types.pyi": "value: int\n",
        "cutlass/cute/__init__.py": "cute = True\n",
        "tvm/__init__.py": "tvm = True\n",
    }.items():
        assert (target / path).read_text() == contents
    assert not (target / "example-1.0.data").exists()


def test_real_decorator_overlay_preserves_direct_and_factory_signatures(tmp_path: Path):
    wheel = _wheel(
        tmp_path,
        {
            "flashinfer/api_logging.py": (
                "from collections.abc import Callable\n"
                "def flashinfer_api(func: Callable | None = None, *, trace=None) -> Callable:\n"
                "    return func if func is not None else lambda wrapped: wrapped\n"
            ),
            "flashinfer/api.py": (
                "from .api_logging import flashinfer_api\n"
                "@flashinfer_api\n"
                "def direct(value: int) -> int: return value\n"
                "@flashinfer_api(trace=object())\n"
                "def factory(value: str) -> str: return value\n"
            ),
        },
    )
    _lock(tmp_path, {"flashinfer-python": wheel})
    stub = Path(__file__).resolve().parents[1] / "type_stubs/flashinfer/api_logging.pyi"
    _overlay(tmp_path, stub.read_text())
    target = sync(tmp_path)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from typing import assert_type\n"
        "from flashinfer.api import direct, factory\n"
        "assert_type(direct(3), int)\n"
        "assert_type(factory('ok'), str)\n"
        "direct('wrong')\n"
        "factory(3)\n"
    )
    command = [sys.executable, "-m", "ty", "check"]
    command.extend(("--python", sys.prefix, "--extra-search-path", str(target)))
    command.extend(("--output-format", "concise", str(probe)))
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 1, diagnostics
    assert diagnostics.count("error[invalid-argument-type]") == 2, diagnostics
    assert diagnostics.count("error[") == 2, diagnostics
