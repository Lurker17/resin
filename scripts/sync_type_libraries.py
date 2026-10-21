"""Expose locked Linux-library sources to ty without installing or importing them."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import re
import shutil
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
# These upstream packages add import roots using executable .pth/setup code.
# Relocate their sources instead of running that code on the development host.
IMPORT_ROOTS = (
    PurePosixPath("nvidia_cutlass_dsl/dsl_packages"),
    PurePosixPath("tilelang/3rdparty/tvm/python"),
)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def linux_packages(lock):
    """Follow workspace Linux dependencies, using installed ordinary dependencies."""

    installed = {
        normalize(distribution.metadata["Name"])
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    roots = {
        dependency["name"]
        for package in lock["package"]
        if package.get("source", {}).keys() & {"editable", "virtual"}
        for dependency in package.get("dependencies", [])
        if "sys_platform == 'linux'" in dependency.get("marker", "")
    }
    if not roots:
        raise ValueError("uv.lock has no workspace Linux dependencies")
    pending = [{"name": name} for name in sorted(roots)]
    packages = {}
    while pending:
        dependency = pending.pop()
        name = dependency["name"]
        if name in packages or (name not in roots and name in installed):
            continue
        matches = [
            package
            for package in lock["package"]
            if package["name"] == name
            and (
                "version" not in dependency
                or package["version"] == dependency["version"]
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one locked version of {name}")
        package = matches[0]
        packages[name] = package
        pending.extend(package.get("dependencies", []))
    return [packages[name] for name in sorted(packages)]


def wheel_bytes(package, cache: Path) -> bytes:
    # Native wheels also contain Python wrappers and .pyi files. Their ABI does
    # not need to match the host because we never extract or load the binaries.
    wheels = [
        wheel
        for wheel in package.get("wheels", [])
        if urlsplit(wheel["url"]).path.endswith("-none-any.whl")
        or "linux" in urlsplit(wheel["url"]).path
    ]
    if not wheels:
        raise ValueError(f"no universal or Linux wheel available for {package['name']}")
    wheel = min(
        wheels,
        key=lambda item: (
            not item["url"].endswith("-none-any.whl"),
            item.get("size", 0),
            item["url"],
        ),
    )
    algorithm, digest = wheel["hash"].split(":", 1)
    if (
        algorithm != "sha256"
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid wheel hash for {package['name']}")
    cached = cache / f"{digest}.whl"
    if cached.exists():
        data = cached.read_bytes()
    else:
        print(f"Downloading {package['name']}=={package['version']}", flush=True)
        data = download(wheel["url"])
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"wheel checksum mismatch for {package['name']}")
    if not cached.exists():
        temporary = cached.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(cached)
    return data


def extract(data: bytes, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe wheel path: {member.filename}")
            if member.is_dir() or (
                path.suffix not in {".py", ".pyi"} and path.name != "py.typed"
            ):
                continue
            if len(path.parts) > 2 and path.parts[0].endswith(".data"):
                if path.parts[1] not in {"purelib", "platlib"}:
                    continue
                path = PurePosixPath(*path.parts[2:])
            for prefix in IMPORT_ROOTS:
                if path.is_relative_to(prefix):
                    path = path.relative_to(prefix)
                    break
            first = path.parts[0]
            # Build-system helpers are bundled by several unrelated wheels.
            # They are not library interfaces and would shadow one another.
            if len(path.parts) == 1 and path.name in {"build_backend.py", "setup.py"}:
                continue
            if not (
                first.isidentifier()
                or (len(path.parts) == 1 and path.stem.isidentifier())
            ):
                continue
            output = destination / path
            contents = archive.read(member)
            if output.exists() and output.read_bytes() != contents:
                raise ValueError(f"conflicting type-library source: {path}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(contents)
            count += 1
    return count


def sync(root: Path = ROOT) -> Path:
    """Refresh all Linux interfaces and local overlays from uv.lock.

    Verified wheels are cached for offline reuse. Namespace packages are merged
    before the generated source tree is replaced, so failures preserve the old
    interfaces. Nothing is installed, imported, or added to sys.path.
    """

    lock = tomllib.loads((root / "uv.lock").read_text())
    directory = root / ".type-libs"
    target, cache = directory / "sources", directory / ".wheels"
    cache.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=directory) as name:
        stage = Path(name)
        source = stage / "sources"
        source.mkdir()
        for package in linux_packages(lock):
            count = extract(wheel_bytes(package, cache), source)
            reports.append(
                f"{package['name']}=={package['version']}: {count} Python/type files"
            )
        overlays = root / "type_stubs"
        for overlay in sorted(overlays.rglob("*.pyi")):
            output = source / overlay.relative_to(overlays)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(overlay, output)
        # Keep the configured search directory present on a fresh checkout.
        if (target / ".gitignore").exists():
            shutil.copyfile(target / ".gitignore", source / ".gitignore")
        previous = stage / "previous"
        if target.exists():
            target.rename(previous)
        try:
            source.rename(target)
        except BaseException:
            if previous.exists():
                previous.rename(target)
            raise
    print("\n".join(reports))
    return target


if __name__ == "__main__":
    sync()
