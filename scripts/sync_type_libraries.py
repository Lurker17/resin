"""Expose locked Linux-library sources to ty without installing or importing them."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import re
import shutil
import stat
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

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


def git_bytes(package, cache: Path) -> tuple[bytes, str]:
    """Fetch an immutable source snapshot without running repository code."""

    source = urlsplit(package["source"]["git"])
    commit = source.fragment
    query = parse_qs(source.query)
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or query not in (
        {},
        {"rev": [commit]},
    ):
        raise ValueError(f"Git source must pin a full commit for {package['name']}")
    parts = source.path.removesuffix(".git").strip("/").split("/")
    if (
        source.scheme != "https"
        or source.netloc != "github.com"
        or len(parts) != 2
        or any(
            part in {".", ".."} or not re.fullmatch(r"[\w.-]+", part) for part in parts
        )
    ):
        raise ValueError(f"unsupported Git source for {package['name']}")
    url = f"https://codeload.github.com/{'/'.join(parts)}/zip/{commit}"
    identity = hashlib.sha256(url.encode()).hexdigest()
    cached, checksum = cache / f"git-{identity}.zip", cache / f"git-{identity}.sha256"
    if cached.exists():
        data = cached.read_bytes()
        if (
            not checksum.exists()
            or hashlib.sha256(data).hexdigest() != checksum.read_text().strip()
        ):
            raise ValueError(f"Git snapshot checksum mismatch for {package['name']}")
    else:
        print(f"Downloading {package['name']}@{commit}", flush=True)
        data = download(url)
        temporary = cached.with_suffix(".tmp")
        temporary.write_bytes(data)
        checksum.write_text(hashlib.sha256(data).hexdigest() + "\n")
        temporary.replace(cached)
    return data, commit


def extract_git(data: bytes, commit: str, destination: Path) -> int:
    """Expose package sources, excluding repository tooling and native code."""

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = []
        roots = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe Git snapshot path: {member.filename}")
            roots.add(path.parts[0])
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"unsafe Git snapshot symlink: {member.filename}")
            if not member.is_dir() and len(path.parts) > 1:
                members.append((member, PurePosixPath(*path.parts[1:])))
        if len(roots) != 1 or not next(iter(roots)).endswith(f"-{commit}"):
            raise ValueError("Git snapshot does not match its pinned commit")
        # Support ordinary repository packages and the conventional src/ layout.
        packages = {
            path.parent
            for _, path in members
            if path.name in {"__init__.py", "__init__.pyi", "py.typed"}
            and len(path.parts) == (3 if path.parts[0] == "src" else 2)
        }
        excluded = {"tests", "scripts", "examples", "benchmarks", "docs", "tools"}
        count = 0
        for member, path in members:
            if path.parts[0] == "src":
                if len(path.parts) == 1:
                    continue
                path = PurePosixPath(*path.parts[1:])
                root = PurePosixPath("src", path.parts[0])
            else:
                root = PurePosixPath(path.parts[0])
            if (
                root not in packages
                or not path.parts[0].isidentifier()
                or path.parts[0] in excluded
                or (path.suffix not in {".py", ".pyi"} and path.name != "py.typed")
            ):
                continue
            output = destination / path
            contents = archive.read(member)
            if output.exists() and output.read_bytes() != contents:
                raise ValueError(f"conflicting type-library source: {path}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(contents)
            count += 1
        if not count:
            raise ValueError("Git snapshot contains no importable Python packages")
        return count


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

    Wheels and pinned Git snapshots are cached for offline reuse. Packages merge
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
            if "git" in package.get("source", {}):
                data, commit = git_bytes(package, cache)
                count = extract_git(data, commit, source)
            else:
                count = extract(wheel_bytes(package, cache), source)
            reports.append(
                f"{package['name']}=={package['version']}: {count} Python/type files"
            )
        overlays = directory / "patches"
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
