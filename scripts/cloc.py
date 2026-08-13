"""Count production Rust and Python code with Tokei."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_files(path: Path, pattern: str, *, recursive: bool = True) -> list[Path]:
    files = path.rglob(pattern) if recursive else path.glob(pattern)
    return sorted(file for file in files if "tests" not in file.parts)


def rust_item_end(source: str, start: int) -> int:
    """Find the end of the item following a cfg(test) attribute."""

    index = start
    depth = 0
    block_comments = 0
    while index < len(source):
        if block_comments:
            if source.startswith("/*", index):
                block_comments += 1
                index += 2
            elif source.startswith("*/", index):
                block_comments -= 1
                index += 2
            else:
                index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            block_comments = 1
            index += 2
            continue

        raw = re.match(r'(?:br|r)(?P<marks>#{0,255})"', source[index:])
        if raw:
            terminator = '"' + raw.group("marks")
            index += raw.end()
            end = source.find(terminator, index)
            index = len(source) if end < 0 else end + len(terminator)
            continue
        if source[index] == '"':
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            continue

        character = re.match(r"'(?:\\.|[^\\'])'", source[index:])
        if character:
            index += character.end()
            continue
        token = source[index]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        elif depth == 0 and token in ";,":
            return index + 1
        index += 1
    raise ValueError("unterminated #[cfg(test)] item")


def strip_rust_tests(source: str) -> str:
    attribute = re.compile(r"(?m)^[ \t]*#\[cfg\(test\)\][ \t]*\n")
    while match := attribute.search(source):
        end = rust_item_end(source, match.end())
        source = (
            source[: match.start()]
            + "\n" * source[match.start() : end].count("\n")
            + source[end:]
        )
    return source


def rust_without_tests(paths: list[Path], destination: Path) -> list[Path]:
    stripped = []
    for source in paths:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(strip_rust_tests(source.read_text()))
        stripped.append(target)
    return stripped


def count(paths: list[Path], language: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as temporary:
        inputs = (
            rust_without_tests(paths, Path(temporary)) if language == "Rust" else paths
        )
        result = subprocess.run(
            ["tokei", "--output", "json", *map(str, inputs)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    return {"files": len(paths), "code": json.loads(result.stdout)[language]["code"]}


def report() -> dict[str, dict[str, dict[str, int]]]:
    groups = {
        "rust": {
            "resin-core": source_files(ROOT / "crates/resin-core/src", "*.rs"),
            "resin": source_files(ROOT / "src", "*.rs", recursive=False),
        },
        "python": {
            "top-level": source_files(ROOT / "src/resin", "*.py", recursive=False),
            "backend": source_files(ROOT / "src/resin/backend", "*.py"),
            "models": source_files(ROOT / "src/resin/models", "*.py"),
        },
    }
    output = {}
    for section, paths_by_group in groups.items():
        language = section.title()
        rows = {name: count(paths, language) for name, paths in paths_by_group.items()}
        rows["total"] = {
            "files": sum(row["files"] for row in rows.values()),
            "code": sum(row["code"] for row in rows.values()),
        }
        output[section] = rows
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    if shutil.which("tokei") is None:
        parser.error("tokei is required")

    counts = report()
    if args.json:
        print(json.dumps(counts, indent=2))
        return
    print("Production code lines (Tokei; comments, blanks, and test code excluded)")
    for section, rows in counts.items():
        print(f"\n{section.title()}")
        width = max(map(len, rows))
        for name, row in rows.items():
            print(f"  {name:<{width}}  {row['code']:>6,}  ({row['files']} files)")


if __name__ == "__main__":
    main()
