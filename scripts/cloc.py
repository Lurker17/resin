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


def python_package(path: Path) -> dict[str, object]:
    groups: dict[str, object] = {"_self": source_files(path, "*.py", recursive=False)}
    for directory in sorted(path.iterdir()):
        if directory.is_dir() and not directory.name.startswith("__"):
            files = source_files(directory, "*.py")
            if files:
                groups[directory.name.replace("_", "-")] = files
    return groups


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
    if not paths:
        return {"files": 0, "code": 0}
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


def count_tree(groups: dict[str, object], language: str) -> dict[str, object]:
    own = count(groups.get("_self", []), language)
    children = {}
    for name, group in groups.items():
        if name == "_self":
            continue
        children[name] = (
            count(group, language)
            if isinstance(group, list)
            else count_tree(group, language)
        )
    return {
        "files": own["files"] + sum(child["files"] for child in children.values()),
        "code": own["code"] + sum(child["code"] for child in children.values()),
        "children": children,
    }


def report() -> dict[str, dict[str, object]]:
    groups = {
        "rust": {
            "resin-core": {
                "core": source_files(
                    ROOT / "crates/resin-core/src", "*.rs", recursive=False
                ),
                "resources": source_files(
                    ROOT / "crates/resin-core/src/resources", "*.rs"
                ),
            },
            "resin-http": source_files(ROOT / "crates/resin-http/src", "*.rs"),
            "resin-py": source_files(ROOT / "crates/resin-py/src", "*.rs"),
        },
        "python": {
            "resin": {
                "_self": source_files(ROOT / "resin", "*.py", recursive=False),
                "backend": source_files(ROOT / "resin/backend", "*.py"),
            },
            "resin-models-cuda": python_package(ROOT / "models/cuda/resin_models_cuda"),
            "resin-models-torch": python_package(
                ROOT / "models/torch/resin_models_torch"
            ),
        },
    }
    return {
        section: count_tree(section_groups, section.title())
        for section, section_groups in groups.items()
    }


def tree_rows(
    children: dict[str, dict[str, object]], prefix: str = ""
) -> list[tuple[str, dict[str, object]]]:
    rows = []
    items = list(children.items())
    for index, (name, node) in enumerate(items):
        last = index == len(items) - 1
        rows.append((f"{prefix}{'└──' if last else '├──'} {name}", node))
        nested = node.get("children")
        if nested:
            rows.extend(tree_rows(nested, prefix + ("    " if last else "│   ")))
    return rows


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
    for section, root in counts.items():
        print(f"\n{section.title()}")
        children = dict(root["children"])
        children["total"] = {"files": root["files"], "code": root["code"]}
        rows = tree_rows(children)
        width = max(len(label) for label, _ in rows)
        for label, row in rows:
            noun = "file" if row["files"] == 1 else "files"
            print(f"  {label:<{width}}  {row['code']:>6,}  ({row['files']} {noun})")


if __name__ == "__main__":
    main()
