#!/usr/bin/env python3
"""Build Resin's native extension and refresh development type information."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, "scripts/sync_type_libraries.py"], cwd=ROOT, check=True
    )
    subprocess.run(
        ["uv", "run", "maturin", "develop", "--generate-stubs"], cwd=ROOT, check=True
    )
    subprocess.run([sys.executable, "scripts/normalize_stubs.py"], cwd=ROOT, check=True)
    subprocess.run(
        ["uv", "run", "ruff", "format", "resin/_resin.pyi"], cwd=ROOT, check=True
    )


if __name__ == "__main__":
    main()
