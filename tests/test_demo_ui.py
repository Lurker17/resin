"""Exercise the shipped demo's request lifecycle without a browser dependency."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="requires Node.js")
def test_demo_request_error_recovery():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("demo_ui.test.cjs"))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
