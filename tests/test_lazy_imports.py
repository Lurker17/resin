from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


class TestImport:
    def test_import_resin_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; import resin; import resin.backend; "
            "import resin.backend.cuda; import resin.models; "
            "import resin.models.gpt_oss; import resin.models.lfm2; "
            "print(*(int(name in sys.modules) for name in "
            "('torch', 'flashinfer', 'flash_attn')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0 0 0"

    def test_accelerator_dependencies_are_optional(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
        assert project["dependencies"] == ["blake3>=1.0,<2"]
        cuda = project["optional-dependencies"]["cuda"]
        assert any(item.startswith("torch") for item in cuda)
        assert any(item.startswith("flashinfer-python") for item in cuda)
        assert any(item.startswith("quack-kernels==0.6.4") for item in cuda)
        assert any(item.startswith("nvidia-cutlass-dsl") for item in cuda)
        assert any(item.startswith("flash-attn-4") for item in cuda)
