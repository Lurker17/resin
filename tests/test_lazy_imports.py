from __future__ import annotations

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


class ImportTests(unittest.TestCase):
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
        self.assertEqual(result.stdout.strip(), "0 0 0")

    def test_accelerator_dependencies_are_optional(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
        self.assertEqual(project["dependencies"], ["blake3>=1.0,<2"])
        cuda = project["optional-dependencies"]["cuda"]
        self.assertTrue(any(item.startswith("torch") for item in cuda))
        self.assertTrue(any(item.startswith("flashinfer-python") for item in cuda))
        self.assertTrue(any(item.startswith("flash-attn-4") for item in cuda))
        self.assertTrue(any(item.startswith("nvidia-cutlass-dsl") for item in cuda))


if __name__ == "__main__":
    unittest.main()
