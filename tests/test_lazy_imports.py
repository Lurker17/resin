from __future__ import annotations

import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def test_import_resin_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; import resin; "
            "print(int('torch' in sys.modules), int('flashinfer' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        self.assertEqual(result.stdout.strip(), "0 0")


if __name__ == "__main__":
    unittest.main()
