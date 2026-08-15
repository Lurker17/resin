"""Measure cold Python import boundaries in fresh processes."""

from __future__ import annotations

import subprocess
import sys


def measure(statement: str) -> float:
    code = (
        "import time; start=time.perf_counter(); "
        f"{statement}; print((time.perf_counter()-start)*1000)"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    return float(output.strip())


print(f"import resin: {measure('import resin'):.2f} ms")
print(
    "import PyTorch backend module: "
    f"{measure('import resin_models_torch.lfm2.reference'):.2f} ms"
)
print(
    "import managed CUDA model backend: "
    f"{measure('import resin_models_cuda.lfm2.cuda'):.2f} ms"
)
