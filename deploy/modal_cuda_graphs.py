"""Validate Resin's shared prefill graph runtime on a single Blackwell GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "/opt/resin/validation/test_cuda_prefill_graphs.py"
TENSOR_TEST_PATH = "/opt/resin/validation/test_cuda_tensor_graphs.py"
image = (
    build_modal_image(ROOT)
    .uv_pip_install("pytest>=8,<9")
    .add_local_file(ROOT / "tests/test_cuda_prefill_graphs.py", TEST_PATH, copy=True)
    .add_local_file(
        ROOT / "tests/test_cuda_tensor_graphs.py", TENSOR_TEST_PATH, copy=True
    )
)
app = modal.App("resin-cuda-graphs", image=image)


@app.function(gpu="B300:1", cpu=4, memory=8 * 1024, timeout=10 * 60)
def validate() -> dict[str, str]:
    """Run CPU policy checks and CUDA capture/replay checks without model weights."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("CUDA graph validation requires an available Blackwell GPU")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-q",
            TEST_PATH,
            TENSOR_TEST_PATH,
        ],
        # Use the installed Linux wheel, without development pythonpath settings.
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=9 * 60,
        check=False,
    )
    print(result.stdout, flush=True)
    if result.returncode or "skipped" in result.stdout:
        raise RuntimeError("CUDA graph validation did not pass every CPU and GPU case")
    return {"gpu": torch.cuda.get_device_name(), "tests": result.stdout.strip()}
