"""Run owner-sparse expert transport checks on real NVLink GPUs."""

import os
import subprocess
import sys
from pathlib import Path

import modal

from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
    prepare_kernel_cache,
)
from deploy.modal_image import build_modal_image

ROOT = Path(
    os.environ.get("RESIN_MODAL_SOURCE_ROOT", Path(__file__).resolve().parents[1])
)
image = (
    build_modal_image(ROOT)
    .env(KERNEL_CACHE_ENV)
    .uv_pip_install("pytest>=8,<9")
    .add_local_file(
        ROOT / "tests/test_expert_parallel_gpu.py",
        "/opt/resin/validation/test_expert_parallel_gpu.py",
        copy=True,
    )
)
cache = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=KERNEL_CACHE_VOLUME_VERSION,
)
app = modal.App("resin-sparse-expert-transport")


@app.function(
    image=image,
    gpu="H100:4",
    cpu=16,
    memory=64 * 1024,
    timeout=30 * 60,
    volumes={KERNEL_CACHE_MOUNT: cache},
    single_use_containers=True,
)
def validate(selection: str = "", world_size: int = 4):
    prepare_kernel_cache(cache)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-q",
        "-s",
        "-x",
        "/opt/resin/validation/test_expert_parallel_gpu.py",
    ]
    if selection:
        command.extend(("-k", selection))
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "RESIN_EP_TEST_WORLD_SIZE": str(world_size),
            "PYTHONUNBUFFERED": "1",
        },
        timeout=20 * 60,
        check=True,
    )
    cache.commit()
    return result.returncode


@app.local_entrypoint()
def main(selection: str = "", world_size: int = 4):
    if world_size not in (2, 4, 8):
        raise ValueError("validation requires two, four, or eight GPUs")
    validate.with_options(gpu=f"H100:{world_size}").remote(selection, world_size)
