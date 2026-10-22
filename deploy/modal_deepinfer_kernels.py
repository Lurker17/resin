"""Validate the pinned upstream kernels and Resin adapters on B300s."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
DEEPINFER_REVISION = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
# Exact gitlinks in that DeepInfer revision, not independently selected releases.
KERNEL_REVISIONS = {
    "DeepGEMM": "1fc9c66975ee322d628b3288534ca0e9ce12a64d",
    "FlashMLA": "2e5429fc5653bab6e081f09477126f731882a6a9",
    "DeepEP": "93eb6eb238127e96c6d7a4a625a6dad158348509",
    "DeepSelect": "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18",
}
PROBE_PYTHON = "/opt/resin-deps/.venv/bin/python"
CACHE = modal.Volume.from_name("resin-deepinfer-kernels-v1", create_if_missing=True)

image = build_modal_image(
    ROOT, env={"DG_JIT_CACHE_DIR": "/kernel-cache/deep_gemm"}
).run_commands(
    'python -c "import deep_gemm; '
    "assert deep_gemm.__version__ == '2.8.2'; "
    "assert all(callable(getattr(deep_gemm, name)) for name in "
    "('bf16_mega_gate', 'mega_mhc', 'SymmBuffer', 'fp8_fp4_mega_moe'))\""
)
for helper in (
    "deepinfer_kernel_probe",
    "mega_moe_validation",
    "bench",
    "deepseek_v41_validation",
    "v41_engine_smoke",
):
    image = image.add_local_file(
        ROOT / f"deploy/{helper}.py", f"/root/deploy/{helper}.py", copy=True
    )
app = modal.App("resin-deepinfer-kernels", image=image)


@app.function(
    gpu="B300",
    cpu=32,
    memory=131072,
    timeout=30 * 60,
    volumes={"/kernel-cache": CACHE},
    single_use_containers=True,
)
def probe(mode: str = "kernels", tokens_per_rank: int = 8) -> dict:
    if mode not in {"kernels", "mega-moe", "engine-smoke"}:
        raise ValueError("mode must be kernels, mega-moe or engine-smoke")
    if tokens_per_rank < 1:
        raise ValueError("tokens_per_rank must be positive")
    try:
        # Run the same locked environment as the model deployment.
        with tempfile.TemporaryDirectory(prefix="deepinfer-kernels-") as directory:
            output = Path(directory) / "result.json"
            command = [PROBE_PYTHON, "-m"]
            if mode == "engine-smoke":
                command += ["deploy.v41_engine_smoke", str(output)]
            else:
                command += [
                    "deploy.deepinfer_kernel_probe",
                    str(output),
                    mode,
                    "--tokens-per-rank",
                    str(tokens_per_rank),
                ]
            subprocess.run(
                command,
                cwd="/root",
                check=True,
            )
            return {
                "deepinfer_revision": DEEPINFER_REVISION,
                "kernel_revisions": KERNEL_REVISIONS,
                **json.loads(output.read_text()),
            }
    finally:
        CACHE.commit()


@app.local_entrypoint()
def main(
    output: str = "/tmp/resin-deepinfer-kernels.json",
    mode: str = "kernels",
    tokens_per_rank: int = 8,
) -> None:
    result = probe.with_options(gpu="B300" if mode == "kernels" else "B300:2").remote(
        mode, tokens_per_rank
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
