"""Validate the prepared V4.1 forward operators on one real Blackwell GPU."""

import json
import subprocess
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
CACHE = modal.Volume.from_name("resin-deepinfer-kernels-v1", create_if_missing=True)
image = build_modal_image(
    ROOT,
    env={
        "DG_JIT_CACHE_DIR": "/kernel-cache/deep_gemm",
        "TILELANG_CACHE_DIR": "/kernel-cache/tilelang",
        "TRITON_CACHE_DIR": "/kernel-cache/triton",
    },
)
for helper in (
    "v41_indexer_probe",
    "deepinfer_projection_probe",
    "deepinfer_head_probe",
    "deepinfer_compressor_probe",
    "flashmla_validation",
    "deepseek_v41_validation",
    "v41_transform_probe",
    "v41_residual_probe",
):
    image = image.add_local_file(
        ROOT / f"deploy/{helper}.py", f"/root/deploy/{helper}.py", copy=True
    )
app = modal.App("resin-v41-forward-probe", image=image)


@app.function(
    gpu="B300",
    cpu=32,
    memory=131072,
    timeout=1800,
    volumes={"/kernel-cache": CACHE},
    single_use_containers=True,
)
def probe() -> dict:
    results = {}
    try:
        for name, module in (
            ("indexer", "v41_indexer_probe"),
            ("projections", "deepinfer_projection_probe"),
            ("heads", "deepinfer_head_probe"),
            ("compressor", "deepinfer_compressor_probe"),
            ("attention", "flashmla_validation"),
            ("transforms", "v41_transform_probe"),
            ("residuals", "v41_residual_probe"),
        ):
            # Give each check an independent CUDA context, so a device assert
            # yields useful diagnostics for the remaining operator families.
            output = Path(f"/tmp/resin-v41-{name}-probe.json")
            script = (
                "import json; from pathlib import Path; "
                f"from deploy.{module} import run_probe; "
                f"Path({str(output)!r}).write_text(json.dumps(run_probe(), indent=2))"
            )
            process = subprocess.run(
                ["/opt/resin-deps/.venv/bin/python", "-c", script], cwd="/root"
            )
            results[name] = {
                "passed": process.returncode == 0,
                "result": json.loads(output.read_text()) if output.exists() else None,
            }
        return results
    finally:
        CACHE.commit()


@app.local_entrypoint()
def main(output: str = "/tmp/resin-v41-forward-probe.json") -> None:
    result = probe.remote()
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not all(item["passed"] for item in result.values()):
        raise RuntimeError("one or more V4.1 forward probes failed")
