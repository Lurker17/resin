"""Modal transport for scripts/bench.py; measurement lives in resin.bench."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image
from deploy.models import MODEL_SPECS, get_model, image_environment

ROOT = Path(__file__).resolve().parents[1]
TRACE_VOLUME = modal.Volume.from_name(
    "resin-benchmark-traces-v1", create_if_missing=True
)
MODEL_VOLUMES = {
    spec.model_mount: modal.Volume.from_name(
        spec.model_volume, version=spec.model_volume_version, create_if_missing=True
    )
    for spec in MODEL_SPECS.values()
}
KERNEL_VOLUMES = {
    spec.kernel_mount: modal.Volume.from_name(
        spec.kernel_volume, version=spec.kernel_volume_version, create_if_missing=True
    )
    for spec in MODEL_SPECS.values()
    if spec.kernel_mount is not None and spec.kernel_volume is not None
}
image = build_modal_image(ROOT).add_local_file(
    ROOT / "scripts/bench.py", "/root/scripts/bench.py", copy=True
)
app = modal.App(
    "resin-bench",
    image=image,
    volumes={**MODEL_VOLUMES, **KERNEL_VOLUMES, "/bench-traces": TRACE_VOLUME},
)


@app.function(cpu=16, memory=32768, timeout=4 * 60 * 60)
def download(model: str, metadata_only: bool = False) -> dict:
    """Populate the same pinned checkpoint and marker used by serving."""
    from deploy.models import download_checkpoint

    spec = get_model(model)
    volume = MODEL_VOLUMES[spec.model_mount]
    volume.reload()
    result = download_checkpoint(spec, metadata_only=metadata_only)
    volume.commit()
    return result


@app.function(cpu=32, memory=262144, timeout=4 * 60 * 60, single_use_containers=True)
def sweep(model: str, cases: list[dict], topology: dict, options: dict) -> list[dict]:
    spec = get_model(model)
    # Set compiler namespaces before importing any concrete CUDA runner. Each
    # invocation gets a fresh container, so models cannot inherit another arch.
    os.environ.update(image_environment(spec))
    from deploy.models import prepare_runtime, require_checkpoint
    from resin import Parallelism
    from scripts.bench import run_benchmark

    MODEL_VOLUMES[spec.model_mount].reload()
    kernel_volume = KERNEL_VOLUMES.get(spec.kernel_mount)
    if kernel_volume is not None:
        kernel_volume.reload()
    checkpoint = require_checkpoint(spec, draft_tokens=options["draft_tokens"])
    prepare_runtime(spec)
    options = dict(options)
    options["metadata"] = {
        **options.get("metadata", {}),
        "repo_id": spec.repo_id,
        "checkpoint_revision": spec.revision,
    }
    try:
        return run_benchmark(
            model, checkpoint, cases, parallelism=Parallelism(**topology), **options
        )
    finally:
        if kernel_volume is not None:
            kernel_volume.commit()
        if options.get("trace_dir"):
            TRACE_VOLUME.commit()


def download_traces(remote_directory: str, output: str | Path) -> list[str]:
    """Retrieve raw rank traces and merge each prefill/decode case for Perfetto."""
    from scripts.merge_bench_traces import merge_traces

    prefix = Path(remote_directory).relative_to("/bench-traces").as_posix()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    cases: dict[str, list[Path]] = {}
    for entry in TRACE_VOLUME.iterdir(prefix):
        if not entry.path.endswith(".json.gz"):
            continue
        target = destination / Path(entry.path).name
        with target.open("wb") as stream:
            for chunk in TRACE_VOLUME.read_file(entry.path):
                stream.write(chunk)
        cases.setdefault(target.name.rsplit("-rank", 1)[0], []).append(target)
    merged = []
    for case, sources in sorted(cases.items()):
        target = destination / f"{case}-all-ranks.json.gz"
        merge_traces(sources, target)
        merged.append(str(target))
    if not merged:
        raise RuntimeError(f"no benchmark traces found in {remote_directory}")
    return merged
