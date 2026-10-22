"""Resident Engine sweeps on Modal; raw points are appended to local JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal

from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
)
from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
# Share the existing checkpoint volumes, while keeping model identity explicit
# in every result. Repository Python files are never downloaded or executed.
CHECKPOINTS = {
    "gpt-oss-20b": (
        "resin-gpt-oss-models-v2",
        2,
        "/models/gpt",
        "20b",
        "openai/gpt-oss-20b",
        "6cee5e81ee83917806bbde320786a8fb61efebee",
    ),
    "gpt-oss-120b": (
        "resin-gpt-oss-models-v2",
        2,
        "/models/gpt",
        "120b",
        "openai/gpt-oss-120b",
        "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ),
    "deepseek-v4": (
        "resin-deepseek-v4-models-v1",
        1,
        "/models/deepseek",
        "flash-0731",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    ),
}
MODEL_VOLUMES = {
    mount: modal.Volume.from_name(name, version=version, create_if_missing=True)
    for name, version, mount, *_ in CHECKPOINTS.values()
}
KERNEL_VOLUME = modal.Volume.from_name(
    "resin-deepseek-v4-kernels-v2", version=1, create_if_missing=True
)
HOPPER_KERNEL_VOLUME = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    version=KERNEL_CACHE_VOLUME_VERSION,
    create_if_missing=True,
)
image = build_modal_image(
    ROOT,
    env={
        **KERNEL_CACHE_ENV,
        "CC": "gcc",
        "CXX": "g++",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_JIT_CACHE_DIR": "/kernel-cache/.cache/deep_gemm",
        "TILELANG_CACHE_DIR": "/kernel-cache/.cache/tilelang",
        "CUDA_TILE_CACHE_DIR": "/kernel-cache/.cache/cutile-python",
    },
).add_local_file(ROOT / "deploy/bench.py", "/root/deploy/bench.py", copy=True)
app = modal.App(
    "resin-bench",
    image=image,
    volumes={
        **MODEL_VOLUMES,
        "/kernel-cache/.cache": KERNEL_VOLUME,
        KERNEL_CACHE_MOUNT: HOPPER_KERNEL_VOLUME,
    },
)


def _checkpoint(model: str) -> tuple[Path, str, str]:
    _, _, mount, key, repo, revision = CHECKPOINTS[model]
    return Path(mount) / key / revision, repo, revision


@app.function(cpu=16, memory=32768, timeout=4 * 60 * 60)
def download(model: str) -> dict:
    """Publish a complete pinned checkpoint using a CPU worker."""
    from huggingface_hub import snapshot_download

    target, repo, revision = _checkpoint(model)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo,
        revision=revision,
        local_dir=target,
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "inference/config.json",
        ],
    )
    MODEL_VOLUMES[CHECKPOINTS[model][2]].commit()
    return {"repo_id": repo, "revision": revision, "path": str(target)}


@app.function(cpu=32, memory=262144, timeout=4 * 60 * 60, single_use_containers=True)
def sweep(model: str, cases: list[dict], topology: dict, options: dict) -> list[dict]:
    if model.startswith("deepseek"):
        # Hopper's precompiled attention stack has an explicit architecture;
        # DeepSeek's Blackwell kernels infer it from their current device.
        for name in (
            "TORCH_CUDA_ARCH_LIST",
            "FLASHINFER_CUDA_ARCH_LIST",
            "FLASH_ATTENTION_ARCH",
            "CUTE_DSL_ARCH",
        ):
            os.environ.pop(name, None)
        os.environ["TRITON_CACHE_DIR"] = "/kernel-cache/.cache/triton"
    from deploy.bench import run_benchmark
    from deploy.jit_cache import ensure_kernel_cache_dirs, install_flashinfer_hash_cache
    from resin import Parallelism

    target, repo, revision = _checkpoint(model)
    if not (target / "config.json").is_file():
        raise FileNotFoundError(f"download the pinned {model} checkpoint first")
    if model.startswith("gpt-oss"):
        ensure_kernel_cache_dirs()
        install_flashinfer_hash_cache()
    try:
        return run_benchmark(
            model,
            target,
            cases,
            parallelism=Parallelism(**topology),
            metadata={
                **options.pop("metadata", {}),
                "repo_id": repo,
                "checkpoint_revision": revision,
            },
            **options,
        )
    finally:
        KERNEL_VOLUME.commit()
        HOPPER_KERNEL_VOLUME.commit()


@app.local_entrypoint()
def main(
    model: str = "deepseek-v4",
    gpu: str = "B300",
    world_size: int = 4,
    attn_dp: int = 1,
    attn_tp: int = 4,
    cp: int = 1,
    ep: int = 1,
    phases: str = "prefill,decode",
    context_lengths: str = "4096",
    batch_sizes: str = "1,2,4,8",
    draft_tokens: int = 4,
    accepted_drafts: int = -1,
    natural_acceptance: bool = False,
    balanced_experts: bool = True,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_chunk_size: int = 4096,
    prefill_graph: str = "breakable",
    decode_graph: str = "full",
    gpu_memory_utilization: float = 0.9,
    output: str = "bench/results.jsonl",
    download_only: bool = False,
):
    from deploy.bench import write_results
    from resin import Parallelism
    from resin.bench import BenchCase

    if model not in CHECKPOINTS:
        raise ValueError(f"model must be one of {list(CHECKPOINTS)}")
    if download_only:
        print(json.dumps(download.remote(model), indent=2))
        return
    topology = dict(
        world_size=world_size, attn_dp=attn_dp, attn_tp=attn_tp, cp=cp, ep=ep
    )
    Parallelism(**topology)
    cases: list[dict[str, Any]] = [
        dict(phase=phase, context_length=int(context), batch_size=int(batch))
        for phase in phases.split(",")
        for context in context_lengths.split(",")
        for batch in batch_sizes.split(",")
    ]
    for case in cases:
        BenchCase(**case)
    if model.startswith("gpt-oss") and draft_tokens:
        raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    source = hashlib.sha256(git_revision.encode())
    source.update(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    )
    for name in sorted(
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
        ).splitlines()
    ):
        path = ROOT / name
        if path.suffix in {".py", ".rs", ".toml", ".lock", ".cu", ".cuh"}:
            source.update(name.encode())
            source.update(path.read_bytes())
    options = dict(
        draft_tokens=draft_tokens,
        accepted_drafts=(
            None
            if natural_acceptance
            else min(3, draft_tokens)
            if accepted_drafts == -1
            else accepted_drafts
        ),
        balanced_experts=balanced_experts,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        prefill_chunk_size=prefill_chunk_size,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata={
            "git_commit": git_revision,
            "dirty": dirty,
            "source_digest": source.hexdigest(),
        },
    )
    records = sweep.with_options(gpu=f"{gpu}:{world_size}").remote(
        model, cases, topology, options
    )
    write_results(output, records)
    print(f"Appended {len(records)} raw points to {output}")
