"""Serve Thinking Machines' Inkling-Small NVFP4 with gamma-five DSpark."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from deploy.graph_validation import capture_cuda_graphs, validate_prefill_graphs
from deploy.modal_image import build_modal_image
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    default_decode_graph_shapes,
)
from resin.backend.parallel import monitor_process_context
from resin_models_cuda.inkling.checkpoint import MODEL_SPEC

ROOT = Path(__file__).resolve().parents[1]
MODEL_MOUNT = "/models/resin-inkling"
MODEL_VOLUME_NAME = "resin-inkling-models-v1"
KERNEL_MOUNT = "/kernel-cache/resin-inkling"
KERNEL_VOLUME_NAME = "resin-inkling-kernels-v2"
TARGET_PATH = Path(MODEL_MOUNT) / "target" / MODEL_SPEC.target_revision
DRAFT_PATH = Path(MODEL_MOUNT) / "draft" / MODEL_SPEC.draft_revision
MAX_CONTEXT_LENGTH = 131_072
TARGET_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "tokenizer*",
    "model*.safetensors",
)
DRAFT_PATTERNS = ("config.json", "model.safetensors")
_http_server_processes: list[object] = []

image = build_modal_image(
    ROOT,
    cuda_source_ignore=(
        "resin_models_cuda/deepseek_v4",
        "resin_models_cuda/gemma4",
        "resin_models_cuda/gpt_oss",
        "resin_models_cuda/kimi_k25",
        "resin_models_cuda/lfm2",
        "resin_models_cuda/qwen3_5",
    ),
    env={
        "TORCH_CUDA_ARCH_LIST": "10.3a",
        "FLASHINFER_CUDA_ARCH_LIST": "10.3a",
        "FLASHINFER_WORKSPACE_BASE": f"{KERNEL_MOUNT}/flashinfer",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{KERNEL_MOUNT}/flash-attention-4",
        "FLASH_ATTENTION_ARCH": "sm_103a",
        "CUTE_DSL_CACHE_DIR": f"{KERNEL_MOUNT}/cute-dsl",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUDA_TILE_CACHE_DIR": f"{KERNEL_MOUNT}/cutile-python",
        "QUACK_CACHE_ENABLED": "1",
        "QUACK_CACHE_DIR": f"{KERNEL_MOUNT}/quack",
        "TRITON_CACHE_DIR": f"{KERNEL_MOUNT}/triton",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)
model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME, create_if_missing=True, version=1
)
kernel_volume = modal.Volume.from_name(
    KERNEL_VOLUME_NAME, create_if_missing=True, version=1
)
app = modal.App(
    "resin-inkling-small",
    image=image,
    volumes={MODEL_MOUNT: model_volume, KERNEL_MOUNT: kernel_volume},
)


def _inventory() -> dict[str, int]:
    index_path = TARGET_PATH / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the pinned Inkling target index is invalid") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("the pinned Inkling target index contains no tensors")
    target_weight_files = {
        TARGET_PATH / str(filename)
        for name, filename in weight_map.items()
        if isinstance(name, str) and name.startswith(("model.llm.", "model.visual."))
    }
    target_files = {
        TARGET_PATH / "config.json",
        TARGET_PATH / "tokenizer.json",
        TARGET_PATH / "tokenizer_config.json",
        index_path,
        *target_weight_files,
    }
    draft_files = {DRAFT_PATH / "config.json", DRAFT_PATH / "model.safetensors"}
    files = sorted(target_files | draft_files)
    if missing := [str(path) for path in files if not path.is_file()]:
        raise RuntimeError(
            f"the pinned Inkling checkpoints are incomplete: {missing[:8]}"
        )
    if empty := [str(path) for path in files if path.stat().st_size <= 0]:
        raise RuntimeError(
            f"the pinned Inkling checkpoints contain empty files: {empty[:8]}"
        )
    return {str(path.relative_to(MODEL_MOUNT)): path.stat().st_size for path in files}


def _require_models() -> dict[str, int]:
    model_volume.reload()
    inventory = _inventory()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Path(KERNEL_MOUNT).mkdir(parents=True, exist_ok=True)
    return inventory


@app.function(cpu=16, memory=32 * 1024, timeout=24 * 60 * 60)
def download_models() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    model_volume.reload()
    started = time.perf_counter()
    cache_hit = True
    try:
        inventory = _inventory()
    except RuntimeError:
        cache_hit = False
        TARGET_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_SPEC.target_repo,
            revision=MODEL_SPEC.target_revision,
            allow_patterns=list(TARGET_PATTERNS),
            local_dir=TARGET_PATH,
        )
        DRAFT_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_SPEC.draft_repo,
            revision=MODEL_SPEC.draft_revision,
            allow_patterns=list(DRAFT_PATTERNS),
            local_dir=DRAFT_PATH,
        )
        inventory = _inventory()
        model_volume.commit()
    return {
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _build_runtime(
    *, context_length: int = MAX_CONTEXT_LENGTH, validate_graphs: bool = False
):
    import torch
    from transformers import AutoProcessor

    from resin import Engine, SchedulerConfig
    from resin.recipe import Recipe
    from resin_models_cuda.inkling.cuda import InklingCudaRunner

    inventory = _require_models()
    runner = InklingCudaRunner(
        TARGET_PATH,
        DRAFT_PATH,
        device="cuda",
        max_seq_len=context_length,
        max_batch_size=DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization=0.90,
        cuda_graph_decode_shapes=default_decode_graph_shapes(
            context_length + InklingCudaRunner.speculative_tokens,
            speculative=True,
            max_batch_size=DEFAULT_MAX_NUM_SEQS,
        ),
    )
    graph_started = time.perf_counter()
    graph_stats = capture_cuda_graphs(runner)
    graph_capture_s = time.perf_counter() - graph_started
    if validate_graphs:
        validate_prefill_graphs(runner)
    processor = AutoProcessor.from_pretrained(
        TARGET_PATH, local_files_only=True, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    recipe = Recipe(tokenizer, model="inkling")
    engine = Engine(
        runner,
        max_running_requests=DEFAULT_MAX_NUM_SEQS,
        context_length=context_length,
        eos_token_id=runner.eos_token_id,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
    )
    report = {
        "checkpoint_files": len(inventory),
        "checkpoint_bytes": sum(inventory.values()),
        "cache": asdict(runner.cache_capacity_plan),
        "cuda_graph_capture_s": graph_capture_s,
        "cuda_graphs": graph_stats,
        "device": torch.cuda.get_device_name(runner.device),
    }
    return runner, processor, recipe, engine, report


def _prompt_ids(recipe: Any, prompt: str) -> list[int]:
    token_ids = recipe.render_ids(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    if not token_ids:
        raise ValueError("Inkling chat template produced no tokens")
    return [int(token_id) for token_id in token_ids]


@app.function(
    gpu="B300:1",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    single_use_containers=True,
)
def smoke_test(
    max_new_tokens: int = 512,
    *,
    context_length: int = 32_768,
) -> dict[str, Any]:
    import torch

    from resin import SamplingParams

    runner, processor, recipe, engine, report = _build_runtime(
        context_length=context_length, validate_graphs=True
    )
    try:
        prompt_ids = _prompt_ids(
            recipe,
            "Explain why speculative decoding can preserve model quality.",
        )
        lease = engine.submit(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=recipe.get_stop_token_ids(),
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=0.95,
                seed=7,
            ),
        )
        while engine.has_work():
            engine.step()
        output_ids = lease.take_result()
        torch.cuda.synchronize(runner.device)
        report.update(
            {
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "output_ids": output_ids,
                "text": processor.decode(output_ids, skip_special_tokens=False),
                "speculative": runner.speculative_stats,
                "cuda_graphs": runner.cuda_graph_stats,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runner.device),
            }
        )
        kernel_volume.commit()
        return report
    finally:
        if not engine.closed:
            engine.close()


def _serve(_: int) -> None:
    import torch

    from resin import SamplingParams
    from resin.models.inkling.codec import InklingCodec
    from resin.server import HttpServer

    runner, processor, recipe, engine, _ = _build_runtime()
    try:
        warmup = engine.submit(
            _prompt_ids(recipe, "Reply with one word."),
            max_new_tokens=2,
            stop_token_ids=recipe.get_stop_token_ids(),
            sampling_params=SamplingParams(temperature=0.0),
        )
        while engine.has_work():
            engine.step()
        warmup.take_result()
        torch.cuda.synchronize(runner.device)
        HttpServer(
            engine,
            codec=InklingCodec(processor),
            model=MODEL_SPEC.target_repo,
            enable_anthropic=False,
        ).serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()
    os._exit(0)


@app.function(
    gpu="B300:1",
    cpu=16,
    memory=256 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(
    max_inputs=DEFAULT_MAX_NUM_SEQS,
    target_inputs=DEFAULT_MAX_NUM_SEQS,
)
@modal.web_server(
    8000,
    startup_timeout=60 * 60,
    label="inkling-small",
    requires_proxy_auth=False,
)
def serve_inkling() -> None:
    import torch

    process_context = torch.multiprocessing.spawn(_serve, nprocs=1, join=False)
    _http_server_processes.append(process_context)
    monitor_process_context(process_context, "inkling-small-http")


@app.local_entrypoint()
def main(mode: str = "smoke", max_new_tokens: int = 512) -> None:
    if mode == "download":
        report = download_models.remote()
    elif mode == "smoke":
        report = smoke_test.remote(max_new_tokens)
    else:
        raise ValueError("mode must be 'download' or 'smoke'")
    print(json.dumps(report, indent=2, sort_keys=True))
