"""Smoke-test and serve Qwen3.8-27B through the native Qwen3.5 runner."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from deploy.modal_image import build_modal_image
from resin.backend.parallel import monitor_process_context

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_MOUNT = "/models/resin-qwen3-5"
MODEL_VOLUME_NAME = "resin-qwen3-5-models-v1"
KERNEL_MOUNT = "/kernel-cache/resin-qwen3-5"
KERNEL_VOLUME_NAME = "resin-qwen3-5-kernels-v1"
MODEL_PATH = Path(MODEL_MOUNT) / MODEL_REVISION
DEFAULT_CONTEXT_LENGTH = 32_768
DEFAULT_SPECULATIVE_TOKENS = 2
MAX_RUNNING_REQUESTS = 32
MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_http_server_processes: list[object] = []

image = build_modal_image(
    ROOT,
    env={
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "FLASHINFER_CUDA_ARCH_LIST": "10.0a",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{KERNEL_MOUNT}/flash-attention-4",
        "FLASH_ATTENTION_ARCH": "sm_100a",
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
    "resin-qwen3-8-27b",
    image=image,
    volumes={MODEL_MOUNT: model_volume, KERNEL_MOUNT: kernel_volume},
)


def _inventory() -> dict[str, int]:
    index_path = MODEL_PATH / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        filenames = set(index["weight_map"].values())
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the pinned Qwen3.8 checkpoint index is invalid") from error
    if not filenames or not all(
        isinstance(filename, str) and filename.endswith(".safetensors")
        for filename in filenames
    ):
        raise RuntimeError("the pinned Qwen3.8 checkpoint index has invalid shards")
    required = [
        MODEL_PATH / "config.json",
        MODEL_PATH / "generation_config.json",
        MODEL_PATH / "tokenizer.json",
        MODEL_PATH / "tokenizer_config.json",
        index_path,
        *(MODEL_PATH / filename for filename in sorted(filenames)),
    ]
    if missing := [
        str(path.relative_to(MODEL_PATH)) for path in required if not path.is_file()
    ]:
        raise RuntimeError(f"the pinned Qwen3.8 checkpoint is incomplete: {missing}")
    if empty := [
        str(path.relative_to(MODEL_PATH))
        for path in required
        if path.stat().st_size <= 0
    ]:
        raise RuntimeError(f"the pinned Qwen3.8 checkpoint has empty files: {empty}")
    return {str(path.relative_to(MODEL_PATH)): path.stat().st_size for path in required}


def _require_model() -> dict[str, int]:
    model_volume.reload()
    inventory = _inventory()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Path(KERNEL_MOUNT).mkdir(parents=True, exist_ok=True)
    return inventory


@app.function(cpu=16, memory=32 * 1024, timeout=24 * 60 * 60)
def download_model() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    model_volume.reload()
    started = time.perf_counter()
    cache_hit = True
    try:
        inventory = _inventory()
    except RuntimeError:
        cache_hit = False
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_PATTERNS),
            local_dir=MODEL_PATH,
        )
        inventory = _inventory()
        model_volume.commit()
    return {
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _build_runtime(*, context_length: int, speculative_tokens: int):
    import torch
    from renderers import DefaultRendererConfig, create_renderer
    from transformers import AutoTokenizer

    from resin import Engine, SchedulerConfig
    from resin_models_cuda.qwen3_5 import Qwen3_5CudaRunner

    inventory = _require_model()
    started = time.perf_counter()
    runner = Qwen3_5CudaRunner(
        MODEL_PATH,
        max_seq_len=context_length,
        max_batch_size=MAX_RUNNING_REQUESTS,
        speculative_tokens=speculative_tokens,
        gpu_memory_utilization=0.90,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    )
    renderer = create_renderer(tokenizer, DefaultRendererConfig())
    engine = Engine(
        runner,
        max_running_requests=MAX_RUNNING_REQUESTS,
        context_length=context_length,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
        eos_token_id=runner.eos_token_id,
        prefix_cache=True,
    )
    torch.cuda.synchronize(runner.device)
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpoint_files": len(inventory),
        "checkpoint_bytes": sum(inventory.values()),
        "load_s": time.perf_counter() - started,
        "context_length": context_length,
        "speculative_tokens": speculative_tokens,
        "cache": asdict(runner.cache_capacity_plan),
        "device": torch.cuda.get_device_name(runner.device),
        "capability": torch.cuda.get_device_capability(runner.device),
    }
    return torch, runner, tokenizer, renderer, engine, report


def _prompt_ids(renderer: Any, prompt: str) -> list[int]:
    token_ids = renderer.render_ids(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    if not token_ids:
        raise ValueError("Qwen3.8 chat template produced no tokens")
    return [int(token_id) for token_id in token_ids]


@app.function(
    gpu="B200:1",
    cpu=16,
    memory=128 * 1024,
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    single_use_containers=True,
)
def smoke_test(
    max_new_tokens: int = 128,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
) -> dict[str, Any]:
    from resin import SamplingParams

    torch, runner, tokenizer, renderer, engine, report = _build_runtime(
        context_length=context_length,
        speculative_tokens=speculative_tokens,
    )
    try:
        prompt_ids = _prompt_ids(
            renderer,
            "Explain how a recurrent-attention hybrid reduces KV-cache growth.",
        )
        request = engine.submit(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=renderer.get_stop_token_ids(),
            sampling_params=SamplingParams(temperature=0.0),
        )
        started = time.perf_counter()
        while engine.has_work():
            engine.step()
        torch.cuda.synchronize(runner.device)
        output_ids = request.take_result()
        report.update(
            {
                "elapsed_s": time.perf_counter() - started,
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "text": tokenizer.decode(output_ids, skip_special_tokens=True),
                "speculative": runner.speculative_stats,
            }
        )
        return report
    finally:
        engine.close()


def _serve(_: int, context_length: int, speculative_tokens: int) -> None:
    from resin.server import HttpServer, RendererCodec

    _, _, tokenizer, renderer, engine, _ = _build_runtime(
        context_length=context_length,
        speculative_tokens=speculative_tokens,
    )
    try:
        HttpServer(
            engine,
            codec=RendererCodec(renderer, tokenizer=tokenizer),
            model=MODEL_ID,
        ).serve(host="0.0.0.0", port=8000)
    finally:
        engine.close()


@app.function(
    gpu="B200:1",
    cpu=16,
    memory=128 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_RUNNING_REQUESTS, target_inputs=MAX_RUNNING_REQUESTS)
@modal.web_server(
    8000,
    startup_timeout=60 * 60,
    label="qwen3-8-27b",
    requires_proxy_auth=False,
)
def serve_qwen3_5() -> None:
    import torch

    process_context = torch.multiprocessing.spawn(
        _serve,
        args=(DEFAULT_CONTEXT_LENGTH, DEFAULT_SPECULATIVE_TOKENS),
        nprocs=1,
        join=False,
    )
    _http_server_processes.append(process_context)
    monitor_process_context(process_context, "qwen3-8-27b-http")


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    max_new_tokens: int = 128,
    speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
) -> None:
    if mode == "download":
        report = download_model.remote()
    elif mode == "smoke":
        report = smoke_test.remote(
            max_new_tokens,
            speculative_tokens=speculative_tokens,
        )
    else:
        raise ValueError("mode must be 'download' or 'smoke'")
    print(json.dumps(report, indent=2, sort_keys=True))
