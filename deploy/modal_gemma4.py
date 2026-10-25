"""Run NVIDIA Gemma 4 31B NVFP4 multimodal inference on one RTX PRO 6000."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

import resin
from deploy.graph_validation import capture_cuda_graphs, validate_prefill_graphs
from deploy.modal_image import build_modal_image
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    default_decode_graph_shapes,
)
from resin.backend.parallel import monitor_process_context
from resin.recipe import Recipe, get_native_recipe

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/Gemma-4-31B-IT-NVFP4"
MODEL_REVISION = "4135a98a9b728a548947683219633b25682223ac"
MODEL_VOLUME_NAME = "resin-gemma4-models-v1"
MODEL_MOUNT = "/models/resin-gemma4"
MODEL_PATH = Path(MODEL_MOUNT) / MODEL_REVISION
ASSISTANT_ID = "google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"
ASSISTANT_PATH = Path(MODEL_MOUNT) / "assistant" / ASSISTANT_REVISION
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ASSISTANT_FILES = ("config.json", "model.safetensors")
CUDA_GRAPH_HISTORY_PAGES = 4096
PRODUCTION_SPECULATIVE_TOKENS = 3
_http_server_processes: list[object] = []

image = build_modal_image(
    ROOT,
    env={
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "FLASHINFER_CUDA_ARCH_LIST": "12.0",
    },
)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = modal.App(
    "resin-gemma4-nvfp4",
    image=image,
    volumes={MODEL_MOUNT: model_volume},
)


def _inventory(*, include_assistant: bool = True) -> dict[str, int]:
    static_files = [MODEL_PATH / name for name in MODEL_FILES if "*" not in name]
    weights = sorted(MODEL_PATH.glob("model*.safetensors"))
    required = [*static_files, *weights]
    if not weights or any(
        not path.is_file() or path.stat().st_size <= 0 for path in required
    ):
        raise RuntimeError("the pinned Gemma 4 checkpoint is incomplete")

    inventory = {f"target/{path.name}": path.stat().st_size for path in required}
    if include_assistant:
        assistant_required = [ASSISTANT_PATH / name for name in ASSISTANT_FILES]
        if any(
            not path.is_file() or path.stat().st_size <= 0
            for path in assistant_required
        ):
            raise RuntimeError("the pinned Gemma 4 assistant checkpoint is incomplete")
        inventory.update(
            {
                f"assistant/{path.name}": path.stat().st_size
                for path in assistant_required
            }
        )
    return inventory


def _checkpoint_size() -> int:
    index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())
    return int(index["metadata"]["total_size"])


def _require_model(*, speculative_tokens: int) -> None:
    model_volume.reload()
    _inventory(include_assistant=bool(speculative_tokens))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@app.function(cpu=8, memory=16 * 1024, timeout=8 * 60 * 60)
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
            allow_patterns=list(MODEL_FILES),
            local_dir=MODEL_PATH,
        )
        ASSISTANT_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=ASSISTANT_ID,
            revision=ASSISTANT_REVISION,
            allow_patterns=list(ASSISTANT_FILES),
            local_dir=ASSISTANT_PATH,
        )
        inventory = _inventory()
        model_volume.commit()
    return {
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _build_runtime(*, speculative_tokens: int, validate_graphs: bool = False):
    import torch
    from transformers import AutoProcessor

    from resin import Engine, SchedulerConfig
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

    _require_model(speculative_tokens=speculative_tokens)
    load_started = time.perf_counter()
    runner = Gemma4CudaRunner(
        MODEL_PATH,
        device="cuda",
        assistant_model=ASSISTANT_PATH if speculative_tokens else None,
        speculative_tokens=speculative_tokens,
        cuda_graph_decode_shapes=default_decode_graph_shapes(
            CUDA_GRAPH_HISTORY_PAGES,
            speculative=bool(speculative_tokens),
            max_batch_size=DEFAULT_MAX_NUM_SEQS,
        ),
        static_memory_fraction=0.95,
    )
    capacity = runner.cache_capacity_plan
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )

    graph_started = time.perf_counter()
    graph_stats = capture_cuda_graphs(runner)
    graph_capture_s = time.perf_counter() - graph_started
    if validate_graphs:
        validate_prefill_graphs(runner)

    engine = Engine(
        runner,
        max_running_requests=DEFAULT_MAX_NUM_SEQS,
        context_length=runner.config.max_position_embeddings,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
        prefix_cache=True,
    )
    torch.cuda.synchronize()
    report = {
        "load_s": time.perf_counter() - load_started,
        "speculative_tokens": speculative_tokens,
        "capacity": asdict(capacity),
        "checkpoint_bytes": _checkpoint_size(),
        "actual_model_bytes": runner.model_memory_bytes,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "cuda_graph_capture_s": graph_capture_s,
        "cuda_graphs": graph_stats,
    }
    if speculative_tokens:
        report["assistant_checkpoint_bytes"] = (
            (ASSISTANT_PATH / "model.safetensors").stat().st_size
        )
    return runner, processor, engine, report


def _synthetic_image(size: int = 96):
    import numpy as np

    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    horizontal = np.broadcast_to(x[None, :], (size, size))
    vertical = horizontal.T
    return np.stack((horizontal, vertical, 1.0 - horizontal), axis=-1).copy()


def _generate(engine, recipe: Recipe, prompt: str, max_output_tokens: int):
    from resin import SamplingParams

    prepared = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [
                    resin.recipe.Image(source=_synthetic_image()),
                    resin.recipe.Text(prompt),
                ],
            ),
        ]
    )
    media = prepared.media
    lease = engine.submit(
        media=media,
        sampling_params=SamplingParams(
            max_output_tokens=max_output_tokens,
            stop_token_ids=recipe.stop_token_ids,
            temperature=0.0,
        ),
    )
    steps = []
    while engine.has_work():
        started = time.perf_counter()
        events = engine.step()
        steps.append(
            {
                "ms": (time.perf_counter() - started) * 1000,
                "tokens": len(events),
            }
        )
    return media, lease.take_result(), steps


def _warmup_text(engine, recipe: Recipe) -> None:
    from resin import SamplingParams

    prepared = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER, [resin.recipe.Text("Reply with the number 1.")]
            ),
        ]
    )
    lease = engine.submit(
        prepared.token_ids,
        sampling_params=SamplingParams(
            max_output_tokens=3, stop_token_ids=recipe.stop_token_ids, temperature=0.0
        ),
    )
    while engine.has_work():
        engine.step()
    lease.take_result()


@app.function(
    gpu="RTX-PRO-6000:1",
    cpu=16,
    memory=128 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def smoke_test(
    max_output_tokens: int = 16,
    *,
    speculative_tokens: int,
) -> dict[str, Any]:
    import torch

    runner, processor, engine, report = _build_runtime(
        speculative_tokens=speculative_tokens, validate_graphs=True
    )
    try:
        recipe = get_native_recipe(
            "gemma4", processor, stop_token_ids=runner.config.eos_token_ids
        )
        warmup_started = time.perf_counter()
        _, _, warmup_steps = _generate(
            engine,
            recipe,
            "Name the main colors.",
            2,
        )
        torch.cuda.synchronize()
        report["warmup_s"] = time.perf_counter() - warmup_started
        report["warmup_step_ms"] = warmup_steps

        torch.cuda.reset_peak_memory_stats()
        media, token_ids, steps = _generate(
            engine,
            recipe,
            "Describe this image briefly and mention its dominant visual pattern.",
            max_output_tokens,
        )
        torch.cuda.synchronize()
        first_token_step = next(
            index for index, step in enumerate(steps) if step["tokens"]
        )
        decode_ms = [
            step["ms"] for step in steps[first_token_step + 1 :] if step["tokens"]
        ]
        decode_tokens = sum(step["tokens"] for step in steps[first_token_step + 1 :])
        report.update(
            {
                "prompt_tokens": len(media.token_ids),
                "vision_tokens": len(media.image_positions),
                "output_tokens": token_ids,
                "output_text": processor.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ),
                "step_ms": steps,
                "ttft_ms": sum(step["ms"] for step in steps[: first_token_step + 1]),
                "mean_tpot_ms": (
                    sum(decode_ms) / decode_tokens if decode_tokens else 0.0
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "prefix_cache": engine.prefix_cache_stats,
                "cuda_graphs": runner.cuda_graph_stats,
                "speculative": runner.speculative_stats,
            }
        )
        return report
    finally:
        if not engine.closed:
            engine.close()


def _serve(_: int) -> None:
    import torch

    from resin.server import HttpServer

    runner, processor, engine, _ = _build_runtime(
        speculative_tokens=PRODUCTION_SPECULATIVE_TOKENS
    )
    try:
        recipe = get_native_recipe(
            "gemma4", processor, stop_token_ids=runner.config.eos_token_ids
        )
        # Compile/load every exact kernel geometry before opening the socket.
        warmup, _, _ = _generate(engine, recipe, "Name the main colors.", 2)
        del warmup
        _warmup_text(engine, recipe)
        torch.cuda.synchronize(runner.device)
        HttpServer(
            engine,
            recipe=recipe,
            model=MODEL_ID,
            enable_anthropic=False,
        ).serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()


@app.function(
    gpu="RTX-PRO-6000:1",
    cpu=16,
    memory=128 * 1024,
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
    label="gemma4-31b",
    requires_proxy_auth=False,
)
def serve_gemma4() -> None:
    import torch

    process_context = torch.multiprocessing.spawn(
        _serve,
        nprocs=1,
        join=False,
    )
    _http_server_processes.append(process_context)
    monitor_process_context(process_context, "gemma4-http")


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    max_output_tokens: int = 16,
    speculative_tokens: int = PRODUCTION_SPECULATIVE_TOKENS,
) -> None:
    if mode == "download":
        print(json.dumps(download_model.remote(), indent=2))
    elif mode == "smoke":
        print(
            json.dumps(
                smoke_test.remote(
                    max_output_tokens,
                    speculative_tokens=speculative_tokens,
                ),
                indent=2,
            )
        )
    else:
        raise ValueError("mode must be 'download' or 'smoke'")
