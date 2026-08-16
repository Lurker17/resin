"""Run NVIDIA Gemma 4 31B NVFP4 multimodal inference on one RTX PRO 6000."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/Gemma-4-31B-IT-NVFP4"
MODEL_REVISION = "4135a98a9b728a548947683219633b25682223ac"
MODEL_VOLUME_NAME = "resin-gemma4-models-v1"
MODEL_MOUNT = "/models/resin-gemma4"
MODEL_PATH = Path(MODEL_MOUNT) / MODEL_REVISION
PROFILE_VOLUME_NAME = "resin-gemma4-profiles-v1"
PROFILE_MOUNT = "/profiles/resin-gemma4"
MODEL_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
CUDA_GRAPH_DECODE_SHAPES = tuple((batch_size, 4096) for batch_size in (1, 2, 4, 8))

image = build_modal_image(
    ROOT,
    env={
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "FLASHINFER_CUDA_ARCH_LIST": "12.0",
    },
)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
app = modal.App(
    "resin-gemma4-nvfp4",
    image=image,
    volumes={MODEL_MOUNT: model_volume, PROFILE_MOUNT: profile_volume},
)


def _inventory() -> dict[str, int]:
    index = MODEL_PATH / "model.safetensors.index.json"
    config = MODEL_PATH / "config.json"
    weights = sorted(MODEL_PATH.glob("model*.safetensors"))
    required = [index, config, *weights]
    if not weights or any(
        not path.is_file() or path.stat().st_size <= 0 for path in required
    ):
        raise RuntimeError("the pinned Gemma 4 checkpoint is incomplete")
    return {path.name: path.stat().st_size for path in required}


def _checkpoint_size() -> int:
    index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())
    return int(index["metadata"]["total_size"])


def _require_model() -> None:
    model_volume.reload()
    _inventory()
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
        inventory = _inventory()
        model_volume.commit()
    return {
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _build_runtime(*, static_memory_fraction: float = 0.72):
    import torch
    from transformers import AutoProcessor

    from resin import Engine, SchedulerConfig
    from resin_models_cuda.gemma4 import Gemma4CudaRunner
    from resin_models_cuda.gemma4.cuda import plan_gemma4_kv_capacity

    _require_model()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    capacity = plan_gemma4_kv_capacity(
        total_memory,
        _checkpoint_size(),
        static_memory_fraction,
    )
    load_started = time.perf_counter()
    runner = Gemma4CudaRunner(
        MODEL_PATH,
        cache_parent_capacity=capacity.parent_capacity,
        cuda_graph_decode_shapes=CUDA_GRAPH_DECODE_SHAPES,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    graph_started = time.perf_counter()
    graph_stats = runner.capture_cuda_graphs()
    graph_capture_s = time.perf_counter() - graph_started
    engine = Engine(
        runner,
        max_running_requests=8,
        context_length=runner.config.max_position_embeddings,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
        eos_token_id=runner.config.eos_token_ids[0],
        prefix_cache=True,
    )
    torch.cuda.synchronize()
    report = {
        "load_s": time.perf_counter() - load_started,
        "capacity": asdict(capacity),
        "actual_model_bytes": runner.model_memory_bytes,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "cuda_graph_capture_s": graph_capture_s,
        "cuda_graphs": graph_stats,
    }
    return torch, runner, processor, engine, report


def _synthetic_image(size: int = 96):
    import numpy as np

    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    horizontal = np.broadcast_to(x[None, :], (size, size))
    vertical = horizontal.T
    return np.stack((horizontal, vertical, 1.0 - horizontal), axis=-1).copy()


def _generate(
    engine,
    processor,
    prompt: str,
    max_new_tokens: int,
    *,
    profile_steps: bool = False,
):
    from resin import SamplingParams
    from resin_models_cuda.gemma4 import Gemma4Media

    media = Gemma4Media.from_images(
        processor,
        [_synthetic_image()],
        prompt=prompt,
    )
    lease = engine.submit(
        media=media,
        max_new_tokens=max_new_tokens,
        stop_token_ids=(1, 106),
        sampling_params=SamplingParams(temperature=0.0),
    )
    steps = []
    while engine.has_work():
        started = time.perf_counter()
        if profile_steps:
            import torch

            with torch.profiler.record_function(f"resin.engine.step.{len(steps)}"):
                events = engine.step()
        else:
            events = engine.step()
        steps.append(
            {
                "ms": (time.perf_counter() - started) * 1000,
                "tokens": len(events),
            }
        )
    token_ids = lease.take_result()
    return media, token_ids, steps


def _text_batch(
    engine, processor, batch_size: int, prompt_words: int
) -> dict[str, Any]:
    from resin import SamplingParams
    from resin_models_cuda.gemma4 import Gemma4Media

    prompt = " ".join(["word"] * prompt_words)
    leases = []
    prompt_tokens = 0
    for index in range(batch_size):
        media = Gemma4Media.from_messages(
            processor,
            [
                {
                    "role": "user",
                    "content": f"{prompt}\nReply with the number {index}.",
                }
            ],
        )
        prompt_tokens = len(media.token_ids)
        leases.append(
            engine.submit(
                media=media,
                max_new_tokens=3,
                stop_token_ids=(1, 106),
                sampling_params=SamplingParams(temperature=0.0),
            )
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
    outputs = [lease.take_result() for lease in leases]
    return {
        "batch_size": batch_size,
        "prompt_words": prompt_words,
        "prompt_tokens": prompt_tokens,
        "steps": steps,
        "output_lengths": list(map(len, outputs)),
    }


@app.function(
    gpu="RTX-PRO-6000:1",
    cpu=16,
    memory=128 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def smoke_test(max_new_tokens: int = 16) -> dict[str, Any]:
    torch, runner, processor, engine, report = _build_runtime()
    try:
        warmup_started = time.perf_counter()
        _, _, warmup_steps = _generate(
            engine,
            processor,
            "Name the main colors.",
            2,
        )
        torch.cuda.synchronize()
        report["warmup_s"] = time.perf_counter() - warmup_started
        report["warmup_step_ms"] = warmup_steps
        torch.cuda.reset_peak_memory_stats()
        media, token_ids, steps = _generate(
            engine,
            processor,
            "Describe this image briefly and mention its dominant visual pattern.",
            max_new_tokens,
        )
        torch.cuda.synchronize()
        first_token_step = next(
            index for index, step in enumerate(steps) if step["tokens"]
        )
        decode_ms = [
            step["ms"] for step in steps[first_token_step + 1 :] if step["tokens"]
        ]
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
                "mean_tpot_ms": sum(decode_ms) / len(decode_ms) if decode_ms else 0.0,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "prefix_cache": engine.prefix_cache_stats,
            }
        )
        return report
    finally:
        if not engine.closed:
            engine.close()


@app.function(
    gpu="RTX-PRO-6000:1",
    cpu=16,
    memory=128 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def shape_profile() -> dict[str, Any]:
    torch, _, processor, engine, report = _build_runtime()
    try:
        _generate(engine, processor, "Name the main colors.", 2)
        cases = [(1, 8), (2, 8), (4, 8), (8, 8), (1, 128), (1, 512)]
        cold = []
        warm = []
        for batch_size, prompt_words in cases:
            engine.clear_prefix_cache()
            cold.append(_text_batch(engine, processor, batch_size, prompt_words))
            engine.clear_prefix_cache()
            warm.append(_text_batch(engine, processor, batch_size, prompt_words))
        torch.cuda.synchronize()
        report["cold"] = cold
        report["warm"] = warm
        return report
    finally:
        if not engine.closed:
            engine.close()


@app.function(
    gpu="RTX-PRO-6000:1",
    cpu=16,
    memory=128 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def perfetto_trace(max_new_tokens: int = 5, *, detailed: bool = True) -> dict[str, Any]:
    torch, _, processor, engine, report = _build_runtime()
    try:
        # Populate every lazy kernel cache before profiling, then force the
        # measured request to recompute its complete multimodal prefix.
        _generate(engine, processor, "Name the main colors.", 2)
        engine.clear_prefix_cache()
        torch.cuda.synchronize()

        detail = "detailed" if detailed else "low-overhead"
        trace_name = f"gemma4-sm120-{detail}-{int(time.time())}.json.gz"
        trace_path = Path(PROFILE_MOUNT) / trace_name
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=detailed,
            profile_memory=detailed,
            with_flops=detailed,
        ) as profile:
            with torch.profiler.record_function("resin.gemma4.multimodal_request"):
                media, token_ids, steps = _generate(
                    engine,
                    processor,
                    "Describe the gradient and its dominant colors in one sentence.",
                    max_new_tokens,
                    profile_steps=True,
                )
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(trace_path))
        profile_volume.commit()

        cuda_table = profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
        report.update(
            {
                "trace_volume": PROFILE_VOLUME_NAME,
                "trace_file": trace_name,
                "prompt_tokens": len(media.token_ids),
                "vision_tokens": len(media.image_positions),
                "output_tokens": token_ids,
                "step_ms": steps,
                "cuda_table": cuda_table,
            }
        )
        return report
    finally:
        if not engine.closed:
            engine.close()


def _serve() -> None:
    from resin.server import HttpServer
    from resin_models_cuda.gemma4 import Gemma4Codec

    torch, runner, processor, engine, _ = _build_runtime()
    try:
        # Compile/load every exact kernel geometry before opening the socket.
        warmup, _, _ = _generate(engine, processor, "Name the main colors.", 2)
        del warmup
        _text_batch(engine, processor, batch_size=1, prompt_words=8)
        torch.cuda.synchronize(runner.device)
        HttpServer(
            engine,
            codec=Gemma4Codec(processor, eos_token_ids=runner.config.eos_token_ids),
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
@modal.concurrent(max_inputs=32, target_inputs=8)
@modal.web_server(
    8000,
    startup_timeout=60 * 60,
    label="gemma4-nvfp4",
    requires_proxy_auth=False,
)
def serve_gemma4() -> None:
    _serve()


@app.local_entrypoint()
def main(mode: str = "smoke", max_new_tokens: int = 16) -> None:
    if mode == "download":
        print(json.dumps(download_model.remote(), indent=2))
    elif mode == "smoke":
        print(json.dumps(smoke_test.remote(max_new_tokens), indent=2))
    elif mode == "profile":
        print(json.dumps(shape_profile.remote(), indent=2))
    elif mode == "trace":
        print(
            json.dumps(perfetto_trace.remote(max_new_tokens, detailed=True), indent=2)
        )
    elif mode == "trace-lite":
        print(
            json.dumps(perfetto_trace.remote(max_new_tokens, detailed=False), indent=2)
        )
    else:
        raise ValueError(
            "mode must be 'download', 'smoke', 'profile', 'trace', or 'trace-lite'"
        )
