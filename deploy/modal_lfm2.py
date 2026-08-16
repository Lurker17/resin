"""Run the native LFM2 engine on a Modal H100."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
MODEL = "LiquidAI/LFM2.5-230M"
REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
PROMPT = [
    1,
    6,
    24131,
    708,
    4083,
    938,
    768,
    11538,
    16701,
    12216,
    968,
    59929,
    14009,
    523,
    7,
    708,
    6,
    6423,
    708,
    3493,
    856,
    779,
    5706,
    803,
    4481,
    540,
    7,
    708,
    6,
    64015,
    708,
]


def cache_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "model.safetensors.index.json",
        ],
    )


image = build_modal_image(ROOT).run_function(cache_model)
app = modal.App("resin-lfm2", image=image)


def _run_cuda() -> dict[str, object]:
    import time

    import torch

    from resin import Engine
    from resin_models_cuda.lfm2 import Lfm2CudaRunner

    started = time.perf_counter()
    runner = Lfm2CudaRunner(MODEL, revision=REVISION, max_pages=256)
    engine = Engine(
        runner,
        context_length=32768,
        eos_token_id=runner.eos_token_id,
        prefix_cache=True,
    )
    torch.cuda.synchronize()
    loaded = time.perf_counter()

    request = engine.submit(PROMPT, max_new_tokens=8)
    batch_ids = []
    pending_polls = 0
    while engine.has_work():
        if not engine.in_flight_batch_count:
            batch_ids.append(engine.dispatch())
        events = engine.poll()
        if engine.in_flight_batch_count:
            if events:
                raise AssertionError("pending poll returned token events")
            pending_polls += 1
    torch.cuda.synchronize()
    finished = time.perf_counter()
    output = request.take_result()
    report = {
        "device": torch.cuda.get_device_name(),
        "model_load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "output_ids": output,
        "batch_ids": batch_ids,
        "pending_polls": pending_polls,
        "cache_parent_capacity": engine.cache_parent_capacity,
        "cache_free_parent_count": engine.cache_free_parent_count,
        "prefix_cache_stats": engine.prefix_cache_stats,
    }
    engine.close()
    return report


@app.function(gpu="H100!", timeout=20 * 60)
def benchmark() -> dict[str, object]:
    return _run_cuda()


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(benchmark.remote(), indent=2))
