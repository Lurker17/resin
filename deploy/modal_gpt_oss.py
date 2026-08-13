"""Run GPT-OSS on Modal with Resin."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
    install_flashinfer_hash_cache,
    run_cache_prewarm,
    run_cache_reuse_probe,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_CACHE_VOLUME_NAME = "resin-gpt-oss-models-v2"
MODEL_CACHE_VOLUME_VERSION = 2
MODEL_CACHE_MOUNT = "/models/resin-gpt-oss"
MODEL_SPECS = {
    "20b": {
        "repo_id": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
    },
    "120b": {
        "repo_id": "openai/gpt-oss-120b",
        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    },
}
MODEL_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)

PAGE_TOKENS = 128
WINDOW_RESIDENT_BLOCKS = 2


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "MAX_JOBS": "12",
            "CMAKE_BUILD_PARALLEL_LEVEL": "12",
        }
    )
    .uv_pip_install(
        "maturin>=1.9,<2",
        "torch==2.13.0",
        "numpy==2.5.1",
        "huggingface-hub==1.27.0",
        "hf-xet==1.6.0",
        "safetensors==0.8.0",
        "kernels==0.15.2",
        "triton==3.7.1",
        "flashinfer-python==0.6.15.post1",
        "flash-attn-4==4.0.0b22",
        "nvidia-cutlass-dsl==4.6.0.dev0",
        "apache-tvm-ffi==0.1.13.post2",
        "quack-kernels==0.5.3",
        "blake3==1.0.9",
        "renderers==0.1.9",
    )
    .uv_pip_install(
        "flashinfer-cubin==0.6.15.post1",
        extra_index_url="https://flashinfer.ai/whl",
    )
    .uv_pip_install(
        "flashinfer-jit-cache==0.6.15.post1+cu130",
        extra_index_url="https://flashinfer.ai/whl/cu130",
    )
    .add_local_file(ROOT / "deploy/__init__.py", "/root/deploy/__init__.py", copy=True)
    .add_local_file(
        ROOT / "deploy/jit_cache.py", "/root/deploy/jit_cache.py", copy=True
    )
    .add_local_file(ROOT / "Cargo.toml", "/opt/resin/Cargo.toml", copy=True)
    .add_local_file(ROOT / "Cargo.lock", "/opt/resin/Cargo.lock", copy=True)
    .add_local_file(ROOT / "pyproject.toml", "/opt/resin/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/resin/README.md", copy=True)
    .add_local_dir(ROOT / "src", "/opt/resin/src", copy=True)
    .run_commands("python -m pip install --no-deps /opt/resin")
    .env(KERNEL_CACHE_ENV)
)

kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=KERNEL_CACHE_VOLUME_VERSION,
)
model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=MODEL_CACHE_VOLUME_VERSION,
)
single_use_options = (
    {"single_use_containers": True}
    if "single_use_containers" in inspect.signature(modal.App.function).parameters
    else {"max_inputs": 1}
)
app = modal.App(
    "resin-gpt-oss-flat",
    image=image,
    volumes={
        KERNEL_CACHE_MOUNT: kernel_cache_volume,
        MODEL_CACHE_MOUNT: model_cache_volume,
    },
)


def _model_path(model_key: str) -> Path:
    try:
        revision = MODEL_SPECS[model_key]["revision"]
    except KeyError as error:
        raise ValueError("model must be '20b' or '120b'") from error
    return Path(MODEL_CACHE_MOUNT) / model_key / revision


def _model_marker(model_key: str) -> Path:
    return _model_path(model_key) / ".resin-checkpoint.json"


def _checkpoint_inventory(model_key: str) -> dict[str, int]:
    root = _model_path(model_key)
    config = root / "config.json"
    weights = sorted(root.glob("model*.safetensors"))
    if not config.is_file() or config.stat().st_size <= 0 or not weights:
        raise RuntimeError("checkpoint is missing config or model weights")
    files = [config, *weights]
    index = root / "model.safetensors.index.json"
    if len(weights) > 1:
        if not index.is_file() or index.stat().st_size <= 0:
            raise RuntimeError("sharded checkpoint is missing its weight index")
        files.append(index)
    if any(path.stat().st_size <= 0 for path in files):
        raise RuntimeError("checkpoint contains an empty required file")
    return {path.name: path.stat().st_size for path in files}


def _model_is_complete(model_key: str) -> bool:
    try:
        marker = json.loads(_model_marker(model_key).read_text())
        inventory = _checkpoint_inventory(model_key)
        spec = MODEL_SPECS[model_key]
    except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
        return False
    return marker == {
        "schema": 1,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "files": inventory,
    }


def _tokenizer_is_complete(model_key: str) -> bool:
    tokenizer = _model_path(model_key) / "tokenizer.json"
    return tokenizer.is_file() and tokenizer.stat().st_size > 0


def _download_model(model_key: str) -> dict[str, object]:
    """Populate one exact checkpoint on CPU and publish its marker last."""

    _model_path(model_key)  # Validate before touching the Volume.
    model_cache_volume.reload()
    cache_hit = _model_is_complete(model_key) and _tokenizer_is_complete(model_key)
    start = time.perf_counter()
    if not cache_hit:
        from huggingface_hub import snapshot_download

        spec = MODEL_SPECS[model_key]
        target = _model_path(model_key)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=[*MODEL_FILES, *TOKENIZER_FILES],
            local_dir=target,
        )
        marker = {
            "schema": 1,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "files": _checkpoint_inventory(model_key),
        }
        temporary = target / f".resin-checkpoint.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(marker, sort_keys=True))
        temporary.replace(_model_marker(model_key))
        model_cache_volume.commit()
    if not _model_is_complete(model_key) or not _tokenizer_is_complete(model_key):
        raise RuntimeError("checkpoint publication did not complete")
    inventory = _checkpoint_inventory(model_key)
    return {
        "model": model_key,
        "path": str(_model_path(model_key)),
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }


def _require_cached_model(model_key: str) -> tuple[Path, dict[str, object]]:
    """Fail closed unless the complete pinned checkpoint is already present."""

    path = _model_path(model_key)
    model_cache_volume.reload()
    if not _model_is_complete(model_key) or not _tokenizer_is_complete(model_key):
        raise RuntimeError(
            "pinned checkpoint or tokenizer is absent/incomplete; run mode=download first"
        )
    # The GPU process receives a local directory and may never fall back to Hub I/O.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    inventory = _checkpoint_inventory(model_key)
    return path, {
        "volume": MODEL_CACHE_VOLUME_NAME,
        "version": MODEL_CACHE_VOLUME_VERSION,
        "path": str(path),
        "cache_hit": True,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
    }


def _load_renderer(model_path: Path):
    from renderers import GptOssRendererConfig, create_renderer
    from transformers import AutoTokenizer

    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise RuntimeError(
            "tokenizer.json is absent; run mode=download to populate the tokenizer"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    renderer = create_renderer(
        tokenizer,
        GptOssRendererConfig(reasoning_effort="medium", auto_drop_analysis=False),
    )
    return tokenizer, renderer


def _encode_prompt(renderer: Any, prompt: str) -> list[int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    token_ids = renderer.render_ids(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    return [int(token_id) for token_id in token_ids]


def _generation_prefix_ids(renderer: Any, prompt: str) -> list[int]:
    """Return the harmony scaffolding the model continues from.

    The Engine returns sampled completion IDs only.  GPT-OSS starts its
    reasoning channel in the prompt, so the renderer needs that short
    ``assistant/analysis/message`` prefix when parsing a completion back
    into reasoning and final content.  Keeping it separate also lets the
    raw output report show the special tokens without counting them as
    generated tokens.
    """

    messages = [{"role": "user", "content": prompt}]
    conversation_ids = [
        int(token_id)
        for token_id in renderer.render_ids(messages, add_generation_prompt=False)
    ]
    prompt_ids = [
        int(token_id)
        for token_id in renderer.render_ids(messages, add_generation_prompt=True)
    ]
    if prompt_ids[: len(conversation_ids)] != conversation_ids:
        raise RuntimeError("renderer generation prompt is not a conversation suffix")
    return prompt_ids[len(conversation_ids) :]


def _bench_prompt_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    seed = (
        "Resin benchmark paragraph. Measure useful language-model work with "
        "stable token IDs, deterministic scheduling, and no tokenizer work in "
        "the timed GPU interval. "
    )
    token_ids = tokenizer.encode(seed, add_special_tokens=False).ids
    if not token_ids:
        raise RuntimeError("benchmark seed produced no tokenizer IDs")
    repeats = (prompt_tokens + len(token_ids) - 1) // len(token_ids)
    return [int(token_id) for token_id in (token_ids * repeats)[:prompt_tokens]]


def _required_parent_capacity(
    prompt_tokens: int,
    max_new_tokens: int,
) -> int:
    maximum_input = prompt_tokens + max_new_tokens - 1
    history_blocks = (maximum_input + PAGE_TOKENS - 1) // PAGE_TOKENS
    return history_blocks + WINDOW_RESIDENT_BLOCKS


def _required_history_width(prompt_tokens: int, max_new_tokens: int) -> int:
    maximum_input = prompt_tokens + max_new_tokens - 1
    return (maximum_input + PAGE_TOKENS - 1) // PAGE_TOKENS


def _autotune_path(model_key: str, protocol: str) -> Path:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    return (
        Path(KERNEL_CACHE_ROOT)
        / "flashinfer-autotune"
        / (f"{protocol}-{model_key}-{revision}.json")
    )


def _workload_name(model_key: str, protocol: str) -> str:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    return f"{protocol}-{model_key}-{revision}"


def _new_cuda_runner(
    model_path: Path,
    cache_parent_capacity: int,
    cuda_graph_decode_shapes: tuple[tuple[int, int], ...] = (),
):
    from resin.backends.gpt_oss.cuda import GptOssCudaRunner
    from resin.backends.gpt_oss.flashinfer import make_flashinfer_sm90_mxfp4_kernel

    return GptOssCudaRunner(
        model_path,
        cache_parent_capacity=cache_parent_capacity,
        device="cuda",
        moe_kernel=make_flashinfer_sm90_mxfp4_kernel(),
        cuda_graph_decode_shapes=cuda_graph_decode_shapes,
    )


def _run_prompt(
    model_key: str,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
) -> dict[str, object]:
    import torch

    from resin import Engine

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model_path, model_cache = _require_cached_model(model_key)
    tokenizer, renderer = _load_renderer(model_path)
    prompt_ids = _encode_prompt(renderer, prompt)
    generation_prefix_ids = _generation_prefix_ids(renderer, prompt)
    if len(prompt_ids) + max_new_tokens - 1 > context_length:
        raise ValueError("prompt plus max_new_tokens exceeds context_length")
    cache_parent_capacity = _required_parent_capacity(len(prompt_ids), max_new_tokens)
    runner = _new_cuda_runner(
        model_path,
        cache_parent_capacity,
        cuda_graph_decode_shapes=(
            (1, _required_history_width(len(prompt_ids), max_new_tokens)),
        ),
    )
    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=context_length,
        eos_token_id=runner.eos_token_id,
    )
    output_events = []
    first_token_time = None
    try:
        # Warm kernels and capture the configured B=1 decode graph before timing.
        warmup = engine.submit(prompt_ids, max_new_tokens=2)
        while engine.has_work():
            engine.step()
        warmup.take_result()
        torch.cuda.synchronize()

        start = time.perf_counter()
        request = engine.submit(prompt_ids, max_new_tokens=max_new_tokens)
        while engine.has_work():
            events = engine.step()
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
            output_events.extend(events)
        torch.cuda.synchronize()
        output_ids = request.take_result()
        rendered_output_ids = generation_prefix_ids + output_ids
        parsed = renderer.parse_response(rendered_output_ids)
        elapsed_s = time.perf_counter() - start
        return {
            "model": model_key,
            "device": torch.cuda.get_device_name(),
            "checkpoint": model_cache,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "output_ids": output_ids,
            "text": tokenizer.decode(rendered_output_ids, skip_special_tokens=False),
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": [tool_call.__dict__ for tool_call in parsed.tool_calls],
            "cuda_graph": runner.cuda_graph_stats,
            "time_to_first_token_ms": (
                None if first_token_time is None else (first_token_time - start) * 1000
            ),
            "elapsed_ms": elapsed_s * 1000,
            "output_tokens_per_second": len(output_ids) / elapsed_s,
            "event_count": len(output_events),
        }
    finally:
        if not engine.closed:
            engine.close()


def _bench_cuda(
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
) -> dict[str, object]:
    import torch

    from resin import Engine

    tokenizer, _renderer = _load_renderer(model_path)
    prompt_ids = _bench_prompt_ids(tokenizer, prompt_tokens)
    cache_parent_capacity = _required_parent_capacity(
        prompt_tokens, max_new_tokens
    ) * max(batch_sizes)
    graph_width = _required_history_width(prompt_tokens, max_new_tokens)
    runner = _new_cuda_runner(
        model_path,
        cache_parent_capacity,
        cuda_graph_decode_shapes=tuple(
            (batch_size, graph_width) for batch_size in batch_sizes
        ),
    )
    engine = Engine(
        runner,
        max_running_requests=max(batch_sizes),
        context_length=context_length,
        eos_token_id=runner.eos_token_id,
    )

    def execute_batch(batch_size: int) -> dict[str, object]:
        requests = [
            engine.submit(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                batch_key=prompt_tokens,
            )
            for _ in range(batch_size)
        ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        first_token_time = None
        event_count = 0
        while engine.has_work():
            events = engine.step()
            event_count += len(events)
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
        torch.cuda.synchronize()
        end = time.perf_counter()
        for request in requests:
            if len(request.take_result()) != max_new_tokens:
                raise RuntimeError("benchmark request returned the wrong token count")
        elapsed_s = end - start
        first_s = elapsed_s if first_token_time is None else first_token_time - start
        decode_s = max(elapsed_s - first_s, 1e-9)
        return {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "output_tokens": max_new_tokens,
            "time_to_first_token_ms": first_s * 1000,
            "total_ms": elapsed_s * 1000,
            "per_user_output_tok_s": max_new_tokens / elapsed_s,
            "aggregate_output_tok_s": batch_size * max_new_tokens / elapsed_s,
            "per_user_decode_tok_s": (max_new_tokens - 1) / decode_s,
            "event_count": event_count,
        }

    try:
        # Warm each batch shape once so the reported rows represent steady
        # execution rather than first-use allocation and kernel setup.
        for batch_size in batch_sizes:
            execute_batch(batch_size)
        rows = [execute_batch(batch_size) for batch_size in batch_sizes]
        return {
            "protocol": "gpt-oss-throughput-v1",
            "model": model_key,
            "device": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "checkpoint": model_cache,
            "cache_parent_capacity": cache_parent_capacity,
            "rows": rows,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    finally:
        if not engine.closed:
            engine.close()


def _run_bench(
    model_key: str,
    cache_mode: str,
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
) -> dict[str, object]:
    if model_key not in MODEL_SPECS:
        raise ValueError("model must be '20b' or '120b'")
    if cache_mode not in {"prewarm", "verify"}:
        raise ValueError("cache_mode must be 'prewarm' or 'verify'")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise ValueError("prompt_tokens must be an integer")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise ValueError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if prompt_tokens + max_new_tokens - 1 > context_length:
        raise ValueError("prompt_tokens plus max_new_tokens exceeds context_length")
    if not batch_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in batch_sizes
    ):
        raise ValueError("batch_sizes must contain positive integers")
    if tuple(sorted(set(batch_sizes))) != batch_sizes:
        raise ValueError("batch_sizes must be sorted and contain no duplicates")

    model_path, model_cache = _require_cached_model(model_key)
    protocol = "gpt-oss-throughput-v1"

    def workload() -> dict[str, object]:
        install_flashinfer_hash_cache()

        import flashinfer
        from flashinfer.autotuner import AutoTuner

        path = _autotune_path(model_key, protocol)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_loaded = None
        if path.exists():
            cache_loaded = AutoTuner.get().load_configs(str(path))
            if not cache_loaded:
                raise RuntimeError(f"FlashInfer rejected its saved tactics: {path}")
        elif cache_mode == "verify":
            raise RuntimeError("saved FlashInfer tactics are absent; prewarm first")
        tune_missing = cache_mode == "prewarm"
        tuning_buckets = tuple(sorted({1, *batch_sizes, 128}))
        with flashinfer.autotune(
            tune_missing,
            cache=str(path) if tune_missing else None,
            tuning_buckets=tuning_buckets,
            round_up=True,
        ):
            report = _bench_cuda(
                model_key,
                model_path,
                model_cache,
                prompt_tokens,
                max_new_tokens,
                batch_sizes,
                context_length,
            )
        if not path.is_file():
            raise RuntimeError("FlashInfer did not publish its benchmark tactic file")
        report["flashinfer_autotune"] = {
            "mode": "tune_missing" if tune_missing else "immutable_reuse",
            "cache_loaded": cache_loaded,
            "path": str(path),
            "buckets": list(tuning_buckets),
        }
        return report

    if cache_mode == "prewarm":
        return run_cache_prewarm(
            kernel_cache_volume, _workload_name(model_key, protocol), workload
        )
    report = run_cache_reuse_probe(
        kernel_cache_volume, _workload_name(model_key, protocol), workload
    )
    cache = report["kernel_cache"]
    if (
        cache["cache_hit"] is not True
        or cache["stable_artifact_changes"]
        or cache["flashinfer_hash_cache_misses"] != 0
    ):
        raise AssertionError("fresh-container benchmark reuse was not immutable")
    return report


@app.function(timeout=4 * 60 * 60, cpu=4, memory=8 * 1024)
def cache_checkpoint(model_key: str) -> dict[str, object]:
    return _download_model(model_key)


@app.function(
    gpu="H200",
    cpu=16,
    memory=256 * 1024,
    timeout=4 * 60 * 60,
    **single_use_options,
)
def run_model(
    model_key: str,
    prompt: str,
    max_new_tokens: int = 256,
    context_length: int = 32768,
) -> dict[str, object]:
    return _run_prompt(model_key, prompt, max_new_tokens, context_length)


@app.function(
    gpu="H200",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    **single_use_options,
)
def bench_model(
    model_key: str,
    cache_mode: str = "prewarm",
    prompt_tokens: int = 8192,
    max_new_tokens: int = 256,
    batch_sizes: tuple[int, ...] = (1, 2, 8, 16, 32),
    context_length: int = 32768,
) -> dict[str, object]:
    return _run_bench(
        model_key,
        cache_mode,
        prompt_tokens,
        max_new_tokens,
        batch_sizes,
        context_length,
    )


@app.local_entrypoint()
def main(
    mode: str = "run",
    model: str = "120b",
    cache_mode: str = "prewarm",
    prompt: str = "Explain why batching improves language-model throughput.",
    max_new_tokens: int = 256,
    prompt_tokens: int = 8192,
    batch_sizes: str = "1,2,8,16,32",
    context_length: int = 32768,
) -> None:
    """Download a checkpoint, run one prompt, or benchmark throughput."""

    if mode == "download":
        report = cache_checkpoint.remote(model)
    elif mode == "run":
        report = run_model.remote(model, prompt, max_new_tokens, context_length)
    elif mode == "bench":
        try:
            parsed_batch_sizes = tuple(int(value) for value in batch_sizes.split(","))
        except ValueError as error:
            raise ValueError(
                "batch_sizes must be a comma-separated integer list"
            ) from error
        report = bench_model.remote(
            model,
            cache_mode,
            prompt_tokens,
            max_new_tokens,
            parsed_batch_sizes,
            context_length,
        )
    else:
        raise ValueError("mode must be 'download', 'run', or 'bench'")
    print(json.dumps(report, indent=2, sort_keys=True))
