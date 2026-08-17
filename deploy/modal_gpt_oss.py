"""Run GPT-OSS on Modal with Resin."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import TypedDict, cast

import modal
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from renderers import GptOssRendererConfig, Message, Renderer, create_renderer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_ROOT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
    flashinfer_autotune_context,
    install_flashinfer_hash_cache,
    prepare_kernel_cache,
    run_cache_prewarm,
    run_cache_reuse_probe,
    run_cached_workload,
)
from deploy.modal_image import build_modal_image
from resin import Engine, SamplingParams, SchedulerConfig, TokenEvent
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    CudaGraphStats,
    default_decode_graph_shapes,
)
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.parallel import (
    TensorParallelRunner,
    follow_tensor_parallel_runner,
    monitor_process_context,
)
from resin.server import HttpServer, RendererCodec

ROOT = Path(__file__).resolve().parents[1]

MODEL_CACHE_VOLUME_NAME = "resin-gpt-oss-models-v2"
MODEL_CACHE_VOLUME_VERSION = 2
MODEL_CACHE_MOUNT = "/models/resin-gpt-oss"
PROFILE_VOLUME_NAME = "resin-gpt-oss-profiles"
PROFILE_MOUNT = "/profiles/resin-gpt-oss"


class ModelSpec(TypedDict):
    repo_id: str
    revision: str


class KernelCacheVerification(TypedDict):
    cache_hit: bool
    stable_artifact_changes: list[str]
    flashinfer_hash_cache_misses: int


MODEL_SPECS: dict[str, ModelSpec] = {
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

image = build_modal_image(
    ROOT,
    env={"MAX_JOBS": "12", "CMAKE_BUILD_PARALLEL_LEVEL": "12"},
).env(KERNEL_CACHE_ENV)

with image.imports():
    from resin_models_cuda.gpt_oss import GptOssCudaRunner
    from resin_models_cuda.gpt_oss.flashinfer import make_flashinfer_sm90_mxfp4_kernel

# Mirror the runner's frozen paged-cache geometry without reading names from
# ``image.imports()`` while Modal is still parsing this file locally.
PAGE_TOKENS = 128
WINDOW_RESIDENT_BLOCKS = 65

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
profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
app = modal.App(
    "resin-gpt-oss-flat",
    image=image,
    volumes={
        KERNEL_CACHE_MOUNT: kernel_cache_volume,
        MODEL_CACHE_MOUNT: model_cache_volume,
        PROFILE_MOUNT: profile_volume,
    },
)

_http_server_processes: list[object] = []


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


def _load_renderer(model_path: Path) -> tuple[PreTrainedTokenizerBase, Renderer]:
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise RuntimeError(
            "tokenizer.json is absent; run mode=download to populate the tokenizer"
        )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    renderer = create_renderer(
        tokenizer,
        GptOssRendererConfig(reasoning_effort="medium", auto_drop_analysis=False),
    )
    return tokenizer, renderer


def _encode_prompt(renderer: Renderer, prompt: str) -> list[int]:
    messages: list[Message] = [{"role": "user", "content": prompt}]
    token_ids = renderer.render_ids(messages, add_generation_prompt=True)
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    return [int(token_id) for token_id in token_ids]


def _generation_prefix_ids(renderer: Renderer, prompt: str) -> list[int]:
    """Return the harmony scaffolding the model continues from.

    The Engine returns sampled completion IDs only.  GPT-OSS starts its
    reasoning channel in the prompt, so the renderer needs that short
    ``assistant/analysis/message`` prefix when parsing a completion back
    into reasoning and final content.  Keeping it separate also lets the
    raw output report show the special tokens without counting them as
    generated tokens.
    """

    messages: list[Message] = [{"role": "user", "content": prompt}]
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


def _bench_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, prompt_tokens: int
) -> list[int]:
    seed = (
        "Resin benchmark paragraph. Measure useful language-model work with "
        "stable token IDs, deterministic scheduling, and no tokenizer work in "
        "the timed GPU interval. "
    )
    token_ids = tokenizer.encode(seed, add_special_tokens=False)
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


def _autotune_path(
    model_key: str,
    protocol: str,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> Path:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    parallel = (
        ""
        if tensor_parallel_size == 1
        else f"-tp{tensor_parallel_size}-rank{tensor_parallel_rank}"
    )
    return (
        Path(KERNEL_CACHE_ROOT)
        / "flashinfer-autotune"
        / (f"{protocol}-{model_key}-{revision}{parallel}.json")
    )


def _workload_name(model_key: str, protocol: str, tensor_parallel_size: int) -> str:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    parallel = "" if tensor_parallel_size == 1 else f"-tp{tensor_parallel_size}"
    return f"{protocol}-{model_key}-{revision}{parallel}"


def _new_cuda_runner(
    model_path: Path,
    cache_parent_capacity: int | None,
    tensor_parallel: TensorParallelContext,
    cuda_graph_decode_shapes: tuple[tuple[int, int], ...] = (),
) -> GptOssCudaRunner:
    moe_kernel = make_flashinfer_sm90_mxfp4_kernel()
    if tensor_parallel.rank == 0:
        print(
            f"[model-load] Loading TP={tensor_parallel.size} model from "
            f"{model_path}...",
            flush=True,
        )
    model_load_start = time.perf_counter()
    runner = GptOssCudaRunner(
        model_path,
        cache_parent_capacity=cache_parent_capacity,
        device=f"cuda:{tensor_parallel.rank}",
        moe_kernel=moe_kernel,
        cuda_graph_decode_shapes=cuda_graph_decode_shapes,
        tensor_parallel=tensor_parallel,
    )
    torch.cuda.synchronize(runner.device)
    tensor_parallel.barrier()
    model_load_elapsed_s = time.perf_counter() - model_load_start
    if tensor_parallel.rank == 0:
        print(
            f"[allreduce] backend={runner.allreduce_backend}",
            flush=True,
        )
        print(
            f"[model-load] TP={tensor_parallel.size} model loaded in "
            f"{model_load_elapsed_s:.2f} seconds",
            flush=True,
        )
    return runner


def _capture_cuda_graphs(runner: GptOssCudaRunner) -> CudaGraphStats:
    stats = runner.cuda_graph_stats
    shapes = stats["configured_shapes"]
    if not shapes:
        return stats
    if runner.tensor_parallel_rank == 0:
        print(
            f"[cuda-graphs] Capturing {len(shapes)} TP={runner.tensor_parallel_size} "
            "decode shape(s)...",
            flush=True,
        )
    capture_start = time.perf_counter()
    stats = runner.capture_cuda_graphs()
    runner.tensor_parallel.barrier()
    capture_elapsed_s = time.perf_counter() - capture_start
    if runner.tensor_parallel_rank == 0:
        print(
            f"[cuda-graphs] Captured {stats['capture_count']} shape(s) in "
            f"{capture_elapsed_s:.2f} seconds",
            flush=True,
        )
    return stats


def _run_prompt(
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    schedule_conservativeness: float,
    sampling_params: SamplingParams,
    tensor_parallel: TensorParallelContext,
) -> dict[str, object]:
    if len(prompt_ids) + max_new_tokens - 1 > context_length:
        raise ValueError("prompt plus max_new_tokens exceeds context_length")
    cache_parent_capacity = _required_parent_capacity(len(prompt_ids), max_new_tokens)
    runner = _new_cuda_runner(
        model_path,
        cache_parent_capacity,
        tensor_parallel,
        cuda_graph_decode_shapes=(
            (1, _required_history_width(len(prompt_ids), max_new_tokens)),
        ),
    )
    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=context_length,
        scheduler_config=SchedulerConfig(
            schedule_conservativeness=schedule_conservativeness
        ),
        eos_token_id=runner.eos_token_id,
    )
    output_events: list[TokenEvent] = []
    first_token_time: float | None = None
    try:
        # Graph capture is an explicit initialization phase and never consumes
        # the first request batch.
        _capture_cuda_graphs(runner)
        # Warm the real prefill path before timing request execution.
        warmup = engine.submit(
            prompt_ids,
            max_new_tokens=2,
            sampling_params=sampling_params,
        )
        while engine.has_work():
            engine.step()
        warmup.take_result()
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()

        start = time.perf_counter()
        request = engine.submit(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )
        while engine.has_work():
            events = engine.step()
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
            output_events.extend(events)
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()
        output_ids = request.take_result()
        if tensor_parallel.enabled:
            output_tensor = torch.full(
                (max_new_tokens,),
                -1,
                dtype=torch.int64,
                device=runner.device,
            )
            output_tensor[: len(output_ids)] = torch.tensor(
                output_ids,
                dtype=torch.int64,
                device=runner.device,
            )
            gathered_outputs = torch.empty(
                tensor_parallel.size * max_new_tokens,
                dtype=torch.int64,
                device=runner.device,
            )
            assert tensor_parallel.process_group is not None
            dist.all_gather_single(
                gathered_outputs,
                output_tensor,
                group=tensor_parallel.process_group,
            )
            output_rows = gathered_outputs.view(
                tensor_parallel.size,
                max_new_tokens,
            )
            if any(
                not torch.equal(output_rows[0], output_rows[rank])
                for rank in range(1, tensor_parallel.size)
            ):
                raise RuntimeError("tensor-parallel ranks produced different tokens")
            if tensor_parallel.rank == 0:
                print(
                    f"[tensor-parallel] Verified output agreement across "
                    f"{tensor_parallel.size} ranks",
                    flush=True,
                )
        elapsed_s = time.perf_counter() - start
        return {
            "model": model_key,
            "device": torch.cuda.get_device_name(),
            "tensor_parallel_size": tensor_parallel.size,
            "allreduce_backend": runner.allreduce_backend,
            "checkpoint": model_cache,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "sampling": {
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "seed": sampling_params.seed,
            },
            "output_tokens": len(output_ids),
            "_output_ids": output_ids,
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


def _render_prompt_report(
    report: dict[str, object],
    tokenizer: PreTrainedTokenizerBase,
    renderer: Renderer,
    generation_prefix_ids: list[int],
) -> dict[str, object]:
    raw_output_ids = report.pop("_output_ids", None)
    output_ids = cast(list[int], raw_output_ids)
    rendered_output_ids = generation_prefix_ids + output_ids
    parsed = renderer.parse_response(rendered_output_ids)
    report.update(
        {
            "text": tokenizer.decode(
                rendered_output_ids,
                skip_special_tokens=False,
            ),
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": [asdict(tool_call) for tool_call in parsed.tool_calls],
        }
    )
    return report


def _bench_cuda(
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
    schedule_conservativeness: float,
    profile_decode_steps: int,
    tensor_parallel: TensorParallelContext,
) -> dict[str, object]:
    if len(prompt_ids) != prompt_tokens:
        raise ValueError("benchmark prompt IDs do not match prompt_tokens")
    cache_parent_capacity = _required_parent_capacity(
        prompt_tokens, max_new_tokens
    ) * max(batch_sizes)
    graph_width = _required_history_width(prompt_tokens, max_new_tokens)
    runner = _new_cuda_runner(
        model_path,
        cache_parent_capacity,
        tensor_parallel,
        cuda_graph_decode_shapes=tuple(
            (batch_size, graph_width) for batch_size in batch_sizes
        ),
    )
    engine = Engine(
        runner,
        max_running_requests=max(batch_sizes),
        context_length=context_length,
        scheduler_config=SchedulerConfig(
            schedule_conservativeness=schedule_conservativeness
        ),
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
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()
        start = time.perf_counter()
        first_token_time = None
        event_count = 0
        while engine.has_work():
            events = engine.step()
            event_count += len(events)
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()
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

    def profile_decode(batch_size: int) -> dict[str, object]:
        profiled_steps = min(profile_decode_steps, max_new_tokens - 1)
        if profiled_steps <= 0:
            raise ValueError("profile_decode_steps requires at least two output tokens")
        requests = [
            engine.submit(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                batch_key=prompt_tokens,
            )
            for _ in range(batch_size)
        ]
        # Advance through every prefill chunk and stop immediately after the
        # first sampled token. Every profiled step after this point is a
        # fixed-shape CUDA-graph decode replay.
        while engine.has_work():
            if engine.step():
                break
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()

        if tensor_parallel.rank == 0:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as cuda_profile:
                for _ in range(profiled_steps):
                    engine.step()
                torch.cuda.synchronize(runner.device)
        else:
            for _ in range(profiled_steps):
                engine.step()
            torch.cuda.synchronize(runner.device)
            cuda_profile = None
        tensor_parallel.barrier()

        while engine.has_work():
            engine.step()
        for request in requests:
            if len(request.take_result()) != max_new_tokens:
                raise RuntimeError("profile request returned the wrong token count")
        if cuda_profile is None:
            return {}

        trace_name = (
            f"gpt-oss-{model_key}-tp{tensor_parallel.size}-batch{batch_size}-"
            f"decode-{time.time_ns()}.perfetto-trace.json.gz"
        )
        trace_path = Path(PROFILE_MOUNT) / trace_name
        cuda_profile.export_chrome_trace(str(trace_path))

        events = sorted(
            (
                event
                for event in cuda_profile.key_averages()
                if event.self_device_time_total > 0
            ),
            key=lambda event: event.self_device_time_total,
            reverse=True,
        )
        total_cuda_time_us = sum(event.self_device_time_total for event in events)
        return {
            "batch_size": batch_size,
            "profiled_decode_steps": profiled_steps,
            "perfetto_trace": trace_name,
            "total_cuda_time_us": total_cuda_time_us,
            "cuda_time_per_step_us": total_cuda_time_us / profiled_steps,
            "top_cuda_kernels": [
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cuda_time_us": event.self_device_time_total,
                    "cuda_time_per_call_us": (
                        event.self_device_time_total / event.count
                    ),
                    "cuda_time_percent": (
                        100.0 * event.self_device_time_total / total_cuda_time_us
                    ),
                }
                for event in events[:50]
            ],
        }

    try:
        _capture_cuda_graphs(runner)
        # Warm each batch shape once so the reported rows represent steady
        # execution rather than first-use allocation and kernel setup.
        for batch_size in batch_sizes:
            execute_batch(batch_size)
        rows = [execute_batch(batch_size) for batch_size in batch_sizes]
        report: dict[str, object] = {
            "protocol": "gpt-oss-throughput-v1",
            "model": model_key,
            "device": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "tensor_parallel_size": tensor_parallel.size,
            "allreduce_backend": runner.allreduce_backend,
            "checkpoint": model_cache,
            "cache_parent_capacity": cache_parent_capacity,
            "rows": rows,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
        if profile_decode_steps > 0:
            report["decode_profile"] = profile_decode(batch_sizes[0])
        return report
    finally:
        if not engine.closed:
            engine.close()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _initialize_tensor_parallel(
    rank: int, tensor_parallel_size: int, master_port: int
) -> TensorParallelContext:
    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(
            f"TP={tensor_parallel_size} requires {tensor_parallel_size} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if tensor_parallel_size == 1:
        return TensorParallelContext()
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=tensor_parallel_size,
        timeout=timedelta(minutes=30),
        device_id=device,
    )
    process_group = dist.group.WORLD
    if process_group is None:
        raise RuntimeError("PyTorch did not create the tensor-parallel process group")
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=tensor_parallel_size),
        process_group,
    )


def _run_prompt_rank(
    rank: int,
    tensor_parallel_size: int,
    master_port: int,
    report_path: Path,
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    schedule_conservativeness: float,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(
        rank, tensor_parallel_size, master_port
    )
    install_flashinfer_hash_cache()
    report = _run_prompt(
        model_key,
        model_path,
        model_cache,
        prompt_ids,
        max_new_tokens,
        context_length,
        schedule_conservativeness,
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        ),
        tensor_parallel,
    )
    tensor_parallel.barrier()
    if rank == 0:
        report_path.write_text(json.dumps(report))
    # Decode graphs own NCCL work. These are dedicated single-use workers, so
    # process exit is the communicator lifetime boundary; explicit group
    # destruction can wait forever on graph-owned communicators.
    os._exit(0)


def _launch_tensor_parallel_prompt(
    tensor_parallel_size: int,
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    schedule_conservativeness: float,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
) -> dict[str, object]:
    master_port = _free_local_port()
    with tempfile.TemporaryDirectory(prefix="resin-tp-") as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_prompt_rank,
            args=(
                tensor_parallel_size,
                master_port,
                report_path,
                model_key,
                model_path,
                model_cache,
                prompt_ids,
                max_new_tokens,
                context_length,
                schedule_conservativeness,
                temperature,
                top_p,
                top_k,
                seed,
            ),
            nprocs=tensor_parallel_size,
            join=True,
        )
        return json.loads(report_path.read_text())


def _bench_rank_workload(
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    cache_mode: str,
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
    schedule_conservativeness: float,
    profile_decode_steps: int,
    tensor_parallel: TensorParallelContext,
) -> dict[str, object]:
    install_flashinfer_hash_cache()
    protocol = "gpt-oss-throughput-v1"
    path = _autotune_path(
        model_key,
        protocol,
        tensor_parallel.size,
        tensor_parallel.rank,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tune_missing = cache_mode == "prewarm"
    tuning_buckets = tuple(sorted({1, *batch_sizes, 128}))
    with flashinfer_autotune_context(
        path,
        tune_missing=tune_missing,
        tuning_buckets=tuning_buckets,
    ) as cache_loaded:
        report = _bench_cuda(
            model_key,
            model_path,
            model_cache,
            prompt_ids,
            prompt_tokens,
            max_new_tokens,
            batch_sizes,
            context_length,
            schedule_conservativeness,
            profile_decode_steps,
            tensor_parallel,
        )
    if not path.is_file():
        raise RuntimeError("FlashInfer did not publish its benchmark tactic file")
    report["flashinfer_autotune"] = {
        "mode": "tune_missing" if tune_missing else "immutable_reuse",
        "cache_loaded": cache_loaded,
        "path": str(path),
        "rank_paths": [
            str(
                _autotune_path(
                    model_key,
                    protocol,
                    tensor_parallel.size,
                    rank,
                )
            )
            for rank in range(tensor_parallel.size)
        ],
        "buckets": list(tuning_buckets),
    }
    return report


def _run_bench_rank(
    rank: int,
    tensor_parallel_size: int,
    master_port: int,
    report_path: Path,
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    cache_mode: str,
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
    schedule_conservativeness: float,
    profile_decode_steps: int,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(
        rank, tensor_parallel_size, master_port
    )
    report = _bench_rank_workload(
        model_key,
        model_path,
        model_cache,
        prompt_ids,
        cache_mode,
        prompt_tokens,
        max_new_tokens,
        batch_sizes,
        context_length,
        schedule_conservativeness,
        profile_decode_steps,
        tensor_parallel,
    )
    tensor_parallel.barrier()
    if rank == 0:
        report_path.write_text(json.dumps(report))
    os._exit(0)


def _launch_tensor_parallel_bench(
    tensor_parallel_size: int,
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    prompt_ids: list[int],
    cache_mode: str,
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
    schedule_conservativeness: float,
    profile_decode_steps: int,
) -> dict[str, object]:
    master_port = _free_local_port()
    with tempfile.TemporaryDirectory(prefix="resin-tp-") as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_bench_rank,
            args=(
                tensor_parallel_size,
                master_port,
                report_path,
                model_key,
                model_path,
                model_cache,
                prompt_ids,
                cache_mode,
                prompt_tokens,
                max_new_tokens,
                batch_sizes,
                context_length,
                schedule_conservativeness,
                profile_decode_steps,
            ),
            nprocs=tensor_parallel_size,
            join=True,
        )
        return json.loads(report_path.read_text())


def _run_bench(
    model_key: str,
    cache_mode: str,
    prompt_tokens: int,
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    context_length: int,
    schedule_conservativeness: float,
    profile_decode_steps: int,
) -> dict[str, object]:
    if model_key not in MODEL_SPECS:
        raise ValueError("model must be '20b' or '120b'")
    if cache_mode not in {"prewarm", "verify"}:
        raise ValueError("cache_mode must be 'prewarm' or 'verify'")
    if prompt_tokens + max_new_tokens - 1 > context_length:
        raise ValueError("prompt_tokens plus max_new_tokens exceeds context_length")
    if tuple(sorted(set(batch_sizes))) != batch_sizes:
        raise ValueError("batch_sizes must be sorted and contain no duplicates")
    if profile_decode_steps < 0:
        raise ValueError("profile_decode_steps cannot be negative")

    model_path, model_cache = _require_cached_model(model_key)
    tokenizer, _renderer = _load_renderer(model_path)
    prompt_ids = _bench_prompt_ids(tokenizer, prompt_tokens)
    protocol = "gpt-oss-throughput-v1"

    def workload() -> dict[str, object]:
        return _launch_tensor_parallel_bench(
            2,
            model_key,
            model_path,
            model_cache,
            prompt_ids,
            cache_mode,
            prompt_tokens,
            max_new_tokens,
            batch_sizes,
            context_length,
            schedule_conservativeness,
            profile_decode_steps,
        )

    if cache_mode == "prewarm":
        report = run_cache_prewarm(
            kernel_cache_volume,
            _workload_name(model_key, protocol, 2),
            workload,
        )
    else:
        report = run_cache_reuse_probe(
            kernel_cache_volume,
            _workload_name(model_key, protocol, 2),
            workload,
        )
        cache = cast(KernelCacheVerification, report["kernel_cache"])
        if (
            cache["cache_hit"] is not True
            or cache["stable_artifact_changes"]
            or cache["flashinfer_hash_cache_misses"] != 0
        ):
            raise AssertionError("fresh-container benchmark reuse was not immutable")
    if profile_decode_steps > 0:
        profile_volume.commit()
    return report


@app.function(timeout=4 * 60 * 60, cpu=4, memory=8 * 1024)
def cache_checkpoint(model_key: str) -> dict[str, object]:
    return _download_model(model_key)


def _serve_http_rank(
    rank: int,
    master_port: int,
    model_key: str,
    model_path: Path,
    tensor_parallel_size: int,
) -> None:
    """Run one model rank; rank zero additionally owns the HTTP listener."""

    tensor_parallel = _initialize_tensor_parallel(
        rank, tensor_parallel_size, master_port
    )
    control_group = (
        dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
        if tensor_parallel.enabled
        else None
    )
    install_flashinfer_hash_cache()
    runner = _new_cuda_runner(
        model_path,
        None,
        tensor_parallel,
        cuda_graph_decode_shapes=default_decode_graph_shapes(
            _required_history_width(131072, 1),
            speculative=False,
            max_batch_size=DEFAULT_MAX_NUM_SEQS,
        ),
    )
    if rank == 0:
        cache = runner.cache_capacity_plan
        print(
            "[http-startup] fungible KV cache "
            f"parents={runner.cache_parent_capacity:,}; "
            f"parent={cache.bytes_per_parent / 2**20:.2f} MiB; "
            f"allocation={cache.allocation_bytes / 2**30:.2f} GiB; "
            f"gpu_memory_utilization={cache.gpu_memory_utilization:.2f}",
            flush=True,
        )
    _capture_cuda_graphs(runner)

    if rank != 0:
        assert control_group is not None
        print("[http-startup] follower ready for rank-zero commands", flush=True)
        follow_tensor_parallel_runner(runner, control_group, dist)
        os._exit(0)

    print("[http-startup] loading tokenizer and renderer...", flush=True)
    tokenizer, renderer = _load_renderer(model_path)
    print("[http-startup] tokenizer and renderer loaded", flush=True)
    coordinated_runner = (
        TensorParallelRunner(runner, control_group, dist)
        if control_group is not None
        else None
    )
    engine_runner = coordinated_runner or runner
    print("[http-startup] creating engine and binding shared cache...", flush=True)
    engine = Engine(
        engine_runner,
        max_running_requests=DEFAULT_MAX_NUM_SEQS,
        context_length=131072,
        eos_token_id=runner.eos_token_id,
        prefix_cache=True,
    )
    print("[http-startup] engine created; running warmup...", flush=True)
    warmup = engine.submit(
        _encode_prompt(renderer, "Reply with one word."),
        max_new_tokens=2,
        sampling_params=SamplingParams(temperature=0.0),
    )
    while engine.has_work():
        engine.step()
    warmup.take_result()
    torch.cuda.synchronize(runner.device)
    print("[http-startup] warmup complete; starting HTTP server", flush=True)
    codec = RendererCodec(
        renderer,
        tokenizer=tokenizer,
        include_generation_prefix_in_parser=True,
    )
    server = HttpServer(engine, codec=codec, model=MODEL_SPECS[model_key]["repo_id"])
    try:
        server.serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()
        if coordinated_runner is not None:
            coordinated_runner.close_followers()
    os._exit(0)


def _start_http_server(model_key: str) -> None:
    """Spawn the model-specific ranks backing one GPT-OSS HTTP deployment."""

    prepare_kernel_cache(kernel_cache_volume)
    model_path, _ = _require_cached_model(model_key)
    tensor_parallel_size = 1 if model_key == "20b" else 2
    process_context = torch.multiprocessing.spawn(
        _serve_http_rank,
        args=(
            _free_local_port(),
            model_key,
            model_path,
            tensor_parallel_size,
        ),
        nprocs=tensor_parallel_size,
        join=False,
    )
    _http_server_processes.append(process_context)
    monitor_process_context(process_context, f"gpt-oss-{model_key}-http")


@app.function(
    gpu="H200:1",
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
    label="gpt-oss-20b",
    requires_proxy_auth=False,
)
def serve_gpt_oss_20b() -> None:
    _start_http_server("20b")


@app.function(
    gpu="H200:2",
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
    label="gpt-oss-120b",
    requires_proxy_auth=False,
)
def serve_gpt_oss_120b() -> None:
    _start_http_server("120b")


def _run_model(
    model_key: str,
    prompt: str,
    max_new_tokens: int = 16384,
    context_length: int = 131072,
    schedule_conservativeness: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    def workload() -> dict[str, object]:
        # Volume v2 does not preserve the nanosecond mtimes Ninja uses to
        # validate FlashInfer JIT outputs. Install Resin's content-addressed
        # loader before importing GPU packages so a fresh single-use container
        # can load the cached shared library directly instead of rebuilding it.
        model_path, model_cache = _require_cached_model(model_key)
        tokenizer, renderer = _load_renderer(model_path)
        prompt_ids = _encode_prompt(renderer, prompt)
        generation_prefix_ids = _generation_prefix_ids(renderer, prompt)
        report = _launch_tensor_parallel_prompt(
            2,
            model_key,
            model_path,
            model_cache,
            prompt_ids,
            max_new_tokens,
            context_length,
            schedule_conservativeness,
            temperature,
            top_p,
            top_k,
            seed,
        )
        return _render_prompt_report(
            report,
            tokenizer,
            renderer,
            generation_prefix_ids,
        )

    return run_cached_workload(kernel_cache_volume, workload)


@app.function(
    gpu="H200:2",
    cpu=16,
    memory=256 * 1024,
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def run_model(
    model_key: str,
    prompt: str,
    max_new_tokens: int = 16384,
    context_length: int = 131072,
    schedule_conservativeness: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    return _run_model(
        model_key,
        prompt,
        max_new_tokens,
        context_length,
        schedule_conservativeness,
        temperature,
        top_p,
        top_k,
        seed,
    )


@app.function(
    gpu="H200:2",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    single_use_containers=True,
)
def bench_model(
    model_key: str,
    cache_mode: str = "prewarm",
    prompt_tokens: int = 8192,
    max_new_tokens: int = 256,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    context_length: int = 32768,
    schedule_conservativeness: float = 1.0,
    profile_decode_steps: int = 0,
) -> dict[str, object]:
    return _run_bench(
        model_key,
        cache_mode,
        prompt_tokens,
        max_new_tokens,
        batch_sizes,
        context_length,
        schedule_conservativeness,
        profile_decode_steps,
    )


@app.local_entrypoint()
def main(
    mode: str = "run",
    model: str = "120b",
    cache_mode: str = "prewarm",
    prompt: str = "Explain why batching improves language-model throughput.",
    max_new_tokens: int = 16384,
    prompt_tokens: int = 8192,
    batch_sizes: str = "1,2,8,16,32",
    context_length: int = 131072,
    schedule_conservativeness: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    profile_decode_steps: int = 0,
) -> None:
    """Download a checkpoint, run one prompt, or benchmark throughput."""

    if mode == "download":
        report = cache_checkpoint.remote(model)
    elif mode == "run":
        report = run_model.remote(
            model,
            prompt,
            max_new_tokens,
            context_length,
            schedule_conservativeness,
            temperature,
            top_p,
            top_k,
            seed,
        )
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
            schedule_conservativeness,
            profile_decode_steps,
        )
    else:
        raise ValueError("mode must be 'download', 'run', or 'bench'")
    print(json.dumps(report, indent=2, sort_keys=True))
