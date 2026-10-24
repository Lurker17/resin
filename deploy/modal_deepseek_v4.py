"""Cache DeepSeek checkpoints and retain legacy TP-only diagnostic commands."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import modal
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import resin
from deploy.checkpoint_cache import (
    checkpoint_inventory,
    checkpoint_is_complete,
    publish_checkpoint,
)
from deploy.graph_validation import capture_cuda_graphs, validate_prefill_graphs
from deploy.modal_image import build_modal_image
from resin import Engine, SamplingParams, SchedulerConfig
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
)
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.recipe import DeepSeekV4Recipe, DeepSeekV41Recipe

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_MOUNT = "/models/resin-deepseek-v4"
MODEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-models-v1"
KERNEL_CACHE_MOUNT = "/kernel-cache/.cache"
KERNEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-kernels-v2"
PROFILE_MOUNT = "/profiles"
PROFILE_VOLUME_NAME = "resin-deepseek-v4-profiles-v1"
DEFAULT_MODEL_KEY = "flash-0731"
PRO_MODEL_KEY = "pro-0813"
V41_MODEL_KEY = "flash-1017"
DEFAULT_TP_SIZE = 4
DSPARK_TOKENS = 5
BENCHMARK_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
CHECKPOINT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference/config.json",
)

image = (
    build_modal_image(
        ROOT,
        env={
            "CC": "gcc-13",
            "CXX": "g++-13",
            "TORCH_CUDA_ARCH_LIST": "10.3",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "FLASHINFER_WORKSPACE_BASE": "/kernel-cache",
            # Keep lazy kernel work visible without DeepGEMM's per-launch debug
            # dump. Compiler commands appear only for an actual cache miss.
            "FLASHINFER_LOGGING_LEVEL": "INFO",
            "FLASHINFER_LOG_LEVEL": "INFO",
            "FLASHINFER_JIT_VERBOSE": "1",
            "CUDA_TILE_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/cutile-python",
            "DG_JIT_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/deep_gemm",
            "DG_JIT_DEBUG": "0",
            "DG_JIT_PRINT_COMPILER_COMMAND": "1",
            "DG_JIT_PRINT_LOAD_TIME": "1",
            "TRITON_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/triton",
            "TILELANG_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/tilelang",
            "RESIN_DEEPSEEK_MODEL": os.environ.get(
                "RESIN_DEEPSEEK_MODEL", DEFAULT_MODEL_KEY
            ),
        },
    )
    .add_local_file(
        ROOT / "deploy/deepseek_v4_parallel_validation.py",
        "/root/deploy/deepseek_v4_parallel_validation.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "deploy/deepseek_v4_expert_probe.py",
        "/root/deploy/deepseek_v4_expert_probe.py",
        copy=True,
    )
)

with image.imports():
    from resin_models_cuda.deepseek_v4.checkpoint import DEFAULT_MODEL, DEFAULT_REVISION
    from resin_models_cuda.deepseek_v41.checkpoint import (
        DEFAULT_MODEL as V41_MODEL,
    )
    from resin_models_cuda.deepseek_v41.checkpoint import (
        DEFAULT_REVISION as V41_REVISION,
    )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable identity and storage key for one published checkpoint."""

    key: str
    repo_id: str
    revision: str


MODEL_SPECS = {
    DEFAULT_MODEL_KEY: ModelSpec(DEFAULT_MODEL_KEY, DEFAULT_MODEL, DEFAULT_REVISION),
    PRO_MODEL_KEY: ModelSpec(
        PRO_MODEL_KEY,
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "72e1d3230f6c080a530b0a1d46f8eb4602340597",
    ),
    V41_MODEL_KEY: ModelSpec(V41_MODEL_KEY, V41_MODEL, V41_REVISION),
}

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME, create_if_missing=True, version=1
)
kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME, create_if_missing=True, version=1
)
profile_volume = modal.Volume.from_name(
    PROFILE_VOLUME_NAME, create_if_missing=True, version=1
)
app = modal.App(
    "resin-deepseek-v4-flash-dspark",
    image=image,
    volumes={
        MODEL_CACHE_MOUNT: model_cache_volume,
        KERNEL_CACHE_MOUNT: kernel_cache_volume,
        PROFILE_MOUNT: profile_volume,
    },
)


def _model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"model must be one of: {choices}") from error


def _model_path(model: ModelSpec) -> Path:
    return Path(MODEL_CACHE_MOUNT) / model.key / model.revision


def _check_legacy_topology(model: ModelSpec, attn_tp: int) -> None:
    if attn_tp > 1:
        raise ValueError(
            f"{model.key} has one KV head and requires attn_tp=1; this legacy "
            f"deployment uses attention TP{attn_tp}. Use deploy/modal_bench.py "
            "with --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 instead."
        )


def _implementation(model_path: Path):
    """Select the native architecture from checkpoint data, without remote code."""
    values = json.loads((model_path / "config.json").read_text())
    if values.get("model_type") == "deepseek_v41":
        from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner

        return DeepSeekV41CudaRunner, DeepSeekV41Recipe
    from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

    return DeepSeekV4CudaRunner, DeepSeekV4Recipe


def _checkpoint_inventory(model: ModelSpec) -> dict[str, int]:
    return checkpoint_inventory(_model_path(model))


def _checkpoint_is_complete(model: ModelSpec) -> bool:
    return checkpoint_is_complete(_model_path(model), model.repo_id, model.revision)


def _download_checkpoint(model: ModelSpec) -> dict[str, object]:
    model_cache_volume.reload()
    cache_hit = _checkpoint_is_complete(model)
    start = time.perf_counter()
    if not cache_hit:
        target = _model_path(model)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model.repo_id,
            revision=model.revision,
            allow_patterns=list(CHECKPOINT_PATTERNS),
            local_dir=target,
        )
        publish_checkpoint(target, model.repo_id, model.revision)
        model_cache_volume.commit()
    if not _checkpoint_is_complete(model):
        raise RuntimeError("checkpoint publication did not complete")
    inventory = _checkpoint_inventory(model)
    return {
        "repo_id": model.repo_id,
        "revision": model.revision,
        "path": str(_model_path(model)),
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }


def _require_checkpoint(model: ModelSpec) -> tuple[Path, dict[str, object]]:
    model_cache_volume.reload()
    if not _checkpoint_is_complete(model):
        raise RuntimeError(
            f"run mode=download --model={model.key} before requesting B300 inference"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    inventory = _checkpoint_inventory(model)
    return _model_path(model), {
        "repo_id": model.repo_id,
        "revision": model.revision,
        "path": str(_model_path(model)),
        "cache_hit": True,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
    }


def _encode_prompt(
    recipe: resin.recipe.Recipe,
    prompt: str,
    thinking_mode: str,
    reasoning_effort: str,
) -> list[int]:
    return recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(prompt)])],
        options=resin.recipe.RenderOptions(
            thinking_mode=thinking_mode, reasoning_effort=reasoning_effort
        ),
    ).token_ids


def _real_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    input_tokens: int,
) -> list[int]:
    """Build the same real-text 8K prompt used by the SGLang comparison."""

    prefix = (
        "You are reviewing Resin, a language-model inference engine. Read the "
        "project documentation and implementation excerpts below carefully. "
        "Focus on correctness, scheduling invariants, resource ownership, and "
        "maintainability.\n\n"
    )
    suffix = (
        "\n\nBased on the complete material above, write a prioritized engineering "
        "review. Explain the most consequential correctness risks first, cite "
        "specific mechanisms from the supplied design and code, and recommend "
        "concrete fixes and tests."
    )
    prompt_root = Path("/opt/resin")
    prompt_files = (
        "README.md",
        "docs/architecture.md",
        "docs/scheduler.md",
        "crates/resin-py/src/lib.rs",
    )
    body = "\n\n".join(
        f"===== {relative_path} =====\n{(prompt_root / relative_path).read_text()}"
        for relative_path in prompt_files
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body_budget = input_tokens - len(prefix_ids) - len(suffix_ids)
    if body_budget <= 0:
        raise ValueError("input token budget is too small for the real prompt")
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    if len(body_ids) < body_budget:
        raise ValueError(
            f"real prompt corpus has only {len(body_ids)} body tokens; "
            f"need {body_budget}"
        )
    return [
        *map(int, prefix_ids),
        *map(int, body_ids[:body_budget]),
        *map(int, suffix_ids),
    ]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_tensor_parallel(
    rank: int, master_port: int, tensor_parallel_size: int
) -> TensorParallelContext:
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=tensor_parallel_size,
        timeout=timedelta(minutes=30),
        device_id=torch.device(f"cuda:{rank}"),
    )
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=tensor_parallel_size),
        dist.group.WORLD,
    )


def _log_cache_capacity(runner) -> None:
    cache = runner.cache_capacity_plan
    print(
        "[deepseek-v4] cache "
        f"parents={runner.cache_layout.parent_capacity:,}; "
        f"parent={cache.bytes_per_parent / 2**20:.2f} MiB; "
        f"allocation={cache.allocation_bytes / 2**30:.2f} GiB; "
        f"packings={runner.cache_geometry.packings}",
        flush=True,
    )


def _run_rank_workload(
    rank: int,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    stop_token_ids: list[int] | None,
    tensor_parallel: TensorParallelContext,
    profile: bool,
    profile_run_id: str | None,
    speculative_tokens: int,
) -> dict[str, object]:
    if rank == 0:
        print(
            f"[deepseek-v4] loading TP={tensor_parallel.size} checkpoint",
            flush=True,
        )
    load_start = time.perf_counter()
    batch_size = len(prompt_rows)
    graph_capacity = (
        max(len(row) for row in prompt_rows) + max_new_tokens + speculative_tokens + 1
    )
    runner_type, _ = _implementation(model_path)
    runner = runner_type(
        model_path,
        max_seq_len=context_length,
        max_batch_size=batch_size,
        tensor_parallel=tensor_parallel,
        device=f"cuda:{rank}",
        cuda_graph_decode_shapes=tuple(
            (rows, max(2, graph_capacity)) for rows in range(1, batch_size + 1)
        ),
        speculative_tokens=speculative_tokens,
        profile_graph_phases=profile,
    )
    if rank == 0:
        _log_cache_capacity(runner)
    torch.cuda.synchronize(runner.device)
    tensor_parallel.barrier()
    load_elapsed = time.perf_counter() - load_start
    if rank == 0:
        print(
            f"[deepseek-v4] model loaded in {load_elapsed:.2f}s; "
            f"allreduce={runner.allreduce_backend}; capturing graph",
            flush=True,
        )

    tensor_parallel.barrier()
    capture_start = time.perf_counter()
    capture_cuda_graphs(runner)
    tensor_parallel.barrier()
    capture_elapsed = time.perf_counter() - capture_start
    if rank == 0:
        print(
            f"[deepseek-v4] graph captured in {capture_elapsed:.2f}s",
            flush=True,
        )

    validate_prefill_graphs(runner)

    engine = Engine(
        runner,
        max_running_requests=batch_size,
        context_length=context_length,
        prefix_cache=profile,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=batch_size * context_length,
            prefill_chunk_size=context_length,
        ),
    )
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        stop_token_ids=[runner.eos_token_id]
        if stop_token_ids is None
        else stop_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    first_token_time: float | None = None
    generated_steps = 0
    try:
        profile_tables: dict[str, str] = {}
        profile_trace_files: list[dict[str, object]] = []
        profile_phase_timings: list[dict[str, float]] = []
        prefix_cache_validation: dict[str, object] | None = None
        if profile:
            if profile_run_id is None:
                raise ValueError("profile_run_id is required when profiling")
            if rank == 0:
                print(
                    "[deepseek-v4] warming profile prefill, then tracing decode",
                    flush=True,
                )
            profile_requests = [
                engine.submit(
                    row,
                    sampling_params=sampling_params,
                )
                for row in prompt_rows
            ]
            # Advance through all prefill chunks. The first emitted event marks
            # the transition to fixed-shape CUDA-graph decode replay.
            while engine.has_work():
                if engine.step():
                    break
            torch.cuda.synchronize(runner.device)
            tensor_parallel.barrier()

            profiled_steps = 3
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=True,
            ) as decode_profile:
                for _ in range(profiled_steps):
                    with torch.profiler.record_function(
                        f"step[DSPARK_DECODE bs={batch_size}]"
                    ):
                        engine.step()
                    if runner.last_dspark_phase_ms is not None:
                        profile_phase_timings.append(runner.last_dspark_phase_ms)
                torch.cuda.synchronize(runner.device)
            tensor_parallel.barrier()

            profile_dir = Path(PROFILE_MOUNT) / profile_run_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            trace_path = profile_dir / (
                f"{profile_run_id}-TP-{rank}-DECODE.trace.json.gz"
            )
            decode_profile.export_chrome_trace(str(trace_path))
            tensor_parallel.barrier()
            profile_trace_files = [
                {
                    "rank": trace_rank,
                    "path": str(
                        profile_dir
                        / f"{profile_run_id}-TP-{trace_rank}-DECODE.trace.json.gz"
                    ),
                    "bytes": (
                        profile_dir
                        / f"{profile_run_id}-TP-{trace_rank}-DECODE.trace.json.gz"
                    )
                    .stat()
                    .st_size,
                }
                for trace_rank in range(tensor_parallel.size)
            ]
            if rank == 0:
                profile_tables["decode"] = decode_profile.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            while engine.has_work():
                engine.step()
            profile_outputs = [request.take_result() for request in profile_requests]
            tensor_parallel.barrier()
            before_prefix = engine.prefix_cache_stats
            continuation_rows = [
                [*row, *output_ids, 0]
                for row, output_ids in zip(prompt_rows, profile_outputs, strict=True)
            ]
            probes = [
                engine.submit(
                    row,
                    sampling_params=SamplingParams(
                        max_new_tokens=1,
                        stop_token_ids=sampling_params.stop_token_ids,
                        temperature=0.0,
                    ),
                )
                for row in continuation_rows
            ]
            while engine.has_work():
                engine.step()
            probe_outputs = [probe.take_result() for probe in probes]
            after_prefix = engine.prefix_cache_stats
            if (
                before_prefix is None
                or after_prefix is None
                or after_prefix[1] <= before_prefix[1]
            ):
                raise RuntimeError("DeepSeek prefix-cache continuation did not hit")
            engine.clear_prefix_cache()
            fresh_probes = [
                engine.submit(
                    row,
                    sampling_params=SamplingParams(
                        max_new_tokens=1,
                        stop_token_ids=sampling_params.stop_token_ids,
                        temperature=0.0,
                    ),
                )
                for row in continuation_rows
            ]
            while engine.has_work():
                engine.step()
            fresh_outputs = [probe.take_result() for probe in fresh_probes]
            if probe_outputs != fresh_outputs:
                raise RuntimeError(
                    "DeepSeek prefix-cache continuation differs from full prefill"
                )
            prefix_cache_validation = {
                "before": before_prefix,
                "after": after_prefix,
                "output_tokens": tuple(map(len, probe_outputs)),
                "fresh_parity": True,
            }
            if rank == 0:
                print(
                    f"[deepseek-v4] prefix continuation passed: {prefix_cache_validation}",
                    flush=True,
                )
                print("[deepseek-v4] profiling complete", flush=True)

        tensor_parallel.barrier()
        if rank == 0:
            print("[deepseek-v4] starting clean timed request", flush=True)
        clean_phase_timings: list[dict[str, float]] = []
        start = time.perf_counter()
        requests = [
            engine.submit(
                row,
                sampling_params=sampling_params,
            )
            for row in prompt_rows
        ]
        while engine.has_work():
            replay_count = runner.cuda_graph_stats["decode"]["replay_count"]
            events = engine.step()
            if (
                runner.cuda_graph_stats["decode"]["replay_count"] > replay_count
                and runner.last_dspark_phase_ms is not None
            ):
                clean_phase_timings.append(runner.last_dspark_phase_ms)
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
            if events:
                generated_steps += 1
                if tensor_parallel.rank == 0 and generated_steps % 256 == 0:
                    print(
                        f"[deepseek-v4] batch={batch_size}: "
                        f"generated {generated_steps}/{max_new_tokens}",
                        flush=True,
                    )
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()
        elapsed = time.perf_counter() - start
        decode_elapsed = (
            None if first_token_time is None else time.perf_counter() - first_token_time
        )
        output_rows = [request.take_result() for request in requests]

        output_tensor = torch.full(
            (batch_size, max_new_tokens),
            -1,
            dtype=torch.int64,
            device=runner.device,
        )
        for row, output_ids in enumerate(output_rows):
            output_tensor[row, : len(output_ids)] = torch.tensor(
                output_ids, dtype=torch.int64, device=runner.device
            )
        gathered = torch.empty(
            tensor_parallel.size * batch_size * max_new_tokens,
            dtype=torch.int64,
            device=runner.device,
        )
        dist.all_gather_single(
            gathered, output_tensor.flatten(), group=dist.group.WORLD
        )
        rank_outputs = gathered.view(tensor_parallel.size, batch_size, max_new_tokens)
        if any(
            not torch.equal(rank_outputs[0], rank_outputs[index])
            for index in range(1, tensor_parallel.size)
        ):
            raise RuntimeError("tensor-parallel ranks sampled different tokens")

        output_count = sum(map(len, output_rows))

        return {
            "model": checkpoint["repo_id"],
            "revision": checkpoint["revision"],
            "device": torch.cuda.get_device_name(runner.device),
            "tensor_parallel_size": tensor_parallel.size,
            "allreduce_backend": runner.allreduce_backend,
            "checkpoint": checkpoint,
            "weight_load": runner.weight_load,
            "model_load_ms": load_elapsed * 1000,
            "cuda_graph_capture_ms": capture_elapsed * 1000,
            "cuda_graph": runner.cuda_graph_stats,
            "speculative": runner.speculative_stats,
            "torch_profile": profile_tables,
            "graph_phase_timings_ms": profile_phase_timings,
            "clean_graph_phase_timings_ms": clean_phase_timings,
            "profile_run_id": profile_run_id,
            "perfetto_traces": profile_trace_files,
            "prefix_cache_validation": prefix_cache_validation,
            "batch_size": batch_size,
            "prompt_tokens": [len(row) for row in prompt_rows],
            "output_tokens": [len(row) for row in output_rows],
            "_output_ids": output_rows,
            "sampling": {
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "seed": sampling_params.seed,
            },
            "time_to_first_token_ms": (
                None if first_token_time is None else (first_token_time - start) * 1000
            ),
            "elapsed_ms": elapsed * 1000,
            "output_tokens_per_second": output_count / elapsed,
            "decode_tokens_per_second": (
                None
                if decode_elapsed is None
                or decode_elapsed <= 0
                or output_count <= batch_size
                else (output_count - batch_size) / decode_elapsed
            ),
        }
    finally:
        if not engine.closed:
            engine.close()


def _run_rank(
    rank: int,
    master_port: int,
    tensor_parallel_size: int,
    report_path: Path,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    stop_token_ids: list[int] | None,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
    profile_run_id: str | None,
    speculative_tokens: int,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(
        rank, master_port, tensor_parallel_size
    )
    report = _run_rank_workload(
        rank,
        model_path,
        checkpoint,
        prompt_rows,
        max_new_tokens,
        context_length,
        temperature,
        top_p,
        top_k,
        seed,
        stop_token_ids,
        tensor_parallel,
        profile,
        profile_run_id,
        speculative_tokens,
    )
    tensor_parallel.barrier()
    if rank == 0:
        report_path.write_text(json.dumps(report))
    os._exit(0)


def _launch_tensor_parallel(
    tensor_parallel_size: int,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    stop_token_ids: list[int] | None,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
    profile_run_id: str | None = None,
    speculative_tokens: int = DSPARK_TOKENS,
) -> dict[str, object]:
    master_port = _free_local_port()
    with tempfile.TemporaryDirectory(
        prefix=f"resin-dsv4-tp{tensor_parallel_size}-"
    ) as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_rank,
            args=(
                master_port,
                tensor_parallel_size,
                report_path,
                model_path,
                checkpoint,
                prompt_rows,
                stop_token_ids,
                max_new_tokens,
                context_length,
                temperature,
                top_p,
                top_k,
                seed,
                profile,
                profile_run_id,
                speculative_tokens,
            ),
            nprocs=tensor_parallel_size,
            join=True,
        )
        return cast(dict[str, object], json.loads(report_path.read_text()))


def _benchmark_point(
    runner,
    tensor_parallel: TensorParallelContext,
    prompt_ids: list[int],
    batch_size: int,
    max_new_tokens: int,
    context_length: int,
    sampling_params: SamplingParams,
) -> dict[str, object]:
    """Measure one synchronized batch after model and graph initialization."""

    engine = Engine(
        runner,
        max_running_requests=batch_size,
        context_length=context_length,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=batch_size * len(prompt_ids),
            prefill_chunk_size=len(prompt_ids),
        ),
    )
    first_token_time: float | None = None
    try:
        tensor_parallel.barrier()
        start = time.perf_counter()
        requests = [
            engine.submit(
                prompt_ids,
                sampling_params=sampling_params,
            )
            for _ in range(batch_size)
        ]
        events = engine.step()
        if not events:
            raise RuntimeError("benchmark prefill produced no token event")
        first_token_time = time.perf_counter()

        while engine.has_work():
            engine.step()
        torch.cuda.synchronize(runner.device)
        tensor_parallel.barrier()
        end = time.perf_counter()
        output_rows = [request.take_result() for request in requests]
        output_lengths = [len(row) for row in output_rows]
        if any(length != max_new_tokens for length in output_lengths):
            raise RuntimeError(
                f"benchmark generated {output_lengths}, expected {max_new_tokens} each"
            )
        if first_token_time is None:
            raise RuntimeError("benchmark produced no first-token event")

        output_count = sum(output_lengths)
        decode_token_count = output_count - batch_size
        decode_elapsed = end - first_token_time
        decode_throughput = decode_token_count / decode_elapsed
        return {
            "batch_size": batch_size,
            "input_tokens_per_request": len(prompt_ids),
            "output_tokens_per_request": max_new_tokens,
            "time_to_first_token_ms": (first_token_time - start) * 1000,
            "elapsed_ms": (end - start) * 1000,
            "decode_elapsed_ms": decode_elapsed * 1000,
            "decode_throughput_tokens_per_second": decode_throughput,
            "interactivity_tokens_per_second": decode_throughput / batch_size,
        }
    finally:
        if not engine.closed:
            engine.close()


def _run_rank_benchmark(
    rank: int,
    master_port: int,
    tensor_parallel_size: int,
    report_path: Path,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_ids: list[int],
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    use_cuda_graphs: bool,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(
        rank, master_port, tensor_parallel_size
    )
    context_length = len(prompt_ids) + max_new_tokens
    max_batch_size = max(batch_sizes)
    if rank == 0:
        print(
            f"[deepseek-v4] loading TP={tensor_parallel_size} benchmark runner "
            f"for batches {batch_sizes}",
            flush=True,
        )
    load_start = time.perf_counter()
    runner_type, _ = _implementation(model_path)
    runner = runner_type(
        model_path,
        max_seq_len=context_length,
        max_batch_size=max_batch_size,
        tensor_parallel=tensor_parallel,
        device=f"cuda:{rank}",
        cuda_graph_decode_shapes=(
            tuple(
                (batch_size, context_length + DSPARK_TOKENS + 1)
                for batch_size in batch_sizes
            )
            if use_cuda_graphs
            else ()
        ),
        cuda_graph_prefill="breakable" if use_cuda_graphs else "disabled",
        cuda_graph_decode="full" if use_cuda_graphs else "disabled",
        speculative_tokens=DSPARK_TOKENS,
    )
    if rank == 0:
        _log_cache_capacity(runner)
    torch.cuda.synchronize(runner.device)
    tensor_parallel.barrier()
    load_elapsed = time.perf_counter() - load_start

    capture_elapsed = 0.0
    if use_cuda_graphs:
        capture_start = time.perf_counter()
        capture_cuda_graphs(runner)
        capture_elapsed = time.perf_counter() - capture_start
        validate_prefill_graphs(runner)
    # Force exactly max_new_tokens so every benchmark has the same denominator.
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        stop_token_ids=(),
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        seed=0,
    )

    points = []
    for batch_size in batch_sizes:
        if rank == 0:
            print(
                f"[deepseek-v4] TP={tensor_parallel_size} batch={batch_size} "
                f"{len(prompt_ids)}/{max_new_tokens}",
                flush=True,
            )
        point = _benchmark_point(
            runner,
            tensor_parallel,
            prompt_ids,
            batch_size,
            max_new_tokens,
            context_length,
            sampling_params,
        )
        points.append(point)
        if rank == 0:
            print(
                f"[deepseek-v4] batch={batch_size}: "
                f"interactivity={point['interactivity_tokens_per_second']:.2f} "
                f"tok/s, throughput="
                f"{point['decode_throughput_tokens_per_second']:.2f} tok/s",
                flush=True,
            )

    tensor_parallel.barrier()
    if rank == 0:
        report = {
            "model": checkpoint["repo_id"],
            "revision": checkpoint["revision"],
            "device": torch.cuda.get_device_name(runner.device),
            "tensor_parallel_size": tensor_parallel_size,
            "allreduce_backend": runner.allreduce_backend,
            "checkpoint": checkpoint,
            "input_tokens_per_request": len(prompt_ids),
            "output_tokens_per_request": max_new_tokens,
            "sampling": {
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "seed": sampling_params.seed,
            },
            "model_load_ms": load_elapsed * 1000,
            "cuda_graph_capture_ms": capture_elapsed * 1000,
            "use_cuda_graphs": use_cuda_graphs,
            "cuda_graph": runner.cuda_graph_stats,
            "points": points,
        }
        report_path.write_text(json.dumps(report))
    os._exit(0)


def _launch_batch_benchmark(
    tensor_parallel_size: int,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_ids: list[int],
    max_new_tokens: int,
    batch_sizes: tuple[int, ...],
    use_cuda_graphs: bool,
) -> dict[str, object]:
    master_port = _free_local_port()
    with tempfile.TemporaryDirectory(
        prefix=f"resin-dsv4-benchmark-tp{tensor_parallel_size}-"
    ) as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_rank_benchmark,
            args=(
                master_port,
                tensor_parallel_size,
                report_path,
                model_path,
                checkpoint,
                prompt_ids,
                max_new_tokens,
                batch_sizes,
                use_cuda_graphs,
            ),
            nprocs=tensor_parallel_size,
            join=True,
        )
        return cast(dict[str, object], json.loads(report_path.read_text()))


def _benchmark_remote(
    model: ModelSpec,
    tensor_parallel_size: int,
    input_tokens: int = 8192,
    max_new_tokens: int = 1024,
    batch_sizes: tuple[int, ...] = BENCHMARK_BATCH_SIZES,
    use_cuda_graphs: bool = True,
) -> dict[str, object]:
    if input_tokens <= 0 or max_new_tokens <= 1:
        raise ValueError("benchmark token counts must be positive")
    _check_legacy_topology(model, tensor_parallel_size)
    model_path, checkpoint = _require_checkpoint(model)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    _, codec_type = _implementation(model_path)
    recipe = codec_type(tokenizer)
    seed_prompt = _encode_prompt(
        recipe,
        "Measure inference throughput with a stable synthetic prompt.",
        "chat",
        "high",
    )
    repeats = (input_tokens + len(seed_prompt) - 1) // len(seed_prompt)
    prompt_ids = (seed_prompt * repeats)[:input_tokens]
    return _launch_batch_benchmark(
        tensor_parallel_size,
        model_path,
        checkpoint,
        prompt_ids,
        max_new_tokens,
        batch_sizes,
        use_cuda_graphs,
    )


@app.function(timeout=24 * 60 * 60, cpu=16, memory=32 * 1024)
def cache_checkpoint(model_key: str = DEFAULT_MODEL_KEY) -> dict[str, object]:
    return _download_checkpoint(_model_spec(model_key))


@app.function(
    gpu="B300:4",
    cpu=32,
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
    label="deepseek-v4",
    requires_proxy_auth=False,
)
def serve_deepseek_v4() -> None:
    """Reject the retired query-head-only TP endpoint before loading a model."""
    model = _model_spec(os.environ.get("RESIN_DEEPSEEK_MODEL", DEFAULT_MODEL_KEY))
    _check_legacy_topology(model, DEFAULT_TP_SIZE)
    raise ValueError("This HTTP endpoint is retired; use deploy/modal_deepseek_v41.py")


@app.function(
    gpu="B300:2",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def benchmark_tp2(model_key: str = DEFAULT_MODEL_KEY) -> dict[str, object]:
    return _benchmark_remote(_model_spec(model_key), 2)


@app.function(
    image=image.uv_pip_install("pytest>=8,<9")
    .add_local_file(
        ROOT / "models/cuda/tests/test_deepseek_v4_window.py",
        "/opt/resin/validation/test_deepseek_v4_window.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "models/cuda/tests/test_deepseek_v4_kernels.py",
        "/opt/resin/validation/test_deepseek_v4_kernels.py",
        copy=True,
    ),
    gpu="B300:1",
    cpu=4,
    memory=8 * 1024,
    timeout=10 * 60,
)
def validate_kernels() -> dict[str, object]:
    """Check fused math, packed cache writes, and speculative attention windows."""

    import subprocess

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("kernel validation requires an available Blackwell GPU")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            # The source checkout may contain a host-built native extension.
            # Validate the image's installed Linux wheels, without the repo's
            # development-only pytest pythonpath setting.
            "-c",
            "/dev/null",
            "-q",
            "/opt/resin/validation/test_deepseek_v4_window.py",
            "/opt/resin/validation/test_deepseek_v4_kernels.py",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=9 * 60,
        check=False,
    )
    print(result.stdout, flush=True)
    if result.returncode or "skipped" in result.stdout:
        raise RuntimeError("DeepSeek-V4 kernel validation did not pass")
    kernel_cache_volume.commit()
    return {"gpu": torch.cuda.get_device_name(), "tests": result.stdout.strip()}


@app.function(
    gpu=["B300:1", "B200:1"],
    env={"TORCH_CUDA_ARCH_LIST": "10.0;10.3"},
    cpu=4,
    memory=16 * 1024,
    timeout=20 * 60,
    single_use_containers=True,
)
def probe_experts() -> dict[str, object]:
    """Compare FP4 kernel shard and batch semantics without loading a model."""

    from deploy.deepseek_v4_expert_probe import run_probe

    kernel_cache_volume.reload()
    try:
        report = {
            "small": run_probe(),
            "model_geometry": run_probe(
                hidden_size=4096,
                intermediate_size=2048,
                experts=8,
                top_k=6,
                row_counts=(1, 8, 32),
            ),
            "checkpoint": run_probe(
                checkpoint_path=_require_checkpoint(_model_spec(DEFAULT_MODEL_KEY))[0],
                experts=256,
                row_counts=(1, 8, 32),
                probe_deep_gemm=True,
            ),
        }
        destination = Path(PROFILE_MOUNT) / datetime.now(timezone.utc).strftime(
            "expert-probe-%Y%m%dT%H%M%SZ.json"
        )
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
        report["remote_report_path"] = str(destination)
        profile_volume.commit()
        return report
    finally:
        kernel_cache_volume.commit()


@app.function(
    image=image.uv_pip_install("pytest>=8,<9").add_local_file(
        ROOT / "tests/test_expert_parallel_gpu.py",
        "/opt/resin/validation/test_expert_parallel_gpu.py",
        copy=True,
    ),
    gpu=["B300:4", "B200:4"],
    env={"TORCH_CUDA_ARCH_LIST": "10.0;10.3"},
    cpu=32,
    memory=256 * 1024,
    timeout=2 * 60 * 60,
    single_use_containers=True,
)
def validate_parallel(
    model_key: str = DEFAULT_MODEL_KEY,
    include_tp_baseline: bool = False,
) -> dict[str, object]:
    """Validate NCCL routing and complete-checkpoint EP/attention-DP serving."""

    return _validate_parallel(model_key, include_tp_baseline=include_tp_baseline)


def _validate_parallel(
    model_key: str,
    *,
    include_tp_baseline: bool = False,
    composable: bool = False,
    case_names: tuple[str, ...] | None = None,
    baseline_path: str | None = None,
    graphs: bool = True,
    decode_capacity: int = 512,
    speculative_tokens: int = 0,
) -> dict[str, object]:
    """Share checkpoint, routing reference and report handling across GPU sizes."""

    if model_key == V41_MODEL_KEY:
        raise ValueError("legacy parallel validation targets V4-0731, not V4.1-1017")

    import subprocess

    from deploy.deepseek_v4_parallel_validation import (
        run_composable_validation,
        run_parallel_validation,
    )

    model = _model_spec(model_key)
    model_path, checkpoint = _require_checkpoint(model)
    world_size = torch.cuda.device_count()
    expected_sizes = (2, 4) if composable else (4,)
    if not torch.cuda.is_available() or world_size not in expected_sizes:
        raise RuntimeError(
            f"parallel validation requires Blackwell GPU counts {expected_sizes}"
        )
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-q",
            "-s",
            "/opt/resin/validation/test_expert_parallel_gpu.py",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "RESIN_EP_TEST_WORLD_SIZE": str(world_size)},
        timeout=8 * 60,
        check=False,
    )
    print(tests.stdout, flush=True)
    if tests.returncode or "skipped" in tests.stdout:
        raise RuntimeError(
            f"{world_size}-GPU expert dispatch reference validation did not pass"
        )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    recipe = DeepSeekV4Recipe(tokenizer)
    texts = {
        "short": "Continue the sequence with the next number: 2, 4, 6,",
        "long": (
            "Review this inference rule: every request owns its cache pages; "
            "expert workers may process tokens from independent request replicas. "
        )
        * 6
        + "Explain why those cache namespaces must remain independent.",
        "late": "Give one practical reason to batch requests with unequal prompt lengths.",
        "probe": "What is the next integer after 41? Answer briefly.",
    }
    prompts = {
        name: _encode_prompt(recipe, text, "chat", "low")
        for name, text in texts.items()
    }
    print(
        f"[deepseek-v4] validating complete Flash checkpoint on {world_size} Blackwell GPUs",
        flush=True,
    )
    if composable:
        diagnostic_directory = Path(PROFILE_MOUNT) / datetime.now(
            timezone.utc
        ).strftime("composable-%Y%m%dT%H%M%SZ")
        print(
            f"[deepseek-v4] persistent diagnostics: {diagnostic_directory}", flush=True
        )
        try:
            report = run_composable_validation(
                model_path,
                prompts,
                vocab_size=len(tokenizer),
                diagnostic_path=diagnostic_directory,
                case_names=case_names,
                baseline_path=baseline_path,
                graphs=graphs,
                decode_capacity=decode_capacity,
                speculative_tokens=speculative_tokens,
            )
        finally:
            # Keep the baseline, failing tensors and metrics when a check raises.
            profile_volume.commit()
            kernel_cache_volume.commit()
    else:
        report = run_parallel_validation(
            model_path,
            prompts,
            vocab_size=len(tokenizer),
            include_tp_baseline=include_tp_baseline,
        )
    report.update(
        {
            "checkpoint": checkpoint,
            "dispatch_reference_tests": tests.stdout.strip(),
            "prompt_token_counts": {name: len(ids) for name, ids in prompts.items()},
        }
    )
    for case in report["cases"]:
        for rank in case["ranks"]:
            result = rank["driver_result"]
            if result is not None:
                for output in result["outputs"].values():
                    output["text"] = tokenizer.decode(
                        output["token_ids"], skip_special_tokens=False
                    )
                for probe in result["probes"]:
                    probe["text"] = tokenizer.decode(
                        probe["token_ids"], skip_special_tokens=False
                    )
    run_id = datetime.now(timezone.utc).strftime("parallel-%Y%m%dT%H%M%SZ")
    remote_report = Path(PROFILE_MOUNT) / f"{run_id}.json"
    remote_report.write_text(json.dumps(report, indent=2, sort_keys=True))
    report["remote_report_path"] = str(remote_report)
    profile_volume.commit()
    kernel_cache_volume.commit()
    return report


@app.function(
    image=image.uv_pip_install("pytest>=8,<9").add_local_file(
        ROOT / "tests/test_expert_parallel_gpu.py",
        "/opt/resin/validation/test_expert_parallel_gpu.py",
        copy=True,
    ),
    gpu=["B300:2", "B200:2", "B300:4", "B200:4"],
    env={"TORCH_CUDA_ARCH_LIST": "10.0;10.3"},
    cpu=32,
    memory=256 * 1024,
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def validate_composable(
    model_key: str = DEFAULT_MODEL_KEY,
    case_names: tuple[str, ...] | None = None,
    baseline_path: str | None = None,
    graphs: bool = True,
    decode_capacity: int = 512,
    speculative_tokens: int = 0,
) -> dict[str, object]:
    """Run real FP4 models with composed meshes, graph replay and logit checks."""

    if model_key != DEFAULT_MODEL_KEY:
        raise ValueError(
            "composable validation currently requires the Flash checkpoint"
        )
    if not 0 <= speculative_tokens <= DSPARK_TOKENS:
        raise ValueError(f"speculative_tokens must be in [0, {DSPARK_TOKENS}]")
    return _validate_parallel(
        model_key,
        composable=True,
        case_names=case_names,
        baseline_path=baseline_path,
        graphs=graphs,
        decode_capacity=decode_capacity,
        speculative_tokens=speculative_tokens,
    )


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def benchmark_tp4(model_key: str = DEFAULT_MODEL_KEY) -> dict[str, object]:
    return _benchmark_remote(_model_spec(model_key), 4)


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def run_model_tp4(
    prompt: str,
    max_new_tokens: int = 16384,
    context_length: int = 32768,
    thinking_mode: str = "chat",
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
    profile_input_tokens: int | None = None,
    speculative_tokens: int = DSPARK_TOKENS,
    model_key: str = DEFAULT_MODEL_KEY,
) -> dict[str, object]:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")
    if reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be 'low', 'high', or 'max'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= speculative_tokens <= DSPARK_TOKENS:
        raise ValueError(f"speculative_tokens must be in [0, {DSPARK_TOKENS}]")
    if profile and max_new_tokens < 32:
        raise ValueError("profile runs require at least 32 output tokens")
    model = _model_spec(model_key)
    _check_legacy_topology(model, DEFAULT_TP_SIZE)
    model_path, checkpoint = _require_checkpoint(model)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    _, codec_type = _implementation(model_path)
    recipe = codec_type(tokenizer)
    if profile_input_tokens is None:
        rendered = recipe.render(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(prompt)])],
            options=resin.recipe.RenderOptions(
                thinking_mode=thinking_mode, reasoning_effort=reasoning_effort
            ),
        )
        prompt_ids = rendered.token_ids
    else:
        rendered = None
        prompt_ids = _real_prompt_ids(tokenizer, profile_input_tokens)
    prompt_rows = [prompt_ids.copy() for _ in range(batch_size)]
    if max(map(len, prompt_rows)) + max_new_tokens - 1 > context_length:
        raise ValueError("prompt plus max_new_tokens exceeds context_length")

    profile_run_id = (
        datetime.now(timezone.utc).strftime("resin-0731-%Y%m%dT%H%M%SZ")
        if profile
        else None
    )
    report = _launch_tensor_parallel(
        DEFAULT_TP_SIZE,
        model_path,
        checkpoint,
        prompt_rows,
        None if rendered is None else rendered.stop_token_ids,
        max_new_tokens,
        context_length,
        temperature,
        top_p,
        top_k,
        seed,
        profile,
        profile_run_id,
        speculative_tokens,
    )
    output_rows = cast(list[list[int]], report.pop("_output_ids"))
    completions = [
        tokenizer.decode(output_ids, skip_special_tokens=False)
        for output_ids in output_rows
    ]
    report["text"] = completions[0] if batch_size == 1 else completions
    report["thinking_mode"] = thinking_mode
    if batch_size == 1 and rendered is not None:
        try:
            report["message"] = recipe.parse_response(output_rows[0], prompt=rendered)
        except (AssertionError, ValueError) as error:
            report["parse_error"] = str(error)
    kernel_cache_volume.commit()
    if profile:
        profile_volume.commit()
    return report


@app.local_entrypoint()
def main(
    mode: str = "run",
    prompt: str = "Explain why batching improves language-model throughput.",
    max_new_tokens: int = 16384,
    context_length: int = 32768,
    thinking_mode: str = "chat",
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
    input_tokens: int = 8192,
    speculative_tokens: int = DSPARK_TOKENS,
    model: str = DEFAULT_MODEL_KEY,
    parallel_baseline: bool = False,
    parallel_cases: str | None = None,
    parallel_baseline_path: str | None = None,
    parallel_graphs: bool = True,
    parallel_decode_capacity: int = 512,
    parallel_speculative_tokens: int = 0,
    parallel_world_size: int = 2,
    parallel_gpu_type: str = "B300",
    report_path: str | None = None,
) -> None:
    model_spec = _model_spec(model)
    if mode in {"bench-tp2", "bench-tp4", "run", "real", "profile"}:
        _check_legacy_topology(model_spec, 2 if mode == "bench-tp2" else 4)
    if model == V41_MODEL_KEY and mode in {
        "validate-kernels",
        "validate-window",
        "probe-experts",
        "validate-parallel",
        "validate-composable",
    }:
        raise ValueError("legacy validation/probe modes target V4-0731, not V4.1-1017")
    if mode == "download":
        report = cache_checkpoint.remote(model_spec.key)
    elif mode in {"validate-kernels", "validate-window"}:
        report = validate_kernels.remote()
    elif mode == "probe-experts":
        report = probe_experts.remote()
    elif mode == "validate-parallel":
        report = validate_parallel.remote(model_spec.key, parallel_baseline)
    elif mode == "validate-composable":
        if parallel_world_size not in (2, 4) or parallel_gpu_type not in (
            "B200",
            "B300",
        ):
            raise ValueError("composable validation uses two/four B200 or B300 GPUs")
        case_names = (
            None
            if parallel_cases is None
            else tuple(
                name.strip() for name in parallel_cases.split(",") if name.strip()
            )
        )
        function = validate_composable
        if parallel_world_size != 2 or parallel_gpu_type != "B300":
            function = function.with_options(
                gpu=f"{parallel_gpu_type}:{parallel_world_size}",
                cpu=4 * parallel_world_size,
                memory=32 * 1024 * parallel_world_size,
            )
        report = function.remote(
            model_spec.key,
            case_names,
            parallel_baseline_path,
            parallel_graphs,
            parallel_decode_capacity,
            parallel_speculative_tokens,
        )
    elif mode == "bench-tp2":
        if model_spec.key not in {DEFAULT_MODEL_KEY, V41_MODEL_KEY}:
            raise ValueError("bench-tp2 supports flash-0731 or flash-1017")
        report = benchmark_tp2.remote(model_spec.key)
    elif mode == "bench-tp4":
        if model_spec.key not in {DEFAULT_MODEL_KEY, V41_MODEL_KEY}:
            raise ValueError("bench-tp4 supports flash-0731 or flash-1017")
        report = benchmark_tp4.remote(model_spec.key)
    elif mode in {"run", "real", "profile"}:
        capture_profile = profile or mode == "profile"
        report = run_model_tp4.remote(
            prompt,
            max_new_tokens,
            context_length,
            thinking_mode,
            reasoning_effort,
            temperature,
            top_p,
            top_k,
            seed,
            capture_profile,
            batch_size,
            input_tokens if mode in {"real", "profile"} else None,
            speculative_tokens,
            model_spec.key,
        )
    else:
        raise ValueError(
            "mode must be 'download', 'validate-kernels', 'probe-experts', 'validate-parallel', 'validate-composable', 'bench-tp2', 'bench-tp4', 'run', 'real', "
            "or 'profile'"
        )
    if mode in {"validate-parallel", "validate-composable"} or report_path is not None:
        destination = (
            (
                Path(report_path)
                if report_path is not None
                else ROOT
                / "artifacts"
                / datetime.now(timezone.utc).strftime(
                    "deepseek-v4-parallel-%Y%m%dT%H%M%SZ.json"
                )
            )
            .expanduser()
            .resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Validation report: {destination}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
