"""Run DeepSeek-V4-Flash-0731 on one Modal 4xB300 node with TP=4."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import cast

import modal
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from resin import Engine, SamplingParams, SchedulerConfig
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.models.deepseek_v4.checkpoint import DEFAULT_MODEL, DEFAULT_REVISION

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_MOUNT = "/models/resin-deepseek-v4"
MODEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-models-v1"
KERNEL_CACHE_MOUNT = "/root/.cache"
KERNEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-kernels-v1"
MODEL_KEY = "flash-0731"
TP_SIZE = 4
CHECKPOINT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference/config.json",
    "encoding/*.py",
    "encoding/*.md",
)

image = (
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "curl", "git", "ninja-build")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "MAX_JOBS": "16",
            "CMAKE_BUILD_PARALLEL_LEVEL": "16",
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": "10.3",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .uv_sync(
        str(ROOT),
        extras=["cuda", "demo"],
        extra_options="--no-default-groups",
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
        ROOT / "deploy/modal_deepseek_v4.py",
        "/root/deploy/modal_deepseek_v4.py",
        copy=True,
    )
    .add_local_file(ROOT / "Cargo.toml", "/opt/resin/Cargo.toml", copy=True)
    .add_local_file(ROOT / "Cargo.lock", "/opt/resin/Cargo.lock", copy=True)
    .add_local_file(ROOT / "pyproject.toml", "/opt/resin/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/resin/README.md", copy=True)
    .add_local_dir(ROOT / "crates", "/opt/resin/crates", copy=True)
    .add_local_dir(ROOT / "src", "/opt/resin/src", copy=True)
    .uv_pip_install("/opt/resin", extra_options="--no-deps")
)

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME, create_if_missing=True, version=1
)
kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME, create_if_missing=True, version=1
)
app = modal.App(
    "resin-deepseek-v4-flash-0731",
    image=image,
    volumes={
        MODEL_CACHE_MOUNT: model_cache_volume,
        KERNEL_CACHE_MOUNT: kernel_cache_volume,
    },
)


def _model_path() -> Path:
    return Path(MODEL_CACHE_MOUNT) / MODEL_KEY / DEFAULT_REVISION


def _marker_path() -> Path:
    return _model_path() / ".resin-checkpoint.json"


def _checkpoint_inventory() -> dict[str, int]:
    root = _model_path()
    index = root / "model.safetensors.index.json"
    weights = sorted(root.glob("model-*.safetensors"))
    required = [
        root / "config.json",
        index,
        root / "tokenizer.json",
        root / "tokenizer_config.json",
        root / "inference/config.json",
        root / "encoding/encoding_dsv4.py",
    ]
    if len(weights) != 48:
        raise RuntimeError(f"expected 48 checkpoint shards, found {len(weights)}")
    files = [*required, *weights]
    if missing := [str(path.relative_to(root)) for path in files if not path.is_file()]:
        raise RuntimeError(f"checkpoint is missing files: {missing}")
    if empty := [
        str(path.relative_to(root)) for path in files if path.stat().st_size <= 0
    ]:
        raise RuntimeError(f"checkpoint contains empty files: {empty}")
    return {str(path.relative_to(root)): path.stat().st_size for path in sorted(files)}


def _checkpoint_is_complete() -> bool:
    try:
        marker = json.loads(_marker_path().read_text())
        inventory = _checkpoint_inventory()
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    return marker == {
        "schema": 1,
        "repo_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "files": inventory,
    }


def _download_checkpoint() -> dict[str, object]:
    model_cache_volume.reload()
    cache_hit = _checkpoint_is_complete()
    start = time.perf_counter()
    if not cache_hit:
        target = _model_path()
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            allow_patterns=list(CHECKPOINT_PATTERNS),
            local_dir=target,
        )
        marker = {
            "schema": 1,
            "repo_id": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "files": _checkpoint_inventory(),
        }
        temporary = target / f".resin-checkpoint.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(marker, sort_keys=True))
        temporary.replace(_marker_path())
        model_cache_volume.commit()
    if not _checkpoint_is_complete():
        raise RuntimeError("checkpoint publication did not complete")
    inventory = _checkpoint_inventory()
    return {
        "repo_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "path": str(_model_path()),
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }


def _require_checkpoint() -> tuple[Path, dict[str, object]]:
    model_cache_volume.reload()
    if not _checkpoint_is_complete():
        raise RuntimeError("run mode=download before requesting B300 inference")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    inventory = _checkpoint_inventory()
    return _model_path(), {
        "repo_id": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "path": str(_model_path()),
        "cache_hit": True,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
    }


def _load_encoding(model_path: Path) -> ModuleType:
    path = model_path / "encoding/encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("_resin_encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DeepSeek-V4 encoding: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    encoding: ModuleType,
    prompt: str,
    thinking_mode: str,
    reasoning_effort: str,
) -> list[int]:
    rendered = encoding.encode_messages(
        [{"role": "user", "content": prompt}],
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    token_ids = tokenizer.encode(rendered)
    if not token_ids:
        raise ValueError("prompt produced no tokens")
    return [int(token_id) for token_id in token_ids]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_tensor_parallel(rank: int, master_port: int) -> TensorParallelContext:
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=TP_SIZE,
        timeout=timedelta(minutes=30),
        device_id=torch.device(f"cuda:{rank}"),
    )
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=TP_SIZE),
        dist.group.WORLD,
    )


def _run_rank_workload(
    rank: int,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    max_new_tokens: int,
    context_length: int,
    sampling_params: SamplingParams,
    tensor_parallel: TensorParallelContext,
    profile: bool,
) -> dict[str, object]:
    from resin.models.deepseek_v4.cuda import DeepSeekV4FlashCudaRunner

    if rank == 0:
        print("[deepseek-v4] loading TP=4 checkpoint", flush=True)
    load_start = time.perf_counter()
    batch_size = len(prompt_rows)
    graph_capacity = max(len(row) for row in prompt_rows) + max_new_tokens
    runner = DeepSeekV4FlashCudaRunner(
        model_path,
        max_seq_len=context_length,
        max_batch_size=batch_size,
        tensor_parallel=tensor_parallel,
        device=f"cuda:{rank}",
        cuda_graph_decode_shapes=tuple(
            (rows, max(2, graph_capacity)) for rows in range(1, batch_size + 1)
        ),
    )
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
    runner.capture_cuda_graphs()
    tensor_parallel.barrier()
    capture_elapsed = time.perf_counter() - capture_start
    if rank == 0:
        print(
            f"[deepseek-v4] graph captured in {capture_elapsed:.2f}s",
            flush=True,
        )

    engine = Engine(
        runner,
        max_running_requests=batch_size,
        context_length=context_length,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=batch_size * context_length,
            prefill_chunk_size=context_length,
        ),
        eos_token_id=runner.eos_token_id,
    )
    first_token_time: float | None = None
    try:
        profile_tables: dict[str, str] = {}
        if profile:
            if rank == 0:
                print("[deepseek-v4] profiling warm prefill/decode", flush=True)
            profile_requests = [
                engine.submit(
                    row,
                    max_new_tokens=min(max_new_tokens, 8),
                    sampling_params=sampling_params,
                )
                for row in prompt_rows
            ]
            if rank == 0:
                with torch.profiler.profile(
                    activities=(
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ),
                    record_shapes=True,
                ) as prefill_profile:
                    engine.step()
                    torch.cuda.synchronize(runner.device)
            else:
                engine.step()
            if rank == 0:
                profile_tables["prefill"] = prefill_profile.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            if engine.has_work():
                if rank == 0:
                    with torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ),
                        record_shapes=True,
                    ) as decode_profile:
                        while engine.has_work():
                            engine.step()
                        torch.cuda.synchronize(runner.device)
                else:
                    while engine.has_work():
                        engine.step()
                if rank == 0:
                    profile_tables["decode"] = decode_profile.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=30,
                    )
            for request in profile_requests:
                request.take_result()
            tensor_parallel.barrier()
            if rank == 0:
                print("[deepseek-v4] profiling complete", flush=True)

        tensor_parallel.barrier()
        if rank == 0:
            print("[deepseek-v4] starting clean timed request", flush=True)
        start = time.perf_counter()
        requests = [
            engine.submit(
                row,
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params,
            )
            for row in prompt_rows
        ]
        while engine.has_work():
            events = engine.step()
            if events and first_token_time is None:
                first_token_time = time.perf_counter()
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
            TP_SIZE * batch_size * max_new_tokens,
            dtype=torch.int64,
            device=runner.device,
        )
        dist.all_gather_single(gathered, output_tensor.flatten(), group=dist.group.WORLD)
        rank_outputs = gathered.view(TP_SIZE, batch_size, max_new_tokens)
        if any(
            not torch.equal(rank_outputs[0], rank_outputs[index])
            for index in range(1, TP_SIZE)
        ):
            raise RuntimeError("tensor-parallel ranks sampled different tokens")

        output_count = sum(map(len, output_rows))

        return {
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "device": torch.cuda.get_device_name(runner.device),
            "tensor_parallel_size": TP_SIZE,
            "allreduce_backend": runner.allreduce_backend,
            "checkpoint": checkpoint,
            "weight_load": runner.weight_load,
            "model_load_ms": load_elapsed * 1000,
            "cuda_graph_capture_ms": capture_elapsed * 1000,
            "cuda_graph": runner.cuda_graph_stats,
            "torch_profile": profile_tables,
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
                if decode_elapsed is None or decode_elapsed <= 0 or output_count <= batch_size
                else (output_count - batch_size) / decode_elapsed
            ),
        }
    finally:
        if not engine.closed:
            engine.close()


def _run_rank(
    rank: int,
    master_port: int,
    report_path: Path,
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(rank, master_port)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    report = _run_rank_workload(
        rank,
        model_path,
        checkpoint,
        prompt_rows,
        max_new_tokens,
        context_length,
        sampling_params,
        tensor_parallel,
        profile,
    )
    tensor_parallel.barrier()
    if rank == 0:
        report_path.write_text(json.dumps(report))
    os._exit(0)


def _launch_tp4(
    model_path: Path,
    checkpoint: dict[str, object],
    prompt_rows: list[list[int]],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
) -> dict[str, object]:
    master_port = _free_local_port()
    with tempfile.TemporaryDirectory(prefix="resin-dsv4-tp4-") as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_rank,
            args=(
                master_port,
                report_path,
                model_path,
                checkpoint,
                prompt_rows,
                max_new_tokens,
                context_length,
                temperature,
                top_p,
                top_k,
                seed,
                profile,
            ),
            nprocs=TP_SIZE,
            join=True,
        )
        return cast(dict[str, object], json.loads(report_path.read_text()))


@app.function(timeout=24 * 60 * 60, cpu=16, memory=32 * 1024)
def cache_checkpoint() -> dict[str, object]:
    return _download_checkpoint()


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def run_model_tp4(
    prompt: str,
    max_new_tokens: int = 256,
    context_length: int = 32768,
    thinking_mode: str = "chat",
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
) -> dict[str, object]:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")
    if reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be 'low', 'high', or 'max'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_path, checkpoint = _require_checkpoint()
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    encoding = _load_encoding(model_path)
    prompt_ids = _encode_prompt(
        tokenizer,
        encoding,
        prompt,
        thinking_mode,
        reasoning_effort,
    )
    prompt_rows = [prompt_ids.copy() for _ in range(batch_size)]
    if max(map(len, prompt_rows)) + max_new_tokens - 1 > context_length:
        raise ValueError("prompt plus max_new_tokens exceeds context_length")

    report = _launch_tp4(
        model_path,
        checkpoint,
        prompt_rows,
        max_new_tokens,
        context_length,
        temperature,
        top_p,
        top_k,
        seed,
        profile,
    )
    output_rows = cast(list[list[int]], report.pop("_output_ids"))
    completions = [
        tokenizer.decode(output_ids, skip_special_tokens=False)
        for output_ids in output_rows
    ]
    report["text"] = completions[0] if batch_size == 1 else completions
    report["thinking_mode"] = thinking_mode
    if batch_size == 1:
        try:
            report["message"] = encoding.parse_message_from_completion_text(
                completions[0], thinking_mode=thinking_mode
            )
        except (AssertionError, ValueError) as error:
            report["parse_error"] = str(error)
    kernel_cache_volume.commit()
    return report


@app.local_entrypoint()
def main(
    mode: str = "run",
    prompt: str = "Explain why batching improves language-model throughput.",
    max_new_tokens: int = 256,
    context_length: int = 32768,
    thinking_mode: str = "chat",
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
) -> None:
    if mode == "download":
        report = cache_checkpoint.remote()
    elif mode == "run":
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
            profile,
            batch_size,
        )
    else:
        raise ValueError("mode must be 'download' or 'run'")
    print(json.dumps(report, indent=2, sort_keys=True))
