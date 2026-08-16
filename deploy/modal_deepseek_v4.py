"""Run and benchmark DeepSeek-V4-Flash-DSpark on Modal B300 nodes."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
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

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_MOUNT = "/models/resin-deepseek-v4"
MODEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-models-v1"
KERNEL_CACHE_MOUNT = "/root/.cache"
KERNEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-kernels-v1"
PROFILE_MOUNT = "/profiles"
PROFILE_VOLUME_NAME = "resin-deepseek-v4-profiles-v1"
MODEL_KEY = "flash-0731"
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
            "DG_JIT_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/deep_gemm",
            "TRITON_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/triton",
            "TILELANG_CACHE_DIR": f"{KERNEL_CACHE_MOUNT}/tilelang",
        }
    )
    .uv_sync(
        str(ROOT),
        extras=["gpu", "dev"],
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
    .add_local_dir(ROOT / "docs", "/opt/resin/docs", copy=True)
    .add_local_dir(ROOT / "crates", "/opt/resin/crates", copy=True)
    .add_local_dir(ROOT / "resin", "/opt/resin/resin", copy=True)
    .add_local_dir(
        ROOT / "models/cuda",
        "/opt/resin/models/cuda",
        copy=True,
    )
    .uv_pip_install("/opt/resin", extra_options="--no-deps")
    .uv_pip_install(
        "/opt/resin/models/cuda",
        extra_options="--no-deps",
    )
)

with image.imports():
    from resin_models_cuda.deepseek_v4.checkpoint import DEFAULT_MODEL, DEFAULT_REVISION

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


def _diagnose_verify_parity(
    runner,
    prompt_ids: list[int],
    context_length: int,
    tensor_parallel: TensorParallelContext,
) -> dict[str, object]:
    """Compare one six-token verify pass with causal one-token target steps."""

    model = runner._model
    device = runner.device
    block_length = runner.speculative_tokens + 1
    if block_length <= 1:
        raise ValueError("verify parity diagnosis requires speculative decoding")
    graph_seq_len = min(
        context_length,
        len(prompt_ids) + block_length + 64,
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.inference_mode(), torch.device(device):
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None]
            prefill_logits, main_hidden = model.forward_target(prompt)
            if main_hidden is None:
                raise RuntimeError("target did not produce DSpark prefill states")
            model.prefill_dspark(main_hidden)
            current = prefill_logits.argmax(dim=-1, keepdim=True)
            prefill_anchor = current.clone()
            del prefill_logits
            state = [
                (buffer, buffer.clone())
                for _, buffer in runner._sequence_state_buffers()
            ]

            sequential_inputs: list[torch.Tensor] = []
            sequential_logits: list[torch.Tensor] = []
            sequential_layers: list[list[torch.Tensor]] = [[] for _ in model.layers]
            for step in range(block_length):
                token = current.clone()
                sequential_inputs.append(token)
                position = torch.tensor(
                    [len(prompt_ids) + step], dtype=torch.long, device=device
                )
                hidden = model.embed(token)
                hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
                for layer_id, layer in enumerate(model.layers):
                    hidden = layer.forward_decode(
                        hidden,
                        position,
                        token,
                        graph_seq_len,
                    )
                    sequential_layers[layer_id].append(hidden.clone())
                collapsed = model.layers[-1].hc_head(
                    hidden,
                    model.hc_head_fn,
                    model.hc_head_scale,
                    model.hc_head_base,
                )
                logits = model.head(model.norm(collapsed), full_logits=True)
                sequential_logits.append(logits)
                current = logits[:, -1].argmax(dim=-1, keepdim=True)

            def compressor_parity(
                compressor, inputs: torch.Tensor
            ) -> dict[str, object]:
                """Compare the block compressor with repeated one-token updates."""

                assert compressor.kv_cache is not None
                initial_value_state = compressor.kv_state.clone()
                initial_score_state = compressor.score_state.clone()
                initial_cache = compressor.kv_cache.clone()
                sequential_outputs: list[torch.Tensor] = []
                for token_index in range(block_length):
                    token_position = torch.tensor(
                        [len(prompt_ids) + token_index],
                        dtype=torch.long,
                        device=device,
                    )
                    sequential_outputs.append(
                        compressor.forward_decode(
                            inputs[:, token_index : token_index + 1], token_position
                        ).clone()
                    )
                sequential_value_state = compressor.kv_state.clone()
                sequential_score_state = compressor.score_state.clone()
                sequential_cache = compressor.kv_cache.clone()

                compressor.kv_state.copy_(initial_value_state)
                compressor.score_state.copy_(initial_score_state)
                compressor.kv_cache.copy_(initial_cache)
                block_positions = torch.arange(block_length, device=device).unsqueeze(
                    0
                ) + len(prompt_ids)
                block_output = compressor.forward_verify(inputs, block_positions)

                def difference(actual: torch.Tensor, expected: torch.Tensor):
                    delta = (actual.float() - expected.float()).abs()
                    matching_empty = torch.isneginf(actual) & torch.isneginf(expected)
                    delta = torch.where(matching_empty, 0.0, delta)
                    return {
                        "maximum_absolute": float(delta.max().item()),
                        "mean_absolute": float(delta.mean().item()),
                    }

                result = {
                    "outputs": difference(
                        block_output, torch.cat(sequential_outputs, dim=1)
                    ),
                    "final_value_state": difference(
                        compressor.kv_state, sequential_value_state
                    ),
                    "final_score_state": difference(
                        compressor.score_state, sequential_score_state
                    ),
                    "final_cache": difference(compressor.kv_cache, sequential_cache),
                }
                compressor.kv_state.copy_(initial_value_state)
                compressor.score_state.copy_(initial_score_state)
                compressor.kv_cache.copy_(initial_cache)
                return result

            # Layers 0 and 1 are uncompressed. Feed layer 2's compressors the
            # exact per-token normalized inputs from the sequential baseline so
            # this isolates the block scan from all other batched math.
            compressed_layer = model.layers[2]
            compressor_inputs: list[torch.Tensor] = []
            for token_index in range(block_length):
                layer_input, _, _ = compressed_layer.hc_pre(
                    sequential_layers[1][token_index],
                    compressed_layer.hc_attn_fn,
                    compressed_layer.hc_attn_scale,
                    compressed_layer.hc_attn_base,
                )
                compressor_inputs.append(compressed_layer.attn_norm(layer_input))
            compressor_input = torch.cat(compressor_inputs, dim=1)
            compressor_differences = {
                "attention": compressor_parity(
                    compressed_layer.attn.compressor, compressor_input
                ),
                "indexer": compressor_parity(
                    compressed_layer.attn.indexer.compressor, compressor_input
                ),
            }
            torch._foreach_copy_(
                [buffer for buffer, _ in state],
                [snapshot for _, snapshot in state],
            )
            verify_ids = torch.cat(sequential_inputs, dim=1)
            one_block_verify_ids = verify_ids
            base_position = torch.tensor(
                [len(prompt_ids)], dtype=torch.long, device=device
            )
            token_positions = base_position[:, None] + torch.arange(
                block_length, device=device
            )
            hidden = model.embed(verify_ids)
            hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            layer_differences: list[dict[str, float | int]] = []
            for layer_id, layer in enumerate(model.layers):
                hidden = layer.forward_verify(
                    hidden,
                    token_positions,
                    verify_ids,
                    graph_seq_len,
                )
                expected = torch.cat(sequential_layers[layer_id], dim=1)
                difference = (hidden.float() - expected.float()).abs()
                layer_differences.append(
                    {
                        "layer": layer_id,
                        "maximum_absolute": float(difference.max().item()),
                        "mean_absolute": float(difference.mean().item()),
                    }
                )
            collapsed = model.layers[-1].hc_head(
                hidden,
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
            )
            verify_logits = model.head(model.norm(collapsed), full_logits=True)
            expected_logits = torch.cat(sequential_logits, dim=1)
            sequential_top = expected_logits.argmax(dim=-1)
            verify_top = verify_logits.argmax(dim=-1)
            logits_difference = (verify_logits - expected_logits).abs()

            def restore_prefill_state() -> None:
                torch._foreach_copy_(
                    [buffer for buffer, _ in state],
                    [snapshot for _, snapshot in state],
                )

            parity_tokens = 64
            restore_prefill_state()
            sequential_completion = [int(prefill_anchor.item())]
            sequential_anchor = prefill_anchor.clone()
            sequential_position = len(prompt_ids)
            while len(sequential_completion) < parity_tokens:
                position = torch.tensor(
                    [sequential_position], dtype=torch.long, device=device
                )
                logits, _ = model.forward_decode_target(
                    sequential_anchor, position, graph_seq_len
                )
                sequential_anchor = logits.argmax(dim=-1, keepdim=True)
                sequential_completion.append(int(sequential_anchor.item()))
                sequential_position += 1

            restore_prefill_state()
            speculative_completion = [int(prefill_anchor.item())]
            speculative_anchor = prefill_anchor.clone()
            speculative_position = len(prompt_ids)
            speculative_acceptance: list[int] = []
            while len(speculative_completion) < parity_tokens:
                position = torch.tensor(
                    [speculative_position], dtype=torch.long, device=device
                )
                drafts = model.draft(speculative_anchor, position)
                block_verify_ids = torch.cat((speculative_anchor, drafts), dim=1)
                logits, verify_hidden = model.forward_verify(
                    block_verify_ids, position, graph_seq_len
                )
                if verify_hidden is None:
                    raise RuntimeError("target did not produce DSpark verify states")
                model.commit_dspark_main_block(verify_hidden, position)
                targets = logits.argmax(dim=-1)
                accepted = 0
                while (
                    accepted < block_length - 1
                    and targets[0, accepted] == drafts[0, accepted]
                ):
                    accepted += 1
                emitted = [
                    *drafts[0, :accepted].tolist(),
                    int(targets[0, accepted].item()),
                ]
                model.clear_rejected_speculative_state(
                    position,
                    torch.tensor([accepted + 1], dtype=torch.int32, device=device),
                    block_length,
                )
                speculative_acceptance.append(accepted)
                speculative_completion.extend(map(int, emitted))
                speculative_anchor.fill_(emitted[-1])
                speculative_position += len(emitted)

            speculative_completion = speculative_completion[:parity_tokens]
            first_mismatch = next(
                (
                    index
                    for index, (sequential, speculative) in enumerate(
                        zip(sequential_completion, speculative_completion)
                    )
                    if sequential != speculative
                ),
                None,
            )
            restore_prefill_state()
            tensor_parallel.barrier()
            return {
                "prompt_tokens": len(prompt_ids),
                "verify_ids": one_block_verify_ids[0].tolist(),
                "sequential_top": sequential_top[0].tolist(),
                "verify_top": verify_top[0].tolist(),
                "top1_matches": (sequential_top == verify_top)[0].tolist(),
                "logits_maximum_absolute": float(logits_difference.max().item()),
                "logits_mean_absolute": float(logits_difference.mean().item()),
                "compressor_differences": compressor_differences,
                "layer_differences": layer_differences,
                "decode_parity": {
                    "tokens": parity_tokens,
                    "first_mismatch": first_mismatch,
                    "speculative_acceptance": speculative_acceptance,
                    "sequential": sequential_completion,
                    "speculative": speculative_completion,
                },
            }
    finally:
        torch.set_default_dtype(previous_dtype)


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
    profile_run_id: str | None,
    speculative_tokens: int,
    diagnose_verify: bool,
) -> dict[str, object]:
    from resin_models_cuda.deepseek_v4 import DeepSeekV4FlashCudaRunner

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
    runner = DeepSeekV4FlashCudaRunner(
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
    torch.cuda.synchronize(runner.device)
    tensor_parallel.barrier()
    load_elapsed = time.perf_counter() - load_start
    if rank == 0:
        print(
            f"[deepseek-v4] model loaded in {load_elapsed:.2f}s; "
            f"allreduce={runner.allreduce_backend}; capturing graph",
            flush=True,
        )

    if diagnose_verify:
        return _diagnose_verify_parity(
            runner,
            prompt_rows[0],
            context_length,
            tensor_parallel,
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
    generated_steps = 0
    try:
        profile_tables: dict[str, str] = {}
        profile_trace_files: list[dict[str, object]] = []
        profile_phase_timings: list[dict[str, float]] = []
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
                    max_new_tokens=max_new_tokens,
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
            for request in profile_requests:
                while engine.has_work():
                    engine.step()
                request.take_result()
            tensor_parallel.barrier()
            if rank == 0:
                print("[deepseek-v4] profiling complete", flush=True)

        tensor_parallel.barrier()
        if rank == 0:
            print("[deepseek-v4] starting clean timed request", flush=True)
        clean_phase_timings: list[dict[str, float]] = []
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
            replay_count = runner.cuda_graph_stats["replay_count"]
            events = engine.step()
            if (
                runner.cuda_graph_stats["replay_count"] > replay_count
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
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
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
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
    profile_run_id: str | None,
    speculative_tokens: int,
    diagnose_verify: bool,
) -> None:
    tensor_parallel = _initialize_tensor_parallel(
        rank, master_port, tensor_parallel_size
    )
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
        profile_run_id,
        speculative_tokens,
        diagnose_verify,
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
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    profile: bool,
    profile_run_id: str | None = None,
    speculative_tokens: int = DSPARK_TOKENS,
    diagnose_verify: bool = False,
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
                max_new_tokens,
                context_length,
                temperature,
                top_p,
                top_k,
                seed,
                profile,
                profile_run_id,
                speculative_tokens,
                diagnose_verify,
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
        # Force exactly max_new_tokens so every point has the same denominator.
        eos_token_id=None,
    )
    first_token_time: float | None = None
    try:
        tensor_parallel.barrier()
        start = time.perf_counter()
        requests = [
            engine.submit(
                prompt_ids,
                max_new_tokens=max_new_tokens,
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
    from resin_models_cuda.deepseek_v4 import DeepSeekV4FlashCudaRunner

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
    runner = DeepSeekV4FlashCudaRunner(
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
        speculative_tokens=DSPARK_TOKENS,
    )
    torch.cuda.synchronize(runner.device)
    tensor_parallel.barrier()
    load_elapsed = time.perf_counter() - load_start

    capture_elapsed = 0.0
    sampling_params = SamplingParams(
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
        if use_cuda_graphs:
            capture_start = time.perf_counter()
            shape = (batch_size, context_length + DSPARK_TOKENS + 1)
            runner._graphs.capture(  # noqa: SLF001 - benchmark shape isolation
                shape, runner._graph_capture_inputs(shape, runner.vocab_size)
            )
            runner._reset_model_state()  # noqa: SLF001 - reset capture mutations
            torch.cuda.synchronize(runner.device)
            tensor_parallel.barrier()
            capture_elapsed += time.perf_counter() - capture_start
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
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
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
    tensor_parallel_size: int,
    input_tokens: int = 8192,
    max_new_tokens: int = 1024,
    batch_sizes: tuple[int, ...] = BENCHMARK_BATCH_SIZES,
    use_cuda_graphs: bool = True,
) -> dict[str, object]:
    if input_tokens <= 0 or max_new_tokens <= 1:
        raise ValueError("benchmark token counts must be positive")
    model_path, checkpoint = _require_checkpoint()
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    encoding = _load_encoding(model_path)
    seed_prompt = _encode_prompt(
        tokenizer,
        encoding,
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
def cache_checkpoint() -> dict[str, object]:
    return _download_checkpoint()


@app.function(
    gpu="B300:2",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def benchmark_tp2() -> dict[str, object]:
    return _benchmark_remote(2)


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def benchmark_tp4() -> dict[str, object]:
    return _benchmark_remote(4)


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
    reasoning_effort: str = "high",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
    profile_input_tokens: int | None = None,
    speculative_tokens: int = DSPARK_TOKENS,
    diagnose_verify: bool = False,
) -> dict[str, object]:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")
    if reasoning_effort not in {"high", "max"}:
        raise ValueError("reasoning_effort must be 'high' or 'max'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= speculative_tokens <= DSPARK_TOKENS:
        raise ValueError(f"speculative_tokens must be in [0, {DSPARK_TOKENS}]")
    if profile and max_new_tokens < 32:
        raise ValueError("profile runs require at least 32 output tokens")
    model_path, checkpoint = _require_checkpoint()
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        ),
    )
    encoding = _load_encoding(model_path)
    prompt_ids = (
        _real_prompt_ids(tokenizer, profile_input_tokens)
        if profile_input_tokens is not None
        else _encode_prompt(
            tokenizer,
            encoding,
            prompt,
            thinking_mode,
            reasoning_effort,
        )
    )
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
        max_new_tokens,
        context_length,
        temperature,
        top_p,
        top_k,
        seed,
        profile,
        profile_run_id,
        speculative_tokens,
        diagnose_verify,
    )
    if diagnose_verify:
        return report
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
    if profile:
        profile_volume.commit()
    return report


@app.local_entrypoint()
def main(
    mode: str = "run",
    prompt: str = "Explain why batching improves language-model throughput.",
    max_new_tokens: int = 256,
    context_length: int = 32768,
    thinking_mode: str = "chat",
    reasoning_effort: str = "high",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int | None = 0,
    profile: bool = False,
    batch_size: int = 1,
    input_tokens: int = 8192,
    speculative_tokens: int = DSPARK_TOKENS,
) -> None:
    if mode == "download":
        report = cache_checkpoint.remote()
    elif mode == "bench-tp2":
        report = benchmark_tp2.remote()
    elif mode == "bench-tp4":
        report = benchmark_tp4.remote()
    elif mode in {"run", "real", "profile", "diagnose"}:
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
            input_tokens if mode in {"real", "profile", "diagnose"} else None,
            speculative_tokens,
            mode == "diagnose",
        )
    else:
        raise ValueError(
            "mode must be 'download', 'bench-tp2', 'bench-tp4', 'run', 'real', "
            "'profile', or 'diagnose'"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
