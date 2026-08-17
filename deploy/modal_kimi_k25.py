"""Download and run Kimi K2.5-family NVFP4 targets with LightSeek EAGLE."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import cast

import modal
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from deploy.http import (
    KimiK25Codec,
    TensorParallelServerRunner,
    follow_tensor_parallel_server,
    monitor_spawned_server,
)
from deploy.modal_image import build_modal_image
from resin import Engine, SamplingParams, SchedulerConfig
from resin.backend.cuda.graphs import DEFAULT_MAX_NUM_SEQS
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.server import HttpServer

ROOT = Path(__file__).resolve().parents[1]
MODEL_MOUNT = "/models/resin-kimi-k2"
MODEL_VOLUME_NAME = "resin-kimi-k2-models-v1"
KERNEL_MOUNT = "/kernel-cache/.cache"
KERNEL_VOLUME_NAME = "resin-kimi-k2-kernels-v1"
TP_SIZE = 4
SPECULATIVE_TOKENS = 3

image = build_modal_image(
    ROOT,
    env={
        "CC": "gcc",
        "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "10.3",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "FLASHINFER_WORKSPACE_BASE": "/kernel-cache",
        "FLASHINFER_LOGGING_LEVEL": "INFO",
        "FLASHINFER_LOG_LEVEL": "INFO",
        "FLASHINFER_JIT_VERBOSE": "1",
        "TRITON_CACHE_DIR": f"{KERNEL_MOUNT}/triton",
        "CUDA_TILE_CACHE_DIR": f"{KERNEL_MOUNT}/cutile-python",
    },
)

with image.imports():
    from resin_models_cuda.kimi_k25.checkpoint import (
        DEFAULT_MODEL_KEY,
        MODEL_SPECS,
        KimiModelSpec,
        model_spec,
    )

model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME, create_if_missing=True, version=1
)
kernel_volume = modal.Volume.from_name(
    KERNEL_VOLUME_NAME, create_if_missing=True, version=1
)
app = modal.App(
    "resin-kimi-k25",
    image=image,
    volumes={MODEL_MOUNT: model_volume, KERNEL_MOUNT: kernel_volume},
)

_http_server_processes: list[object] = []

TARGET_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "model*.safetensors",
)
DRAFT_PATTERNS = ("config.json", "model.safetensors")


def _paths(spec: KimiModelSpec) -> tuple[Path, Path]:
    root = Path(MODEL_MOUNT) / spec.key
    return root / "target" / spec.target_revision, root / "draft" / spec.draft_revision


def _marker(spec: KimiModelSpec) -> Path:
    return Path(MODEL_MOUNT) / spec.key / ".resin-checkpoint.json"


def _inventory(spec: KimiModelSpec) -> dict[str, int]:
    target, draft = _paths(spec)
    index_path = target / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Kimi target has an invalid safetensor index") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Kimi target safetensor index has no weights")
    shards = sorted({target / str(name) for name in weight_map.values()})
    files = [
        target / "config.json",
        target / "configuration_deepseek.py",
        target / "configuration_kimi_k25.py",
        target / "kimi_k25_processor.py",
        target / "kimi_k25_vision_processing.py",
        target / "media_utils.py",
        target / "modeling_deepseek.py",
        target / "modeling_kimi_k25.py",
        target / "preprocessor_config.json",
        target / "processor_config.json",
        target / "tokenization_kimi.py",
        target / "tokenizer_config.json",
        target / "tiktoken.model",
        target / "tool_declaration_ts.py",
        index_path,
        *shards,
        draft / "config.json",
        draft / "model.safetensors",
    ]
    if missing := [str(path) for path in files if not path.is_file()]:
        raise RuntimeError(f"Kimi checkpoint is missing files: {missing[:8]}")
    if empty := [str(path) for path in files if path.stat().st_size <= 0]:
        raise RuntimeError(f"Kimi checkpoint has empty files: {empty[:8]}")
    root = Path(MODEL_MOUNT) / spec.key
    return {str(path.relative_to(root)): path.stat().st_size for path in sorted(files)}


def _complete(spec: KimiModelSpec) -> bool:
    try:
        marker = json.loads(_marker(spec).read_text())
        inventory = _inventory(spec)
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    return marker == {
        "schema": 1,
        "target_repo": spec.target_repo,
        "target_revision": spec.target_revision,
        "draft_repo": spec.draft_repo,
        "draft_revision": spec.draft_revision,
        "eagle_version": spec.eagle_version.value,
        "files": inventory,
    }


def _download(spec: KimiModelSpec) -> dict[str, object]:
    model_volume.reload()
    cache_hit = _complete(spec)
    started = time.perf_counter()
    if not cache_hit:
        target, draft = _paths(spec)
        target.mkdir(parents=True, exist_ok=True)
        draft.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.target_repo,
            revision=spec.target_revision,
            allow_patterns=list(TARGET_PATTERNS),
            local_dir=target,
        )
        snapshot_download(
            repo_id=spec.draft_repo,
            revision=spec.draft_revision,
            allow_patterns=list(DRAFT_PATTERNS),
            local_dir=draft,
        )
        marker = {
            "schema": 1,
            "target_repo": spec.target_repo,
            "target_revision": spec.target_revision,
            "draft_repo": spec.draft_repo,
            "draft_revision": spec.draft_revision,
            "eagle_version": spec.eagle_version.value,
            "files": _inventory(spec),
        }
        temporary = _marker(spec).with_suffix(f".{os.getpid()}.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(marker, sort_keys=True))
        temporary.replace(_marker(spec))
        model_volume.commit()
    if not _complete(spec):
        raise RuntimeError("Kimi checkpoint publication did not complete")
    inventory = _inventory(spec)
    return {
        "model": spec.key,
        "target_repo": spec.target_repo,
        "target_revision": spec.target_revision,
        "draft_repo": spec.draft_repo,
        "draft_revision": spec.draft_revision,
        "eagle_version": spec.eagle_version.value,
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def _require(spec: KimiModelSpec) -> tuple[Path, Path, dict[str, object]]:
    model_volume.reload()
    if not _complete(spec):
        raise RuntimeError(
            f"run --mode download --model {spec.key} before Kimi inference"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    target, draft = _paths(spec)
    return (
        target,
        draft,
        {
            "target_repo": spec.target_repo,
            "target_revision": spec.target_revision,
            "draft_repo": spec.draft_repo,
            "draft_revision": spec.draft_revision,
            "eagle_version": spec.eagle_version.value,
        },
    )


@app.function(cpu=8, memory=16 * 1024, timeout=24 * 60 * 60)
def cache_checkpoint(model_key: str = DEFAULT_MODEL_KEY) -> dict[str, object]:
    return _download(model_spec(model_key))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_tp(rank: int, port: int) -> TensorParallelContext:
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=TP_SIZE,
        timeout=timedelta(minutes=30),
        device_id=torch.device(f"cuda:{rank}"),
    )
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=TP_SIZE), dist.group.WORLD
    )


def _decode_graph_shapes(max_batch_size: int) -> tuple[tuple[int], ...]:
    """Use a few batch buckets without specializing on sequence length."""

    buckets = (1, 2, 4, 8, 16, 32, 64, 128)
    return tuple((size,) for size in buckets if size <= max_batch_size)


def _activation_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    selected = values[finite]
    return {
        "min": float(selected.min()) if selected.size else float("nan"),
        "max": float(selected.max()) if selected.size else float("nan"),
        "mean": float(selected.mean()) if selected.size else float("nan"),
        "variance": float(selected.var()) if selected.size else float("nan"),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def _compare_activations(
    reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, object]:
    reference_values = reference.cpu().numpy()
    actual_values = actual.cpu().numpy()
    difference = actual_values - reference_values
    reference_norm = float(np.linalg.norm(reference_values))
    actual_norm = float(np.linalg.norm(actual_values))
    denominator = reference_norm * actual_norm
    return {
        "reference": _activation_stats(reference_values),
        "decode": _activation_stats(actual_values),
        "max_abs_error": float(np.abs(difference).max()),
        "mean_abs_error": float(np.abs(difference).mean()),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(reference_norm, 1e-30)
        ),
        "cosine_similarity": (
            float(np.dot(reference_values, actual_values) / denominator)
            if denominator
            else float("nan")
        ),
    }


def _diagnose_target_decode(
    runner, prompt_ids: list[int], context_length: int, rank: int
) -> tuple[list[int], dict[str, object] | None]:
    """Compare cached one-token decode with a full-prefill oracle."""

    captures: dict[str, dict[str, torch.Tensor]] = {
        "reference": {},
        "decode": {},
        "naive": {},
    }
    capture_mode: list[str | None] = [None]
    handles = []

    def capture(name: str, *, output: bool = True):
        def hook(_module, inputs, result=None) -> None:
            mode = capture_mode[0]
            if mode is None or name in captures[mode] or rank != 0:
                return
            tensor = result if output else inputs[0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"diagnostic hook {name} did not receive a tensor")
            if tensor.ndim == 3:
                tensor = tensor[0, -1]
            elif tensor.ndim == 2:
                tensor = tensor[0]
            captures[mode][name] = tensor.detach().float().clone()

        return hook

    if rank == 0:
        target = runner._model.target
        for layer_id, layer in enumerate(target.layers):
            handles.append(
                layer.register_forward_pre_hook(
                    capture(f"layer.{layer_id:02d}.input", output=False)
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    capture(f"layer.{layer_id:02d}.attention")
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(capture(f"layer.{layer_id:02d}.mlp"))
            )
        handles.append(target.norm.register_forward_hook(capture("final.norm")))
        handles.append(target.lm_head.register_forward_hook(capture("final.logits")))

    token_count = len(prompt_ids) + 1
    page_count = (token_count + 63) // 64
    required_parents = 4 * page_count
    if required_parents > runner.cache_layout.parent_capacity:
        raise RuntimeError("diagnostic scratch pages exceed Kimi cache capacity")

    def scratch_cache(target_parent: int, aux_parent: int):
        runner._cache.storage[target_parent : target_parent + page_count].zero_()
        runner._cache.storage[aux_parent : aux_parent + page_count].zero_()
        cache = runner._cache.graph_cache(1, runner.physical_seq_len)
        for group in cache.groups:
            group.parent_ids.zero_()
            group.child_slots.zero_()
        pages = torch.arange(
            target_parent,
            target_parent + page_count,
            dtype=torch.int32,
            device=runner.device,
        )
        cache.groups[0].parent_ids[0, :page_count].copy_(pages)
        aux_pages = torch.arange(
            aux_parent,
            aux_parent + page_count,
            dtype=torch.int32,
            device=runner.device,
        )
        cache.groups[2].parent_ids[0, :page_count].copy_(aux_pages)
        return cache

    actual_cache = scratch_cache(1, 1 + page_count)
    reference_cache = scratch_cache(1 + 2 * page_count, 1 + 3 * page_count)
    target = runner._model.target
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=runner.device)
    prompt_positions = torch.arange(
        len(prompt_ids), dtype=torch.long, device=runner.device
    )[None]
    prompt_length = torch.tensor(
        [len(prompt_ids)], dtype=torch.int32, device=runner.device
    )
    previous_dtype = torch.get_default_dtype()
    original_attention = []
    layer35_inputs: dict[str, torch.Tensor] = {}
    kernel_probes: dict[str, dict[str, float | int] | str] = {}
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.inference_mode(), torch.device(runner.device):
            prompt_logits, _ = target(
                prompt,
                prompt_positions,
                actual_cache,
                prompt_length,
                runner.physical_seq_len,
                full_logits=False,
                prefill_length=len(prompt_ids),
            )
            anchor_tensor = prompt_logits.argmax(dim=-1)
            anchor = int(anchor_tensor[0].item())
            complete_tokens = torch.cat((prompt, anchor_tensor[:, None]), dim=1)
            complete_positions = torch.arange(
                token_count, dtype=torch.long, device=runner.device
            )[None]
            complete_length = torch.tensor(
                [token_count], dtype=torch.int32, device=runner.device
            )

            capture_mode[0] = "reference"
            reference_logits, _ = target(
                complete_tokens,
                complete_positions,
                reference_cache,
                complete_length,
                runner.physical_seq_len,
                full_logits=False,
                prefill_length=token_count,
            )
            capture_mode[0] = None

            layer35_attention = target.layers[35].self_attn
            layer35_method = layer35_attention._run_decode_attention

            def capture_layer35_inputs(
                _attention,
                query: torch.Tensor,
                kv_cache: torch.Tensor,
                block_tables: torch.Tensor,
                sequence_lengths: torch.Tensor,
                max_sequence_length: int,
            ) -> torch.Tensor:
                layer35_inputs.update(
                    query=query.detach().clone(),
                    kv_cache=kv_cache.detach(),
                    block_tables=block_tables.detach().clone(),
                    sequence_lengths=sequence_lengths.detach().clone(),
                    max_sequence_length=torch.tensor(max_sequence_length),
                )
                return layer35_method(
                    query,
                    kv_cache,
                    block_tables,
                    sequence_lengths,
                    max_sequence_length,
                )

            layer35_attention._run_decode_attention = MethodType(
                capture_layer35_inputs, layer35_attention
            )
            try:
                capture_mode[0] = "decode"
                decode_logits, _ = target(
                    anchor_tensor[:, None],
                    torch.tensor([[len(prompt_ids)]], device=runner.device),
                    actual_cache,
                    complete_length,
                    token_count,
                    full_logits=False,
                )
                capture_mode[0] = None
            finally:
                layer35_attention._run_decode_attention = layer35_method

            def naive_attention(
                attention,
                query: torch.Tensor,
                kv_cache: torch.Tensor,
                block_tables: torch.Tensor,
                sequence_lengths: torch.Tensor,
                _max_sequence_length: int,
            ) -> torch.Tensor:
                outputs = []
                for row in range(query.shape[0]):
                    sequence_length = int(sequence_lengths[row].item())
                    pages = (sequence_length + 63) // 64
                    page_ids = block_tables[row, :pages].long()
                    history = kv_cache[page_ids].flatten(0, 1)[:sequence_length]
                    scores = torch.einsum(
                        "thd,sd->ths", query[row].float(), history.float()
                    )
                    scores.mul_(attention.scale)
                    probabilities = scores.softmax(dim=-1)
                    outputs.append(
                        torch.einsum(
                            "ths,sc->thc",
                            probabilities,
                            history[:, : attention.kv_lora_rank].float(),
                        )
                    )
                return torch.stack(outputs).to(torch.bfloat16)

            from tokenspeed_mla import tokenspeed_mla_decode

            probe_query = layer35_inputs["query"]
            probe_lengths = layer35_inputs["sequence_lengths"].to(torch.int32)
            probe_length = int(probe_lengths.max().item())
            probe_pages = (probe_length + 63) // 64
            probe_page_ids = layer35_inputs["block_tables"][0, :probe_pages].long()
            probe_cache = layer35_inputs["kv_cache"][probe_page_ids].clone()
            probe_table = torch.arange(
                probe_pages, dtype=torch.int32, device=runner.device
            )[None]
            fp8_limit = torch.finfo(torch.float8_e4m3fn).max
            fp8_query = probe_query.clamp(-fp8_limit, fp8_limit).to(torch.float8_e4m3fn)
            zero_padded_cache = probe_cache.clone()
            zero_padded_cache.flatten(0, 1)[probe_length:].zero_()

            def tensor_stats(tensor: torch.Tensor) -> dict[str, float | int]:
                return _activation_stats(tensor.detach().float().cpu().numpy())

            kernel_probes.update(
                query_bf16=tensor_stats(probe_query),
                query_fp8=tensor_stats(fp8_query),
                query_fp8_saturated=int(
                    (probe_query.float().abs() > fp8_limit).sum().item()
                ),
                cache_active=tensor_stats(probe_cache.flatten(0, 1)[:probe_length]),
                cache_padding=tensor_stats(probe_cache.flatten(0, 1)[probe_length:]),
            )

            def run_kernel_probe(
                name: str,
                query: torch.Tensor,
                cache: torch.Tensor,
                *,
                variable: bool,
                pdl: bool,
            ) -> None:
                try:
                    output = tokenspeed_mla_decode(
                        query=query,
                        kv_cache=cache,
                        workspace_buffer=layer35_attention.workspace,
                        kv_lora_rank=layer35_attention.kv_lora_rank,
                        qk_rope_head_dim=layer35_attention.qk_rope_head_dim,
                        block_tables=probe_table,
                        seq_lens=probe_lengths,
                        max_seq_len=probe_length,
                        softmax_scale=layer35_attention.scale,
                        output_scale=1.0,
                        is_var_seq=variable,
                        causal_mask=True,
                        enable_pdl=pdl,
                    )
                    torch.cuda.synchronize(runner.device)
                    kernel_probes[name] = tensor_stats(output)
                except Exception as error:
                    kernel_probes[name] = f"{type(error).__name__}: {error}"

            for variable, pdl in (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ):
                suffix = f"{'variable' if variable else 'persistent'}_pdl_{pdl}"
                run_kernel_probe(
                    f"fp8_{suffix}",
                    fp8_query,
                    probe_cache,
                    variable=variable,
                    pdl=pdl,
                )
            run_kernel_probe(
                "fp8_variable_pdl_true_zero_padding",
                fp8_query,
                zero_padded_cache,
                variable=True,
                pdl=True,
            )
            run_kernel_probe(
                "bf16_variable_pdl_true",
                probe_query,
                probe_cache.to(torch.bfloat16),
                variable=True,
                pdl=True,
            )
            kernel_probes["fp8_naive"] = tensor_stats(
                naive_attention(
                    layer35_attention,
                    fp8_query,
                    probe_cache,
                    probe_table,
                    probe_lengths,
                    probe_length,
                )
            )

            for layer in target.layers:
                attention = layer.self_attn
                original_attention.append((attention, attention._run_decode_attention))
                attention._run_decode_attention = MethodType(naive_attention, attention)
            capture_mode[0] = "naive"
            naive_logits, _ = target(
                anchor_tensor[:, None],
                torch.tensor([[len(prompt_ids)]], device=runner.device),
                actual_cache,
                complete_length,
                token_count,
                full_logits=False,
            )
            capture_mode[0] = None
            reference_output = reference_logits.argmax(dim=-1).tolist()
            decode_next = decode_logits.argmax(dim=-1).tolist()
            naive_output = naive_logits.argmax(dim=-1).tolist()
            decoded_output = [anchor, *decode_next]
    finally:
        for attention, method in original_attention:
            attention._run_decode_attention = method
        torch.set_default_dtype(previous_dtype)
        capture_mode[0] = None
        for handle in handles:
            handle.remove()

    torch.cuda.synchronize(runner.device)
    runner.tensor_parallel.barrier()
    if rank != 0:
        return decoded_output, None
    for name, result in kernel_probes.items():
        print(f"[kimi-diagnose:layer35-kernel] {name}={result}", flush=True)
    expected = set(captures["reference"])
    if mismatched := {
        mode: sorted(expected ^ set(captures[mode]))
        for mode in ("decode", "naive")
        if expected ^ set(captures[mode])
    }:
        raise RuntimeError(f"diagnostic activation sets differ: {mismatched}")
    comparisons = {
        mode: {
            name: _compare_activations(reference_tensor, captures[mode][name])
            for name, reference_tensor in captures["reference"].items()
        }
        for mode in ("decode", "naive")
    }
    for mode, mode_comparisons in comparisons.items():
        for name, comparison in mode_comparisons.items():
            reference_stats = comparison["reference"]
            actual_stats = comparison["decode"]
            print(
                f"[kimi-diagnose:{mode}] {name} "
                f"ref=({reference_stats['min']:.6g},{reference_stats['max']:.6g},"
                f"var={reference_stats['variance']:.6g},"
                f"nan={reference_stats['nan_count']},inf={reference_stats['inf_count']}) "
                f"actual=({actual_stats['min']:.6g},{actual_stats['max']:.6g},"
                f"var={actual_stats['variance']:.6g},"
                f"nan={actual_stats['nan_count']},inf={actual_stats['inf_count']}) "
                f"max_err={comparison['max_abs_error']:.6g} "
                f"rel_l2={comparison['relative_l2_error']:.6g} "
                f"cos={comparison['cosine_similarity']:.8f}",
                flush=True,
            )

    def comparison_summary(mode: str) -> dict[str, object]:
        values = comparisons[mode]
        first_nonfinite = next(
            (
                name
                for name, comparison in values.items()
                if comparison["decode"]["nan_count"]
                or comparison["decode"]["inf_count"]
            ),
            None,
        )
        return {
            "first_nonfinite": first_nonfinite,
            "final_norm": values["final.norm"],
            "final_logits": values["final.logits"],
        }

    return decoded_output, {
        "anchor_token": anchor,
        "reference_output": reference_output,
        "decode_output": decoded_output,
        "naive_output": [anchor, *naive_output],
        "layer35_kernel_probes": kernel_probes,
        "activation_summary": {mode: comparison_summary(mode) for mode in comparisons},
    }


def _run_rank(
    rank: int,
    port: int,
    report_path: Path,
    target_path: Path,
    draft_path: Path,
    spec: KimiModelSpec,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    seed: int,
    speculative_tokens: int,
    cuda_graph: bool,
    diagnose_target: bool,
) -> None:
    from resin_models_cuda.kimi_k25 import KimiK25CudaRunner

    tp = _initialize_tp(rank, port)
    if rank == 0:
        print(
            f"[kimi] loading {spec.key} TP=4 with {spec.eagle_version.value}",
            flush=True,
        )
    started = time.perf_counter()
    runner = KimiK25CudaRunner(
        target_path,
        draft_path,
        max_seq_len=context_length,
        max_batch_size=1,
        tensor_parallel=tp,
        device=f"cuda:{rank}",
        cuda_graph_decode_shapes=((1,),) if cuda_graph else (),
        speculative_tokens=speculative_tokens,
        eagle_version=spec.eagle_version,
    )
    torch.cuda.synchronize(runner.device)
    tp.barrier()
    load_elapsed = time.perf_counter() - started
    if rank == 0:
        cache = runner.cache_capacity_plan
        print(
            f"[kimi] loaded in {load_elapsed:.2f}s; cache="
            f"{cache.allocation_bytes / 2**30:.2f} GiB; capturing graph",
            flush=True,
        )
    if diagnose_target:
        output_ids, diagnostic = _diagnose_target_decode(
            runner, prompt_ids, context_length, rank
        )
        if rank == 0:
            report_path.write_text(
                json.dumps(
                    {
                        "model": spec.key,
                        "device": torch.cuda.get_device_name(runner.device),
                        "model_load_ms": load_elapsed * 1000,
                        "weight_load": runner.weight_load,
                        "diagnostic": diagnostic,
                        "_output_ids": output_ids,
                    }
                )
            )
        os._exit(0)
    capture_started = time.perf_counter()
    runner.capture_cuda_graphs()
    tp.barrier()
    capture_elapsed = time.perf_counter() - capture_started

    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=context_length,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
        eos_token_id=runner.eos_token_id,
    )
    request_started = time.perf_counter()
    first_token_at: float | None = None
    try:
        request = engine.submit(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling_params=SamplingParams(
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            ),
        )
        while engine.has_work():
            if engine.step() and first_token_at is None:
                first_token_at = time.perf_counter()
        torch.cuda.synchronize(runner.device)
        tp.barrier()
        output_ids = request.take_result()
    finally:
        if not engine.closed:
            engine.close()
    request_elapsed = time.perf_counter() - request_started
    if rank == 0:
        report_path.write_text(
            json.dumps(
                {
                    "model": spec.key,
                    "eagle_version": spec.eagle_version.value,
                    "device": torch.cuda.get_device_name(runner.device),
                    "allreduce_backend": runner.allreduce_backend,
                    "weight_load": runner.weight_load,
                    "model_load_ms": load_elapsed * 1000,
                    "cuda_graph_capture_ms": capture_elapsed * 1000,
                    "cuda_graph": runner.cuda_graph_stats,
                    "cuda_graph_requested": cuda_graph,
                    "speculative": runner.speculative_stats,
                    "speculative_tokens": speculative_tokens,
                    "prompt_tokens": len(prompt_ids),
                    "output_tokens": len(output_ids),
                    "time_to_first_token_ms": (
                        None
                        if first_token_at is None
                        else (first_token_at - request_started) * 1000
                    ),
                    "elapsed_ms": request_elapsed * 1000,
                    "_output_ids": output_ids,
                }
            )
        )
    os._exit(0)


def _serve_http_rank(
    rank: int,
    port: int,
    target_path: Path,
    draft_path: Path,
    spec: KimiModelSpec,
) -> None:
    """Run one TP rank; rank zero additionally owns the public HTTP listener."""

    from resin_models_cuda.kimi_k25 import KimiK25CudaRunner

    tp = _initialize_tp(rank, port)
    control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    if rank == 0:
        print(
            f"[http-startup] loading {spec.key} TP={TP_SIZE} with "
            f"{spec.eagle_version.value}",
            flush=True,
        )
    runner = KimiK25CudaRunner(
        target_path,
        draft_path,
        max_seq_len=131072,
        max_batch_size=DEFAULT_MAX_NUM_SEQS,
        tensor_parallel=tp,
        device=f"cuda:{rank}",
        cuda_graph_decode_shapes=_decode_graph_shapes(DEFAULT_MAX_NUM_SEQS),
        speculative_tokens=SPECULATIVE_TOKENS,
        eagle_version=spec.eagle_version,
    )
    if rank == 0:
        cache = runner.cache_capacity_plan
        print(
            "[http-startup] fungible cache "
            f"parents={cache.parent_capacity:,}; "
            f"parent={cache.bytes_per_parent / 2**20:.2f} MiB; "
            f"allocation={cache.allocation_bytes / 2**30:.2f} GiB",
            flush=True,
        )
    torch.cuda.synchronize(runner.device)
    tp.barrier()
    runner.capture_cuda_graphs()
    tp.barrier()

    if rank != 0:
        follow_tensor_parallel_server(runner, control_group, dist)
        os._exit(0)

    processor = AutoProcessor.from_pretrained(
        str(target_path), local_files_only=True, trust_remote_code=True
    )
    codec = KimiK25Codec(processor)
    coordinated_runner = TensorParallelServerRunner(runner, control_group, dist)
    engine = Engine(
        coordinated_runner,
        max_running_requests=DEFAULT_MAX_NUM_SEQS,
        context_length=131072,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=DEFAULT_MAX_NUM_SEQS * runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
        eos_token_id=runner.eos_token_id,
    )
    warmup_requests = (
        {
            "messages": [{"role": "user", "content": "word " * 4096}],
            "reasoning_effort": None,
            "stop": [],
        },
        {
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "reasoning_effort": "low",
            "stop": [],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": np.full(
                                (192, 256, 3),
                                (0.90, 0.10, 0.14),
                                dtype=np.float32,
                            ),
                        },
                        {"type": "text", "text": "Name the color."},
                    ],
                }
            ],
            "reasoning_effort": None,
            "stop": [],
        },
    )
    for request in warmup_requests:
        rendered = codec.render(request)
        prompt_ids = rendered["token_ids"]
        started = time.perf_counter()
        warmup = engine.submit(
            prompt_ids,
            media=rendered.get("media"),
            max_new_tokens=2,
            sampling_params=SamplingParams(temperature=0.0),
        )
        while engine.has_work():
            engine.step()
        warmup.take_result()
        print(
            f"[http-startup] warmed {len(prompt_ids):,}-token prompt in "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
    torch.cuda.synchronize(runner.device)
    kernel_volume.commit()
    print("[http-startup] warmup complete; starting HTTP server", flush=True)
    try:
        HttpServer(engine, codec=codec, model=spec.target_repo).serve(
            host="0.0.0.0", port=8000
        )
    finally:
        if not engine.closed:
            engine.close()
        coordinated_runner.close_followers()
    os._exit(0)


@app.function(
    gpu="B200:4",
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
    label="kimi-k2-6",
    requires_proxy_auth=False,
)
def serve_kimi_k26() -> None:
    spec = model_spec(DEFAULT_MODEL_KEY)
    target, draft, _ = _require(spec)
    kernel_volume.reload()
    process_context = torch.multiprocessing.spawn(
        _serve_http_rank,
        args=(_free_port(), target, draft, spec),
        nprocs=TP_SIZE,
        join=False,
    )
    _http_server_processes.append(process_context)
    monitor_spawned_server(process_context, "kimi-k2.6-http")


@app.function(
    gpu="B200:4",
    cpu=32,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    single_use_containers=True,
)
def run_model_tp4(
    prompt: str,
    model_key: str = DEFAULT_MODEL_KEY,
    max_new_tokens: int = 512,
    context_length: int = 32768,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    speculative_tokens: int = SPECULATIVE_TOKENS,
    cuda_graph: bool = True,
    diagnose_target: bool = False,
) -> dict[str, object]:
    if not 0 <= speculative_tokens <= SPECULATIVE_TOKENS:
        raise ValueError(
            f"speculative_tokens must be between zero and {SPECULATIVE_TOKENS}"
        )
    spec = model_spec(model_key)
    target, draft, checkpoint = _require(spec)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            str(target), local_files_only=True, trust_remote_code=True
        ),
    )
    encoded_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        thinking=thinking,
    )
    if isinstance(encoded_prompt, Mapping):
        encoded_prompt = encoded_prompt["input_ids"]
    if torch.is_tensor(encoded_prompt):
        encoded_prompt = encoded_prompt.flatten().tolist()
    elif encoded_prompt and isinstance(encoded_prompt[0], list):
        [encoded_prompt] = encoded_prompt
    prompt_ids = [int(token_id) for token_id in encoded_prompt]
    if len(prompt_ids) + max_new_tokens > context_length:
        raise ValueError("prompt plus output exceeds context_length")
    with tempfile.TemporaryDirectory(prefix="resin-kimi-") as directory:
        report_path = Path(directory) / "report.json"
        torch.multiprocessing.spawn(
            _run_rank,
            args=(
                _free_port(),
                report_path,
                target,
                draft,
                spec,
                prompt_ids,
                max_new_tokens,
                context_length,
                temperature,
                top_p,
                seed,
                speculative_tokens,
                cuda_graph,
                diagnose_target,
            ),
            nprocs=TP_SIZE,
            join=True,
        )
        report = cast(dict[str, object], json.loads(report_path.read_text()))
    output_ids = cast(list[int], report.pop("_output_ids"))
    report["output_ids"] = output_ids
    report["text"] = tokenizer.decode(output_ids, skip_special_tokens=False)
    report["checkpoint"] = checkpoint
    kernel_volume.commit()
    return report


@app.local_entrypoint()
def main(
    mode: str = "run",
    model: str = DEFAULT_MODEL_KEY,
    prompt: str = "Explain speculative decoding in two paragraphs.",
    max_new_tokens: int = 512,
    context_length: int = 32768,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    speculative_tokens: int = SPECULATIVE_TOKENS,
    cuda_graph: bool = True,
) -> None:
    if model not in MODEL_SPECS:
        raise ValueError(f"model must be one of: {', '.join(MODEL_SPECS)}")
    if mode == "download":
        report = cache_checkpoint.remote(model)
    elif mode in {"run", "diagnose"}:
        diagnose_target = mode == "diagnose"
        report = run_model_tp4.remote(
            prompt,
            model,
            max_new_tokens,
            context_length,
            thinking,
            temperature,
            top_p,
            seed,
            0 if diagnose_target else speculative_tokens,
            False if diagnose_target else cuda_graph,
            diagnose_target,
        )
    else:
        raise ValueError("mode must be 'download', 'run', or 'diagnose'")
    print(json.dumps(report, indent=2, sort_keys=True))
