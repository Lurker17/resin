"""Serve Thinking Machines' Inkling-Small NVFP4 with gamma-five DSpark."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from deploy.http import monitor_spawned_server
from deploy.modal_image import build_modal_image
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    default_decode_graph_shapes,
)
from resin_models_cuda.inkling.checkpoint import MODEL_SPEC

ROOT = Path(__file__).resolve().parents[1]
MODEL_MOUNT = "/models/resin-inkling"
MODEL_VOLUME_NAME = "resin-inkling-models-v1"
KERNEL_MOUNT = "/kernel-cache/resin-inkling"
KERNEL_VOLUME_NAME = "resin-inkling-kernels-v1"
PROFILE_MOUNT = "/profiles/resin-inkling"
PROFILE_VOLUME_NAME = "resin-inkling-profiles-v1"
TARGET_PATH = Path(MODEL_MOUNT) / "target" / MODEL_SPEC.target_revision
DRAFT_PATH = Path(MODEL_MOUNT) / "draft" / MODEL_SPEC.draft_revision
MAX_CONTEXT_LENGTH = 131_072
SPECULATIVE_TOKENS = 5
TARGET_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "tokenizer*",
    "model*.safetensors",
)
DRAFT_PATTERNS = ("config.json", "model.safetensors")
_http_server_processes: list[object] = []

image = build_modal_image(
    ROOT,
    env={
        "TORCH_CUDA_ARCH_LIST": "10.3a",
        "FLASHINFER_CUDA_ARCH_LIST": "10.3a",
        "FLASHINFER_WORKSPACE_BASE": f"{KERNEL_MOUNT}/flashinfer",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{KERNEL_MOUNT}/flash-attention-4",
        "FLASH_ATTENTION_ARCH": "sm_103a",
        "CUTE_DSL_CACHE_DIR": f"{KERNEL_MOUNT}/cute-dsl",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUDA_TILE_CACHE_DIR": f"{KERNEL_MOUNT}/cutile-python",
        "QUACK_CACHE_ENABLED": "1",
        "QUACK_CACHE_DIR": f"{KERNEL_MOUNT}/quack",
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
profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
app = modal.App(
    "resin-inkling-small",
    image=image,
    volumes={
        MODEL_MOUNT: model_volume,
        KERNEL_MOUNT: kernel_volume,
        PROFILE_MOUNT: profile_volume,
    },
)


def _inventory() -> dict[str, int]:
    index_path = TARGET_PATH / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the pinned Inkling target index is invalid") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("the pinned Inkling target index contains no tensors")
    target_weight_files = {
        TARGET_PATH / str(filename)
        for name, filename in weight_map.items()
        if isinstance(name, str) and name.startswith(("model.llm.", "model.visual."))
    }
    target_files = {
        TARGET_PATH / "config.json",
        TARGET_PATH / "tokenizer.json",
        TARGET_PATH / "tokenizer_config.json",
        index_path,
        *target_weight_files,
    }
    draft_files = {DRAFT_PATH / "config.json", DRAFT_PATH / "model.safetensors"}
    files = sorted(target_files | draft_files)
    if missing := [str(path) for path in files if not path.is_file()]:
        raise RuntimeError(
            f"the pinned Inkling checkpoints are incomplete: {missing[:8]}"
        )
    if empty := [str(path) for path in files if path.stat().st_size <= 0]:
        raise RuntimeError(
            f"the pinned Inkling checkpoints contain empty files: {empty[:8]}"
        )
    return {str(path.relative_to(MODEL_MOUNT)): path.stat().st_size for path in files}


def _require_models() -> dict[str, int]:
    model_volume.reload()
    inventory = _inventory()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Path(KERNEL_MOUNT).mkdir(parents=True, exist_ok=True)
    return inventory


@app.function(cpu=16, memory=32 * 1024, timeout=24 * 60 * 60)
def download_models() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    model_volume.reload()
    started = time.perf_counter()
    cache_hit = True
    try:
        inventory = _inventory()
    except RuntimeError:
        cache_hit = False
        TARGET_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_SPEC.target_repo,
            revision=MODEL_SPEC.target_revision,
            allow_patterns=list(TARGET_PATTERNS),
            local_dir=TARGET_PATH,
        )
        DRAFT_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_SPEC.draft_repo,
            revision=MODEL_SPEC.draft_revision,
            allow_patterns=list(DRAFT_PATTERNS),
            local_dir=DRAFT_PATH,
        )
        inventory = _inventory()
        model_volume.commit()
    return {
        "cache_hit": cache_hit,
        "files": len(inventory),
        "bytes": sum(inventory.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _build_runtime(
    *,
    context_length: int = MAX_CONTEXT_LENGTH,
    cuda_graphs: bool = True,
    gpu_memory_utilization: float = 0.90,
):
    import torch
    from renderers import DefaultRendererConfig, create_renderer
    from transformers import AutoProcessor

    from resin import Engine, SchedulerConfig
    from resin_models_cuda.inkling import InklingCudaRunner

    inventory = _require_models()
    runner = InklingCudaRunner(
        TARGET_PATH,
        DRAFT_PATH,
        max_seq_len=context_length,
        max_batch_size=DEFAULT_MAX_NUM_SEQS,
        speculative_tokens=SPECULATIVE_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
        cuda_graph_decode_shapes=(
            default_decode_graph_shapes(
                context_length + SPECULATIVE_TOKENS,
                speculative=True,
                max_batch_size=DEFAULT_MAX_NUM_SEQS,
            )
            if cuda_graphs
            else ()
        ),
    )
    graph_started = time.perf_counter()
    graph_stats = runner.capture_cuda_graphs()
    graph_capture_s = time.perf_counter() - graph_started
    processor = AutoProcessor.from_pretrained(
        TARGET_PATH, local_files_only=True, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    renderer = create_renderer(tokenizer, DefaultRendererConfig())
    engine = Engine(
        runner,
        max_running_requests=DEFAULT_MAX_NUM_SEQS,
        context_length=context_length,
        eos_token_id=runner.eos_token_id,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=runner.prefill_chunk_size,
            prefill_chunk_size=runner.prefill_chunk_size,
        ),
    )
    report = {
        "checkpoint_files": len(inventory),
        "checkpoint_bytes": sum(inventory.values()),
        "cache": asdict(runner.cache_capacity_plan),
        "cuda_graph_capture_s": graph_capture_s,
        "cuda_graphs": graph_stats,
        "device": torch.cuda.get_device_name(runner.device),
    }
    return torch, runner, processor, renderer, engine, report


def _prompt_ids(renderer: Any, prompt: str) -> list[int]:
    token_ids = renderer.render_ids(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    if not token_ids:
        raise ValueError("Inkling chat template produced no tokens")
    return [int(token_id) for token_id in token_ids]


@app.function(
    gpu="B300:1",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    single_use_containers=True,
)
def smoke_test(
    max_new_tokens: int = 512,
    *,
    context_length: int = 32_768,
    cuda_graphs: bool = True,
) -> dict[str, Any]:
    from resin import SamplingParams

    torch, runner, processor, renderer, engine, report = _build_runtime(
        context_length=context_length,
        cuda_graphs=cuda_graphs,
    )
    try:
        prompt_ids = _prompt_ids(
            renderer,
            "Explain why speculative decoding can preserve model quality.",
        )
        lease = engine.submit(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=renderer.get_stop_token_ids(),
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=0.95,
                seed=7,
            ),
        )
        while engine.has_work():
            engine.step()
        output_ids = lease.take_result()
        torch.cuda.synchronize(runner.device)
        report.update(
            {
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "output_ids": output_ids,
                "text": processor.decode(output_ids, skip_special_tokens=False),
                "speculative": runner.speculative_stats,
                "cuda_graphs": runner.cuda_graph_stats,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runner.device),
            }
        )
        kernel_volume.commit()
        return report
    finally:
        if not engine.closed:
            engine.close()


@app.function(
    gpu="B300:1",
    cpu=4,
    memory=16 * 1024,
    timeout=30 * 60,
    startup_timeout=20 * 60,
    single_use_containers=True,
)
def cutile_kernel_test() -> dict[str, Any]:
    """Compile and numerically validate the persistent KV/sconv fusion."""
    import torch
    import torch.nn.functional as F

    from resin_models_cuda.inkling.cache import (
        HEAD_DIM,
        KV_HEADS,
        PAGE_TOKENS,
        SLAB_ELEMENTS,
        SLAB_UNITS,
        STATE_LENGTH,
    )
    from resin_models_cuda.inkling.cutile import (
        kv_sconv_norm_store,
        router_choice,
        router_weights,
        sconv_residual,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    width = KV_HEADS * HEAD_DIM
    epsilon = 1e-6
    results = []
    for batch, tokens, layer_count, layer, slotnum in (
        (1, 6, 7, 3, 4),
        (2, 6, 35, 17, 1),
        (1, 3, 7, 5, 7),
    ):
        key_input = torch.randn(
            batch, tokens, width, dtype=torch.bfloat16, device=device
        )
        value_input = torch.randn_like(key_input)
        key_state = torch.randn(
            batch, STATE_LENGTH, width, dtype=torch.bfloat16, device=device
        )
        value_state = torch.randn_like(key_state)
        key_weight = torch.randn(width, 1, 4, dtype=torch.float32, device=device) / 8
        value_weight = torch.randn_like(key_weight) / 8
        norm_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)

        physical_slabs = batch + 1
        storage = torch.full(
            (physical_slabs, SLAB_ELEMENTS),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        flat = storage.flatten()
        page_elements = PAGE_TOKENS * KV_HEADS * HEAD_DIM
        page_shape = (
            physical_slabs * SLAB_UNITS,
            PAGE_TOKENS,
            KV_HEADS,
            HEAD_DIM,
        )
        page_stride = (2 * page_elements, KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
        cache_keys = torch.as_strided(flat, page_shape, page_stride)
        cache_values = torch.as_strided(
            flat, page_shape, page_stride, storage_offset=page_elements
        )
        slabs = torch.arange(1, batch + 1, dtype=torch.int32, device=device)[:, None]
        slots = torch.full_like(slabs, slotnum)
        first_blocks = torch.ones(batch, dtype=torch.int32, device=device)
        positions = 128 + torch.arange(tokens, dtype=torch.long, device=device)
        positions = positions[None].expand(batch, -1).contiguous()

        next_key, next_value = kv_sconv_norm_store(
            key_input,
            value_input,
            key_state,
            value_state,
            key_weight,
            value_weight,
            norm_weight,
            cache_keys,
            cache_values,
            slabs,
            slots,
            first_blocks,
            positions,
            page_tokens=PAGE_TOKENS,
            slab_units=SLAB_UNITS,
            layer_count=layer_count,
            layer=layer,
            epsilon=epsilon,
        )
        torch.cuda.synchronize(device)

        def reference(
            inputs: torch.Tensor,
            state: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            combined = torch.cat((state, inputs), dim=1)
            convolved = F.conv1d(
                combined.transpose(1, 2).float(),
                weight,
                groups=width,
            ).transpose(1, 2)
            output = (convolved + inputs.float()).to(torch.bfloat16)
            return output, combined[:, -STATE_LENGTH:].to(state.dtype)

        expected_key, expected_next_key = reference(key_input, key_state, key_weight)
        expected_value, expected_next_value = reference(
            value_input, value_state, value_weight
        )
        shaped_key = expected_key.view(batch, tokens, KV_HEADS, HEAD_DIM)
        key_float = shaped_key.float()
        expected_key = (
            key_float
            * torch.rsqrt(key_float.square().mean(-1, keepdim=True) + epsilon)
            * norm_weight.float()
        ).to(torch.bfloat16)
        pages = slabs[:, 0] * SLAB_UNITS + slotnum * layer_count + layer
        actual_key = torch.stack(
            [
                cache_keys[pages[row], positions[row] % PAGE_TOKENS]
                for row in range(batch)
            ]
        )
        actual_value = torch.stack(
            [
                cache_values[pages[row], positions[row] % PAGE_TOKENS]
                for row in range(batch)
            ]
        )

        errors = {
            "key": float((actual_key - expected_key).abs().max()),
            "value": float(
                (actual_value - expected_value.view(batch, tokens, KV_HEADS, HEAD_DIM))
                .abs()
                .max()
            ),
            "next_key": float((next_key - expected_next_key).abs().max()),
            "next_value": float((next_value - expected_next_value).abs().max()),
        }
        if errors["key"] > 0.03125 or errors["value"] > 0.03125:
            raise AssertionError(f"persistent KV/sconv numerical mismatch: {errors}")
        if errors["next_key"] or errors["next_value"]:
            raise AssertionError(f"persistent KV/sconv state mismatch: {errors}")
        results.append(
            {
                "batch": batch,
                "tokens": tokens,
                "layer_count": layer_count,
                "errors": errors,
                "finite_key_elements": int(actual_key.isfinite().sum()),
                "finite_value_elements": int(actual_value.isfinite().sum()),
            }
        )
    sconv_results = []
    for batch, tokens in ((1, 6), (2, 6), (1, 3)):
        width = 4096
        inputs = torch.randn(batch, tokens, width, dtype=torch.bfloat16, device=device)
        state = torch.randn(
            batch, STATE_LENGTH, width, dtype=torch.bfloat16, device=device
        )
        weight = torch.randn(width, 1, 4, dtype=torch.float32, device=device) / 8
        residual = torch.randn_like(inputs)
        output, next_state = sconv_residual(inputs, state, weight, residual)
        torch.cuda.synchronize(device)
        expected, expected_state = reference(inputs, state, weight)
        expected = residual + expected
        errors = {
            "output": float((output - expected).abs().max()),
            "next_state": float((next_state - expected_state).abs().max()),
        }
        if errors["output"] or errors["next_state"]:
            raise AssertionError(f"persistent sconv/residual mismatch: {errors}")
        sconv_results.append({"batch": batch, "tokens": tokens, "errors": errors})
    router_results = []
    for rows, routed_experts, shared_experts, top_k in (
        (1, 256, 2, 6),
        (12, 256, 2, 6),
        (4, 64, 1, 4),
        (8, 128, 2, 4),
    ):
        logits = torch.randn(
            rows,
            routed_experts + shared_experts,
            dtype=torch.bfloat16,
            device=device,
        )
        bias = torch.randn(routed_experts, dtype=torch.bfloat16, device=device) / 8
        global_scale = torch.tensor([1.125], dtype=torch.bfloat16, device=device)
        route_scale = 1.3
        choice = router_choice(logits, bias)
        indices = torch.topk(choice, top_k, dim=-1, sorted=False).indices
        float_logits = logits.float()
        routed, shared = router_weights(
            float_logits,
            indices,
            global_scale,
            route_scale,
            routed_experts=routed_experts,
            shared_experts=shared_experts,
        )
        torch.cuda.synchronize(device)

        expected_choice = torch.sigmoid(float_logits[:, :routed_experts]) + bias.float()
        expected_indices = torch.topk(
            expected_choice, top_k, dim=-1, sorted=False
        ).indices
        selected = torch.cat(
            (
                float_logits[:, :routed_experts].gather(1, expected_indices),
                float_logits[:, routed_experts:],
            ),
            dim=-1,
        )
        expected_weights = torch.softmax(F.logsigmoid(selected), dim=-1)
        expected_weights *= route_scale * global_scale.float()
        expected_weights = expected_weights.to(torch.bfloat16)
        errors = {
            "choice": float((choice - expected_choice).abs().max()),
            "routed": float((routed - expected_weights[:, :top_k]).abs().max()),
            "shared": float((shared - expected_weights[:, top_k:]).abs().max()),
        }
        if not torch.equal(indices, expected_indices):
            raise AssertionError("persistent router changed top-k selection")
        if (
            errors["choice"] > 1e-6
            or max(errors["routed"], errors["shared"]) > 0.00390625
        ):
            raise AssertionError(f"persistent router numerical mismatch: {errors}")
        router_results.append(
            {
                "rows": rows,
                "routed_experts": routed_experts,
                "shared_experts": shared_experts,
                "top_k": top_k,
                "errors": errors,
            }
        )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_results = {}
    schedule_programs = sorted({64, 128, 256, 2 * sm_count})
    for batch in (1, 128):
        tokens = 6
        width = 4096
        inputs = torch.randn(batch, tokens, width, dtype=torch.bfloat16, device=device)
        state = torch.randn(
            batch, STATE_LENGTH, width, dtype=torch.bfloat16, device=device
        )
        weight = torch.randn(width, 1, 4, dtype=torch.float32, device=device) / 8
        residual = torch.randn_like(inputs)
        timings = {}
        for programs in schedule_programs:
            for _ in range(3):
                sconv_residual(
                    inputs,
                    state,
                    weight,
                    residual,
                    program_count=programs,
                )
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                sconv_residual(
                    inputs,
                    state,
                    weight,
                    residual,
                    program_count=programs,
                )
            end.record()
            end.synchronize()
            timings[str(programs)] = start.elapsed_time(end) * 10.0
        schedule_results[str(batch)] = timings
    kernel_volume.commit()
    return {
        "device": torch.cuda.get_device_name(device),
        "kv_sconv_cases": results,
        "router_cases": router_results,
        "schedule": {
            "sm_count": sm_count,
            "occupancy": 2,
            "sconv_us_by_batch_and_programs": schedule_results,
        },
        "sconv_residual_cases": sconv_results,
    }


@app.function(
    gpu="B300:1",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    single_use_containers=True,
)
def perfetto_trace(
    max_new_tokens: int = 1024,
    *,
    context_length: int = 32_768,
    warmup_steps: int = 16,
    benchmark_steps: int = 64,
    profile_steps: int = 32,
) -> dict[str, Any]:
    """Benchmark and trace steady-state batch-one speculative decode."""
    from resin import SamplingParams

    if warmup_steps < 0 or benchmark_steps <= 0 or profile_steps <= 0:
        raise ValueError(
            "warmup_steps must be nonnegative and measured steps must be positive"
        )
    total_steps = warmup_steps + benchmark_steps + profile_steps
    if max_new_tokens < 1 + 6 * total_steps:
        raise ValueError("max_new_tokens is too small for the requested decode windows")

    torch, runner, processor, renderer, engine, report = _build_runtime(
        context_length=context_length,
        cuda_graphs=True,
    )
    lease = engine.submit(
        _prompt_ids(
            renderer,
            "Write an exhaustive technical tutorial on speculative decoding, "
            "including derivations, examples, and implementation details.",
        ),
        max_new_tokens=max_new_tokens,
        stop_token_ids=(),
        sampling_params=SamplingParams(temperature=1.0, top_p=0.95, seed=7),
    )
    try:
        # Cross prefill and warm several fixed-shape graph replays. A gamma-five
        # decode step can emit as many as six output tokens.
        while engine.has_work():
            if engine.step():
                break
        for _ in range(warmup_steps):
            if not engine.has_work():
                raise RuntimeError("profile request ended during decode warmup")
            engine.step()
        torch.cuda.synchronize(runner.device)

        benchmark_before = dict(runner.speculative_stats)
        benchmark_events = 0
        benchmark_started = time.perf_counter()
        for _ in range(benchmark_steps):
            if not engine.has_work():
                raise RuntimeError("profile request ended during benchmark window")
            benchmark_events += len(engine.step())
        torch.cuda.synchronize(runner.device)
        benchmark_elapsed_s = time.perf_counter() - benchmark_started
        benchmark_after = dict(runner.speculative_stats)

        profile_events = 0
        profile_started = time.perf_counter()
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
        ) as profile:
            for step in range(profile_steps):
                if not engine.has_work():
                    raise RuntimeError("profile request ended during trace window")
                with torch.profiler.record_function(
                    f"step[INKLING_DSPARK_DECODE bs=1 index={step}]"
                ):
                    profile_events += len(engine.step())
            torch.cuda.synchronize(runner.device)
        profile_elapsed_s = time.perf_counter() - profile_started

        while engine.has_work():
            engine.step()
        output_ids = lease.take_result()

        trace_name = f"inkling-small-bs1-decode-{time.time_ns()}.trace.json.gz"
        trace_path = Path(PROFILE_MOUNT) / trace_name
        profile.export_chrome_trace(str(trace_path))
        profile_volume.commit()

        speculative_delta = {
            key: benchmark_after[key] - benchmark_before[key]
            for key in ("blocks", "drafts", "accepted", "emitted")
        }
        cuda_events = sorted(
            (
                event
                for event in profile.key_averages()
                if event.self_device_time_total > 0
            ),
            key=lambda event: event.self_device_time_total,
            reverse=True,
        )
        total_cuda_time_us = sum(event.self_device_time_total for event in cuda_events)
        report.update(
            {
                "benchmark_decode_steps": benchmark_steps,
                "benchmark_output_tokens": benchmark_events,
                "benchmark_elapsed_s": benchmark_elapsed_s,
                "benchmark_output_tok_s": benchmark_events / benchmark_elapsed_s,
                "benchmark_step_ms": (benchmark_elapsed_s * 1000 / benchmark_steps),
                "speculative_delta": speculative_delta,
                "speculative_acceptance_rate": (
                    speculative_delta["accepted"] / speculative_delta["drafts"]
                    if speculative_delta["drafts"]
                    else 0.0
                ),
                "profiled_decode_steps": profile_steps,
                "profiled_output_tokens": profile_events,
                "profiled_elapsed_s": profile_elapsed_s,
                "output_tokens": len(output_ids),
                "output_ids_tail": output_ids[-256:],
                "text_tail": processor.decode(
                    output_ids[-256:], skip_special_tokens=False
                ),
                "perfetto_trace": trace_name,
                "trace_bytes": trace_path.stat().st_size,
                "trace_volume": PROFILE_VOLUME_NAME,
                "total_cuda_time_us": total_cuda_time_us,
                "cuda_time_per_step_us": total_cuda_time_us / profile_steps,
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
                    for event in cuda_events[:40]
                ],
            }
        )
        return report
    finally:
        if not engine.closed:
            engine.close()


@app.function(
    gpu="B300:1",
    cpu=16,
    memory=256 * 1024,
    timeout=12 * 60 * 60,
    startup_timeout=60 * 60,
    single_use_containers=True,
)
def throughput_curve(
    *,
    context_length: int = 32_768,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 96, 128),
    warmup_steps: int = 8,
    benchmark_steps: int = 32,
) -> dict[str, Any]:
    """Measure the interactive-throughput frontier for live speculative batches."""
    from resin import SamplingParams

    if (
        not batch_sizes
        or any(size <= 0 or size > DEFAULT_MAX_NUM_SEQS for size in batch_sizes)
        or tuple(sorted(set(batch_sizes))) != batch_sizes
    ):
        raise ValueError("batch sizes must be unique, increasing, and within capacity")
    if warmup_steps < 0 or benchmark_steps <= 0:
        raise ValueError("invalid throughput benchmark window")

    torch, runner, _, renderer, engine, report = _build_runtime(
        context_length=context_length,
        cuda_graphs=True,
    )
    prompt_ids = _prompt_ids(
        renderer,
        "Write a detailed explanation of efficient GPU inference.",
    )
    results = []
    try:
        for batch_size in batch_sizes:
            leases = [
                engine.submit(
                    prompt_ids,
                    max_new_tokens=1 + 6 * (warmup_steps + benchmark_steps + 2),
                    stop_token_ids=(),
                    sampling_params=SamplingParams(
                        temperature=1.0,
                        top_p=0.95,
                        seed=10_000 * batch_size + row,
                    ),
                )
                for row in range(batch_size)
            ]

            # Admit and prefill the complete batch before warming decode replays.
            engine.step()
            for _ in range(warmup_steps):
                if any(lease.done for lease in leases):
                    raise RuntimeError("throughput request ended during warmup")
                engine.step()
            torch.cuda.synchronize(runner.device)

            speculative_before = dict(runner.speculative_stats)
            output_tokens = 0
            started = time.perf_counter()
            for _ in range(benchmark_steps):
                if any(lease.done for lease in leases):
                    raise RuntimeError("throughput request ended during measurement")
                output_tokens += len(engine.step())
            torch.cuda.synchronize(runner.device)
            elapsed_s = time.perf_counter() - started
            speculative_after = dict(runner.speculative_stats)
            speculative_delta = {
                key: speculative_after[key] - speculative_before[key]
                for key in ("blocks", "drafts", "accepted", "emitted")
            }
            aggregate_tok_s = output_tokens / elapsed_s
            results.append(
                {
                    "batch_size": batch_size,
                    "benchmark_steps": benchmark_steps,
                    "elapsed_s": elapsed_s,
                    "output_tokens": output_tokens,
                    "aggregate_output_tok_s": aggregate_tok_s,
                    "interactive_output_tok_s": aggregate_tok_s / batch_size,
                    "step_ms": elapsed_s * 1000 / benchmark_steps,
                    "speculative_acceptance_rate": (
                        speculative_delta["accepted"] / speculative_delta["drafts"]
                        if speculative_delta["drafts"]
                        else 0.0
                    ),
                    "speculative_delta": speculative_delta,
                }
            )

            for lease in leases:
                engine.cancel(lease)
            while engine.has_work():
                engine.step()

        report.update(
            {
                "prompt_tokens": len(prompt_ids),
                "warmup_steps": warmup_steps,
                "benchmark_steps": benchmark_steps,
                "results": results,
                "cuda_graphs": runner.cuda_graph_stats,
            }
        )
        return report
    finally:
        if not engine.closed:
            engine.close()


def _serve(_: int) -> None:
    import os

    from resin import SamplingParams
    from resin.server import HttpServer
    from resin_models_cuda.inkling import InklingCodec

    torch, runner, processor, renderer, engine, _ = _build_runtime()
    try:
        warmup = engine.submit(
            _prompt_ids(renderer, "Reply with one word."),
            max_new_tokens=2,
            stop_token_ids=renderer.get_stop_token_ids(),
            sampling_params=SamplingParams(temperature=0.0),
        )
        while engine.has_work():
            engine.step()
        warmup.take_result()
        torch.cuda.synchronize(runner.device)
        HttpServer(
            engine,
            codec=InklingCodec(processor),
            model=MODEL_SPEC.target_repo,
            enable_anthropic=False,
        ).serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()
    os._exit(0)


@app.function(
    gpu="B300:1",
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
    label="inkling-small",
    requires_proxy_auth=False,
)
def serve_inkling() -> None:
    import torch

    process_context = torch.multiprocessing.spawn(_serve, nprocs=1, join=False)
    _http_server_processes.append(process_context)
    monitor_spawned_server(process_context, "inkling-small-http")


@app.local_entrypoint()
def main(mode: str = "smoke", max_new_tokens: int = 512) -> None:
    if mode == "download":
        report = download_models.remote()
    elif mode == "smoke":
        report = smoke_test.remote(max_new_tokens)
    elif mode == "eager-smoke":
        report = smoke_test.remote(max_new_tokens, cuda_graphs=False)
    elif mode == "profile":
        report = perfetto_trace.remote(max_new_tokens)
    elif mode == "curve":
        report = throughput_curve.remote()
    elif mode == "cutile-test":
        report = cutile_kernel_test.remote()
    else:
        raise ValueError(
            "mode must be 'download', 'smoke', 'eager-smoke', 'profile', "
            "'curve', or 'cutile-test'"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
