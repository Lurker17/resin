"""Run one tokenizer-free GPT-OSS request over Resin's flat cache on Modal.

The GPU path is deliberately small: it loads a pinned checkpoint only from the
model volume, creates one ``GptOssCudaRunner`` and one ``Engine``, then audits a
cold prompt plus fixed greedy decode. Kernel prewarming and fresh-container
reuse use the shared content-validated cache helpers in :mod:`deploy.jit_cache`.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from types import MethodType
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

SMOKE_PROTOCOL = "gpt-oss-flat-cache-smoke-v1"
PROMPT_TOKENS = 257
MAX_NEW_TOKENS = 4
PAGE_TOKENS = 128
WINDOW_RESIDENT_BLOCKS = 2
DEFAULT_CACHE_PARENT_CAPACITY = 8
AUTOTUNE_BUCKETS = (1, 128)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal"
    )
    .env(
        {
            "PATH": (
                "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
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
    )
    .run_commands(
        "python -m pip install flashinfer-cubin==0.6.15.post1 "
        "--index-url https://flashinfer.ai/whl"
    )
    .run_commands(
        "python -m pip install 'flashinfer-jit-cache==0.6.15.post1+cu130' "
        "--index-url https://flashinfer.ai/whl/cu130"
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


def _download_model(model_key: str) -> dict[str, object]:
    """Populate one exact checkpoint on CPU and publish its marker last."""

    _model_path(model_key)  # Validate before touching the Volume.
    model_cache_volume.reload()
    cache_hit = _model_is_complete(model_key)
    start = time.perf_counter()
    if not cache_hit:
        from huggingface_hub import snapshot_download

        spec = MODEL_SPECS[model_key]
        target = _model_path(model_key)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=list(MODEL_FILES),
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
    if not _model_is_complete(model_key):
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
    if not _model_is_complete(model_key):
        raise RuntimeError(
            "pinned checkpoint is absent or incomplete; run mode=download first"
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


def _required_parent_capacity() -> int:
    maximum_input = PROMPT_TOKENS + MAX_NEW_TOKENS - 1
    history_blocks = (maximum_input + PAGE_TOKENS - 1) // PAGE_TOKENS
    return history_blocks + WINDOW_RESIDENT_BLOCKS


def _validate_inputs(model_key: str, cache_mode: str, capacity: int) -> None:
    if model_key not in MODEL_SPECS:
        raise ValueError("model must be '20b' or '120b'")
    if cache_mode not in {"prewarm", "verify"}:
        raise ValueError("cache_mode must be 'prewarm' or 'verify'")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("cache_parent_capacity must be a positive integer")
    if capacity < _required_parent_capacity():
        raise ValueError(
            f"cache_parent_capacity must be at least {_required_parent_capacity()}"
        )


def _prompt_token_ids(vocab_size: int) -> list[int]:
    """Create one fixed raw-ID prompt without importing a tokenizer."""

    usable_vocab = min(vocab_size, 100_000)
    if usable_vocab <= 256:
        raise ValueError("GPT-OSS vocabulary is unexpectedly small")
    state = 0x6D2B79F5
    prompt = []
    for _ in range(PROMPT_TOKENS):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        prompt.append(256 + state % (usable_vocab - 256))
    return prompt


def _token_digest(token_ids: list[int]) -> str:
    from blake3 import blake3

    digest = blake3()
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def _integer_tuple(name: str, value: Any) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise AssertionError(f"{name} is not iterable") from error
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise AssertionError(f"{name} contains a non-integer")
    return result


def _validate_flat_plan(plan: Any, runner: Any) -> dict[str, object]:
    """Audit the exact sparse History/Window plan accepted by the runner."""

    if plan.cache_pool_id != runner._cache_pool_id:  # noqa: SLF001
        raise AssertionError("plan and runner cache pool identities differ")
    sequence_ids = _integer_tuple("sequence_ids", plan.sequence_ids)
    token_ids = _integer_tuple("token_ids", plan.token_ids)
    positions = _integer_tuple("positions", plan.positions)
    context_lengths = _integer_tuple("context_lengths", plan.context_lengths)
    sequence_indptr = _integer_tuple("sequence_indptr", plan.sequence_indptr)
    if len(sequence_ids) != 1 or sequence_indptr != (0, len(token_ids)):
        raise AssertionError("smoke plan must contain exactly one complete row")
    if not positions or len(positions) != len(token_ids):
        raise AssertionError("plan positions do not cover every query token")
    if positions != tuple(
        range(context_lengths[0], context_lengths[0] + len(token_ids))
    ):
        raise AssertionError("plan positions are not contiguous with their context")

    expected_phase = "prefill" if positions[0] < PROMPT_TOKENS else "decode"
    phase_rows = (
        _integer_tuple("prefill_sequence_indices", plan.prefill_sequence_indices),
        _integer_tuple("decode_sequence_indices", plan.decode_sequence_indices),
    )
    if plan.mode != expected_phase or phase_rows != (
        (0,) if expected_phase == "prefill" else (),
        (0,) if expected_phase == "decode" else (),
    ):
        raise AssertionError("plan phase partition is not exact")
    expected_sample_rows = (
        (0,) if expected_phase == "decode" or positions[-1] + 1 == PROMPT_TOKENS else ()
    )
    if (
        _integer_tuple("sample_sequence_indices", plan.sample_sequence_indices)
        != expected_sample_rows
    ):
        raise AssertionError("smoke plan sampling does not match prompt progress")

    layouts = tuple(runner.cache_layout.groups)
    groups = tuple(plan.cache_groups)
    if len(layouts) != 2 or len(groups) != 2:
        raise AssertionError("GPT-OSS requires ordered History and Window groups")
    if (layouts[0].retention, layouts[1].retention) != ("history", "window"):
        raise AssertionError("runner cache layout order is not History then Window")

    all_blocks: set[int] = set()
    first_logical_blocks = []
    group_block_ids = []
    post_length = positions[-1] + 1
    for group_index, (layout, group) in enumerate(zip(layouts, groups, strict=True)):
        if group.group_index != group_index or layout.blocks_per_parent != 1:
            raise AssertionError("cache group identity or parent packing is invalid")
        block_ids = _integer_tuple("block_ids", group.block_ids)
        block_indptr = _integer_tuple("block_indptr", group.block_indptr)
        first = _integer_tuple("first_logical_blocks", group.first_logical_blocks)
        append_ids = _integer_tuple("append_block_ids", group.append_block_ids)
        append_offsets = _integer_tuple(
            "append_block_offsets", group.append_block_offsets
        )
        if block_indptr != (0, len(block_ids)) or len(first) != 1:
            raise AssertionError("cache group CSR does not cover its only row")
        if len(set(block_ids)) != len(block_ids) or any(
            block <= 0 or block > runner.cache_parent_capacity for block in block_ids
        ):
            raise AssertionError("cache group contains an invalid physical parent")
        if all_blocks.intersection(block_ids):
            raise AssertionError("History and Window cannot share a live parent")
        all_blocks.update(block_ids)

        block_tokens = layout.block_tokens
        total_blocks = (post_length + block_tokens - 1) // block_tokens
        resident = (
            total_blocks
            if layout.retention == "history"
            else min(total_blocks, layout.window_resident_block_count)
        )
        expected_first = total_blocks - resident
        if first != (expected_first,) or len(block_ids) != resident:
            raise AssertionError("cache group does not expose its chronological suffix")
        if len(append_ids) != len(positions) or len(append_offsets) != len(positions):
            raise AssertionError("cache append metadata does not cover every token")

        logical_to_physical = {}
        for offset, block_id in enumerate(block_ids):
            logical = expected_first + offset
            slot = logical if layout.retention == "history" else logical % resident
            logical_to_physical[slot] = block_id
        for position, block_id, block_offset in zip(
            positions, append_ids, append_offsets, strict=True
        ):
            logical = position // block_tokens
            slot = logical if layout.retention == "history" else logical % resident
            if (
                logical_to_physical[slot] != block_id
                or block_offset != position % block_tokens
            ):
                raise AssertionError("cache append destination is not exact")
        first_logical_blocks.append(expected_first)
        group_block_ids.append(list(block_ids))

    try:
        plan.mode = plan.mode
    except AttributeError:
        frozen = True
    else:
        raise AssertionError("native BatchPlan unexpectedly accepted mutation")
    return {
        "batch_id": plan.batch_id,
        "mode": plan.mode,
        "context_length": context_lengths[0],
        "query_tokens": len(token_ids),
        "position_start": positions[0],
        "position_end": positions[-1],
        "first_logical_blocks": first_logical_blocks,
        "group_block_ids": group_block_ids,
        "frozen": frozen,
    }


def _install_plan_audit(runner: Any) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    forward = runner.forward

    def audited_forward(self, plan):
        traces.append(_validate_flat_plan(plan, self))
        return forward(plan)

    runner.forward = MethodType(audited_forward, runner)
    return traces


def _cold_smoke_report(
    model_key: str,
    model_path: Path,
    model_cache: dict[str, object],
    cache_parent_capacity: int,
) -> dict[str, object]:
    import sys

    import torch

    from resin import Engine
    from resin.backends.gpt_oss_cuda import GptOssCudaRunner
    from resin.backends.gpt_oss_flashinfer import make_flashinfer_sm90_mxfp4_kernel

    if tuple(torch.cuda.get_device_capability()) != (9, 0):
        raise RuntimeError("the GPT-OSS smoke harness requires an SM90 GPU")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    config = json.loads((model_path / "config.json").read_text())
    prompt = _prompt_token_ids(config["vocab_size"])

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    runner = GptOssCudaRunner(
        model_path,
        cache_parent_capacity=cache_parent_capacity,
        device="cuda",
        moe_kernel=make_flashinfer_sm90_mxfp4_kernel(),
        validate_first_logits=True,
    )
    traces = _install_plan_audit(runner)
    engine = Engine(runner)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000

    if (
        runner.bind_cache_pool(engine.cache_pool_id, runner.cache_layout)
        != engine.cache_pool_id
    ):
        raise AssertionError("runner cache binding is not idempotent")
    if (
        engine.cache_parent_capacity != cache_parent_capacity
        or engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != cache_parent_capacity
    ):
        raise AssertionError("new Engine cache state is not empty and exact")

    output_ids = []
    run_start = time.perf_counter()
    try:
        request_id = engine.submit(prompt, max_new_tokens=MAX_NEW_TOKENS)
        expected_reserved = _required_parent_capacity()
        if (
            engine.cache_occupied_block_count != expected_reserved
            or engine.cache_free_parent_count
            != cache_parent_capacity - expected_reserved
        ):
            raise AssertionError("request reservation does not match both cache groups")

        step_count = 0
        while engine.has_work():
            step_count += 1
            if step_count > 2 * _required_parent_capacity():
                raise AssertionError("smoke Engine exceeded its bounded step budget")
            events = engine.step()
            if not events:
                continue
            if len(events) != 1 or events[0].request_id != request_id:
                raise AssertionError("sampled smoke steps must emit one request event")
            event = events[0]
            generation_index = len(output_ids)
            if event.index != generation_index or event.finished != (
                generation_index == MAX_NEW_TOKENS - 1
            ):
                raise AssertionError("token event progress is not exact")
            output_ids.append(event.token_id)
        torch.cuda.synchronize()

        expected_contexts = [
            0,
            PAGE_TOKENS,
            2 * PAGE_TOKENS,
            *range(PROMPT_TOKENS, PROMPT_TOKENS + MAX_NEW_TOKENS - 1),
        ]
        expected_query_tokens = [PAGE_TOKENS, PAGE_TOKENS, 1, 1, 1, 1]
        expected_modes = [
            "prefill",
            "prefill",
            "prefill",
            *("decode" for _ in range(MAX_NEW_TOKENS - 1)),
        ]
        if (
            engine.has_work()
            or request_id.take_result() != output_ids
            or len(output_ids) != MAX_NEW_TOKENS
            or [trace["context_length"] for trace in traces] != expected_contexts
            or [trace["query_tokens"] for trace in traces] != expected_query_tokens
            or [trace["mode"] for trace in traces] != expected_modes
        ):
            raise AssertionError("cold prompt/decode trajectory is not exact")
        if (
            engine.cache_occupied_block_count != 0
            or engine.cache_free_parent_count != cache_parent_capacity
            or runner.forward_calls != len(expected_contexts)
        ):
            raise AssertionError("completed request did not release the flat cache")
        engine.close()
        if not engine.closed:
            raise AssertionError("Engine did not close")
    finally:
        if not engine.closed:
            engine.close()

    tokenizer_modules = [
        name for name in ("transformers", "tokenizers") if name in sys.modules
    ]
    if tokenizer_modules:
        raise AssertionError(
            f"tokenizer modules entered the GPU path: {tokenizer_modules}"
        )
    return {
        "protocol": SMOKE_PROTOCOL,
        "model": {
            "key": model_key,
            "repo_id": MODEL_SPECS[model_key]["repo_id"],
            "revision": MODEL_SPECS[model_key]["revision"],
            "checkpoint": model_cache,
        },
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "prompt": {
            "tokens": len(prompt),
            "blake3": _token_digest(prompt),
            "tokenizer": None,
        },
        "output_ids": output_ids,
        "output_blake3": _token_digest(output_ids),
        "plan_trace": traces,
        "cache": {
            "parent_capacity": cache_parent_capacity,
            "groups": ["history", "window"],
            "occupied_after_completion": engine.cache_occupied_block_count,
            "free_after_completion": engine.cache_free_parent_count,
            "content_reuse_claimed": False,
        },
        "engine_closed": engine.closed,
        "load_ms": load_ms,
        "run_ms": (time.perf_counter() - run_start) * 1000,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "tokenizer_modules_loaded": tokenizer_modules,
    }


def _autotune_path(model_key: str) -> Path:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    return (
        Path(KERNEL_CACHE_ROOT)
        / "flashinfer-autotune"
        / (f"{SMOKE_PROTOCOL}-{model_key}-{revision}.json")
    )


def _workload_name(model_key: str) -> str:
    revision = MODEL_SPECS[model_key]["revision"][:12]
    return f"{SMOKE_PROTOCOL}-{model_key}-{revision}"


def _run_smoke(
    model_key: str, cache_mode: str, cache_parent_capacity: int
) -> dict[str, object]:
    _validate_inputs(model_key, cache_mode, cache_parent_capacity)
    model_path, model_cache = _require_cached_model(model_key)

    def workload() -> dict[str, object]:
        install_flashinfer_hash_cache()

        import flashinfer
        from flashinfer.autotuner import AutoTuner

        path = _autotune_path(model_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_loaded = None
        if path.exists():
            cache_loaded = AutoTuner.get().load_configs(str(path))
            if not cache_loaded:
                raise RuntimeError(f"FlashInfer rejected its saved tactics: {path}")
        elif cache_mode == "verify":
            raise RuntimeError("saved FlashInfer tactics are absent; prewarm first")
        tune_missing = cache_mode == "prewarm"
        with flashinfer.autotune(
            tune_missing,
            cache=str(path) if tune_missing else None,
            tuning_buckets=AUTOTUNE_BUCKETS,
            round_up=True,
        ):
            report = _cold_smoke_report(
                model_key,
                model_path,
                model_cache,
                cache_parent_capacity,
            )
        if not path.is_file():
            raise RuntimeError("FlashInfer did not publish its tactic file")
        report["flashinfer_autotune"] = {
            "mode": "tune_missing" if tune_missing else "immutable_reuse",
            "cache_loaded": cache_loaded,
            "path": str(path),
            "buckets": list(AUTOTUNE_BUCKETS),
        }
        return report

    if cache_mode == "prewarm":
        report = run_cache_prewarm(
            kernel_cache_volume, _workload_name(model_key), workload
        )
    else:
        report = run_cache_reuse_probe(
            kernel_cache_volume, _workload_name(model_key), workload
        )
        cache = report["kernel_cache"]
        if (
            cache["cache_hit"] is not True
            or cache["stable_artifact_changes"]
            or cache["flashinfer_hash_cache_misses"] != 0
        ):
            raise AssertionError("fresh-container kernel reuse was not immutable")
    if report["kernel_cache"]["stable_artifact_count"] <= 0:
        raise AssertionError("kernel cache contains no stable artifacts")
    return report


@app.function(timeout=4 * 60 * 60, cpu=4, memory=8 * 1024)
def cache_checkpoint(model_key: str) -> dict[str, object]:
    return _download_model(model_key)


@app.function(
    gpu="H100!",
    cpu=16,
    memory=128 * 1024,
    timeout=4 * 60 * 60,
    **single_use_options,
)
def smoke_model(
    model_key: str,
    cache_mode: str,
    cache_parent_capacity: int = DEFAULT_CACHE_PARENT_CAPACITY,
) -> dict[str, object]:
    return _run_smoke(model_key, cache_mode, cache_parent_capacity)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    model: str = "20b",
    cache_mode: str = "prewarm",
    cache_parent_capacity: int = DEFAULT_CACHE_PARENT_CAPACITY,
) -> None:
    """Use ``download`` once, then ``smoke`` with prewarm or verify."""

    if mode == "download":
        report = cache_checkpoint.remote(model)
    elif mode == "smoke":
        _validate_inputs(model, cache_mode, cache_parent_capacity)
        report = smoke_model.remote(model, cache_mode, cache_parent_capacity)
    else:
        raise ValueError("mode must be 'download' or 'smoke'")
    print(json.dumps(report, indent=2, sort_keys=True))
