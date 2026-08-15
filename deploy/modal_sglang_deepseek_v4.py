"""Benchmark and profile SGLang's verified DeepSeek-V4-0731 DSpark path."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

MODEL_CACHE_MOUNT = "/models/resin-deepseek-v4"
MODEL_CACHE_VOLUME_NAME = "resin-deepseek-v4-models-v1"
MODEL_KEY = "flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
LOCAL_MODEL_ROOT = "/local-model"
PROFILE_MOUNT = "/profiles"
PROFILE_VOLUME_NAME = "resin-deepseek-v4-sglang-profiles-v1"
SERVER_PORT = 30000
PROMPT_ROOT = Path("/benchmark/resin")
PROMPT_FILES = (
    "README.md",
    "docs/architecture.md",
    "docs/scheduler.md",
    "crates/resin-py/src/lib.rs",
)

image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .env({"SGLANG_TORCH_PROFILER_DIR": PROFILE_MOUNT})
    .add_local_file("README.md", f"{PROMPT_ROOT}/README.md")
    .add_local_file("docs/architecture.md", f"{PROMPT_ROOT}/docs/architecture.md")
    .add_local_file("docs/scheduler.md", f"{PROMPT_ROOT}/docs/scheduler.md")
    .add_local_file(
        "crates/resin-py/src/lib.rs",
        f"{PROMPT_ROOT}/crates/resin-py/src/lib.rs",
    )
)
model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME, create_if_missing=False, version=1
)
profile_volume = modal.Volume.from_name(
    PROFILE_VOLUME_NAME, create_if_missing=True, version=1
)
app = modal.App(
    "resin-sglang-deepseek-v4-profile",
    image=image,
    volumes={
        MODEL_CACHE_MOUNT: model_cache_volume,
        PROFILE_MOUNT: profile_volume,
    },
)


def _model_path() -> Path:
    return Path(MODEL_CACHE_MOUNT) / MODEL_KEY / MODEL_REVISION


def _stage_checkpoint(source: Path, destination: Path) -> None:
    """Copy independent checkpoint shards from the volume to local NVMe."""

    files = [path for path in source.rglob("*") if path.is_file()]
    for path in files:
        (destination / path.relative_to(source)).parent.mkdir(
            parents=True, exist_ok=True
        )

    def copy(path: Path) -> None:
        shutil.copy2(path, destination / path.relative_to(source))

    with ThreadPoolExecutor(max_workers=12) as executor:
        tuple(executor.map(copy, files))


def _request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 60.0,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        content_type = response.headers.get_content_type()
    if not body:
        return None
    if content_type == "application/json":
        return json.loads(body)
    return body.decode(errors="replace")


def _wait_for_server(process: subprocess.Popen[str], timeout: float = 2700.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited during startup with {process.returncode}"
            )
        try:
            _request_json("/health", timeout=5.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2.0)
    raise TimeoutError("SGLang did not become healthy within 45 minutes")


def _real_prompt_ids(model_path: Path, input_tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    body = "\n\n".join(
        f"===== {relative_path} =====\n{(PROMPT_ROOT / relative_path).read_text()}"
        for relative_path in PROMPT_FILES
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
    return [*prefix_ids, *body_ids[:body_budget], *suffix_ids]


def _stream_generate(input_ids: list[int], output_tokens: int) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    first_event_at: float | None = None
    last_event_at = started_at
    last_event: dict[str, Any] = {}
    event_count = 0
    with urllib.request.urlopen(request, timeout=3600.0) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            now = time.perf_counter()
            if first_event_at is None:
                first_event_at = now
            last_event_at = now
            last_event = event
            event_count += 1

    if first_event_at is None:
        raise RuntimeError("SGLang streaming response contained no token events")
    meta = last_event.get("meta_info") or {}
    completion_tokens = int(meta.get("completion_tokens") or event_count)
    decode_seconds = last_event_at - first_event_at
    return {
        "prompt_tokens": int(meta.get("prompt_tokens") or len(input_ids)),
        "completion_tokens": completion_tokens,
        "event_count": event_count,
        "ttft_seconds": first_event_at - started_at,
        "decode_seconds": decode_seconds,
        "elapsed_seconds": last_event_at - started_at,
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds if decode_seconds > 0 else None
        ),
        "text": last_event.get("text", ""),
    }


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    ephemeral_disk=512 * 1024,
    timeout=8 * 60 * 60,
    single_use_containers=True,
)
def benchmark_and_profile(
    input_tokens: int = 8192,
    output_tokens: int = 1024,
    profile_output_tokens: int = 64,
) -> dict[str, Any]:
    cached_model_path = _model_path()
    if not (cached_model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(
            f"cached 0731 checkpoint is incomplete: {cached_model_path}"
        )
    model_path = Path(LOCAL_MODEL_ROOT) / MODEL_KEY / MODEL_REVISION
    copy_start = time.perf_counter()
    _stage_checkpoint(cached_model_path, model_path)
    copy_elapsed = time.perf_counter() - copy_start
    prompt_ids = _real_prompt_ids(model_path, input_tokens)

    run_id = datetime.now(timezone.utc).strftime("sglang-0731-%Y%m%dT%H%M%SZ")
    profile_dir = Path(PROFILE_MOUNT) / run_id
    profile_dir.mkdir(parents=True)
    server_log_path = profile_dir / "server.log"
    server_command = [
        "sglang",
        "serve",
        "--trust-remote-code",
        "--model-path",
        str(model_path),
        "--tp",
        "4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--speculative-algorithm",
        "DSPARK",
        "--mem-fraction-static",
        "0.90",
        "--swa-full-tokens-ratio",
        "0.1",
        # This harness only measures batch one. Capturing larger tiers changes
        # startup time and memory use, not the batch-one graph being profiled.
        "--cuda-graph-bs-decode",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
    ]

    with server_log_path.open("w") as server_log:
        process = subprocess.Popen(
            server_command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_server(process)
            _request_json(
                "/generate",
                {
                    "text": "Explain why batching improves language-model throughput.",
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 16,
                    },
                },
                timeout=300.0,
            )

            _request_json("/flush_cache", {}, timeout=120.0)
            benchmark = _stream_generate(prompt_ids, output_tokens)

            _request_json("/flush_cache", {}, timeout=120.0)
            profile_response = _request_json(
                "/start_profile",
                {
                    "output_dir": str(profile_dir),
                    "num_steps": 3,
                    "activities": ["CPU", "GPU"],
                    "profile_by_stage": True,
                    "record_shapes": True,
                    "profile_id": run_id,
                    "profile_prefix": "sglang-",
                },
                timeout=120.0,
            )
            trace_benchmark = _stream_generate(prompt_ids, profile_output_tokens)
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                traces = sorted(profile_dir.glob("*.trace.json.gz"))
                if len(traces) >= 8:
                    break
                time.sleep(2.0)
            else:
                traces = sorted(profile_dir.glob("*.trace.json.gz"))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30.0)

    profile_volume.commit()
    return {
        "run_id": run_id,
        "checkpoint_copy_seconds": copy_elapsed,
        "model_path": str(model_path),
        "server_command": server_command,
        "prompt_source_files": list(PROMPT_FILES),
        "prompt_token_count": len(prompt_ids),
        "benchmark": benchmark,
        "profile_response": profile_response,
        "trace_benchmark": trace_benchmark,
        "trace_files": [
            {"path": str(path), "bytes": path.stat().st_size} for path in traces
        ],
        "server_log": str(server_log_path),
    }


@app.local_entrypoint()
def main(
    input_tokens: int = 8192,
    output_tokens: int = 1024,
    profile_output_tokens: int = 64,
) -> None:
    report = benchmark_and_profile.remote(
        input_tokens,
        output_tokens,
        profile_output_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
