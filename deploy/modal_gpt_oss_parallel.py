"""Validate GPT-OSS parallelism and its compositions on Hopper GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import modal

import resin
from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
    prepare_kernel_cache,
)
from deploy.modal_image import build_modal_image
from resin.recipe import GptOssRecipe

ROOT = Path(
    os.environ.get("RESIN_MODAL_SOURCE_ROOT", Path(__file__).resolve().parents[1])
)
MODEL_REPO = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_MOUNT = "/models/resin-gpt-oss"
MODEL_PATH = Path(MODEL_MOUNT) / "20b" / MODEL_REVISION
PROFILE_MOUNT = "/profiles/resin-gpt-oss"

image = (
    build_modal_image(ROOT, env={"MAX_JOBS": "12", "CMAKE_BUILD_PARALLEL_LEVEL": "12"})
    .env(KERNEL_CACHE_ENV)
    .uv_pip_install("pytest>=8,<9")
    .add_local_file(
        ROOT / "deploy/gpt_oss_parallel_validation.py",
        "/root/deploy/gpt_oss_parallel_validation.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "deploy/gpt_oss_expert_probe.py",
        "/root/deploy/gpt_oss_expert_probe.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "tests/test_expert_parallel_gpu.py",
        "/opt/resin/validation/test_expert_parallel_gpu.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "models/cuda/tests/test_gpt_oss_flashinfer_gpu.py",
        "/opt/resin/validation/test_gpt_oss_flashinfer_gpu.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "tests/test_context_parallel_gpu.py",
        "/opt/resin/validation/test_context_parallel_gpu.py",
        copy=True,
    )
)
model_volume = modal.Volume.from_name(
    "resin-gpt-oss-models-v2", create_if_missing=False, version=2
)
kernel_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=KERNEL_CACHE_VOLUME_VERSION,
)
profile_volume = modal.Volume.from_name(
    "resin-gpt-oss-profiles", create_if_missing=True
)
app = modal.App(
    "resin-gpt-oss-parallel-validation",
    image=image,
    volumes={
        MODEL_MOUNT: model_volume,
        KERNEL_CACHE_MOUNT: kernel_volume,
        PROFILE_MOUNT: profile_volume,
    },
)


def _checkpoint_inventory():
    index = MODEL_PATH / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    filenames = set(data["weight_map"].values())
    if not filenames or any(
        Path(name).name != name or not name.endswith(".safetensors")
        for name in filenames
    ):
        raise RuntimeError("cached GPT-OSS20B has an invalid safetensor index")
    files = [
        MODEL_PATH / name
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            ".resin-checkpoint.json",
            *sorted(filenames),
        )
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in files):
        raise RuntimeError("the pinned GPT-OSS20B checkpoint is incomplete")
    return {str(path.relative_to(MODEL_PATH)): path.stat().st_size for path in files}


def _stream_subprocess(command, *, env, timeout):
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []

    def read_output():
        stdout = process.stdout
        assert stdout is not None
        for line in stdout:
            lines.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        raise
    finally:
        reader.join(timeout=10)
    if reader.is_alive():
        raise RuntimeError("GPU reference subprocess output did not finish")
    return subprocess.CompletedProcess(command, returncode, stdout="".join(lines))


@app.function(
    gpu=["H200:2", "H100:2"],
    cpu=16,
    memory=128 * 1024,
    timeout=100 * 60,
    single_use_containers=True,
)
def validate_parallel(
    graphs: bool = False,
    composable: bool = False,
    world_size: int = 2,
    case_names: tuple[str, ...] | None = None,
):
    import torch
    from transformers import AutoTokenizer

    from deploy.gpt_oss_parallel_validation import run_validation

    if torch.cuda.device_count() != world_size or any(
        torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(world_size)
    ):
        raise RuntimeError(f"GPT-OSS validation requires {world_size} Hopper SM90 GPUs")
    graphs = graphs or composable
    print(
        f"[parallel-validation] allocated {[torch.cuda.get_device_name(rank) for rank in range(world_size)]}",
        flush=True,
    )
    model_volume.reload()
    inventory = _checkpoint_inventory()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    prepare_kernel_cache(kernel_volume)
    tests = _stream_subprocess(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-q",
            "-s",
            "/opt/resin/validation/test_expert_parallel_gpu.py",
            "/opt/resin/validation/test_gpt_oss_flashinfer_gpu.py",
            "/opt/resin/validation/test_context_parallel_gpu.py",
        ],
        env={**os.environ, "RESIN_EP_TEST_WORLD_SIZE": "2", "PYTHONUNBUFFERED": "1"},
        timeout=6 * 60,
    )
    if tests.returncode or "skipped" in tests.stdout:
        raise RuntimeError("two-GPU NCCL dispatcher reference validation failed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=False
    )
    if tokenizer is None:
        raise RuntimeError("the GPT-OSS checkpoint did not load a tokenizer")
    recipe = GptOssRecipe(
        tokenizer, options=resin.recipe.RenderOptions(reasoning_effort="medium")
    )
    texts = {
        "short": "Continue the sequence with the next number: 2, 4, 6,",
        "long": (
            "Every request owns its cache pages. Expert workers may process tokens "
            "from independent request replicas. "
        )
        * 6
        + "Explain why cache namespaces must remain independent.",
        "late": "Give one practical reason to batch requests with unequal prompt lengths.",
        "probe": "What is the next integer after 41? Answer briefly.",
    }
    prompts = {
        name: list(
            map(
                int,
                recipe.render_ids(
                    [
                        resin.recipe.Message(
                            resin.recipe.Role.USER, [resin.recipe.Text(text)]
                        )
                    ],
                    options=resin.recipe.RenderOptions(add_generation_prompt=True),
                ),
            )
        )
        for name, text in texts.items()
    }
    prefix = (
        f"gpt-oss-composable{world_size}"
        if composable
        else "gpt-oss-ep-graphs"
        if graphs
        else "gpt-oss-ep"
    )
    run_id = datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%SZ")
    path = Path(PROFILE_MOUNT) / f"{run_id}.json"
    print(f"[parallel-validation] diagnostic report: {path}", flush=True)
    try:
        report = run_validation(
            MODEL_PATH,
            prompts,
            vocab_size=len(tokenizer),
            diagnostic_path=path,
            graphs=graphs,
            composable=composable,
            world_size=world_size,
            case_names=case_names,
        )
    except Exception as error:
        if path.is_file():
            diagnostic = json.loads(path.read_text())
            diagnostic["status"] = "failed"
            diagnostic.setdefault("validation_error", str(error))
            path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True))
        profile_volume.commit()
        kernel_volume.commit()
        raise
    report.update(
        {
            "checkpoint": {
                "repo_id": MODEL_REPO,
                "revision": MODEL_REVISION,
                "files": len(inventory),
                "bytes": sum(inventory.values()),
                "cache_hit": True,
            },
            "dispatcher_reference_tests": tests.stdout.strip(),
            "prompt_token_counts": {name: len(ids) for name, ids in prompts.items()},
        }
    )
    for baseline in report["baseline"]:
        baseline["generation"]["text"] = tokenizer.decode(
            baseline["generation"]["token_ids"], skip_special_tokens=False
        )
    for case in report["cases"]:
        for record in case["ranks"]:
            result = record["driver_result"]
            if result is not None:
                for output in [*result["outputs"].values(), *result["probes"]]:
                    output["text"] = tokenizer.decode(
                        output["token_ids"], skip_special_tokens=False
                    )
    report["remote_report_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    profile_volume.commit()
    kernel_volume.commit()
    return report


@app.function(
    gpu=["H200", "H100"],
    cpu=8,
    memory=32 * 1024,
    timeout=20 * 60,
    single_use_containers=True,
)
def probe_experts(reference_path: str):
    from deploy.gpt_oss_expert_probe import run_expert_probe
    from deploy.jit_cache import ensure_kernel_cache_dirs, install_flashinfer_hash_cache

    model_volume.reload()
    profile_volume.reload()
    inventory = _checkpoint_inventory()
    prepare_kernel_cache(kernel_volume)
    ensure_kernel_cache_dirs()
    install_flashinfer_hash_cache()
    report = run_expert_probe(MODEL_PATH, reference_path)
    report["checkpoint"] = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "bytes": sum(inventory.values()),
    }
    run_id = datetime.now(timezone.utc).strftime("gpt-oss-expert-probe-%Y%m%dT%H%M%SZ")
    path = Path(PROFILE_MOUNT) / f"{run_id}.json"
    report["remote_report_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    profile_volume.commit()
    kernel_volume.commit()
    return report


@app.function(
    gpu="H100", cpu=8, memory=64 * 1024, timeout=15 * 60, single_use_containers=True
)
def probe_prompt_scores(attention_oracle: bool = False):
    """Separate prompt chunking arithmetic from all distributed sharding."""
    import socket

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from deploy.gpt_oss_parallel_validation import (
        _engine_options,
        _initialize,
        _metrics,
        _new_runner,
        _score_prompt,
    )
    from resin import Engine

    model_volume.reload()
    _checkpoint_inventory()
    prepare_kernel_cache(kernel_volume)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    recipe = GptOssRecipe(
        tokenizer, options=resin.recipe.RenderOptions(reasoning_effort="medium")
    )
    text = (
        "Every request owns its cache pages. Expert workers may process tokens "
        "from independent request replicas. "
    ) * 6 + "Explain why cache namespaces must remain independent."
    prompt = list(
        recipe.render_ids(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(text)])],
            options=resin.recipe.RenderOptions(add_generation_prompt=True),
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    _initialize(0, port, 1)
    reports = []
    attention_records, attention_inputs = [], []
    try:
        for chunk in (512, 32):
            runner = _new_runner(0, MODEL_PATH, chunk_size=chunk)
            if attention_oracle:
                original_fa4 = runner._fa4
                num_layers = len(runner._model.model.layers)
                calls = 0

                def observe_attention(query, key_cache, value_cache, **kwargs):
                    nonlocal calls
                    output = original_fa4(query, key_cache, value_cache, **kwargs)
                    layer = calls % num_layers
                    calls += 1
                    if layer:
                        return output
                    count = int(kwargs["seqused_k"][0])
                    pages = kwargs["page_table"][0].long()
                    key = key_cache[pages].flatten(0, 1)[:count]
                    value = value_cache[pages].flatten(0, 1)[:count]
                    repeats = query.shape[1] // key.shape[1]
                    key = key.repeat_interleave(repeats, dim=1).float()
                    value = value.repeat_interleave(repeats, dim=1).float()
                    logits = torch.einsum("qhd,khd->hqk", query.float(), key)
                    logits *= kwargs["softmax_scale"]
                    positions = (
                        count
                        - len(query)
                        + torch.arange(len(query), device=query.device)
                    )
                    keys = torch.arange(count, device=query.device)
                    mask = keys[None, :] <= positions[:, None]
                    left, _right = kwargs["window_size"]
                    if left is not None:
                        mask &= keys[None, :] >= positions[:, None] - left
                    logits.masked_fill_(~mask[None, :, :], -torch.inf)
                    sinks = kwargs["learnable_sink"].float().reshape(-1, 1, 1)
                    weights = torch.cat(
                        (logits, sinks.expand(-1, len(query), 1)), dim=-1
                    ).softmax(-1)[..., :-1]
                    expected = torch.einsum("hqk,khd->qhd", weights, value)
                    actual = output[0] if isinstance(output, tuple) else output
                    delta = actual.float() - expected
                    attention_records.append(
                        {
                            "chunk_size": chunk,
                            "query_rows": len(query),
                            "kv_rows": count,
                            "window": left,
                            "comparison": _metrics(
                                actual, expected, maximum=0.05, relative=0.01
                            ),
                            "row_max_abs": delta.abs()
                            .flatten(1)
                            .amax(1)
                            .cpu()
                            .tolist(),
                        }
                    )
                    attention_inputs.append(
                        {
                            "chunk_size": chunk,
                            "query": query.cpu(),
                            "key": key_cache[pages].flatten(0, 1)[:count].cpu(),
                            "value": value_cache[pages].flatten(0, 1)[:count].cpu(),
                            "actual": actual.cpu(),
                            "sink": kwargs["learnable_sink"].cpu(),
                            "softmax_scale": kwargs["softmax_scale"],
                            "window": left,
                        }
                    )
                    return output

                runner._fa4 = observe_attention
            engine = Engine(runner, **_engine_options(chunk))
            try:
                reports.append(
                    _score_prompt(
                        engine,
                        prompt,
                        oracle_head=runner._model.lm_head if chunk == 512 else None,
                    )
                )
            finally:
                engine.close()
                runner.close_cuda_graphs()
            del engine, runner
            torch.cuda.empty_cache()
        comparison = _metrics(
            torch.tensor(reports[1]["scores"][1:]),
            torch.tensor(reports[0]["scores"][1:]),
            maximum=0.5,
            relative=0.02,
        )
        report = {
            "gpu": torch.cuda.get_device_name(),
            "unchunked": reports[0],
            "chunked": reports[1],
            "comparison": comparison,
            "attention_oracle": attention_records,
        }
        path = Path(PROFILE_MOUNT) / datetime.now(timezone.utc).strftime(
            "prompt-scoring-isolation-%Y%m%dT%H%M%SZ.json"
        )
        path.write_text(json.dumps(report, indent=2))
        if attention_inputs:
            torch.save(attention_inputs, path.with_suffix(".pt"))
        profile_volume.commit()
        kernel_volume.commit()
        return report
    finally:
        dist.destroy_process_group()


@app.function(
    gpu="H100:4",
    cpu=16,
    memory=64 * 1024,
    timeout=20 * 60,
    single_use_containers=True,
)
def probe_expert_transport(reference_path: str):
    from deploy.gpt_oss_parallel_validation import run_expert_transport_probe

    model_volume.reload()
    profile_volume.reload()
    _checkpoint_inventory()
    prepare_kernel_cache(kernel_volume)
    path = Path(PROFILE_MOUNT) / datetime.now(timezone.utc).strftime(
        "expert-transport-isolation-%Y%m%dT%H%M%SZ.json"
    )
    try:
        report = run_expert_transport_probe(
            MODEL_PATH, reference_path, diagnostic_path=path
        )
    finally:
        profile_volume.commit()
        kernel_volume.commit()
    report["remote_report_path"] = str(path)
    return report


@app.local_entrypoint()
def main(
    report_path: str = "artifacts/gpt-oss-parallel-validation.json",
    graphs: bool = False,
    composable: bool = False,
    world_size: int = 0,
    gpu_type: str = "H200",
    expert_probe: bool = False,
    expert_reference_path: str = "",
    prompt_scoring_probe: bool = False,
    expert_transport_probe: bool = False,
    attention_oracle: bool = False,
    parallel_cases: str | None = None,
):
    if gpu_type not in ("H200", "H100"):
        raise ValueError("gpu_type must be H200 or H100")
    if expert_transport_probe:
        if not expert_reference_path:
            raise ValueError(
                "--expert-transport-probe requires --expert-reference-path"
            )
        report = probe_expert_transport.with_options(gpu=f"{gpu_type}:4").remote(
            expert_reference_path
        )
    elif prompt_scoring_probe:
        report = probe_prompt_scores.with_options(gpu=gpu_type).remote(attention_oracle)
    elif expert_probe:
        if not expert_reference_path:
            raise ValueError("--expert-probe requires --expert-reference-path")
        report = probe_experts.with_options(gpu=gpu_type).remote(expert_reference_path)
    else:
        world_size = world_size or (4 if composable else 2)
        if world_size not in ((2, 4, 8) if composable else (2,)):
            raise ValueError(
                "use two GPUs, or --composable with --world-size 2, 4 or 8"
            )
        function = validate_parallel
        if world_size != 2 or gpu_type != "H200":
            function = function.with_options(
                gpu=f"{gpu_type}:{world_size}",
                cpu=4 * world_size,
                memory=16 * 1024 * world_size,
            )
        report = function.remote(
            graphs=graphs,
            composable=composable,
            world_size=world_size,
            case_names=(
                None
                if parallel_cases is None
                else tuple(
                    name.strip() for name in parallel_cases.split(",") if name.strip()
                )
            ),
        )
    path = Path(report_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Validation report: {path}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
