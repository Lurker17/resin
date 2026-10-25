"""Shared serving/benchmark configuration preserves runner and checkpoint contracts."""

import importlib
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest

from deploy.models import (
    MODEL_SPECS,
    create_runner,
    download_checkpoint,
    get_model,
    require_checkpoint,
    validate_gpu,
)


@pytest.mark.parametrize("name", MODEL_SPECS)
def test_factory_arguments_match_native_runner_and_cache_geometry(monkeypatch, name):
    spec = get_model(name)
    module = importlib.import_module(f"resin_models_cuda.{spec.family}.cuda")
    class_name = {
        "gpt_oss": "GptOssCudaRunner",
        "deepseek_v41": "DeepSeekV41CudaRunner",
        "gemma4": "Gemma4CudaRunner",
        "qwen3_5": "Qwen3_5CudaRunner",
        "inkling": "InklingCudaRunner",
    }[spec.family]
    constructor = signature(getattr(module, class_name).__init__)
    groups = None if spec.family == "inkling" else object()
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            constructor.bind(self, *args, **kwargs)
            calls.append((args, kwargs))

    monkeypatch.setattr(module, class_name, Runner)
    if spec.family == "gpt_oss":
        monkeypatch.setattr(
            "resin_models_cuda.gpt_oss.flashinfer.FlashInferSM90MXFP4Kernel",
            lambda: "kernel",
        )
    create_runner(
        spec,
        device="cuda:0",
        groups=groups,
        context_length=8193,
        max_batch_size=8,
        prefill_chunk_size=4096,
        draft_tokens=spec.draft_tokens,
        decode_batch_sizes=(1, 4, 8),
    )
    args, options = calls[0]
    assert args[0] == spec.path
    assert options["device"] == "cuda:0"
    width = {"gpt_oss": 65, "gemma4": 129, "inkling": 8198}.get(spec.family, 8193)
    assert options["cuda_graph_decode_shapes"] == ((1, width), (4, width), (8, width))
    if spec.family != "inkling":
        assert options["parallel_groups"] is groups
    if spec.assistant:
        assert (
            options["assistant_model"] if spec.family == "gemma4" else args[1]
        ) == spec.assistant_path


def make_checkpoint(path, required_files):
    path.mkdir(parents=True)
    for name in required_files:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


@pytest.mark.parametrize("name", MODEL_SPECS)
def test_existing_complete_cache_adoption_never_downloads(monkeypatch, tmp_path, name):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    spec = replace(get_model(name), model_mount=str(tmp_path))
    make_checkpoint(spec.path, spec.checkpoint.required_files)
    if spec.assistant:
        make_checkpoint(spec.assistant_path, spec.assistant.required_files)

    def no_download(*args, **kwargs):
        raise AssertionError("unexpected download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)
    report = download_checkpoint(spec, metadata_only=True)
    assert report["cache_hit"] and report["metadata_only"]
    assert require_checkpoint(spec) == spec.path
    (spec.path / "model.safetensors").write_bytes(b"changed size")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)


def test_local_checkpoint_override_needs_no_volume_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    spec = get_model("gpt-oss-20b")
    make_checkpoint(tmp_path / "local", spec.checkpoint.required_files)
    assert require_checkpoint(spec, checkpoint=tmp_path / "local") == tmp_path / "local"


@pytest.mark.parametrize(
    "draft,groups",
    [(0, None), (5, SimpleNamespace(topology=SimpleNamespace(world_size=2)))],
)
def test_inkling_rejects_unsupported_runtime_options(draft, groups):
    with pytest.raises(ValueError, match="fixed five-token|single GPU"):
        create_runner(
            "inkling",
            device="cuda:0",
            groups=groups,
            context_length=1024,
            max_batch_size=1,
            prefill_chunk_size=4096,
            draft_tokens=draft,
        )


@pytest.mark.parametrize(
    "model,gpu",
    [
        ("gemma4", "B200"),
        ("qwen3_5", "B300"),
        ("inkling", "H200"),
        ("deepseek-v41", "H100"),
    ],
)
def test_gpu_override_cannot_reuse_incompatible_kernel_cache(model, gpu):
    with pytest.raises(ValueError, match="GPU-specific kernel configuration"):
        validate_gpu(model, gpu)
    assert validate_gpu(model) == get_model(model).gpu


@pytest.mark.parametrize("gpu", ["H100", "H200"])
def test_gpt_gpu_override_preserves_sm90_kernel_architecture(gpu):
    assert validate_gpu("gpt-oss-120b", gpu) == gpu
