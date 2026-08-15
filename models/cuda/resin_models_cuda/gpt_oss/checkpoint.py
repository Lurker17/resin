"""Tokenizer-free GPT-OSS checkpoint resolution and model loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import (
    MODEL_FILES,
    materialize_safetensor_model,
    read_config,
    resolve_model,
)
from resin_models_cuda.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.model import GptOssForCausalLM

DEFAULT_MODEL = "openai/gpt-oss-20b"
MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"
MODEL_REVISIONS = {
    MODEL_20B: "6cee5e81ee83917806bbde320786a8fb61efebee",
    MODEL_120B: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
}


def _download_model(
    repo_id: str, revision: str | None, *, allow_mxfp4: bool = False
) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Inspect the tiny config before asking the Hub for tens of gigabytes of
    # packed weights. Without an explicit kernel, fail before shard download.
    config_file = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    _check_quantization(json.loads(config_file.read_text()), allow_mxfp4=allow_mxfp4)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(
    model: str | Path, revision: str | None, *, allow_mxfp4: bool = False
) -> Path:
    repo_id = str(model)
    selected_revision = MODEL_REVISIONS.get(repo_id) if revision is None else revision
    return resolve_model(
        model,
        selected_revision,
        lambda repository, selected: _download_model(
            repository, selected, allow_mxfp4=allow_mxfp4
        ),
    )


def _check_quantization(config: dict[str, Any], *, allow_mxfp4: bool = False) -> None:
    quantization = config.get("quantization_config") or config.get("quantization")
    if not quantization:
        return
    method = quantization.get("quant_method")
    if method == "mxfp4":
        if allow_mxfp4:
            return
        raise NotImplementedError(
            "the released GPT-OSS checkpoint uses MXFP4 expert weights; "
            "Resin's dense reference expert module cannot load packed blocks yet. "
            "An H100 MXFP4 expert kernel must be selected at this boundary"
        )
    raise ValueError(f"unsupported GPT-OSS quantization method: {method!r}")


def load_model(
    model_path: Path,
    *,
    device,
    dtype=None,
    moe_kernel=None,
    tensor_parallel: TensorParallelLayout | None = None,
):
    config = read_config(model_path)
    args = GptOssConfig.from_dict(config)
    _check_quantization(config, allow_mxfp4=moe_kernel is not None)
    quantization = config.get("quantization_config") or config.get("quantization")
    if quantization and quantization.get("quant_method") == "mxfp4":
        from resin_models_cuda.gpt_oss.mxfp4 import load_mxfp4_model

        return load_mxfp4_model(
            model_path,
            device=device,
            kernel=moe_kernel,
            dtype=dtype,
            tensor_parallel=tensor_parallel,
        )

    tensor_parallel = tensor_parallel or TensorParallelLayout()
    if tensor_parallel.enabled:
        raise NotImplementedError(
            "tensor parallel loading currently requires packed MXFP4 experts"
        )

    def restore_norm_dtype(model) -> None:
        # The checkpoint stores BF16 scales, but normalization accumulates and
        # retains its parameters in FP32.
        for module in model.modules():
            if module.__class__.__name__ == "RMSNorm":
                module.float()

    model = materialize_safetensor_model(
        torch,
        model_path,
        device=device,
        dtype=dtype,
        backend="GPT-OSS",
        build=lambda: GptOssForCausalLM(args),
        after_cast=restore_norm_dtype,
    )
    return torch, model, config
