"""Tokenizer-free GPT-OSS checkpoint resolution and model loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import (
    MODEL_FILES,
    materialize_safetensor_model,
    read_config,
    resolve_model,
)
from resin.backend.quantization import BF16, MXFP4, QuantizationConfig
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.model import GptOssForCausalLM

DEFAULT_MODEL = "openai/gpt-oss-20b"
MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"
MODEL_REVISIONS = {
    MODEL_20B: "6cee5e81ee83917806bbde320786a8fb61efebee",
    MODEL_120B: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
}


def _download_model(
    repo_id: str,
    revision: str | None,
    *,
    quantization: QuantizationConfig | None = None,
    kernel_moe=None,
) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Inspect the tiny config before asking the Hub for tens of gigabytes of
    # packed weights. Without an explicit kernel, fail before shard download.
    config_file = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    _check_quantization(
        json.loads(config_file.read_text()),
        quantization=quantization,
        kernel_moe=kernel_moe,
    )
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(
    model: str | Path,
    revision: str | None,
    *,
    quantization: QuantizationConfig | None = None,
    kernel_moe=None,
) -> Path:
    repo_id = str(model)
    selected_revision = MODEL_REVISIONS.get(repo_id) if revision is None else revision
    return resolve_model(
        model,
        selected_revision,
        lambda repository, selected: _download_model(
            repository,
            selected,
            quantization=quantization,
            kernel_moe=kernel_moe,
        ),
    )


def _check_quantization(
    config: dict[str, Any],
    *,
    quantization: QuantizationConfig | None = None,
    kernel_moe=None,
):
    checkpoint_config = config.get("quantization_config") or config.get("quantization")
    method = checkpoint_config.get("quant_method") if checkpoint_config else None
    checkpoint_encoding = BF16
    if method == "mxfp4":
        checkpoint_encoding = MXFP4
    elif method is not None:
        raise ValueError(f"unsupported GPT-OSS quantization method: {method!r}")
    if quantization is not None:
        quantization.require("transformer.experts", checkpoint_encoding)
    if checkpoint_encoding == MXFP4 and kernel_moe is None:
        raise NotImplementedError(
            "the released GPT-OSS checkpoint uses MXFP4 expert weights; "
            "pass an imported CUDA implementation as kernel_moe"
        )
    return checkpoint_encoding


def load_model(
    model_path: Path,
    *,
    device,
    dtype=None,
    quantization: QuantizationConfig | None = None,
    kernel_moe=None,
    tensor_parallel: TensorParallelLayout | None = None,
):
    config = read_config(model_path)
    args = GptOssConfig.from_dict(config)
    checkpoint_encoding = _check_quantization(
        config,
        quantization=quantization,
        kernel_moe=kernel_moe,
    )
    if checkpoint_encoding == MXFP4:
        from resin_models_cuda.gpt_oss.mxfp4 import load_mxfp4_model

        return load_mxfp4_model(
            model_path,
            device=device,
            kernel=kernel_moe,
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
        model_path,
        device=device,
        dtype=dtype,
        backend="GPT-OSS",
        build=lambda: GptOssForCausalLM(args),
        after_cast=restore_norm_dtype,
    )
    return model, config
