"""Readable tokenizer-free GPT-OSS loading and execution on PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import (
    MODEL_FILES,
    default_device,
    load_safetensor_shards,
    read_config,
    resolve_dtype,
    resolve_model,
)
from resin.backend.plans import plan_rows
from resin.models.gpt_oss.config import GptOssConfig
from resin.models.gpt_oss.model import GptOssForCausalLM, GptOssState

if TYPE_CHECKING:
    from resin import BatchPlan


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
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise ImportError(
            "loading GPT-OSS by repository id requires huggingface-hub; "
            "install it or pass a local model directory"
        ) from error

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
        from resin.models.gpt_oss.mxfp4 import load_mxfp4_model

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

    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    with torch.device("meta"):
        model = GptOssForCausalLM(args)

    weights = load_safetensor_shards(model_path, device=load_target, backend="GPT-OSS")
    model.load_state_dict(weights, strict=True, assign=True)
    model.materialize_buffers(load_target)
    if target != load_target:
        model.to(target)

    resolved_dtype = resolve_dtype(torch, dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    # GPT-OSS keeps normalization scales and arithmetic in float32 even though
    # the released non-expert checkpoint tensors are BF16.
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm":
            module.float()
    model.materialize_buffers(target)
    model.eval()
    return torch, model, config


class GptOssRunner:
    """Greedy GPT-OSS runner consuming Rust-built integer-token batches.

    Equal-length sequences with equally sized KV caches execute in one physical
    batch. Heterogeneous chunks are rejected instead of being mislabeled as a
    batched execution.
    """

    supports_continuation_prefill = True
    prefill_chunk_size = 8192

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype=None,
        moe_kernel=None,
    ) -> None:
        self.prefill_chunk_size = type(self).prefill_chunk_size
        self.device = torch.device(device or default_device(torch, require_cuda=True))
        self.model_path = resolve_model_path(
            model,
            revision,
            allow_mxfp4=moe_kernel is not None,
        )
        self._torch, self._model, self.config = load_model(
            self.model_path,
            device=self.device,
            dtype=dtype,
            moe_kernel=moe_kernel,
        )
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = self.config["vocab_size"]
        self._cache: dict[int, GptOssState] = {}

    def forward(self, plan: BatchPlan) -> list[int]:
        torch = self._torch
        rows = plan_rows(plan, self._cache, self.vocab_size)

        lengths = {len(row.token_ids) for row in rows}
        if len(lengths) != 1:
            raise ValueError(
                "physical GPT-OSS batching requires equal token counts per sequence"
            )

        request_caches = [
            row.cache if row.cache is not None else self._model.make_cache()
            for row in rows
        ]
        model_cache = (
            request_caches[0]
            if len(request_caches) == 1
            else self._model.make_batch_cache(request_caches)
        )
        inputs = torch.tensor(
            [row.token_ids for row in rows],
            dtype=torch.long,
            device=self.device,
        )
        position_tensor = torch.tensor(
            [row.positions for row in rows],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self._model(
                inputs,
                model_cache,
                position_tensor,
                logits_to_keep=1,
            )
        output = logits[:, -1, :].argmax(dim=-1).tolist()

        for row, cache in zip(rows, request_caches, strict=True):
            if row.cache is None:
                self._cache[row.request_id] = cache
        return output

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)
