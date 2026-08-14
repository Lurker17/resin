"""Readable tokenizer-free LFM2 loading and execution on PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from resin.backend.loading import (
    MODEL_FILES,
    default_device,
    load_safetensor_shards,
    read_config,
    resolve_dtype,
    resolve_model,
)
from resin.backend.plans import plan_rows
from resin.backend.sampling import TorchSampler
from resin.models.lfm2.config import Lfm2Config
from resin.models.lfm2.model import Lfm2ForCausalLM, Lfm2State

if TYPE_CHECKING:
    from resin import BatchPlan


DEFAULT_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"


def _download_model(repo_id: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "loading an LFM2 model by repository id requires huggingface-hub; "
            "install it or pass a local model directory"
        ) from error
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(model: str | Path, revision: str | None) -> Path:
    return resolve_model(model, revision, _download_model)


def load_model(model_path: Path, *, device, dtype=None):
    config = read_config(model_path)
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError(
            "the PyTorch runner requires an unquantized LFM2 checkpoint; "
            "quantized safetensors use a different physical format"
        )
    args = Lfm2Config.from_dict(config)
    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")

    with torch.device("meta"):
        model = Lfm2ForCausalLM(args)

    weights = load_safetensor_shards(model_path, device=load_target, backend="LFM2")
    model.load_state_dict(weights, strict=True, assign=True)
    model.materialize_buffers(load_target)
    if target != load_target:
        model.to(target)
    resolved_dtype = resolve_dtype(torch, dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    # Module.to(dtype=...) casts floating-point buffers too. RoPE frequencies
    # intentionally remain float32 even when the checkpoint is bfloat16.
    model.materialize_buffers(target)
    model.eval()
    return torch, model, config


class Lfm2TorchRunner:
    """LFM2 runner consuming Rust-built integer-token batches."""

    supports_continuation_prefill = True

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        dtype=None,
    ) -> None:
        self.device = torch.device(device or default_device(torch))
        self.model_path = resolve_model_path(model, revision)
        self._torch, self._model, self.config = load_model(
            self.model_path, device=self.device, dtype=dtype
        )
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = self.config["vocab_size"]
        self._sampler = TorchSampler(torch)
        self._cache: dict[int, Lfm2State] = {}
        self.forward_calls = 0

    def forward(self, plan: BatchPlan) -> list[int]:
        torch = self._torch
        logits_rows = []
        self.forward_calls += 1
        rows = plan_rows(plan, self._cache, self.vocab_size)

        with torch.inference_mode():
            for row in rows:
                cache = row.cache
                if cache is None:
                    cache = self._model.make_cache()
                    self._cache[row.request_id] = cache
                inputs = torch.tensor(
                    [row.token_ids], dtype=torch.long, device=self.device
                )
                position_tensor = torch.tensor(
                    [row.positions], dtype=torch.long, device=self.device
                )
                logits = self._model(inputs, cache, position_tensor, logits_to_keep=1)
                logits_rows.append(logits[:, -1, :])
        sampler = getattr(self, "_sampler", None)
        if sampler is None:
            sampler = self._sampler = TorchSampler(torch)
        return sampler(torch.cat(logits_rows), plan).tolist()

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)

    @property
    def live_cache_count(self) -> int:
        return len(self._cache)
