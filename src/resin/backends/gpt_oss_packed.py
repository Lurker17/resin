"""Kernel-neutral bridge for streaming GPT-OSS MXFP4 checkpoints.

The model remains ordinary Resin-owned PyTorch code.  This module replaces only
the expert implementation with the small :mod:`resin.moe_kernels` contract and
streams official packed tensors through the selected adapter one layer at a
time.  It imports PyTorch only when a packed model is explicitly loaded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernelConfig,
    MoERouting,
)

from .gpt_oss_weights import load_gpt_oss_weights

_CHECKPOINT_TO_STANDARD = {
    "gate_up_proj_blocks": "gate_up",
    "gate_up_proj_scales": "gate_up_scales",
    "gate_up_proj_bias": "gate_up_bias",
    "down_proj_blocks": "down",
    "down_proj_scales": "down_scales",
    "down_proj_bias": "down_bias",
}
_EXPERT_CHECKPOINT_NAMES = tuple(_CHECKPOINT_TO_STANDARD)


def _cuda_architecture(torch, device) -> str:
    if device.type != "cuda":
        raise ValueError("packed GPT-OSS MXFP4 execution requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def _moe_config(args, architecture: str) -> MoEKernelConfig:
    return MoEKernelConfig(
        architecture=architecture,
        quantization="mxfp4",
        activation="swiglu",
        top_k=args.num_experts_per_tok,
        num_experts=args.num_local_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        group_size=32,
        expert_bias=True,
        activation_alpha=1.702,
        activation_beta=1.0,
        activation_limit=args.swiglu_limit,
        gate_up_layout="interleaved_gate_up",
    )


def _make_kernel_expert_factory(torch, kernel, config: MoEKernelConfig):
    class KernelExperts(torch.nn.Module):
        storage_format = config.quantization

        def __init__(self, args) -> None:
            super().__init__()
            actual = (
                args.num_local_experts,
                args.hidden_size,
                args.intermediate_size,
                args.num_experts_per_tok,
            )
            expected = (
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
                config.top_k,
            )
            if actual != expected:
                raise ValueError(
                    f"expert module dimensions {actual} do not match kernel {expected}"
                )
            self._prepared_weights = None

        def install_prepared_weights(self, weights) -> None:
            if self._prepared_weights is not None:
                raise RuntimeError("packed expert weights are already installed")
            self._prepared_weights = weights

        @property
        def is_prepared(self) -> bool:
            return self._prepared_weights is not None

        def forward(self, hidden, indices, weights, logits):
            if self._prepared_weights is None:
                raise RuntimeError("packed expert weights have not been loaded")
            routing = MoERouting(
                router_logits=logits,
                selected_experts=indices,
                expert_weights=weights,
            )
            return kernel.forward(hidden, routing, self._prepared_weights, config)

    return lambda args: KernelExperts(args)


class _PackedExpertCoordinator:
    """Connect the generic tensor streamer to one prepared module per layer."""

    def __init__(self, torch, model, kernel, config: MoEKernelConfig) -> None:
        self._torch = torch
        self._model = model
        self._kernel = kernel
        self._config = config
        self._active_layer: int | None = None
        self._raw: dict[str, Any] = {}

    def expected_tensor_names(
        self, layer_index: int, config: Mapping[str, Any]
    ) -> tuple[str, ...]:
        del layer_index, config
        return _EXPERT_CHECKPOINT_NAMES

    def begin_layer(self, layer_index: int, config: Mapping[str, Any]) -> None:
        del config
        if self._active_layer is not None:
            raise RuntimeError("the previous packed expert layer was not finished")
        module = self._model.model.layers[layer_index].mlp.experts
        if not hasattr(module, "install_prepared_weights"):
            raise TypeError(f"layer {layer_index} does not use a kernel expert module")
        self._active_layer = layer_index
        self._raw = {}

    def consume_tensor(self, layer_index: int, name: str, tensor: Any) -> None:
        if layer_index != self._active_layer:
            raise RuntimeError("packed expert tensor arrived outside its layer scope")
        try:
            standard_name = _CHECKPOINT_TO_STANDARD[name]
        except KeyError as error:
            raise KeyError(f"unknown GPT-OSS expert tensor {name!r}") from error
        if standard_name in self._raw:
            raise ValueError(f"duplicate GPT-OSS expert tensor {name!r}")

        if name.endswith("_blocks"):
            if tensor.ndim != 4 or tensor.shape[-1] != 16:
                raise ValueError(
                    f"{name} must have checkpoint shape [E, N, K/32, 16], "
                    f"got {tuple(tensor.shape)}"
                )
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        self._raw[standard_name] = tensor

    def finish_layer(self, layer_index: int) -> None:
        if layer_index != self._active_layer:
            raise RuntimeError("cannot finish an inactive packed expert layer")
        missing = set(_CHECKPOINT_TO_STANDARD.values()) - self._raw.keys()
        if missing:
            raise ValueError(
                "missing GPT-OSS expert tensors: " + ", ".join(sorted(missing))
            )

        raw = MoEExpertWeights(
            gate_up=self._raw["gate_up"],
            down=self._raw["down"],
            gate_up_scales=self._raw["gate_up_scales"],
            down_scales=self._raw["down_scales"],
            gate_up_bias=self._raw["gate_up_bias"],
            down_bias=self._raw["down_bias"],
        )
        prepared = self._kernel.prepare_weights(raw, self._config)
        module = self._model.model.layers[layer_index].mlp.experts
        module.install_prepared_weights(prepared)

        # The prepared tensors stay referenced by the layer.  Release raw and
        # padding/interleave temporaries before reading the next large layer.
        self._raw = {}
        self._active_layer = None
        del raw, prepared
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def load_mxfp4_model(
    model_path: str | Path,
    *,
    device,
    kernel,
    dtype=None,
):
    """Load an official packed checkpoint without Transformers or tokenizers."""

    try:
        import torch
    except ImportError as error:
        raise ImportError("packed GPT-OSS execution requires PyTorch") from error

    from .gpt_oss import _GptOssArgs, _make_model_class, _resolve_dtype

    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    args = _GptOssArgs.from_config(config)
    quantization = config.get("quantization_config") or config.get("quantization")
    if not quantization or quantization.get("quant_method") != "mxfp4":
        raise ValueError("load_mxfp4_model requires an MXFP4 GPT-OSS checkpoint")

    target = torch.device(device)
    resolved_dtype = _resolve_dtype(torch, dtype)
    if resolved_dtype not in {None, torch.bfloat16}:
        raise ValueError("the packed SM90 GPT-OSS path requires bfloat16 activations")
    architecture = _cuda_architecture(torch, target)
    moe_config = _moe_config(args, architecture)
    kernel.validate(moe_config)

    expert_factory = _make_kernel_expert_factory(torch, kernel, moe_config)
    model_class = _make_model_class(torch, expert_factory=expert_factory)
    with torch.device("meta"):
        model = model_class(args)

    coordinator = _PackedExpertCoordinator(
        torch,
        model,
        kernel,
        moe_config,
    )
    loaded_config, _ = load_gpt_oss_weights(
        path,
        model,
        device=target,
        dtype=torch.bfloat16,
        expert_builder=coordinator,
    )
    if dict(loaded_config) != config:
        raise RuntimeError("checkpoint config changed while weights were loading")

    model.materialize_buffers(target)
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm":
            module.float()
    unprepared = [
        index
        for index, layer in enumerate(model.model.layers)
        if not layer.mlp.experts.is_prepared
    ]
    if unprepared:
        raise RuntimeError(f"packed expert layers were not prepared: {unprepared}")
    remaining_meta = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            "ordinary model parameters remain on meta: " + ", ".join(remaining_meta[:8])
        )
    model.eval()
    return torch, model, config


__all__ = ["load_mxfp4_model"]
