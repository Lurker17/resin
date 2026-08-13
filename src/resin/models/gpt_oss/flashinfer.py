"""FlashInfer's Hopper MXFP4 fused-MoE adapter for GPT-OSS.

The public boundary is :mod:`resin.moe_kernels`; this module contains only the
FlashInfer-specific cold-path layout conversion and hot-path invocation.

FlashInfer's SM90 W4A16 scale layout groups four consecutive MXFP4 groups.
GPT-OSS uses a width of 2880 (90 groups), so preparation pads the hidden and
intermediate dimensions to 2944 (92 groups).  Gate/up rows are padded in paired
order from 5760 to 5888.  GPT-OSS stores paired ``[gate, up]`` rows, while the
CUTLASS activation consumes ``[all up, all gate]``; preparation applies that
row permutation consistently to weights, scales, and biases.  FlashInfer's
fused runner uses one hidden size for both input and output, so the second
projection's output rows are padded too; ``forward`` slices the 2944-wide
kernel result back to 2880.  These cold-path details are invisible at the
model-facing kernel interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernelCapabilities,
    MoEKernelConfig,
    MoERouting,
    UnsupportedMoEConfiguration,
)

KERNEL_ID = "flashinfer_cutlass_sm90_mxfp4"
_MXFP4_GROUP_SIZE = 32
_SM90_SCALE_ALIGNMENT = 4
_K_ALIGNMENT = _MXFP4_GROUP_SIZE * _SM90_SCALE_ALIGNMENT
_MAX_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class PreparedFlashInferMXFP4Weights:
    """One layer's interleaved tensors, ready for the FlashInfer hot path."""

    fc1_weights: Any
    fc2_weights: Any
    fc1_scales: Any
    fc2_scales: Any
    fc1_bias: Any | None
    fc2_bias: Any | None
    swiglu_alpha: Any
    swiglu_beta: Any
    swiglu_limit: Any
    hidden_size: int
    padded_hidden_size: int
    intermediate_size: int
    padded_intermediate_size: int
    config_key: tuple[object, ...]


def _config_key(config: MoEKernelConfig) -> tuple[object, ...]:
    return (
        config.architecture,
        config.quantization,
        config.activation,
        config.top_k,
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
        config.group_size,
        config.expert_bias,
        float(config.activation_alpha),
        float(config.activation_beta),
        config.activation_limit,
        config.gate_up_layout,
    )


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(tensor.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError("MoE weights and routing values must be tensor-like") from error


def _reinterpret_interleaved_scales(torch: Any, tensor: Any, field: str):
    """Expose FlashInfer's interleaved scale bytes through its tested ABI view."""

    shape = _shape(tensor)
    if len(shape) != 3 or shape[-1] % 4:
        raise RuntimeError(
            f"{field} interleaver returned incompatible shape {shape}; "
            "the last dimension must be divisible by four"
        )
    if tensor.dtype != torch.uint8:
        raise RuntimeError(
            f"{field} interleaver returned dtype {tensor.dtype}, expected uint8"
        )
    return tensor.view(torch.int32)


def _copy_into_padded(torch: Any, tensor: Any, shape: tuple[int, ...], fill: int):
    if _shape(tensor) == shape:
        return tensor.contiguous()
    padded = torch.full(
        shape,
        fill_value=fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    source_shape = _shape(tensor)
    destination = padded[tuple(slice(0, size) for size in source_shape)]
    destination.copy_(tensor)
    return padded


def _reorder_and_pad_gate_up(
    torch: Any, tensor: Any, shape: tuple[int, ...], fill: int
):
    """Convert paired ``[gate, up]`` rows to CUTLASS ``[up..., gate...]``."""

    source_shape = _shape(tensor)
    if len(source_shape) not in {2, 3} or len(source_shape) != len(shape):
        raise ValueError("gate/up tensors must be rank two or three")
    source_rows = source_shape[1]
    target_rows = shape[1]
    if source_rows % 2 or target_rows % 2:
        raise ValueError("gate/up row counts must be even")
    intermediate = source_rows // 2
    padded_intermediate = target_rows // 2

    padded = torch.full(
        shape,
        fill_value=fill,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    trailing = tuple(slice(0, size) for size in source_shape[2:])
    source_trailing = (slice(None),) * (len(source_shape) - 2)
    up_destination = (slice(None), slice(0, intermediate), *trailing)
    gate_destination = (
        slice(None),
        slice(padded_intermediate, padded_intermediate + intermediate),
        *trailing,
    )
    up_source = (slice(None), slice(1, None, 2), *source_trailing)
    gate_source = (slice(None), slice(0, None, 2), *source_trailing)
    padded[up_destination].copy_(tensor[up_source])
    padded[gate_destination].copy_(tensor[gate_source])
    return padded


class FlashInferSM90MXFP4Kernel:
    """Adapter for ``flashinfer.fused_moe.cutlass_fused_moe`` on H100.

    The optional underscored constructor arguments are dependency-injection
    seams for CPU contract tests.  Production callers should use the default
    constructor, which validates CUDA compute capability before transforming
    any weights.
    """

    capabilities = MoEKernelCapabilities(
        kernel_id=KERNEL_ID,
        architectures={"sm90"},
        quantizations={"mxfp4"},
        activations={"swiglu"},
        top_k_values={4},
        group_sizes={_MXFP4_GROUP_SIZE},
        supports_expert_bias=True,
        min_experts=4,
        max_experts=128,
        min_tokens=1,
        max_tokens=_MAX_TOKENS,
        gate_up_layouts={"interleaved_gate_up"},
    )

    def __init__(
        self,
        *,
        _torch_module: Any | None = None,
        _fused_moe_api: Any | None = None,
    ) -> None:
        self._torch = _torch_module
        self._fused_moe = _fused_moe_api

    def supports(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> bool:
        if not self.capabilities.supports(config, num_tokens=num_tokens):
            return False
        return config.activation_limit is not None

    def validate(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> None:
        self.capabilities.validate(config, num_tokens=num_tokens)
        if config.activation_limit is None:
            raise UnsupportedMoEConfiguration(
                f"kernel {KERNEL_ID!r} requires activation_limit for SwiGLU"
            )

    def _get_torch(self):
        if self._torch is None:
            self._torch = torch
        return self._torch

    def _get_fused_moe(self):
        if self._fused_moe is None:
            try:
                from flashinfer import fused_moe
            except ImportError as error:
                raise ImportError(
                    f"{KERNEL_ID} requires flashinfer-python with SM90 support"
                ) from error
            self._fused_moe = fused_moe

        required = (
            "ActivationType",
            "cutlass_fused_moe",
            "interleave_moe_scales_for_sm90_mixed_gemm",
            "interleave_moe_weights_for_sm90_mixed_gemm",
        )
        missing = [name for name in required if not hasattr(self._fused_moe, name)]
        if missing:
            raise RuntimeError(
                "FlashInfer does not expose the pinned SM90 MXFP4 API: "
                + ", ".join(missing)
            )
        return self._fused_moe

    def _validate_sm90_tensor(self, tensor: Any, field: str) -> None:
        torch = self._get_torch()
        if not bool(getattr(tensor, "is_cuda", False)):
            raise ValueError(f"{field} must be a CUDA tensor")
        capability = tuple(torch.cuda.get_device_capability(tensor.device))
        if capability != (9, 0):
            raise UnsupportedMoEConfiguration(
                f"kernel {KERNEL_ID!r} requires SM90, got sm{capability[0]}{capability[1]}"
            )

    def _expected_shapes(self, config: MoEKernelConfig) -> dict[str, tuple[int, ...]]:
        experts = config.num_experts
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        group = _MXFP4_GROUP_SIZE
        return {
            "gate_up": (experts, 2 * intermediate, hidden // 2),
            "down": (experts, hidden, intermediate // 2),
            "gate_up_scales": (experts, 2 * intermediate, hidden // group),
            "down_scales": (experts, hidden, intermediate // group),
            "gate_up_bias": (experts, 2 * intermediate),
            "down_bias": (experts, hidden),
        }

    def _validate_weight_field(
        self, field: str, tensor: Any, config: MoEKernelConfig
    ) -> None:
        torch = self._get_torch()
        expected = self._expected_shapes(config)[field]
        actual = _shape(tensor)
        if actual != expected:
            raise ValueError(f"{field} must have shape {expected}, got {actual}")
        if field in {"gate_up", "down", "gate_up_scales", "down_scales"}:
            if tensor.dtype != torch.uint8:
                raise ValueError(f"{field} must have dtype uint8, got {tensor.dtype}")
        else:
            allowed = {torch.float16, torch.bfloat16, torch.float32}
            if tensor.dtype not in allowed:
                raise ValueError(
                    f"{field} must have a floating dtype, got {tensor.dtype}"
                )
        self._validate_sm90_tensor(tensor, field)

    def _validate_weight_devices(self, tensors: list[tuple[str, Any]]) -> None:
        if not tensors:
            return
        expected = tensors[0][1].device
        for field, tensor in tensors[1:]:
            if tensor.device != expected:
                raise ValueError(
                    f"all expert tensors must share one device; {field} is on "
                    f"{tensor.device}, expected {expected}"
                )

    def builder(self, config: MoEKernelConfig) -> FlashInferMXFP4WeightBuilder:
        """Create a per-layer receiver for tensors arriving shard by shard."""

        self.validate(config)
        return FlashInferMXFP4WeightBuilder(self, config)

    def prepare_weights(
        self, weights: MoEExpertWeights[Any], config: MoEKernelConfig
    ) -> PreparedFlashInferMXFP4Weights:
        self.validate(config)
        torch = self._get_torch()
        fused_moe = self._get_fused_moe()

        required = {
            "gate_up": weights.gate_up,
            "down": weights.down,
            "gate_up_scales": weights.gate_up_scales,
            "down_scales": weights.down_scales,
        }
        if config.expert_bias:
            required.update(
                gate_up_bias=weights.gate_up_bias,
                down_bias=weights.down_bias,
            )
        missing = [field for field, tensor in required.items() if tensor is None]
        if missing:
            raise ValueError("missing MXFP4 expert tensors: " + ", ".join(missing))

        present = [
            (field, tensor) for field, tensor in required.items() if tensor is not None
        ]
        for field, tensor in present:
            self._validate_weight_field(field, tensor, config)
        self._validate_weight_devices(present)

        hidden = config.hidden_size
        intermediate = config.intermediate_size
        padded_hidden = _round_up(hidden, _K_ALIGNMENT)
        padded_intermediate = _round_up(intermediate, _K_ALIGNMENT)
        experts = config.num_experts

        # Transform one tensor at a time.  Holding all four padded inputs while
        # an interleaver allocates its output would add several unnecessary
        # gigabytes to GPT-OSS-120B's load-time peak.
        fc1 = _reorder_and_pad_gate_up(
            torch,
            weights.gate_up,
            (experts, 2 * padded_intermediate, padded_hidden // 2),
            0,
        )
        fc1 = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(fc1, "fp4")
        fc1_scales = _reorder_and_pad_gate_up(
            torch,
            weights.gate_up_scales,
            (experts, 2 * padded_intermediate, padded_hidden // _MXFP4_GROUP_SIZE),
            127,
        )
        fc1_scales = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            fc1_scales, _MXFP4_GROUP_SIZE
        )
        fc1_scales = _reinterpret_interleaved_scales(
            torch, fc1_scales, "gate_up_scales"
        )
        fc2 = _copy_into_padded(
            torch,
            weights.down,
            (experts, padded_hidden, padded_intermediate // 2),
            0,
        )
        fc2 = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(fc2, "fp4")
        fc2_scales = _copy_into_padded(
            torch,
            weights.down_scales,
            (
                experts,
                padded_hidden,
                padded_intermediate // _MXFP4_GROUP_SIZE,
            ),
            127,
        )
        fc2_scales = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
            fc2_scales, _MXFP4_GROUP_SIZE
        )
        fc2_scales = _reinterpret_interleaved_scales(torch, fc2_scales, "down_scales")

        device = weights.gate_up.device
        fc1_bias = None
        fc2_bias = None
        if config.expert_bias:
            fc1_bias = weights.gate_up_bias.to(
                device=device, dtype=torch.bfloat16
            ).contiguous()
            fc1_bias = _reorder_and_pad_gate_up(
                torch, fc1_bias, (experts, 2 * padded_intermediate), 0
            )
            fc2_bias = weights.down_bias.to(
                device=device, dtype=torch.bfloat16
            ).contiguous()
            fc2_bias = _copy_into_padded(torch, fc2_bias, (experts, padded_hidden), 0)

        activation_shape = (experts,)
        alpha = torch.full(
            activation_shape,
            fill_value=float(config.activation_alpha),
            dtype=torch.float32,
            device=device,
        )
        beta = torch.full(
            activation_shape,
            fill_value=float(config.activation_beta),
            dtype=torch.float32,
            device=device,
        )
        limit = torch.full(
            activation_shape,
            fill_value=float(config.activation_limit),
            dtype=torch.float32,
            device=device,
        )
        return PreparedFlashInferMXFP4Weights(
            fc1_weights=fc1,
            fc2_weights=fc2,
            fc1_scales=fc1_scales,
            fc2_scales=fc2_scales,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            swiglu_alpha=alpha,
            swiglu_beta=beta,
            swiglu_limit=limit,
            hidden_size=hidden,
            padded_hidden_size=padded_hidden,
            intermediate_size=intermediate,
            padded_intermediate_size=padded_intermediate,
            config_key=_config_key(config),
        )

    def forward(
        self,
        hidden_states: Any,
        routing: MoERouting[Any],
        weights: PreparedFlashInferMXFP4Weights,
        config: MoEKernelConfig,
    ):
        tokens_shape = _shape(hidden_states)
        if len(tokens_shape) != 2:
            raise ValueError(
                f"hidden_states must have shape [tokens, hidden], got {tokens_shape}"
            )
        num_tokens, hidden_size = tokens_shape
        self.validate(config, num_tokens=num_tokens)
        if weights.config_key != _config_key(config):
            raise ValueError("prepared expert weights do not match this configuration")
        if hidden_size != config.hidden_size:
            raise ValueError(
                f"hidden_states width must be {config.hidden_size}, got {hidden_size}"
            )

        torch = self._get_torch()
        fused_moe = self._get_fused_moe()
        self._validate_sm90_tensor(hidden_states, "hidden_states")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"hidden_states must have dtype bfloat16, got {hidden_states.dtype}"
            )
        if hidden_states.device != weights.fc1_weights.device:
            raise ValueError(
                "hidden_states and prepared expert weights must share a device"
            )

        expected_routing = {
            "router_logits": (num_tokens, config.num_experts),
            "selected_experts": (num_tokens, config.top_k),
            "expert_weights": (num_tokens, config.top_k),
        }
        routing_values = {
            "router_logits": routing.router_logits,
            "selected_experts": routing.selected_experts,
            "expert_weights": routing.expert_weights,
        }
        for field, expected in expected_routing.items():
            tensor = routing_values[field]
            actual = _shape(tensor)
            if actual != expected:
                raise ValueError(f"{field} must have shape {expected}, got {actual}")
            if tensor.device != hidden_states.device:
                raise ValueError(f"{field} must be on {hidden_states.device}")
        if routing.selected_experts.dtype not in {torch.int32, torch.int64}:
            raise ValueError("selected_experts must have an integer dtype")
        floating = {torch.float16, torch.bfloat16, torch.float32}
        if routing.router_logits.dtype not in floating:
            raise ValueError("router_logits must have a floating dtype")
        if routing.expert_weights.dtype not in floating:
            raise ValueError("expert_weights must have a floating dtype")

        selected_experts = routing.selected_experts.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        expert_weights = routing.expert_weights.to(
            device=hidden_states.device, dtype=torch.float32
        ).contiguous()

        kernel_input = _copy_into_padded(
            torch,
            hidden_states,
            (num_tokens, weights.padded_hidden_size),
            0,
        )
        # The pinned FlashInfer wrapper returns a list of auxiliary tensors in
        # normal-latency mode even though its public annotation says Tensor.
        # Supplying our own output buffer gives this adapter a stable contract
        # and avoids depending on that return-value discrepancy.
        output = torch.empty(
            (num_tokens, weights.padded_hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        fused_moe.cutlass_fused_moe(
            input=kernel_input,
            token_selected_experts=selected_experts,
            token_final_scales=expert_weights,
            fc1_expert_weights=weights.fc1_weights,
            fc2_expert_weights=weights.fc2_weights,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=[weights.fc1_scales, weights.fc2_scales],
            fc1_expert_biases=weights.fc1_bias,
            fc2_expert_biases=weights.fc2_bias,
            swiglu_alpha=weights.swiglu_alpha,
            swiglu_beta=weights.swiglu_beta,
            swiglu_limit=weights.swiglu_limit,
            use_w4_group_scaling=True,
            use_packed_weights=False,
            tune_max_num_tokens=_MAX_TOKENS,
            activation_type=fused_moe.ActivationType.Swiglu,
            use_fused_finalize=True,
        )
        output_shape = _shape(output)
        expected_kernel_output = (num_tokens, weights.padded_hidden_size)
        if output_shape != expected_kernel_output:
            raise RuntimeError(
                "FlashInfer returned shape "
                f"{output_shape}, expected {expected_kernel_output}"
            )
        if output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"FlashInfer returned dtype {output.dtype}, expected bfloat16"
            )
        if weights.padded_hidden_size != config.hidden_size:
            output = output[:, : config.hidden_size].contiguous()
        return output


class FlashInferMXFP4WeightBuilder:
    """Incremental, single-layer receiver for a shard-streaming model loader."""

    _FIELDS = frozenset(
        {
            "gate_up",
            "down",
            "gate_up_scales",
            "down_scales",
            "gate_up_bias",
            "down_bias",
        }
    )

    def __init__(
        self, kernel: FlashInferSM90MXFP4Kernel, config: MoEKernelConfig
    ) -> None:
        self._kernel = kernel
        self._config = config
        self._tensors: dict[str, Any] = {}
        self._finished = False

    def add(self, field: str, tensor: Any) -> FlashInferMXFP4WeightBuilder:
        if self._finished:
            raise RuntimeError("this MXFP4 weight builder is already finished")
        if field not in self._FIELDS:
            raise KeyError(f"unknown MXFP4 expert tensor: {field!r}")
        if field in self._tensors:
            raise ValueError(f"duplicate MXFP4 expert tensor: {field}")
        self._kernel._validate_weight_field(field, tensor, self._config)
        if self._tensors:
            expected_device = next(iter(self._tensors.values())).device
            if tensor.device != expected_device:
                raise ValueError(
                    f"all expert tensors must share one device; {field} is on "
                    f"{tensor.device}, expected {expected_device}"
                )
        self._tensors[field] = tensor
        return self

    def finish(self) -> PreparedFlashInferMXFP4Weights:
        if self._finished:
            raise RuntimeError("this MXFP4 weight builder is already finished")
        required = {"gate_up", "down", "gate_up_scales", "down_scales"}
        if self._config.expert_bias:
            required.update({"gate_up_bias", "down_bias"})
        missing = sorted(required - self._tensors.keys())
        if missing:
            raise ValueError("missing MXFP4 expert tensors: " + ", ".join(missing))
        weights = MoEExpertWeights(
            gate_up=self._tensors["gate_up"],
            down=self._tensors["down"],
            gate_up_scales=self._tensors["gate_up_scales"],
            down_scales=self._tensors["down_scales"],
            gate_up_bias=self._tensors.get("gate_up_bias"),
            down_bias=self._tensors.get("down_bias"),
        )
        prepared = self._kernel.prepare_weights(weights, self._config)
        self._tensors.clear()
        self._finished = True
        return prepared


def make_flashinfer_sm90_mxfp4_kernel() -> FlashInferSM90MXFP4Kernel:
    """Return the production adapter without importing accelerator packages."""

    return FlashInferSM90MXFP4Kernel()


__all__ = [
    "KERNEL_ID",
    "FlashInferMXFP4WeightBuilder",
    "FlashInferSM90MXFP4Kernel",
    "PreparedFlashInferMXFP4Weights",
    "make_flashinfer_sm90_mxfp4_kernel",
]
