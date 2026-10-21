"""Accelerator-neutral interface for fused mixture-of-experts kernels.

This module describes the boundary between model code and an optional kernel
adapter.  It intentionally imports no tensor framework or kernel compiler.
Adapters may therefore depend on PyTorch, FlashInfer, Triton, or another stack
without making those dependencies part of :mod:`resin`'s import path.

The model supplies weights in canonical logical projection order.  For a gated
MoE with ``E`` experts, hidden size ``H``, and intermediate size ``I``:

* ``gate_up`` has logical shape ``[E, 2I, H]``. Its row ordering is declared by
  ``MoEKernelConfig.gate_up_layout``; ``"interleaved_gate_up"`` means
  ``[gate_0, up_0, gate_1, up_1, ...]``;
* ``down`` has logical shape ``[E, H, I]``;
* projection biases have shapes ``[E, 2I]`` and ``[E, H]``.

Four-bit tensors pack the final (input/K) axis two values per byte, so their
physical shapes are ``[E, 2I, H/2]`` and ``[E, H, I/2]``.  Group scales have
logical shape ``[E, N, K/group_size]``.  ``prepare_weights`` is the cold-path
seam where an adapter may wrap, swizzle, interleave, or otherwise transform
those tensors.  ``forward`` is the hot-path seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

TensorT = TypeVar("TensorT")
PreparedWeightsT = TypeVar("PreparedWeightsT")


class UnsupportedMoEConfiguration(ValueError):
    """Raised when a kernel cannot execute a requested MoE configuration."""


def _require_name(field: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_positive_int(field: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _freeze_names(field: str, values: object) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be a collection of names, not one string")
    try:
        frozen = frozenset(values)  # ty: ignore[invalid-argument-type]  # Validate untrusted input below.
    except TypeError as error:
        raise ValueError(f"{field} must be a collection of names") from error
    if not frozen:
        raise ValueError(f"{field} cannot be empty")
    for value in frozen:
        _require_name(field, value)
    return frozen


def _format_values(values: frozenset[object]) -> str:
    return "{" + ", ".join(sorted(repr(value) for value in values)) + "}"


@dataclass(frozen=True, slots=True)
class MoEKernelConfig:
    """One model layer's framework-independent fused-MoE requirements.

    Names such as ``"sm90"``, ``"mxfp4"``, and ``"swiglu"`` are deliberately
    open strings rather than enums: Resin does not own the set of accelerators,
    number formats, or activations that adapters may add.
    """

    architecture: str
    quantization: str
    activation: str
    top_k: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    group_size: int | None = None
    expert_bias: bool = False
    activation_alpha: float = 1.0
    activation_beta: float = 1.0
    activation_limit: float | None = None
    gate_up_layout: str = "interleaved_gate_up"

    def __post_init__(self) -> None:
        for field in (
            "architecture",
            "quantization",
            "activation",
            "gate_up_layout",
        ):
            _require_name(field, getattr(self, field))
        for field in (
            "top_k",
            "num_experts",
            "hidden_size",
            "intermediate_size",
        ):
            _require_positive_int(field, getattr(self, field))
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.group_size is not None:
            _require_positive_int("group_size", self.group_size)
            if self.hidden_size % self.group_size:
                raise ValueError("group_size must divide hidden_size")
            if self.intermediate_size % self.group_size:
                raise ValueError("group_size must divide intermediate_size")
        if not isinstance(self.expert_bias, bool):
            raise TypeError("expert_bias must be a boolean")
        for field in ("activation_alpha", "activation_beta"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field} must be finite")
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        if self.activation_limit is not None:
            if (
                isinstance(self.activation_limit, bool)
                or not isinstance(self.activation_limit, (int, float))
                or not math.isfinite(self.activation_limit)
            ):
                raise ValueError("activation_limit must be finite")
            if self.activation_limit <= 0:
                raise ValueError("activation_limit must be positive")


@dataclass(frozen=True, slots=True)
class MoEKernelCapabilities:
    """Immutable, inspectable operating envelope for one kernel adapter.

    The named sets describe a Cartesian operating envelope.  An adapter with
    additional layout or shape restrictions should reject them in its own
    ``supports`` and ``validate`` methods before calling the kernel.
    """

    kernel_id: str
    architectures: frozenset[str]
    quantizations: frozenset[str]
    activations: frozenset[str]
    top_k_values: frozenset[int]
    group_sizes: frozenset[int | None] = frozenset({None})
    supports_expert_bias: bool = False
    min_experts: int = 1
    max_experts: int | None = None
    min_tokens: int = 1
    max_tokens: int | None = None
    gate_up_layouts: frozenset[str] = frozenset({"interleaved_gate_up"})
    supports_fp32_partial: bool = False

    def __post_init__(self) -> None:
        _require_name("kernel_id", self.kernel_id)
        for field in (
            "architectures",
            "quantizations",
            "activations",
            "gate_up_layouts",
        ):
            object.__setattr__(self, field, _freeze_names(field, getattr(self, field)))

        top_k_values = frozenset(self.top_k_values)
        if not top_k_values:
            raise ValueError("top_k_values cannot be empty")
        for value in top_k_values:
            _require_positive_int("top_k_values", value)
        object.__setattr__(self, "top_k_values", top_k_values)

        group_sizes = frozenset(self.group_sizes)
        if not group_sizes:
            raise ValueError("group_sizes cannot be empty")
        for value in group_sizes:
            if value is not None:
                _require_positive_int("group_sizes", value)
        object.__setattr__(self, "group_sizes", group_sizes)

        if not isinstance(self.supports_expert_bias, bool):
            raise TypeError("supports_expert_bias must be a boolean")
        if not isinstance(self.supports_fp32_partial, bool):
            raise TypeError("supports_fp32_partial must be a boolean")
        _require_positive_int("min_experts", self.min_experts)
        if self.max_experts is not None:
            _require_positive_int("max_experts", self.max_experts)
            if self.max_experts < self.min_experts:
                raise ValueError("max_experts cannot be smaller than min_experts")
        _require_positive_int("min_tokens", self.min_tokens)
        if self.max_tokens is not None:
            _require_positive_int("max_tokens", self.max_tokens)
            if self.max_tokens < self.min_tokens:
                raise ValueError("max_tokens cannot be smaller than min_tokens")

    def supports(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> bool:
        """Return whether the declared operating envelope accepts a request."""

        return not self._unsupported_reasons(config, num_tokens=num_tokens)

    def validate(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> None:
        """Fail with actionable detail when this adapter cannot run a request."""

        reasons = self._unsupported_reasons(config, num_tokens=num_tokens)
        if reasons:
            detail = "; ".join(reasons)
            raise UnsupportedMoEConfiguration(
                f"kernel {self.kernel_id!r} does not support this MoE "
                f"configuration: {detail}"
            )

    def _unsupported_reasons(
        self, config: MoEKernelConfig, *, num_tokens: int | None
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        checks = (
            ("architecture", config.architecture, self.architectures),
            ("quantization", config.quantization, self.quantizations),
            ("activation", config.activation, self.activations),
            ("gate_up_layout", config.gate_up_layout, self.gate_up_layouts),
            ("top_k", config.top_k, self.top_k_values),
            ("group_size", config.group_size, self.group_sizes),
        )
        for field, value, supported in checks:
            if value not in supported:
                reasons.append(
                    f"{field}={value!r} is unsupported "
                    f"(supported: {_format_values(supported)})"
                )
        if config.expert_bias and not self.supports_expert_bias:
            reasons.append("expert_bias=True is unsupported")
        if config.num_experts < self.min_experts:
            reasons.append(
                f"num_experts={config.num_experts} is below {self.min_experts}"
            )
        if self.max_experts is not None and config.num_experts > self.max_experts:
            reasons.append(
                f"num_experts={config.num_experts} exceeds {self.max_experts}"
            )
        if num_tokens is not None:
            if (
                isinstance(num_tokens, bool)
                or not isinstance(num_tokens, int)
                or num_tokens <= 0
            ):
                reasons.append("num_tokens must be a positive integer")
            elif num_tokens < self.min_tokens:
                reasons.append(f"num_tokens={num_tokens} is below {self.min_tokens}")
            elif self.max_tokens is not None and num_tokens > self.max_tokens:
                reasons.append(f"num_tokens={num_tokens} exceeds {self.max_tokens}")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class MoEExpertWeights(Generic[TensorT]):
    """Canonical layer weights before adapter-specific cold-path preparation."""

    gate_up: TensorT
    down: TensorT
    gate_up_scales: TensorT | None = None
    down_scales: TensorT | None = None
    gate_up_bias: TensorT | None = None
    down_bias: TensorT | None = None


@dataclass(frozen=True, slots=True)
class MoERouting(Generic[TensorT]):
    """All routing representations an adapter may need for one invocation.

    ``router_logits`` is ``[tokens, experts]``.  ``selected_experts`` and
    ``expert_weights`` are both ``[tokens, top_k]``.  Supplying all three keeps
    routing policy in model code while accommodating kernels that build their
    own gather/scatter metadata from the full logits.

    For a configuration with ``E`` experts, selected ID ``E`` with zero weight
    denotes an inactive assignment. Adapters must ignore these slots without
    rerouting or renormalizing the remaining weights. Active IDs are in
    ``[0, E)``. This allows an EP owner to execute its subset of global routes.
    """

    router_logits: TensorT
    selected_experts: TensorT
    expert_weights: TensorT


@runtime_checkable
class MoEKernel(Protocol[TensorT, PreparedWeightsT]):
    """Structural interface implemented by fused-MoE kernel adapters.

    Adapters advertising ``supports_fp32_partial`` also implement
    ``forward_partial`` with the same signature as ``forward``. It returns
    weighted local expert sums in FP32, including for empty inputs, without
    rounding the sum to the activation dtype before cross-rank combination.
    The adapter owns any expansion/chunking needed to respect its token limit.
    """

    capabilities: MoEKernelCapabilities

    def supports(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> bool:
        """Return whether this adapter accepts the complete request."""

    def validate(
        self, config: MoEKernelConfig, *, num_tokens: int | None = None
    ) -> None:
        """Raise before preparation or execution if a request is unsupported."""

    def prepare_weights(
        self, weights: MoEExpertWeights[TensorT], config: MoEKernelConfig
    ) -> PreparedWeightsT:
        """Perform optional wrapping or layout conversion on the cold path."""

    def forward(
        self,
        hidden_states: TensorT,
        routing: MoERouting[TensorT],
        weights: PreparedWeightsT,
        config: MoEKernelConfig,
    ) -> TensorT:
        """Execute the fused experts on the hot path and return hidden states."""


__all__ = [
    "MoEExpertWeights",
    "MoEKernel",
    "MoEKernelCapabilities",
    "MoEKernelConfig",
    "MoERouting",
    "UnsupportedMoEConfiguration",
]
