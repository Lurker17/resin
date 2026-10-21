from __future__ import annotations

import dataclasses
import subprocess
import sys
from dataclasses import replace

import pytest

from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernel,
    MoEKernelCapabilities,
    MoEKernelConfig,
    MoERouting,
    UnsupportedMoEConfiguration,
)


def _config(**changes) -> MoEKernelConfig:
    values = {
        "architecture": "sm90",
        "quantization": "mxfp4",
        "activation": "swiglu",
        "top_k": 4,
        "num_experts": 128,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "group_size": 32,
        "expert_bias": True,
        "activation_alpha": 1.702,
        "activation_beta": 1.0,
        "activation_limit": 7.0,
        "gate_up_layout": "interleaved_gate_up",
    }
    values.update(changes)
    return MoEKernelConfig(**values)


def _capabilities(**changes) -> MoEKernelCapabilities:
    return replace(
        MoEKernelCapabilities(
            kernel_id="test-sm90-mxfp4",
            architectures=frozenset({"sm90"}),
            quantizations=frozenset({"mxfp4"}),
            activations=frozenset({"swiglu"}),
            top_k_values=frozenset({4}),
            group_sizes=frozenset({32}),
            supports_expert_bias=True,
            min_experts=8,
            max_experts=128,
            min_tokens=1,
            max_tokens=8192,
        ),
        **changes,
    )


class TestMoEKernelContract:
    def test_gpt_oss_mxfp4_configuration_is_supported(self) -> None:
        capabilities = _capabilities()
        config = _config()

        assert capabilities.supports(config, num_tokens=16)
        assert capabilities.validate(config, num_tokens=8192) is None

    def test_arch_quantization_activation_and_top_k_fail_explicitly(self) -> None:
        capabilities = _capabilities()
        config = _config()
        changes = {
            "architecture": "sm80",
            "quantization": "int8",
            "activation": "gelu",
            "top_k": 2,
            "gate_up_layout": "concatenated_gate_up",
        }

        for field, value in changes.items():
            unsupported = replace(config, **{field: value})
            assert not capabilities.supports(unsupported)
            with pytest.raises(
                UnsupportedMoEConfiguration, match=rf"{field}={value!r}"
            ):
                capabilities.validate(unsupported)

    def test_shape_bias_group_and_token_envelope_fail_before_execution(self) -> None:
        capabilities = _capabilities()

        cases = (
            (_config(num_experts=256), None, "num_experts=256 exceeds 128"),
            (_config(expert_bias=True), 8193, "num_tokens=8193 exceeds 8192"),
            (
                _config(group_size=None),
                None,
                "group_size=None is unsupported",
            ),
        )
        for config, num_tokens, message in cases:
            with (
                pytest.raises(UnsupportedMoEConfiguration, match=message),
            ):
                capabilities.validate(config, num_tokens=num_tokens)

        no_bias = _capabilities(supports_expert_bias=False)
        assert not no_bias.supports(_config())
        with pytest.raises(
            UnsupportedMoEConfiguration, match="expert_bias=True is unsupported"
        ):
            no_bias.validate(_config())

    def test_config_rejects_invalid_dimensions_and_grouping(self) -> None:
        with pytest.raises(ValueError, match="top_k cannot exceed num_experts"):
            _config(top_k=4, num_experts=2)
        with pytest.raises(ValueError, match="group_size must divide hidden_size"):
            _config(hidden_size=2879)
        with pytest.raises(ValueError, match="activation_limit must be positive"):
            _config(activation_limit=0.0)

    def test_capabilities_are_deeply_immutable(self) -> None:
        architectures = {"sm90"}
        capabilities = _capabilities(architectures=architectures)
        config = _config()
        architectures.add("sm100")

        assert capabilities.architectures == frozenset({"sm90"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(capabilities, "kernel_id", "changed")  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(config, "top_k", 2)  # type: ignore[misc]

    def test_structural_adapter_uses_only_opaque_tensor_objects(self) -> None:
        class FakeKernel:
            capabilities = _capabilities()

            def supports(self, config, *, num_tokens=None):
                return self.capabilities.supports(config, num_tokens=num_tokens)

            def validate(self, config, *, num_tokens=None):
                self.capabilities.validate(config, num_tokens=num_tokens)

            def prepare_weights(self, weights, config):
                self.validate(config)
                return (weights.gate_up, weights.down)

            def forward(self, hidden_states, routing, weights, config):
                self.validate(config, num_tokens=1)
                return (hidden_states, routing, weights)

        kernel = FakeKernel()
        assert isinstance(kernel, MoEKernel)

        raw = MoEExpertWeights(gate_up=object(), down=object())
        prepared = kernel.prepare_weights(raw, _config())
        routing = MoERouting(object(), object(), object())
        hidden = object()
        output = kernel.forward(hidden, routing, prepared, _config())
        assert output[0] is hidden
        assert output[1] is routing
        assert output[2] == prepared

    def test_import_does_not_load_accelerator_or_kernel_stacks(self) -> None:
        code = (
            "import sys; import resin.moe_kernels; "
            "names=('torch','flashinfer','triton'); "
            "print([name for name in names if name in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "[]"
