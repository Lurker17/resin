from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from resin.moe_kernels import (
    MoEExpertWeights,
    MoEKernel,
    MoEKernelConfig,
    MoERouting,
    UnsupportedMoEConfiguration,
)


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _Device:
    def __init__(self, kind: str = "cuda", index: int = 0) -> None:
        self.type = kind
        self.index = index

    def __eq__(self, other) -> bool:
        return isinstance(other, _Device) and (self.type, self.index) == (
            other.type,
            other.index,
        )

    def __repr__(self) -> str:
        return f"{self.type}:{self.index}"


class _FakeView:
    def __init__(self, parent, key) -> None:
        self.parent = parent
        self.key = key
        self.dtype = parent.dtype
        self.device = parent.device
        self.shape = tuple(
            self._slice_size(size, selection)
            for size, selection in zip(parent.shape, key, strict=True)
        )

    @staticmethod
    def _slice_size(size, selection):
        if isinstance(selection, int):
            return None
        start, stop, step = selection.indices(size)
        return len(range(start, stop, step))

    def copy_(self, source):
        self.parent.copies.append((self.key, source))
        return self

    def contiguous(self):
        return _FakeTensor(self.shape, self.dtype, device=self.device)


class _FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        device: _Device | None = None,
        fill_value=None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or _Device()
        self.fill_value = fill_value
        self.copies = []

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def contiguous(self):
        return self

    def to(self, *, device=None, dtype=None):
        return _FakeTensor(
            self.shape,
            self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            fill_value=self.fill_value,
        )

    def view(self, dtype):
        if self.dtype.name == "uint8" and dtype.name == "int32":
            if self.shape[-1] % 4:
                raise ValueError("last dimension is not int32-aligned")
            shape = (*self.shape[:-1], self.shape[-1] // 4)
        else:
            shape = self.shape
        return _FakeTensor(shape, dtype, device=self.device)

    def __getitem__(self, key):
        return _FakeView(self, key)


class _FakeCuda:
    def __init__(self, capability=(9, 0)) -> None:
        self.capability = capability

    def get_device_capability(self, device):
        del device
        return self.capability


class _FakeTorch:
    uint8 = _DType("uint8")
    int32 = _DType("int32")
    int64 = _DType("int64")
    float16 = _DType("float16")
    bfloat16 = _DType("bfloat16")
    float32 = _DType("float32")

    def __init__(self, capability=(9, 0)) -> None:
        self.cuda = _FakeCuda(capability)
        self.full_calls = []
        self.empty_calls = []

    def full(self, shape, *, fill_value, dtype, device):
        tensor = _FakeTensor(
            shape,
            dtype,
            device=device,
            fill_value=fill_value,
        )
        self.full_calls.append(tensor)
        return tensor

    def empty(self, shape, *, dtype, device):
        tensor = _FakeTensor(shape, dtype, device=device)
        self.empty_calls.append(tensor)
        return tensor


class _ActivationType:
    Swiglu = "swiglu-enum"


class _FakeFusedMoE:
    ActivationType = _ActivationType

    def __init__(self) -> None:
        self.weight_calls = []
        self.scale_calls = []
        self.forward_calls = []

    def interleave_moe_weights_for_sm90_mixed_gemm(self, tensor, quant_type):
        self.weight_calls.append((tensor, quant_type))
        return _FakeTensor(tensor.shape, tensor.dtype, device=tensor.device)

    def interleave_moe_scales_for_sm90_mixed_gemm(self, tensor, group_size):
        self.scale_calls.append((tensor, group_size))
        experts, rows, groups = tensor.shape
        return _FakeTensor(
            (experts, groups // 4, rows * 4),
            tensor.dtype,
            device=tensor.device,
        )

    def cutlass_fused_moe(self, **kwargs):
        self.forward_calls.append(kwargs)
        return [kwargs["output"], "auxiliary-output"]


def _config(**changes) -> MoEKernelConfig:
    values: dict[str, Any] = {
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
    }
    values.update(changes)
    return MoEKernelConfig(**values)


def _tensor(runtime, shape, dtype=None, *, device=None):
    return _FakeTensor(
        shape,
        runtime.uint8 if dtype is None else dtype,
        device=device,
    )


def _weights(runtime, config=None):
    config = _config() if config is None else config
    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return MoEExpertWeights(
        gate_up=_tensor(runtime, (experts, 2 * intermediate, hidden // 2)),
        down=_tensor(runtime, (experts, hidden, intermediate // 2)),
        gate_up_scales=_tensor(runtime, (experts, 2 * intermediate, hidden // 32)),
        down_scales=_tensor(runtime, (experts, hidden, intermediate // 32)),
        gate_up_bias=_tensor(runtime, (experts, 2 * intermediate), runtime.float32),
        down_bias=_tensor(runtime, (experts, hidden), runtime.float32),
    )


class TestFlashInferMXFP4Adapter:
    def _kernel(self, monkeypatch, *, capability=(9, 0)):
        import resin_models_cuda.gpt_oss.flashinfer as flashinfer
        from resin_models_cuda.gpt_oss.flashinfer import (
            FlashInferSM90MXFP4Kernel,
        )

        torch = _FakeTorch(capability)
        fused_moe = _FakeFusedMoE()
        monkeypatch.setattr(flashinfer, "torch", torch)
        kernel = FlashInferSM90MXFP4Kernel()
        monkeypatch.setattr(kernel, "_fused_moe", fused_moe, raising=False)
        return kernel, torch, fused_moe

    def test_capabilities_implement_the_standard_contract(self, monkeypatch) -> None:
        kernel, _, _ = self._kernel(monkeypatch)
        assert isinstance(kernel, MoEKernel)
        assert kernel.supports(_config(), num_tokens=16)
        assert not kernel.supports(replace(_config(), architecture="sm100"))
        assert not kernel.supports(replace(_config(), activation_limit=None))
        with pytest.raises(
            UnsupportedMoEConfiguration, match="requires activation_limit"
        ):
            kernel.validate(replace(_config(), activation_limit=None))

    def test_prepare_pads_2880_and_uses_exact_sm90_interleavers(
        self, monkeypatch
    ) -> None:
        kernel, torch, fused_moe = self._kernel(monkeypatch)
        prepared = kernel.prepare_weights(_weights(torch), _config())

        assert [(call[0].shape, call[1]) for call in fused_moe.weight_calls] == [
            ((128, 5888, 1472), "fp4"),
            ((128, 2944, 1472), "fp4"),
        ]
        fc1_before_interleave = fused_moe.weight_calls[0][0]
        assert len(fc1_before_interleave.copies) == 2
        assert [copy[0][1] for copy in fc1_before_interleave.copies] == [
            slice(0, 2880),
            slice(2944, 5824),
        ]
        assert [copy[1].key[1] for copy in fc1_before_interleave.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        assert [(call[0].shape, call[1]) for call in fused_moe.scale_calls] == [
            ((128, 5888, 92), 32),
            ((128, 2944, 92), 32),
        ]
        fc1_scales_before_interleave = fused_moe.scale_calls[0][0]
        assert [copy[1].key[1] for copy in fc1_scales_before_interleave.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        scale_padding = [
            tensor
            for tensor in torch.full_calls
            if tensor.fill_value == 127 and len(tensor.shape) == 3
        ]
        assert [tensor.shape for tensor in scale_padding] == [
            (128, 5888, 92),
            (128, 2944, 92),
        ]
        assert prepared.fc1_bias.shape == (128, 5888)
        assert prepared.fc2_bias.shape == (128, 2944)
        assert [copy[1].key[1] for copy in prepared.fc1_bias.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        assert prepared.fc1_bias.dtype is torch.bfloat16
        assert prepared.fc2_bias.dtype is torch.bfloat16
        assert prepared.swiglu_alpha.shape == (128,)
        assert prepared.swiglu_alpha.fill_value == 1.702
        assert prepared.swiglu_beta.fill_value == 1.0
        assert prepared.swiglu_limit.fill_value == 7.0
        assert prepared.swiglu_alpha.dtype is torch.float32
        assert prepared.padded_hidden_size == 2944
        assert prepared.padded_intermediate_size == 2944

    def test_forward_uses_exact_low_level_api_types_and_scale_order(
        self, monkeypatch
    ) -> None:
        kernel, torch, fused_moe = self._kernel(monkeypatch)
        config = _config()
        prepared = kernel.prepare_weights(_weights(torch), config)
        hidden = _tensor(torch, (16, 2880), torch.bfloat16)
        routing = MoERouting(
            router_logits=_tensor(torch, (16, 128), torch.float32),
            selected_experts=_tensor(torch, (16, 4), torch.int64),
            expert_weights=_tensor(torch, (16, 4), torch.bfloat16),
        )

        output = kernel.forward(hidden, routing, prepared, config)
        assert output.shape == (16, 2880)
        assert output.dtype is torch.bfloat16
        assert len(fused_moe.forward_calls) == 1
        call = fused_moe.forward_calls[0]
        assert call["input"].shape == (16, 2944)
        assert call["token_selected_experts"].dtype is torch.int32
        assert call["token_final_scales"].dtype is torch.float32
        assert call["quant_scales"] == [prepared.fc1_scales, prepared.fc2_scales]
        assert prepared.fc1_scales.dtype is torch.int32
        assert prepared.fc2_scales.dtype is torch.int32
        assert call["output_dtype"] is torch.bfloat16
        assert call["output"].shape == (16, 2944)
        assert call["output"].dtype is torch.bfloat16
        assert call["fc1_expert_biases"].dtype is torch.bfloat16
        assert call["fc2_expert_biases"].dtype is torch.bfloat16
        assert call["swiglu_alpha"].shape == (128,)
        assert call["swiglu_beta"].shape == (128,)
        assert call["swiglu_limit"].shape == (128,)
        assert call["use_w4_group_scaling"]
        assert not call["use_packed_weights"]
        assert call["tune_max_num_tokens"] == 8192
        assert call["activation_type"] == _ActivationType.Swiglu

    def test_shape_and_architecture_fail_early(self, monkeypatch) -> None:
        kernel, torch, _ = self._kernel(monkeypatch)
        config = _config()
        invalid = _weights(torch)
        invalid = replace(
            invalid,
            gate_up=_tensor(torch, (128, 5760, 1439)),
        )
        with pytest.raises(ValueError, match="gate_up must have shape"):
            kernel.prepare_weights(invalid, config)

        wrong_arch_kernel, wrong_arch_torch, _ = self._kernel(
            monkeypatch, capability=(10, 0)
        )
        with pytest.raises(UnsupportedMoEConfiguration, match="requires SM90"):
            wrong_arch_kernel.prepare_weights(_weights(wrong_arch_torch), config)

    def test_inactive_route_sentinel_reaches_cutlass_without_rerouting(
        self, monkeypatch
    ):
        import torch

        import resin_models_cuda.gpt_oss.flashinfer as adapter

        config = _config(num_experts=4, hidden_size=32, intermediate_size=32)
        kernel = adapter.FlashInferSM90MXFP4Kernel()
        monkeypatch.setattr(kernel, "_validate_sm90_tensor", lambda *args: None)
        indices = torch.tensor([[4, 0, 1, 2], [0, 4, 4, 4]], dtype=torch.int64)
        weights = torch.tensor(
            [[0, 0.5, 0.25, 0.25], [1, 0, 0, 0]], dtype=torch.float32
        )
        hidden = torch.ones(2, 32, dtype=torch.bfloat16)

        def cutlass(**kwargs):
            actual_ids = kwargs["token_selected_experts"]
            actual_weights = kwargs["token_final_scales"]
            assert actual_ids.dtype == torch.int32
            assert torch.equal(actual_ids, indices.int())
            assert torch.equal(actual_weights, weights)
            active = actual_ids < config.num_experts
            scales = ((actual_ids + 1) * actual_weights * active).sum(1)
            kwargs["output"].copy_(hidden * scales[:, None])

        monkeypatch.setattr(
            kernel,
            "_get_fused_moe",
            lambda: SimpleNamespace(
                ActivationType=SimpleNamespace(Swiglu=object()),
                cutlass_fused_moe=cutlass,
            ),
        )
        prepared = SimpleNamespace(
            config_key=adapter._config_key(config),
            fc1_weights=torch.empty(1),
            fc2_weights=torch.empty(1),
            fc1_scales=None,
            fc2_scales=None,
            fc1_bias=None,
            fc2_bias=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            padded_hidden_size=32,
        )
        routing = MoERouting(torch.zeros(2, 4), indices, weights)
        # This partial prepared-weight fixture supplies only fields read by forward.
        output = kernel.forward(hidden, routing, prepared, config)  # ty: ignore[invalid-argument-type]
        torch.testing.assert_close(
            output[:, 0], torch.tensor([1.75, 1.0], dtype=torch.bfloat16)
        )

    def test_fp32_partials_preserve_cancellation_and_chunk_expanded_routes(
        self, monkeypatch
    ):
        import torch

        import resin_models_cuda.gpt_oss.flashinfer as adapter

        config = _config(num_experts=4, hidden_size=32, intermediate_size=32)
        kernel = adapter.FlashInferSM90MXFP4Kernel()
        kernel.capabilities = replace(kernel.capabilities, max_tokens=2)
        packed = torch.empty(1, dtype=torch.uint8)
        bias = torch.zeros(4, 32, dtype=torch.bfloat16)
        bias[0] = 0.75
        bias[1] = 1.25
        prepared = adapter.PreparedFlashInferMXFP4Weights(
            packed,
            packed,
            packed,
            packed,
            torch.zeros(4, 64, dtype=torch.bfloat16),
            bias,
            None,
            None,
            None,
            32,
            32,
            32,
            32,
            adapter._config_key(config),
            torch.zeros_like(bias),
        )
        calls = []

        def unweighted(hidden, routing, weights, actual):
            assert actual is config
            assert len(hidden) <= 2
            assert weights.fc1_weights is packed and weights.fc2_weights is packed
            assert weights.fc1_bias is prepared.fc1_bias
            assert not weights.fc2_bias.any()
            assert routing.selected_experts[:, 1:].eq(config.num_experts).all()
            active = routing.selected_experts[:, 0] < config.num_experts
            torch.testing.assert_close(routing.expert_weights[:, 0], active.float())
            assert not routing.expert_weights[:, 1:].any()
            calls.append(len(hidden))
            values = torch.tensor(
                [1000, -1000, 4, 0, float("nan")], dtype=torch.bfloat16
            )
            return values[routing.selected_experts[:, 0].long(), None].expand(-1, 32)

        monkeypatch.setattr(kernel, "forward", unweighted)

        def dynamic_shape(*args, **kwargs):
            # pytest's _WithException wrapper obscures fail's callable signature.
            pytest.fail("partial execution must not compact routes dynamically")  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(torch.Tensor, "nonzero", dynamic_shape)
        hidden = torch.ones(3, 32, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 4, 4, 4], [1, 4, 4, 4], [0, 1, 2, 4]])
        scales = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0.25, 0.5, 0.25, 0]])
        routing = MoERouting(torch.zeros(3, 4), ids, scales)
        output = kernel.forward_partial(hidden, routing, prepared, config)
        assert output.dtype == torch.float32
        torch.testing.assert_close(
            output[:, 0], torch.tensor([1000.75, -998.75, -248.1875])
        )
        assert calls == [2] * 6
        assert output[0, 0] + output[1, 0] == 2
        assert output[:2, 0].bfloat16().float().sum() == 0
        assert prepared.fc2_bias is bias and bias[0, 0] == 0.75
        empty = kernel.forward_partial(
            hidden[:0],
            MoERouting(torch.empty(0, 4), ids[:0], scales[:0]),
            prepared,
            config,
        )
        assert empty.shape == (0, 32) and empty.dtype == torch.float32
        assert calls == [2] * 6

        # Replaying the same row shape with different active routes keeps all
        # kernel shapes and chunk counts stable, including entirely idle input.
        idle = kernel.forward_partial(
            hidden,
            MoERouting(torch.zeros(3, 4), ids, torch.zeros_like(scales)),
            prepared,
            config,
        )
        assert idle.dtype == torch.float32
        assert torch.equal(idle, torch.zeros_like(idle))
        assert calls == [2] * 12
