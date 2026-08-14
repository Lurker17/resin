from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

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
    }
    values.update(changes)
    return MoEKernelConfig(**values)


def _tensor(torch, shape, dtype=None, *, device=None):
    return _FakeTensor(
        shape,
        torch.uint8 if dtype is None else dtype,
        device=device,
    )


def _weights(torch, config=None):
    config = _config() if config is None else config
    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return MoEExpertWeights(
        gate_up=_tensor(torch, (experts, 2 * intermediate, hidden // 2)),
        down=_tensor(torch, (experts, hidden, intermediate // 2)),
        gate_up_scales=_tensor(torch, (experts, 2 * intermediate, hidden // 32)),
        down_scales=_tensor(torch, (experts, hidden, intermediate // 32)),
        gate_up_bias=_tensor(torch, (experts, 2 * intermediate), torch.float32),
        down_bias=_tensor(torch, (experts, hidden), torch.float32),
    )


class TestFlashInferMXFP4Adapter:
    def _kernel(self, *, capability=(9, 0)):
        from resin.models.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        torch = _FakeTorch(capability)
        fused_moe = _FakeFusedMoE()
        kernel = FlashInferSM90MXFP4Kernel(
            _torch_module=torch,
            _fused_moe_api=fused_moe,
        )
        return kernel, torch, fused_moe

    def test_import_loads_torch_but_not_flashinfer(self) -> None:
        code = """
import sys
from resin.models.gpt_oss.flashinfer import make_flashinfer_sm90_mxfp4_kernel
make_flashinfer_sm90_mxfp4_kernel()
assert 'torch' in sys.modules
for name in ('flashinfer', 'triton'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_capabilities_implement_the_standard_contract(self) -> None:
        kernel, _, _ = self._kernel()
        assert isinstance(kernel, MoEKernel)
        assert kernel.supports(_config(), num_tokens=16)
        assert not kernel.supports(replace(_config(), architecture="sm100"))
        assert not kernel.supports(replace(_config(), activation_limit=None))
        with pytest.raises(
            UnsupportedMoEConfiguration, match="requires activation_limit"
        ):
            kernel.validate(replace(_config(), activation_limit=None))

    def test_prepare_pads_2880_and_uses_exact_sm90_interleavers(self) -> None:
        kernel, torch, fused_moe = self._kernel()
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

    def test_forward_uses_exact_low_level_api_types_and_scale_order(self) -> None:
        kernel, torch, fused_moe = self._kernel()
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

    def test_shape_architecture_and_incremental_builder_fail_early(self) -> None:
        kernel, torch, _ = self._kernel()
        config = _config()
        invalid = _weights(torch)
        invalid = replace(
            invalid,
            gate_up=_tensor(torch, (128, 5760, 1439)),
        )
        with pytest.raises(ValueError, match="gate_up must have shape"):
            kernel.prepare_weights(invalid, config)

        wrong_arch_kernel, wrong_arch_torch, _ = self._kernel(capability=(10, 0))
        with pytest.raises(UnsupportedMoEConfiguration, match="requires SM90"):
            wrong_arch_kernel.prepare_weights(_weights(wrong_arch_torch), config)

        builder = kernel.builder(config)
        weights = _weights(torch)
        builder.add("gate_up", weights.gate_up)
        with pytest.raises(ValueError, match="duplicate"):
            builder.add("gate_up", weights.gate_up)
        with pytest.raises(ValueError, match="missing MXFP4"):
            builder.finish()

        complete = kernel.builder(config)
        for field in (
            "gate_up",
            "down",
            "gate_up_scales",
            "down_scales",
            "gate_up_bias",
            "down_bias",
        ):
            complete.add(field, getattr(weights, field))
        prepared = complete.finish()
        assert prepared.padded_hidden_size == 2944
        assert not complete._tensors
        with pytest.raises(RuntimeError, match="already finished"):
            complete.finish()
