"""Dense TP preserves packed checkpoint coordinates and model MLP semantics."""

import copy
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import ProcessGroup
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.loading import prepare_model
from resin_models_cuda.dense_parallel import TensorParallelSum, shard_dense_mlp
from resin_models_cuda.nvfp4 import Nvfp4LinearWeights


class _MLP(nn.Module):
    def __init__(self, hidden=16, intermediate=24, *, quantized=False, bias=True):
        super().__init__()
        self.intermediate_size = intermediate

        def linear(inputs, outputs):
            return (
                Nvfp4LinearWeights(inputs, outputs)
                if quantized
                else nn.Linear(inputs, outputs, bias=bias, dtype=torch.float64)
            )

        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def _context(rank, size):
    return TensorParallelContext(
        TensorParallelLayout(rank, size), cast(ProcessGroup, object())
    )


def _check_reduction(monkeypatch, models, inputs, expected):
    total = sum(model.local(inputs) for model in models)
    torch.testing.assert_close(total, expected, atol=1e-12, rtol=1e-12)
    groups = []

    def reduce(partial, *, group):
        groups.append(group)
        partial.copy_(total)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    for model in models:
        torch.testing.assert_close(model(inputs), expected, atol=1e-12, rtol=1e-12)
    assert groups == [model.parallel.process_group for model in models]


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_shards_sum_to_the_full_mlp_and_bias_is_counted_once(
    monkeypatch, size, bias
):
    torch.manual_seed(31)
    original = _MLP(bias=bias).eval()
    inputs = torch.randn(2, 5, 16, dtype=torch.float64)
    models = [
        shard_dense_mlp(copy.deepcopy(original), _context(rank, size))
        for rank in range(size)
    ]
    _check_reduction(monkeypatch, models, inputs, original(inputs))
    for rank, model in enumerate(models):
        assert isinstance(model, TensorParallelSum)
        assert not model.training
        assert isinstance(model.local, _MLP)
        assert model.local.intermediate_size == 24 // size
        assert model.local.gate_proj.out_features == 24 // size
        assert model.local.down_proj.in_features == 24 // size
        if bias and rank:
            assert torch.count_nonzero(model.local.down_proj.bias) == 0


def _decode(source):
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    nibbles = (
        torch.stack((source.weight & 15, source.weight >> 4), -1).flatten(-2).long()
    )
    return (
        levels[nibbles]
        * source.weight_scale.double().repeat_interleave(16, -1)
        * source.weight_scale_2.double()
    )


def _dequantize(module):
    if not isinstance(module, Nvfp4LinearWeights):
        return module
    linear = nn.Linear(
        module.in_features, module.out_features, bias=False, dtype=torch.float64
    )
    linear.weight = nn.Parameter(_decode(module))
    return linear


def test_nvfp4_slices_packed_columns_and_scales_before_lowering(monkeypatch):
    torch.manual_seed(12)
    original = _MLP(64, 256, quantized=True)
    for layer in (original.gate_proj, original.up_proj, original.down_proj):
        layer.weight.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8))
        layer.weight_scale.copy_(
            (torch.rand(layer.weight_scale.shape) / 8).to(torch.float8_e4m3fn)
        )
        layer.weight_scale_2.fill_(0.125)
        layer.input_scale.fill_(2.5)
    reference = prepare_model(copy.deepcopy(original), _dequantize)
    models = []
    for rank in range(2):
        source = copy.deepcopy(original)
        weights = {
            name: getattr(source, name).weight
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        model = shard_dense_mlp(source, _context(rank, 2))
        for name in weights:
            local = getattr(model.local, name)
            assert (
                local.weight.untyped_storage().nbytes() * 2
                == weights[name].untyped_storage().nbytes()
            )
            assert local.weight.data_ptr() != weights[name].data_ptr()
            assert local.weight.is_contiguous()
            assert local.input_scale.item() == 2.5
            assert local.weight_scale_2.item() == 0.125
            expected = _decode(getattr(original, name))
            expected = (
                expected[rank * 128 : (rank + 1) * 128]
                if name != "down_proj"
                else expected[:, rank * 128 : (rank + 1) * 128]
            )
            torch.testing.assert_close(_decode(local), expected, atol=0, rtol=0)
        models.append(prepare_model(model, _dequantize))
    inputs = torch.randn(3, 64, dtype=torch.float64)
    _check_reduction(monkeypatch, models, inputs, reference(inputs))


def test_disabled_dense_tp_preserves_module_identity():
    mlp = _MLP()
    assert shard_dense_mlp(mlp, TensorParallelContext()) is mlp


@pytest.mark.parametrize(
    "intermediate,quantized,message",
    [(25, False, "divisible"), (96, True, "aligned to 64")],
)
def test_invalid_shard_geometry_is_rejected_before_mutation(
    intermediate, quantized, message
):
    mlp = _MLP(64, intermediate, quantized=quantized)
    before = tuple(mlp.children())
    with pytest.raises(ValueError, match=message):
        shard_dense_mlp(mlp, _context(0, 2))
    assert tuple(mlp.children()) == before


def test_unsupported_or_mismatched_projections_are_rejected():
    mlp = _MLP()
    mlp.up_proj = nn.Identity()
    with pytest.raises(TypeError, match="unprepared"):
        shard_dense_mlp(mlp, _context(0, 2))
    mlp.up_proj = nn.Linear(16, 25)
    with pytest.raises(ValueError, match="incompatible"):
        shard_dense_mlp(mlp, _context(0, 2))
