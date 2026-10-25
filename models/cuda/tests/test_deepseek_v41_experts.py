"""Retained checkpoint storage and FlashInfer expert ABI for DeepSeek V4.1."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin_models_cuda.deepseek_v41 import gemm
from resin_models_cuda.deepseek_v41.experts import MoEWeights
from resin_models_cuda.deepseek_v41.linear import SharedExpert
from resin_models_cuda.deepseek_v41.model import Gate, ModelArgs


@pytest.fixture
def gemm_module(monkeypatch):
    moe = ModuleType("flashinfer.fused_moe")
    for name in (
        "reorder_rows_for_gated_act_gemm",
        "trtllm_fp4_block_scale_routed_moe",
    ):
        setattr(moe, name, None)
    quant = ModuleType("flashinfer.quantization")
    for name in ("block_scale_interleave", "shuffle_matrix_a"):
        setattr(quant, name, None)
    enums = ModuleType("flashinfer.tllm_enums")
    setattr(enums, "ActivationType", SimpleNamespace(Swiglu=SimpleNamespace(value=3)))
    monkeypatch.setitem(sys.modules, moe.__name__, moe)
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    monkeypatch.setitem(sys.modules, enums.__name__, enums)
    return gemm


def test_fp4_fused_moe_uses_flashinfer_exact_mixed_format_path(
    monkeypatch: pytest.MonkeyPatch, gemm_module
) -> None:
    deepseek_gemm = gemm_module

    calls: list[dict[str, object]] = []

    def fused_moe(**kwargs: object) -> list[torch.Tensor]:
        calls.append(kwargs)
        return [torch.zeros(3, 128, dtype=torch.bfloat16)]

    monkeypatch.setattr(
        sys.modules["flashinfer.fused_moe"],
        "trtllm_fp4_block_scale_routed_moe",
        fused_moe,
    )
    left = torch.zeros(3, 128, dtype=torch.float8_e4m3fn)
    left_scales = (
        torch.arange(12, dtype=torch.uint8).view(3, 4).view(torch.float8_e4m3fn)
    )
    topk_ids = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.int32)
    topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)
    w13 = torch.empty(2, 256, 64, dtype=torch.float4_e2m1fn_x2)
    w13_scales = torch.ones(2, 256, 4, dtype=torch.float8_e8m0fnu)
    w2 = torch.empty(2, 128, 64, dtype=torch.float4_e2m1fn_x2)
    w2_scales = torch.ones(2, 128, 4, dtype=torch.float8_e8m0fnu)
    output_scales = torch.ones(2, dtype=torch.float32)
    clamp_limits = torch.full((2,), 10.0, dtype=torch.float32)

    output = deepseek_gemm.fp4_fused_moe(
        left,
        left_scales,
        topk_ids,
        topk_weights,
        w13,
        w13_scales,
        w2,
        w2_scales,
        output_scales,
        clamp_limits,
        num_experts=4,
        local_expert_offset=2,
        scale_dtype=torch.float8_e8m0fnu,
    )

    kwargs = calls[0]
    ids, weights = kwargs["topk_ids"]  # type: ignore[misc]  # ty: ignore[not-iterable]
    assert ids is topk_ids
    assert weights is topk_weights
    assert kwargs["hidden_states_scale"] is left_scales
    assert kwargs["gemm1_weights"].data_ptr() == w13.data_ptr()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
    assert kwargs["gemm2_weights"].data_ptr() == w2.data_ptr()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
    assert kwargs["gemm1_clamp_limit"] is clamp_limits
    assert kwargs["output1_scale_scalar"] is output_scales
    assert kwargs["num_experts"] == 4
    assert kwargs["local_expert_offset"] == 2
    assert kwargs["local_num_experts"] == 2
    assert kwargs["activation_type"] == 3
    assert output.shape == (3, 128)


def test_fused_moe_weight_preparation_is_in_place_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, gemm_module
) -> None:
    deepseek_gemm = gemm_module
    calls: list[tuple[str, tuple[int, ...]]] = []

    def reorder(tensor: torch.Tensor) -> torch.Tensor:
        calls.append(("reorder", tuple(tensor.shape)))
        return tensor.clone()

    def shuffle_weight(tensor: torch.Tensor, tile: int) -> torch.Tensor:
        assert tile == 128
        calls.append(("weight", tuple(tensor.shape)))
        return tensor.clone()

    def shuffle_scale(tensor: torch.Tensor) -> torch.Tensor:
        calls.append(("scale", tuple(tensor.shape)))
        return tensor + 2

    monkeypatch.setattr(
        sys.modules["flashinfer.fused_moe"], "reorder_rows_for_gated_act_gemm", reorder
    )
    monkeypatch.setattr(
        sys.modules["flashinfer.quantization"], "shuffle_matrix_a", shuffle_weight
    )
    monkeypatch.setattr(
        sys.modules["flashinfer.quantization"], "block_scale_interleave", shuffle_scale
    )
    w13 = torch.zeros(2, 256, 64, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    w13_scales = torch.zeros(2, 256, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w2 = torch.zeros(2, 128, 64, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    w2_scales = torch.zeros(2, 128, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    deepseek_gemm.prepare_fp4_fused_moe_weights(w13, w13_scales, w2, w2_scales)
    first_calls = list(calls)
    deepseek_gemm.prepare_fp4_fused_moe_weights(w13, w13_scales, w2, w2_scales)

    assert calls == first_calls
    assert len([call for call in calls if call[0] == "reorder"]) == 1
    assert torch.all(w13.view(torch.uint8) == 0)
    assert torch.all(w2.view(torch.uint8) == 0)
    assert torch.all(w13_scales.view(torch.uint8) == 2)
    assert torch.all(w2_scales.view(torch.uint8) == 2)


def test_grouped_expert_storage_preserves_checkpoint_names_and_aliases():
    args = ModelArgs(
        dim=128,
        moe_inter_dim=128,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_dtype="fp4",
    )
    moe = MoEWeights(
        0,
        args,
        Gate(args),
        SharedExpert(args),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    assert moe.experts[0] is None and moe.experts[1] is None
    first, second = moe.experts[2], moe.experts[3]
    assert first is not None and second is not None
    assert first.w1.weight.data_ptr() == moe._fused_w13_weight[0, 128:].data_ptr()
    assert first.w3.weight.data_ptr() == moe._fused_w13_weight[0, :128].data_ptr()
    assert second.w2.weight.data_ptr() == moe._fused_w2_weight[1].data_ptr()
    state = moe.state_dict()
    assert "experts.2.w1.weight" in state and "experts.2.w1.scale" in state
    assert "experts.3.w3.weight" in state
    assert not any("_fused" in name for name in state)
    with torch.no_grad():
        first.w1.weight.view(torch.uint8).fill_(17)
    assert torch.all(moe._fused_w13_weight[0, 128:].view(torch.uint8) == 17)
