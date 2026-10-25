"""SM100 numerical and changed-input graph checks for GLM mHC and NVFP4.

Weights are synthetic checkpoint-format tensors, independently unpacked for the
reference. The mHC compilation check covers its TileLang readout cache only;
these tests do not claim full-model parity or whole-model compiler stability.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from resin.backend.torch import use_default_dtype
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash.model import (
    Bf16Experts,
    HyperConnection,
    Nvfp4Experts,
    linear,
    swiglu,
)
from resin_models_cuda.nvfp4 import quantize_nvfp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100 Blackwell and the pinned CUDA kernels",
)


def _dequant(packed, scales, alpha):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = levels[codes & 7] * torch.where((codes & 8) != 0, -1, 1)
    return values * scales.float().repeat_interleave(16, -1) * alpha


def _requant_fp32(values, global_scale):
    """Reference fused FP32 activation → NVFP4, without a BF16 intermediate."""
    grouped = values.float().unflatten(-1, (-1, 16))
    scales = (
        (grouped.abs().amax(-1) / (6 * global_scale)).to(torch.float8_e4m3fn).float()
    )
    # CUDA multiplies by a rounded FP32 reciprocal; direct division differs
    # at FP4 midpoints, e.g. 2.34375 / 0.9375 is 2.5 versus 2.5000002384.
    normalized = torch.where(
        scales[..., None] > 0,
        grouped * (scales[..., None] * global_scale).reciprocal(),
        0,
    )
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device, dtype=torch.float32
    )
    distances = (normalized.abs()[..., None] - levels).abs()
    nearest = distances.argmin(-1)
    ties = (distances == distances.amin(-1, keepdim=True)).sum(-1) > 1
    nearest += (ties & (nearest.remainder(2) == 1)).long()
    return (
        levels[nearest] * normalized.sign() * scales[..., None] * global_scale
    ).flatten(-2)


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    return graph, output


def _assert_error(actual, expected, relative_l2=0.02, atol=0.125):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    relative = ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item()
    assert relative < relative_l2, (
        f"relative L2 {relative}, max error {(actual - expected).abs().max().item()}"
    )
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=atol)


class SyntheticReader:
    device = torch.device("cuda")

    def __init__(self, count=8, hidden=256, intermediate=256, up_alpha=2.0):
        self.tensors = {}
        generator = torch.Generator().manual_seed(52)
        for expert in range(count):
            for name in ("gate_proj", "up_proj", "down_proj"):
                out_dim, in_dim = (
                    (hidden, intermediate)
                    if name == "down_proj"
                    else (intermediate, hidden)
                )
                prefix = f"{expert}.{name}."
                self.tensors[prefix + "weight"] = torch.randint(
                    0,
                    256,
                    (out_dim, in_dim // 2),
                    generator=generator,
                    dtype=torch.uint8,
                )
                self.tensors[prefix + "weight_scale"] = torch.full(
                    (out_dim, in_dim // 16), 0.125
                ).to(torch.float8_e4m3fn)
                self.tensors[prefix + "weight_scale_2"] = torch.tensor(
                    {"gate_proj": 2.0, "up_proj": up_alpha, "down_proj": 0.25}[name]
                )
                # Unowned expert zero holds the largest calibration values.
                self.tensors[prefix + "input_scale"] = torch.tensor(
                    0.25 if expert == 0 else 0.125
                )

    def read(self, name, shape=None, dtype=None, *, cpu=False):
        value = self.tensors[name]
        if shape is not None:
            assert tuple(value.shape) == tuple(shape)
        if dtype is not None:
            assert value.dtype == dtype
        return value if cpu else value.to(self.device)


def _moe_reference(reader, hidden, ids, weights, start, count, limit=10):
    a = max(
        float(value)
        for name, value in reader.tensors.items()
        if name.endswith("input_scale") and "down_proj" not in name
    )
    b = max(
        float(value)
        for name, value in reader.tensors.items()
        if name.endswith("down_proj.input_scale")
    )
    packed, scales = quantize_nvfp4(hidden, torch.tensor(1 / a, device="cuda"))
    activation = _dequant(packed, scales, a)
    torch.testing.assert_close(_requant_fp32(hidden, a), activation, atol=0, rtol=0)
    result = torch.zeros_like(hidden, dtype=torch.float32)
    clipped_elements = 0
    for expert in range(start, start + count):
        base = f"{expert}."
        gate_alpha = reader.read(base + "gate_proj.weight_scale_2")
        up_alpha = reader.read(base + "up_proj.weight_scale_2")

        def first_weight(name, alpha):
            # Decode ORIGINAL checkpoint matrices, without using any adapter
            # output or reparameterizing/requantizing their E4M3 block scales.
            return _dequant(
                reader.read(base + name + ".weight"),
                reader.read(base + name + ".weight_scale"),
                alpha,
            )

        # Global alphas belong in the epilogue, not inside the GEMM reduction:
        # a non-power-of-two scale moved into the matrix changes FP4 tie bins.
        gate_raw = (activation / a) @ first_weight("gate_proj", 1.0).T
        up_raw = (activation / a) @ first_weight("up_proj", 1.0).T
        gate = gate_raw * (a * gate_alpha)
        up = up_raw * (a * up_alpha)
        # The official adapter supplies one accumulator-domain clamp limit,
        # calibrated by the gate alpha, and distinct gate/up output scales.
        # Equal alphas (all GLM checkpoint experts) give the usual +/-limit.
        up_limit = limit * up_alpha / gate_alpha
        clipped_elements += int(((gate > limit) | (up.abs() > up_limit)).sum())
        # TRT's GEMM1 activation epilogue quantizes FP32 directly. An extra
        # BF16 rounding here changes FP4 bins and creates cancellation outliers.
        raw_limit = limit / (a * gate_alpha)
        epilogue = F.silu(gate_raw.clamp(max=raw_limit) * (a * gate_alpha))
        epilogue = epilogue * up_raw.clamp(-raw_limit, raw_limit)
        epilogue = epilogue * (a * up_alpha / b)
        intermediate = _requant_fp32(epilogue, 1.0) * b
        down = _dequant(
            reader.read(base + "down_proj.weight"),
            reader.read(base + "down_proj.weight_scale"),
            reader.read(base + "down_proj.weight_scale_2"),
        )
        # TRT's GEMM2 runner stores BF16 before the separate routing finalize.
        # Retaining FP32 here gives cancellation outliers despite exact GEMM1.
        contribution = (intermediate @ down.T).bfloat16().float()
        probability = torch.where(ids == expert, weights.bfloat16().float(), 0).sum(-1)
        result += contribution * probability[:, None]
    assert clipped_elements > 0, "test must exercise the finite SwiGLU clamp"
    return result.bfloat16()


@torch.inference_mode()
@pytest.mark.parametrize("up_alpha", [2.0, 3.25])
def test_nvfp4_partial_experts_match_reference_and_graph_route_changes(up_alpha):
    reader = SyntheticReader(up_alpha=up_alpha)
    config = replace(
        Glm53Config(),
        hidden_size=256,
        moe_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    experts = Nvfp4Experts(config, reader, "", ep_size=2, ep_rank=1, max_tokens=128)
    generator = torch.Generator(device="cuda").manual_seed(44)
    for rows in (3, 17, 65):
        hidden = (
            torch.randn(
                rows, 256, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 3
        )
        ids = torch.stack(
            (
                torch.arange(rows, device="cuda") % 8,
                (torch.arange(rows, device="cuda") + 3) % 8,
            ),
            -1,
        ).int()
        weights = (
            torch.rand(rows, 2, device="cuda", generator=generator).softmax(-1) * 2.5
        )
        reference = _moe_reference(reader, hidden, ids, weights, 4, 4)
        _assert_error(
            experts(hidden, ids, weights), reference, relative_l2=0.025, atol=0.5
        )
        graph, output = _capture(lambda: experts(hidden, ids, weights))
        for phase in range(3):
            if phase == 1:
                ids.fill_(8)  # Inactive EP receive rows use the global sentinel.
                weights.zero_()
            else:
                hidden.mul_(-0.75)
                ids.copy_(
                    torch.stack(
                        (
                            (torch.arange(rows, device="cuda") + phase) % 8,
                            (torch.arange(rows, device="cuda") + 5 + phase) % 8,
                        ),
                        -1,
                    ).int()
                )
                weights.fill_(1.25)
            expected = experts(hidden, ids, weights)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, atol=0.0625, rtol=0.005)
            if phase == 1:
                assert not output.any()


@torch.inference_mode()
def test_nvfp4_distinct_alpha_identity_down_exposes_exact_gemm1_activation():
    reader = SyntheticReader(up_alpha=3.25)
    for expert in range(8):
        base = f"{expert}.down_proj."
        # E2M1 code 2 is exactly one; all other matrix entries are zero.
        packed = torch.zeros(256, 128, dtype=torch.uint8)
        rows = torch.arange(256)
        packed[rows, rows // 2] = (2 << (4 * (rows % 2))).to(torch.uint8)
        reader.tensors[base + "weight"] = packed
        reader.tensors[base + "weight_scale"] = torch.ones(256, 16).to(
            torch.float8_e4m3fn
        )
        reader.tensors[base + "weight_scale_2"] = torch.tensor(1.0)
    config = replace(
        Glm53Config(),
        hidden_size=256,
        moe_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    experts = Nvfp4Experts(config, reader, "", ep_size=2, ep_rank=1, max_tokens=128)
    generator = torch.Generator(device="cuda").manual_seed(44)
    # Reproduce the broad routing test's final cohort, including RNG draws.
    for count in (3, 17, 65):
        hidden = (
            torch.randn(
                count, 256, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 3
        )
        torch.rand(count, 2, device="cuda", generator=generator)
    ids = torch.full((65, 2), 8, device="cuda", dtype=torch.int32)
    weights = torch.zeros(65, 2, device="cuda")
    weights[:, 0] = 1
    for expert in range(4, 8):
        ids[:, 0] = expert
        # Identity GEMM2 and unit routing expose the quantized activation
        # exactly, proving separate gate/up scales and clamp semantics.
        expected = _moe_reference(reader, hidden, ids, weights, 4, 4)
        torch.testing.assert_close(
            experts(hidden, ids, weights), expected, atol=0, rtol=0
        )


@torch.inference_mode()
def test_dense_shared_swiglu_single_cast_and_graph_changed_inputs():
    generator = torch.Generator(device="cuda").manual_seed(390)
    gate = (torch.randn(19, 256, device="cuda", generator=generator) * 6).bfloat16()
    up = (torch.randn(19, 256, device="cuda", generator=generator) * 6).bfloat16()

    def reference():
        # Independent FP64 oracle; only the final product is rounded to BF16.
        g, u = gate.double().clamp(max=10), up.double().clamp(-10, 10)
        return g / (1 + torch.exp(-g)) * u

    def check(actual):
        precise = reference()
        rounded = precise.bfloat16()
        mismatched = actual != rounded
        # FP32 exp/reciprocal may land on the other side of a BF16 midpoint.
        # Accept only adjacent BF16 outputs within four FP32 relative ULPs of
        # that midpoint, not arbitrary one-ULP errors throughout the tensor.
        alternate, nearest = actual[mismatched], rounded[mismatched]
        torch.testing.assert_close(
            torch.nextafter(nearest, alternate), alternate, atol=0, rtol=0
        )
        midpoint = (alternate.double() + nearest.double()) / 2
        exact = precise[mismatched]
        assert torch.all(
            (exact - midpoint).abs() <= 4 * torch.finfo(torch.float32).eps * exact.abs()
        )

    expected = reference().bfloat16()
    legacy = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    assert torch.count_nonzero(legacy != expected) > 700
    check(swiglu(gate, up, 10))
    graph, output = _capture(lambda: swiglu(gate, up, 10))
    for _ in range(3):
        gate.mul_(-0.75)
        up.mul_(1.125)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, swiglu(gate, up, 10), atol=0, rtol=0)
        check(output)


@torch.inference_mode()
def test_nvfp4_dense_is_w4a16_and_replays_changed_inputs():
    reader = SyntheticReader(count=1)
    weights = {
        key.removeprefix("0."): value.cuda() for key, value in reader.tensors.items()
    }
    operation = linear(weights, "gate_proj")
    reference_weight = _dequant(
        weights["gate_proj.weight"],
        weights["gate_proj.weight_scale"],
        weights["gate_proj.weight_scale_2"],
    )
    for rows in (1, 19, 67):
        hidden = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        expected = hidden.float() @ reference_weight.T
        _assert_error(operation(hidden), expected, relative_l2=0.005, atol=0.125)
        graph, output = _capture(lambda: operation(hidden))
        hidden.mul_(-0.625)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, operation(hidden), atol=0, rtol=0)


@torch.inference_mode()
def test_mtp_bf16_experts_preserve_clamped_gate_up_and_local_routing():
    reader = SyntheticReader()
    generator = torch.Generator().manual_seed(59)
    for name, tensor in list(reader.tensors.items()):
        if name.endswith(".weight"):
            reader.tensors[name] = (
                torch.randn(tensor.shape[0], tensor.shape[1] * 2, generator=generator)
                / 4
            ).bfloat16()
    config = replace(
        Glm53Config(),
        hidden_size=256,
        moe_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    with use_default_dtype(torch.bfloat16):
        experts = Bf16Experts(config, reader, "", ep_size=2, ep_rank=1, max_tokens=128)
    assert (
        experts.alpha.dtype
        == experts.beta.dtype
        == experts.clamp.dtype
        == torch.float32
    )
    gpu_generator = torch.Generator(device="cuda").manual_seed(59)
    hidden = (
        torch.randn(
            17, 256, device="cuda", dtype=torch.bfloat16, generator=gpu_generator
        )
        * 3
    )
    ids = torch.stack(
        (
            torch.arange(17, device="cuda") % 8,
            (torch.arange(17, device="cuda") + 3) % 8,
        ),
        -1,
    ).int()
    weights = (
        torch.rand(17, 2, device="cuda", generator=gpu_generator).softmax(-1) * 2.5
    )
    reference = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(4, 8):
        gate = (
            F.linear(hidden.float(), reader.read(f"{expert}.gate_proj.weight").float())
            .bfloat16()
            .float()
            .clamp(max=10)
        )
        up = (
            F.linear(hidden.float(), reader.read(f"{expert}.up_proj.weight").float())
            .bfloat16()
            .float()
            .clamp(-10, 10)
        )
        # CUTLASS's unfused activation path stores GEMM1 and activated rows as
        # BF16; GEMM2 also stores BF16 before the FP32 weighted final reduction.
        contribution = (
            F.linear(
                (F.silu(gate) * up).bfloat16().float(),
                reader.read(f"{expert}.down_proj.weight").float(),
            )
            .bfloat16()
            .float()
        )
        probability = torch.where(ids == expert, weights, 0).sum(-1)
        reference += contribution * probability[:, None]
    _assert_error(experts(hidden, ids, weights), reference, relative_l2=0.012, atol=1)
    graph, output = _capture(lambda: experts(hidden, ids, weights))
    ids.fill_(8)
    weights.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert not output.any()
    ids[:, 0].fill_(7)
    ids[:, 1].fill_(4)
    weights.fill_(1.25)
    hidden.mul_(-0.5)
    expected = experts(hidden, ids, weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_mhc_sm100_matches_cpu_reference_and_graph_changed_streams():
    config = Glm53Config()
    generator = torch.Generator().manual_seed(867)
    weights = {
        "hc_attn_fn": (torch.randn(24, 4 * 4096, generator=generator) / 128).bfloat16(),
        "hc_attn_base": (torch.randn(24, generator=generator) / 3).bfloat16(),
        "hc_attn_scale": torch.tensor([0.2, 0.4, 0.8]).bfloat16(),
    }
    cpu = HyperConnection(config, weights, "attn")
    gpu = HyperConnection(
        config, {key: value.cuda() for key, value in weights.items()}, "attn"
    )
    for rows in (3, 17, 65):
        hidden = torch.randn(rows, 4, 4096, generator=generator).bfloat16().cuda()
        expected, expected_post, expected_comb = cpu(hidden.cpu())
        value, post, comb = gpu(hidden)
        _assert_error(value, expected.cuda(), relative_l2=0.004, atol=0.0625)
        torch.testing.assert_close(post.cpu(), expected_post, rtol=0.002, atol=0.0002)
        torch.testing.assert_close(comb.cpu(), expected_comb, rtol=0.002, atol=0.0002)

        def launch():
            value, post, comb = gpu(hidden)
            return HyperConnection.post(value, hidden, post, comb)

        graph, output = _capture(launch)
        hidden.mul_(-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, launch(), rtol=0, atol=0)
        value, post, comb = cpu(hidden.cpu())
        _assert_error(
            output,
            HyperConnection.post(value, hidden.cpu(), post, comb).cuda(),
            relative_l2=0.006,
            atol=0.125,
        )


def _mhc_known_coefficient_fixture(rows, *, phase=0):
    # sigmoid(0) + 2**-9 is exactly representable. These dyadic BF16 streams
    # keep the four-term FP32 collapse exact, so an FP64 equation can isolate
    # the variance/numerator boundary without GEMM or sigmoid approximation.
    config = replace(Glm53Config(), hc_eps=2**-9, rms_norm_eps=2**-16)
    weights = {
        "hc_attn_fn": torch.zeros(24, 4 * 4096, dtype=torch.float32),
        "hc_attn_base": torch.zeros(24, dtype=torch.float32),
        "hc_attn_scale": torch.ones(3, dtype=torch.float32),
    }
    column = torch.arange(4096)
    row = torch.arange(rows)[:, None, None] % 5
    stream = torch.arange(4)[None, :, None]
    hidden = (
        torch.full((rows, 4, 4096), 0.25)
        if phase == 0
        else (column[None, None, :] % (7 + phase) + 1 + stream + row) / 32
    ).bfloat16()
    norm = (
        torch.ones(4096) if phase == 0 else 0.5 + (column % (13 + phase)) / 16
    ).bfloat16()
    return config, weights, hidden, norm


def _mhc_known_coefficient_reference(hidden, norm, config):
    # Independent known-coefficient equation. Do not call HyperConnection's
    # CPU path: its separate RMS intentionally has a different boundary.
    collapsed = hidden.cpu().double().sum(1) * (0.5 + config.hc_eps)
    torch.testing.assert_close(collapsed, collapsed.float().double(), rtol=0, atol=0)
    numerator = collapsed.bfloat16().double()
    inverse = torch.rsqrt(
        collapsed.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expected = (numerator * inverse * norm.cpu().double()).bfloat16()
    rounded_variance = torch.rsqrt(
        numerator.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    wrong_variance = (numerator * rounded_variance * norm.cpu().double()).bfloat16()
    wrong_numerator = (collapsed * inverse * norm.cpu().double()).bfloat16()
    return expected, wrong_variance, wrong_numerator


def _assert_mhc_rows_exact(actual, expected):
    # Per-row and last-row diagnostics prevent a large early token from hiding
    # a later row's error. This controlled dyadic fixture admits exact BF16 output.
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    per_row = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-8)
    assert torch.equal(actual, expected), (
        f"mHC per-row relative L2={per_row.tolist()}, last={per_row[-1].item()}, "
        f"max absolute={(actual - expected).abs().max().item()}"
    )


@torch.inference_mode()
def test_mhc_fused_norm_fp64_boundary_and_changed_input_weight_graph():
    for rows in (3, 17):
        config, weights, initial, initial_norm = _mhc_known_coefficient_fixture(rows)
        gpu = HyperConnection(
            config, {key: value.cuda() for key, value in weights.items()}, "attn"
        )
        hidden, norm = initial.cuda(), initial_norm.cuda()
        expected, wrong_variance, wrong_numerator = _mhc_known_coefficient_reference(
            initial, initial_norm, config
        )
        # Uniform .25 streams collapse to .501953125, then BF16 rounds to .5.
        # Both incorrect boundaries yield 1; the serving equation yields .99609375.
        assert (expected != wrong_variance).all()
        assert (expected != wrong_numerator).all()
        value, post, _ = gpu(hidden, norm)
        _assert_mhc_rows_exact(value, expected)
        torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=0)
        graph, output = _capture(lambda: gpu(hidden, norm)[0])
        previous = output.clone()
        _, _, changed_hidden, changed_norm = _mhc_known_coefficient_fixture(
            rows, phase=1
        )
        # Mutate inputs and weights separately, then together, to establish that
        # captured kernels read both live buffers instead of retained values.
        for next_hidden, next_norm in (
            (changed_hidden, initial_norm),
            (changed_hidden, changed_norm),
            (initial, changed_norm),
        ):
            hidden.copy_(next_hidden)
            norm.copy_(next_norm)
            graph.replay()
            torch.cuda.synchronize()
            expected, _, _ = _mhc_known_coefficient_reference(
                next_hidden, next_norm, config
            )
            _assert_mhc_rows_exact(output, expected)
            assert not torch.equal(output, previous), "graph fixture must change output"
            previous.copy_(output)


@torch.inference_mode()
def test_mhc_fused_norm_unseen_rows_reuse_tilelang_variant():
    from resin_models_cuda.glm5_3_flash.mhc_cuda import pre_with_norm

    config, weights, hidden, norm = _mhc_known_coefficient_fixture(3, phase=1)
    gpu = HyperConnection(
        config, {key: value.cuda() for key, value in weights.items()}, "attn"
    )
    norm = norm.cuda()
    gpu(hidden.cuda(), norm)
    # TileLang 0.1.14 JITImpl keeps loaded kernels here, including disk-cache
    # hits. This checks the fused readout's variants, not DeepGEMM/AOT caches.
    kernel_cache = getattr(pre_with_norm, "_kernel_cache", None)
    if not isinstance(kernel_cache, dict):
        pytest.skip("TileLang does not expose its loaded-kernel cache")  # ty: ignore[too-many-positional-arguments]
    assert kernel_cache, "warmup must load at least one observable TileLang variant"
    warmed = set(kernel_cache)
    for rows in (1, 17, 65, 129):
        _, _, hidden, _ = _mhc_known_coefficient_fixture(rows, phase=1)
        actual, _, _ = gpu(hidden.cuda(), norm)
        expected, _, _ = _mhc_known_coefficient_reference(hidden, norm.cpu(), config)
        _assert_mhc_rows_exact(actual, expected)
        assert set(kernel_cache) == warmed, (
            f"unseen row count {rows} created a new mHC TileLang variant"
        )


@torch.inference_mode()
def test_mhc_post_serving_fp32_accumulation_and_changed_input_graph():
    from importlib import import_module

    # Install the vendored TileLang package path just as HyperConnection.forward
    # does; this test exercises post directly without invoking the pre-mix GEMM.
    import_module("resin_models_cuda.deepseek_v41.tilelang_kernels")
    generator = torch.Generator(device="cuda").manual_seed(109)
    for rows in (3, 17):
        value = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
        residual = torch.empty(rows, 4, 4096, dtype=torch.bfloat16, device="cuda")
        post = torch.empty(rows, 4, device="cuda")
        comb = torch.empty(rows, 4, 4, device="cuda")

        def change(phase):
            value.normal_(generator=generator)
            residual.normal_(generator=generator)
            post.uniform_(0.25, 1.75, generator=generator)
            comb.uniform_(0.01, 0.9, generator=generator)
            comb.div_(comb.sum(1, keepdim=True))
            # A valid doubly stochastic mix with nearly cancelling terms.
            # Rounding post to BF16 would produce zero instead of the small
            # positive residual required by the serving kernel's FP32 algebra.
            value[0].fill_(128)
            residual[0].fill_(-128)
            comb[0].fill_(0.25)
            post[0].fill_(1.0005 + phase * 0.001)

        def reference():
            # Independent SGLang serving equation: FP32 coefficients and sums,
            # one final BF16 cast. Do not call the HF-style CPU post helper.
            result = post[:, :, None] * value.float()[:, None]
            for stream in range(4):
                result = (
                    result
                    + comb[:, stream, :, None] * residual[:, stream, None, :].float()
                )
            return result.bfloat16()

        change(0)
        graph, output = _capture(
            lambda: HyperConnection.post(value, residual, post, comb)
        )
        for phase in (0, 1):
            change(phase)
            graph.replay()
            torch.cuda.synchronize()
            expected = reference()
            _assert_error(output, expected, relative_l2=0.0005, atol=0.03125)
            torch.testing.assert_close(output[0], expected[0], atol=0, rtol=0)
            assert expected[0].abs().min() > 0.05
            assert not (
                post[0].bfloat16()[:, None] * value[0]
                + torch.matmul(comb[0].bfloat16().T, residual[0])
            ).any(), "cancellation fixture must distinguish BF16 coefficient rounding"
