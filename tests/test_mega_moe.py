"""Prepared-weight, workspace and graph-input contracts without GPU dependencies."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from resin.backend.cuda.mega_moe import MegaMoe, MegaMoeCollective, shared_scale_rows
from resin.backend.topology import RankGroup


@pytest.fixture
def deepgemm(monkeypatch):
    module = SimpleNamespace(calls=[], transforms=[], workspaces=[])

    class Buffer:
        def __init__(
            self, group, experts, capacity, topk, hidden, intermediate, **kwargs
        ):
            self.num_max_tokens_per_rank = capacity
            self.x = torch.zeros(capacity, hidden, dtype=torch.float8_e4m3fn)
            self.x_sf = torch.zeros(capacity, hidden // 128, dtype=torch.int32)
            self.topk_idx = torch.zeros(capacity, topk, dtype=torch.int64)
            self.topk_weights = torch.zeros(capacity, topk, dtype=torch.float32)
            self.shared_l1_acts_sf = torch.zeros(128, hidden // 128, dtype=torch.int32)
            self.destroyed = False
            module.workspaces.append(self)

        def destroy(self):
            assert not self.destroyed
            self.destroyed = True

    def scales(value, n, k, recipe, groups):
        assert recipe == (1, 32)
        assert value.shape[-2:] == (n, k // 32)
        return value.to(torch.float8_e8m0fnu).view(torch.int32).clone()

    def transform(first, second):
        module.transforms.append(
            tuple(tuple(t.clone() for t in pair) for pair in (first, second))
        )
        return tuple(t.clone() for t in first), (second[0], second[1].clone())

    def execute(output, first, second, buffer, **kwargs):
        rows = output.shape[0]
        module.calls.append(
            dict(
                rows=rows,
                indices=buffer.topk_idx[:rows].clone(),
                weights=buffer.topk_weights[:rows].clone(),
                **kwargs,
            )
        )
        output.zero_()

    module.SymmBuffer = Buffer
    module.transform_sf_into_required_layout = scales
    module.transform_weights_for_mega_moe = transform
    module.get_block_m_for_mega_moe = lambda *args: 64
    module.fp8_fp4_mega_moe = execute
    module.gates = []

    def route(hidden, weight, topk, shared, shared_count, scale, rank, **kwargs):
        module.gates.append(dict(rank=rank, mask=kwargs["mask"], out=kwargs["out"]))
        assert not shared and shared_count == 0
        scores = torch.nn.functional.softplus(hidden.float() @ weight.float().T).sqrt()
        indices = (
            kwargs["unmapped_topk_idx"]
            if kwargs["fix_routing_mask"] is not None
            else (scores + kwargs["bias"]).topk(topk, dim=-1).indices
        )
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * scale
        if kwargs["mask"] is not None:
            indices = indices.masked_fill(~kwargs["mask"][:, None], -1)
            weights = weights.masked_fill(~kwargs["mask"][:, None], 0)
        kwargs["out"][0].copy_(indices)
        kwargs["out"][1].copy_(weights)

    module.bf16_mega_gate = route
    monkeypatch.setitem(sys.modules, "deep_gemm", module)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    return module


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 128))
        self.bias = nn.Parameter(torch.zeros(2))
        self.temperature, self.score_func, self.normalized = 1, "sqrtsoftplus", True
        self.topk, self.route_scale = 2, 1.5
        self.benchmark_balanced_experts = False
        self.benchmark_replica, self.benchmark_expert_parallel_size = 0, 1

    def forward(self, hidden, ids):
        return hidden.new_ones((len(hidden), 1)), torch.zeros(
            len(hidden), 1, dtype=torch.int32
        )


def prepared(deepgemm):
    collective = MegaMoeCollective(RankGroup((0,), object(), 0), max_tokens=16)
    weights = (
        torch.zeros(2, 256, 64, dtype=torch.int8),
        torch.ones(2, 256, 1, dtype=torch.int32),
    )
    quantized = []

    def quantize(hidden):
        quantized.append(len(hidden))
        return hidden.to(torch.float8_e4m3fn), torch.ones(
            len(hidden), 4, dtype=torch.float8_e8m0fnu
        )

    def route(hidden, active, out):
        weights, indices = Gate()(hidden, None)
        if active is not None:
            weights = weights.masked_fill(~active[:, None], 0)
            indices = indices.masked_fill(~active[:, None], -1)
        out[0].copy_(indices)
        out[1].copy_(weights)

    layer = MegaMoe(
        Gate(),
        collective,
        quantize,
        route=route,
        gate_up=weights,
        down=weights,
        shared_gate_up=weights,
        shared_down=weights,
        experts=2,
        topk=1,
        hidden=128,
        intermediate=128,
        clamp=10,
    )
    return layer, collective, quantized


def test_shared_expert_scale_rows_follow_upstream_utccp_abi():
    rows = shared_scale_rows(132, 128, device="cpu")
    assert rows[:8].tolist() == [0, 4, 8, 12, 16, 20, 24, 28]
    assert rows[31:34].tolist() == [124, 1, 5]
    assert rows[127:].tolist() == [127, 128, 132, 136, 140]
    assert rows[:128].unique().numel() == 128


def test_idle_rows_enter_mega_collective_and_graph_mask_stays_device_resident(deepgemm):
    layer, collective, quantized = prepared(deepgemm)
    valid = torch.tensor(2, dtype=torch.int32)
    with collective.fixed_capacity(4, valid_rows=valid):
        result = layer(
            torch.ones(2, 2, 128, dtype=torch.bfloat16),
            torch.zeros(2, 2, dtype=torch.long),
        )
        assert result.shape == (2, 2, 128)
        assert deepgemm.calls[-1]["indices"].tolist() == [[0], [0], [-1], [-1]]
        valid.zero_()  # Changed routing on replay uses this same tensor object.
        layer(
            torch.ones(4, 128, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.long)
        )
        assert deepgemm.calls[-1]["indices"].tolist() == [[-1]] * 4
        assert deepgemm.calls[-1]["weights"].count_nonzero() == 0
    layer.forward_empty()
    assert deepgemm.calls[-1]["rows"] == 0
    assert quantized == [4, 4]  # No empty TileLang quantization launch.
    assert len(deepgemm.workspaces) == 1
    assert "shared_l1_weights" in deepgemm.calls[-1]
    assert "shared_l2_weights" in deepgemm.calls[-1]


def test_layers_share_workspace_and_close_releases_it_once(deepgemm):
    layer, collective, _ = prepared(deepgemm)
    assert collective.buffer(2, 1, 128, 128) is layer.workspace
    assert collective.allocated
    collective.close()
    collective.close()
    assert not collective.allocated
    assert deepgemm.workspaces[0].destroyed


def test_v41_factory_preserves_gate_up_order_values_and_adopts_down_storage(
    deepgemm,
):
    from resin_models_cuda.deepseek_v41.experts import prepare_mega_moe

    def linear(value):
        return SimpleNamespace(
            weight=nn.Parameter(
                torch.full((128, 128), value, dtype=torch.float8_e4m3fn)
            ),
            scale=nn.Parameter(torch.full((4, 4), value, dtype=torch.float8_e8m0fnu)),
        )

    source = SimpleNamespace(
        routed_intermediate_size=128,
        dim=128,
        n_routed_experts=2,
        n_activated_experts=2,
        swiglu_limit=10,
        gate=Gate(),
        _fused_w13_weight=torch.cat(
            (
                torch.full((2, 128, 64), 3, dtype=torch.uint8),
                torch.full((2, 128, 64), 5, dtype=torch.uint8),
            ),
            1,
        ),
        _fused_w13_scale=torch.ones(2, 256, 4, dtype=torch.float8_e8m0fnu),
        _fused_w2_weight=torch.full((2, 128, 64), 7, dtype=torch.uint8),
        _fused_w2_scale=torch.ones(2, 128, 4, dtype=torch.float8_e8m0fnu),
        shared_experts=SimpleNamespace(w1=linear(1), w2=linear(2), w3=linear(4)),
    )
    gate_pointer = source.gate.weight.data_ptr()
    collective = MegaMoeCollective(RankGroup((0,), object(), 0), 16)
    layer = prepare_mega_moe(source, collective)
    routed, shared = deepgemm.transforms
    assert (routed[0][0][:, :128] == 5).all()  # Gate precedes up.
    assert (routed[0][0][:, 128:] == 3).all()
    assert (shared[0][0][:128].float() == 1).all()
    assert (shared[0][0][128:].float() == 4).all()
    assert shared[0][1].view(torch.float8_e8m0fnu).float()[:128].eq(1).all()
    assert shared[0][1].view(torch.float8_e8m0fnu).float()[128:].eq(4).all()
    assert layer.gate is source.gate
    assert layer.gate.weight.data_ptr() == gate_pointer
    assert layer.down_weight.data_ptr() == source._fused_w2_weight.data_ptr()
    assert (
        layer.shared_down_weight.data_ptr()
        == source.shared_experts.w2.weight.data_ptr()
    )
    assert layer.down_weight.eq(7).all()  # Prepared weights survive ownership transfer.
    assert layer.shared_down_weight.float().eq(2).all()
    assert all(
        not value.requires_grad and value.grad_fn is None for value in layer.buffers()
    )


@pytest.mark.parametrize("balanced", [False, True])
def test_mega_gate_writes_workspace_masks_rows_and_preserves_bf16_weights(
    deepgemm, monkeypatch, balanced
):
    from resin.backend.torch import benchmark_expert_indices
    from resin_models_cuda.deepseek_v41.experts import mega_gate

    torch.manual_seed(31)
    gate = Gate()
    gate.weight = nn.Parameter(torch.randn(8, 256) * 0.1)
    gate.bias = nn.Parameter(torch.linspace(-0.3, 0.3, 8))
    gate.topk = 3
    gate.benchmark_balanced_experts = balanced
    gate.benchmark_replica, gate.benchmark_expert_parallel_size = 1, 2
    hidden = torch.randn(4, 256).bfloat16()
    scores = torch.nn.functional.softplus(hidden.float() @ gate.weight.T).sqrt()
    expected_indices = (
        benchmark_expert_indices(scores, 3, 1, 2)
        if balanced
        else (scores + gate.bias).topk(3, dim=-1).indices
    )
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = (
        expected_weights / expected_weights.sum(-1, keepdim=True) * gate.route_scale
    ).bfloat16()

    def no_unfused(*_):
        raise AssertionError("unfused gate called")

    monkeypatch.setattr(gate, "forward", no_unfused)
    indices = torch.empty(4, 3, dtype=torch.long)
    weights = torch.empty(4, 3)
    active = torch.tensor([True, True, False, False])
    mega_gate(gate, hidden, active, (indices, weights), ep_rank=1)
    assert deepgemm.gates[-1]["rank"] == 1
    assert deepgemm.gates[-1]["out"][0].data_ptr() == indices.data_ptr()
    assert torch.equal(indices[:2], expected_indices[:2].long())
    assert torch.equal(weights[:2], expected_weights[:2].float())
    assert torch.equal(weights, weights.bfloat16().float())
    assert indices[2:].eq(-1).all() and weights[2:].eq(0).all()
    active[:] = torch.tensor([False, True, True, True])
    mega_gate(gate, hidden, active, (indices, weights), ep_rank=1)
    assert indices[0].eq(-1).all() and weights[0].eq(0).all()
    assert torch.equal(indices[1:], expected_indices[1:].long())


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("temperature", 2), ("score_func", "softmax"), ("normalized", False), ("topk", 1)],
)
def test_mega_factory_rejects_unsupported_gate_before_weight_transforms(
    deepgemm, attribute, value
):
    from resin_models_cuda.deepseek_v41.experts import prepare_mega_moe

    gate = Gate()
    setattr(gate, attribute, value)
    collective = MegaMoeCollective(RankGroup((0,), object(), 0), 16)
    with pytest.raises(ValueError, match="normalized sqrtsoftplus"):
        prepare_mega_moe(SimpleNamespace(gate=gate), collective)
    assert deepgemm.transforms == []


def test_mega_hc_transition_preserves_shifted_pre_and_full_residual(deepgemm):
    from deploy.deepinfer_kernel_probe import _mhc_reference
    from resin_models_cuda.deepseek_v41.hc import mega_ffn_transition

    captured = {}

    def execute(x, residual, shifted_prev_mix, post_mix, comb_res_mix, **kwargs):
        inputs = dict(
            x=x,
            residual=residual,
            shifted_prev_mix=shifted_prev_mix,
            post_mix=post_mix,
            comb_res_mix=comb_res_mix,
            **{
                key: value
                for key, value in kwargs.items()
                if not key.startswith(("new_", "y_"))
            },
        )
        captured.update(inputs)
        for name, value in _mhc_reference(inputs).items():
            kwargs[name].copy_(value)

    deepgemm.mega_mhc = execute
    torch.manual_seed(81)
    dim = 128
    block = SimpleNamespace(
        args=SimpleNamespace(
            dim=dim, hc_mult=4, norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20
        ),
        hc_ffn_fn=torch.randn(24, 4 * dim) * 0.01,
        hc_ffn_scale=torch.randn(3) * 0.1,
        hc_ffn_base=torch.randn(24) * 0.1,
        ffn_norm=SimpleNamespace(weight=torch.randn(dim).bfloat16()),
    )
    output = torch.randn(2, 3, dim).bfloat16()
    residual = torch.randn(2, 3, 4, dim).bfloat16()
    pre, post = torch.rand(2, 3, 4), torch.rand(2, 3, 4)
    comb = torch.rand(2, 3, 4, 4)
    normalized, updated, next_pre, next_post, next_comb = mega_ffn_transition(
        block, output, residual, pre, post, comb
    )
    assert normalized.shape == output.shape and updated.shape == residual.shape
    assert (
        next_pre.shape == next_post.shape == pre.shape and next_comb.shape == comb.shape
    )
    assert captured["shifted_prev_mix"].data_ptr() == pre.data_ptr()
    assert captured["comb_res_mix"].data_ptr() == comb.data_ptr()
    assert captured["hc_norm_eps"] == captured["rmsnorm_eps"] == 1e-20
    assert captured["hc_pre_eps"] == captured["sinkhorn_eps"] == 1e-6
    expected = (
        output.float()[..., None, :] * post[..., None]
        + torch.einsum("...ij,...id->...jd", comb, residual.float())
    ).bfloat16()
    assert torch.equal(updated, expected)
    collapsed = (updated.float() * pre[..., None]).sum(-2).bfloat16().float()
    expected_norm = (
        collapsed
        * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20)
        * block.ffn_norm.weight.float()
    ).bfloat16()
    assert torch.equal(normalized, expected_norm)
