"""Wire precision must not change the prepared local-expert callback ABI."""

import sys
from types import ModuleType

import pytest
import torch

from resin.backend.cuda.expert_parallel import SparseExpertTransport
from resin.backend.topology import RankGroup


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("multipass", [False, True])
def test_sparse_transport_restores_router_dtype_masks_padding_and_combines_fp32(
    monkeypatch, weight_dtype, multipass
):
    import resin.backend.cuda.expert_parallel as implementation

    if multipass:
        monkeypatch.setattr(implementation, "_MAX_DESTINATIONS_PER_PASS", 1)
    api = ModuleType("flashinfer.comm.trtllm_moe_alltoall")
    calls = []
    capacity, world, experts = 2, 2, 4

    def dispatch(routes, payloads, *args):
        indices, weights, hidden = payloads
        assert indices.dtype == torch.int32
        assert weights.dtype == torch.float32  # Required on-wire representation.
        receives = [
            value.unsqueeze(0).expand(world, -1, -1).clone() for value in payloads
        ]
        # This rank receives copies only for routes targeting physical rank 0.
        addressed = (routes == 0).any(-1)
        receives[0].masked_fill_(~addressed[None, :, None], experts)
        # Undefined receive padding may contain arbitrary wire bytes. It must
        # be zeroed before invoking a strict local kernel, in its own dtype.
        receives[1][:, 1].fill_(7)
        return receives, 0

    def combine(partial, *args):
        assert partial.dtype == torch.float32
        return partial.sum(0)

    monkeypatch.setattr(api, "moe_a2a_dispatch", dispatch, raising=False)
    monkeypatch.setattr(api, "moe_a2a_combine", combine, raising=False)
    monkeypatch.setattr(
        api, "moe_a2a_sanitize_expert_ids", lambda *args: None, raising=False
    )
    monkeypatch.setitem(sys.modules, api.__name__, api)
    transport = SparseExpertTransport.__new__(SparseExpertTransport)
    transport.capacity, transport.group = capacity, RankGroup((0, 1), None, 0)
    transport.workspace = transport.metainfo = torch.empty(0)
    hidden = torch.arange(1, 9).reshape(capacity, 4).bfloat16()
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.2001, 0.7999], [0.25, 0.75]], dtype=weight_dtype)
    owners = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)

    def local(values, expert_ids, router_weights):
        assert router_weights.dtype == weight_dtype
        assert values.dtype == hidden.dtype
        inactive = expert_ids[:, 0] == experts
        assert (router_weights[inactive] == 0).all()
        assert (values[inactive] == 0).all()
        for row in router_weights[~inactive]:
            torch.testing.assert_close(row, weights[0], rtol=0, atol=0)
        calls.append(router_weights.dtype)
        owned = expert_ids < experts // world
        return values.float() * (router_weights.float() * owned).sum(1, keepdim=True)

    actual = transport(
        local, hidden, indices, weights, owners, torch.tensor([True, False]), experts
    )
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    expected[0] = hidden[0].float() * weights[0, 0].float() * world
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert calls == [weight_dtype] * (world if multipass else 1)
