"""Independent attention origins compose with expert ownership and expert TP."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from resin import Parallelism
from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.backend.topology import ParallelGroups

EXPERTS, HIDDEN, INTERMEDIATE = 8, 4, 10


def _weights():
    gate = torch.arange(EXPERTS * INTERMEDIATE * HIDDEN).reshape(
        EXPERTS, INTERMEDIATE, HIDDEN
    )
    up = gate.flip(-1)
    down = torch.arange(EXPERTS * HIDDEN * INTERMEDIATE).reshape(
        EXPERTS, HIDDEN, INTERMEDIATE
    )
    return (
        ((gate % 11) - 5).float() / 32,
        ((up % 7) - 3).float() / 16,
        ((down % 13) - 6).float() / 32,
        torch.arange(EXPERTS * INTERMEDIATE).reshape(EXPERTS, INTERMEDIATE).float()
        / 128,
        torch.arange(EXPERTS * INTERMEDIATE).reshape(EXPERTS, INTERMEDIATE).float()
        / 256,
        3 + torch.arange(EXPERTS * HIDDEN).reshape(EXPERTS, HIDDEN).float() / 8,
    )


def _expert(hidden, tensors, expert, shard=slice(None), *, output_bias=True):
    gate, up, down, gate_bias, up_bias, down_bias = tensors
    gated = F.silu(
        F.linear(hidden.float(), gate[expert, shard], gate_bias[expert, shard])
    )
    expanded = gated * F.linear(
        hidden.float(), up[expert, shard], up_bias[expert, shard]
    )
    return F.linear(
        expanded, down[expert, :, shard], down_bias[expert] if output_bias else None
    )


def _reference(hidden, indices, weights, tensors):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(EXPERTS):
        probability = ((indices == expert) * weights).sum(1)
        result += _expert(hidden, tensors, expert) * probability[:, None]
    return result.to(hidden.dtype)


def _inputs(rank, count, *, one_owner=False):
    hidden = torch.arange(count * HIDDEN).reshape(count, HIDDEN).float() / 16 + rank / 4
    row = torch.arange(count)
    ids = torch.stack(
        (row % EXPERTS, row % EXPERTS, (row * 3 + rank + 4) % EXPERTS), dim=1
    )
    if one_owner:
        ids %= 2
    weights = torch.tensor([0.125, 0.375, 0.5]).expand(count, 3).contiguous()
    return hidden, ids, weights


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    tensors = _weights()
    try:
        for topology in (
            Parallelism(world_size=4, ep=1, attn_dp=2, attn_tp=2),
            Parallelism(world_size=4, ep=2, attn_dp=2, cp=2),
            Parallelism(world_size=4, ep=4, attn_tp=2, cp=2),
        ):
            groups = ParallelGroups.create(topology, device_backend="gloo")
            try:
                local_experts = EXPERTS // topology.ep
                first = groups.ep.local_rank * local_experts
                shard = (
                    TensorParallelLayout(
                        rank=groups.expert_tp.local_rank, size=groups.expert_tp.size
                    )
                    .aligned_shard(INTERMEDIATE, alignment=1, name="intermediate")
                    .as_slice()
                )
                bias_owner = groups.expert_tp.local_rank == 0
                received = []
                bias_only = False
                special_bias = torch.zeros(EXPERTS, HIDDEN)
                special_bias[0] = 1000.75
                special_bias[EXPERTS // 2] = -998.75

                def local(hidden, indices, weights):
                    received.append(hidden.shape[0])
                    result = torch.zeros_like(hidden, dtype=torch.float32)
                    for expert in range(first, first + local_experts):
                        probability = ((indices == expert) * weights).sum(1)
                        if bias_only:
                            partial = (
                                special_bias[expert]
                                if bias_owner
                                else torch.zeros(HIDDEN)
                            )
                        else:
                            partial = _expert(
                                hidden, tensors, expert, shard, output_bias=bias_owner
                            )
                        result += partial * probability[:, None]
                    return result

                dispatch = ExpertParallelDispatch(
                    local, num_experts=EXPERTS, groups=groups
                )
                assert dispatch.expert_tp == topology.expert_tp
                for counts, one_owner in (
                    ((0, 3, 1, 4), False),
                    ((4, 0, 2, 1), True),
                    ((0, 0, 0, 0), False),
                ):
                    hidden, indices, weights = _inputs(
                        rank, counts[rank], one_owner=one_owner
                    )
                    for capacity, masked in ((None, False), (4, False), (4, True)):
                        valid = max(counts[rank] - 1, 0) if masked else counts[rank]
                        context = (
                            nullcontext()
                            if capacity is None
                            else dispatch.fixed_capacity(
                                capacity,
                                valid_rows=torch.tensor(valid, dtype=torch.int32),
                            )
                        )
                        with context:
                            result = dispatch(hidden, indices, weights)
                        expected_weights = weights.clone()
                        expected_weights[valid:].zero_()
                        torch.testing.assert_close(
                            result,
                            _reference(hidden, indices, expected_weights, tensors),
                            rtol=1e-5,
                            atol=1e-5,
                        )
                        if capacity is not None:
                            assert received[-1] == 4 * capacity

                # Output bias belongs to one intermediate shard. Preserve its
                # FP32 cancellation across different expert owners until the
                # final BF16 origin result, including entirely idle origins.
                bias_only = True
                count = int(rank == 0)
                hidden = torch.ones(count, HIDDEN, dtype=torch.bfloat16)
                indices = torch.tensor([[0, EXPERTS // 2]], dtype=torch.int64)[:count]
                weights = torch.ones(count, 2)
                for context in (nullcontext(), dispatch.fixed_capacity(2)):
                    with context:
                        result = dispatch(hidden, indices, weights)
                    assert result.dtype == torch.bfloat16
                    torch.testing.assert_close(
                        result, torch.full_like(hidden, 2), rtol=0, atol=0
                    )
            finally:
                groups.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_dense_experts_compose_ep_and_unequal_intermediate_tp_across_attention_meshes(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "expert-meshes")))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
