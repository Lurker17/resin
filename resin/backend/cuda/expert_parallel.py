"""GPU-driven owner-sparse MoE transport using FlashInfer's NVLink workspace."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch
from torch import Tensor

if TYPE_CHECKING:
    from resin.backend.topology import RankGroup

_MAX_DESTINATIONS_PER_PASS = 22


class SparseExpertTransport:
    """Fixed receive storage with device-resident, sparse dispatch/combine counts.

    FlashInfer writes each token only to its selected destination ranks. Static
    receive slots are masked before the local expert operation; their presence
    does not cause padded network transfers. Initialization belongs to warmup,
    and all subsequent routing, synchronization and combination is capturable.
    """

    def __init__(
        self,
        group: RankGroup,
        capacity: int,
        width: int,
        dtype: torch.dtype,
        top_k: int,
        device: torch.device,
    ):
        from flashinfer.comm.trtllm_moe_alltoall import (
            moe_a2a_get_workspace_size_per_rank,
            moe_a2a_initialize,
        )

        from resin.backend.cuda.shared_workspace import SharedWorkspace

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up expert dispatch before CUDA graph capture")
        if group.size > 64:
            raise ValueError(
                "FlashInfer sparse expert transport supports up to 64 ranks"
            )
        if group.ranks != tuple(range(group.size)):
            raise ValueError(
                "CUDA expert transport requires the full rank domain; pass groups=ParallelGroups"
            )
        self.group, self.capacity = group, capacity
        # Physical GPUs are transport destinations, independent of the model's
        # expert numbering and of its attention/expert tensor-parallel meshes.
        workspace_bytes = moe_a2a_get_workspace_size_per_rank(
            group.size,
            capacity,
            width * torch.empty((), dtype=dtype).element_size() + top_k * 8,
            width * 4,  # Preserve FP32 expert partials through combination.
        )
        self.allocation = SharedWorkspace(group, workspace_bytes, device=device)
        self.workspace = self.allocation.tensor
        self.metainfo = moe_a2a_initialize(
            self.workspace, group.local_rank, group.size, capacity
        )
        # No peer may publish routing flags into an allocation another rank is
        # still zeroing during its first warmup call.
        torch.distributed.barrier(group=group.handle)

    def __call__(
        self,
        local: Callable[[Tensor, Tensor, Tensor], Tensor],
        hidden: Tensor,
        indices: Tensor,
        weights: Tensor,
        owners: Tensor,
        active: Tensor,
        num_experts: int,
    ) -> Tensor:
        from flashinfer.comm.trtllm_moe_alltoall import (
            moe_a2a_combine,
            moe_a2a_dispatch,
            moe_a2a_sanitize_expert_ids,
        )

        capacity, world_size = self.capacity, self.group.size
        # The pinned primitive specializes K in {1,2,4,6,8,10,16,22}. Usual
        # top-k routing fits one pass. Large expert-TP fanout is partitioned by
        # physical rank, retaining owner-sparse writes in every pass.
        if owners.shape[1] <= _MAX_DESTINATIONS_PER_PASS:
            routes = [owners.sort(dim=1).values]
        else:
            destinations = torch.zeros(
                (capacity, world_size), device=hidden.device, dtype=torch.bool
            )
            destinations.scatter_(1, owners.long(), True)
            ranks = torch.arange(world_size, device=hidden.device, dtype=torch.int32)
            routes = [
                torch.where(
                    destinations[:, start : start + _MAX_DESTINATIONS_PER_PASS],
                    ranks[start : start + _MAX_DESTINATIONS_PER_PASS],
                    world_size,
                )
                for start in range(0, world_size, _MAX_DESTINATIONS_PER_PASS)
            ]

        result = None
        for route in routes:
            first = route.min(dim=1, keepdim=True).values
            selected = active & (first[:, 0] < world_size)
            route = torch.where(route < world_size, route, first)
            route = torch.where(selected[:, None], route, self.group.local_rank)
            k = next(k for k in (1, 2, 4, 6, 8, 10, 16, 22) if k >= route.shape[1])
            if k != route.shape[1]:
                route = torch.cat(
                    (route, route[:, :1].expand(-1, k - route.shape[1])), 1
                )
            masked_weights = weights.masked_fill(~selected[:, None], 0)
            payloads, combine_offset = moe_a2a_dispatch(
                route.to(torch.int32).contiguous(),
                [
                    indices.masked_fill(~selected[:, None], num_experts).to(
                        torch.int32
                    ),
                    masked_weights.float(),
                    hidden,
                ],
                self.workspace,
                self.metainfo,
                capacity,
                self.group.local_rank,
                world_size,
                k,
                world_size,
            )
            received_indices, received_weights, received_hidden = payloads
            moe_a2a_sanitize_expert_ids(
                received_indices,
                self.workspace,
                self.metainfo,
                self.group.local_rank,
                num_experts,
            )
            valid = received_indices[..., 0] < num_experts
            # The transport ABI stores routing weights in FP32. Local expert
            # kernels retain the caller's dtype contract (BF16 for DeepSeek).
            received_weights = received_weights.masked_fill(~valid[..., None], 0).to(
                weights.dtype
            )
            received_hidden = received_hidden.masked_fill(~valid[..., None], 0)
            partial = local(
                received_hidden.flatten(0, 1),
                received_indices.flatten(0, 1),
                received_weights.flatten(0, 1),
            )
            combined = moe_a2a_combine(
                partial.reshape(world_size, capacity, -1).contiguous(),
                capacity,
                self.workspace,
                self.metainfo,
                capacity,
                self.group.local_rank,
                world_size,
                k,
                combine_offset,
            )
            result = combined if result is None else result + combined
        assert result is not None
        return result


_TRANSPORTS: WeakValueDictionary[tuple, SparseExpertTransport] = WeakValueDictionary()


def sparse_expert_transport(
    group: RankGroup,
    capacity: int,
    width: int,
    dtype: torch.dtype,
    top_k: int,
    device: torch.device,
) -> SparseExpertTransport:
    """Share layer workspaces without retaining closed models or process groups."""
    key = group, capacity, width, dtype, top_k, device
    transport = _TRANSPORTS.get(key)
    if transport is None:
        with torch.cuda.device(device):
            transport = SparseExpertTransport(
                group, capacity, width, dtype, top_k, device
            )
        _TRANSPORTS[key] = transport
    return transport
