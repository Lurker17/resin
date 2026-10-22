"""Expert dispatch over explicit process groups.

Models supply their prepared local expert operation. This transport preserves
local token rows across independent attention replicas and replicated attention.
It supports CPU and CUDA tensors; kernel and quantization choices stay local.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from resin.backend.topology import ParallelGroups, RankGroup


class ExpertParallelDispatch:
    """Dispatch rows to expert owners and return their complete weighted sum.

    Sources may have different batch sizes, including zero. Every rank joins
    the same collectives; top-k indices stay global so the local expert kernel
    can filter its own contiguous expert interval. Eager variable-size exchange
    reads counts on the host. CUDA ``fixed_capacity`` uses GPU-driven sparse
    dispatch instead, so graph replay never reads routing metadata on the host.

    The local callback must return one consistent dtype on all ranks, including
    empty inputs. That dtype is retained through exchange and replica broadcast;
    the combined rows are cast to the input dtype only at the final boundary.
    """

    def __init__(
        self,
        local: Callable[[Tensor, Tensor, Tensor], Tensor],
        *,
        num_experts: int,
        group: RankGroup | None = None,
        replication: RankGroup | tuple[RankGroup, ...] | None = None,
        groups: ParallelGroups | None = None,
    ) -> None:
        if groups is not None:
            if group is not None:
                raise ValueError("pass groups or an expert group, not both")
            group = groups.device_domain
            expert_size = groups.ep.size
            self.expert_tp = groups.expert_tp.size
        else:
            if group is None:
                raise ValueError("expert dispatch requires process groups")
            expert_size = group.size
            self.expert_tp = 1
        if num_experts <= 0 or num_experts % expert_size:
            raise ValueError("the number of experts must be divisible by EP")
        self.local = local
        self.num_experts = num_experts
        self.group = group
        self.replication = (
            ()
            if replication is None
            else replication
            if isinstance(replication, tuple)
            else (replication,)
        )
        self.experts_per_rank = num_experts // expert_size
        self._capacity: int | None = None
        self._valid_rows: Tensor | None = None
        self._transports = {}

    @contextmanager
    def fixed_capacity(self, rows: int, *, valid_rows: Tensor | None = None):
        """Use a common token capacity on every EP rank during this operation.

        CUDA sends rows only to their selected expert owners through a shared
        NVLink workspace. Receive storage has a fixed shape, with unused slots
        masked for the local expert operation. Inactive graph rows and redundant
        attention copies stay local with zero weights. CPU uses equal-size
        collectives as a portable correctness reference. Callers select a common
        capacity before warmup/capture; replay needs no Python dispatch.

        ``valid_rows`` may be a device integer scalar updated before replay.
        It masks routing weights for padded model rows without a host read.
        """

        if type(rows) is not int or rows <= 0:
            raise ValueError("expert dispatch capacity must be a positive integer")
        if valid_rows is not None and (
            valid_rows.numel() != 1
            or valid_rows.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("valid expert rows must be one integer tensor value")
        previous, self._capacity = self._capacity, rows
        previous_valid, self._valid_rows = self._valid_rows, valid_rows
        try:
            yield
        finally:
            self._capacity = previous
            self._valid_rows = previous_valid

    def __call__(self, hidden: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        if self._capacity is not None:
            return self._fixed(
                hidden, indices, weights, self._capacity, self._valid_rows
            )
        group, replication = self.group, self.replication
        # Attention copies execute the same request rows. One copy dispatches;
        # the resulting rows are shared again after the expert exchange.
        source = all(copy.local_rank == 0 for copy in replication)
        send_hidden = hidden if source else hidden[:0]
        send_indices = indices if source else indices[:0]
        send_weights = weights if source else weights[:0]
        owners = send_indices.div(self.experts_per_rank, rounding_mode="floor")
        # Every intermediate shard of an owned expert sees the same token
        # assignments. Attention and expert TP coordinates need not coincide.
        owners = (
            owners[..., None] * self.expert_tp
            + torch.arange(self.expert_tp, device=indices.device)
        ).flatten(1)
        destinations = torch.zeros(
            (send_hidden.shape[0], group.size), device=hidden.device, dtype=torch.bool
        )
        destinations.scatter_(1, owners.long(), True)
        # Transpose orders packed rows by destination, retaining source order.
        _, rows = destinations.t().nonzero(as_tuple=True)
        send_counts = destinations.sum(0, dtype=torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group.handle)
        send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()
        received_rows = sum(recv_splits)

        def exchange(tensor: Tensor) -> Tensor:
            received = tensor.new_empty((received_rows, *tensor.shape[1:]))
            dist.all_to_all_single(
                received,
                tensor[rows].contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=group.handle,
            )
            return received

        received_hidden = exchange(send_hidden)
        received_indices = exchange(send_indices)
        received_weights = exchange(send_weights)
        partial = self.local(received_hidden, received_indices, received_weights)
        returned = partial.new_empty((sum(send_splits), hidden.shape[1]))
        dist.all_to_all_single(
            returned,
            partial.contiguous(),
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=group.handle,
        )
        result = partial.new_zeros(hidden.shape)
        if source and hidden.shape[0]:
            # Packed rows arrive in owner-rank order. Preserve that order
            # within each origin instead of using unordered CUDA atomic sums.
            order = rows.argsort(stable=True)
            result = torch.segment_reduce(  # ty: ignore[unresolved-attribute]  # Missing in Torch stubs.
                returned[order],
                "sum",
                lengths=destinations.sum(1, dtype=torch.int64),
                unsafe=True,  # Lengths partition the packed rows by construction.
            )
        self._broadcast(result)
        return result.to(hidden.dtype)

    def _broadcast(self, result: Tensor) -> None:
        if result.numel():
            for copy in self.replication:
                if copy.size > 1:
                    dist.broadcast(result, src=copy.source, group=copy.handle)

    def _fixed(
        self,
        hidden: Tensor,
        indices: Tensor,
        weights: Tensor,
        capacity: int,
        valid_rows: Tensor | None,
    ) -> Tensor:
        group, replication = self.group, self.replication
        rows, width = hidden.shape
        if rows > capacity:
            raise ValueError(
                f"expert dispatch has {rows} rows but capacity is {capacity}"
            )
        if valid_rows is not None and valid_rows.device != hidden.device:
            raise ValueError("valid expert rows must be on the input device")
        source = all(copy.local_rank == 0 for copy in replication)

        if hidden.is_cuda:
            from resin.backend.cuda.expert_parallel import sparse_expert_transport

            def pad(tensor: Tensor) -> Tensor:
                padded = tensor.new_zeros((capacity, *tensor.shape[1:]))
                if source:
                    padded[:rows].copy_(tensor)
                return padded

            padded_hidden, padded_indices, padded_weights = (
                pad(tensor) for tensor in (hidden, indices, weights)
            )
            active = torch.arange(capacity, device=hidden.device) < (
                valid_rows if valid_rows is not None else rows
            )
            active &= source
            owners = padded_indices.div(self.experts_per_rank, rounding_mode="floor")
            owners = (
                owners[..., None] * self.expert_tp
                + torch.arange(self.expert_tp, device=hidden.device)
            ).flatten(1)
            key = (
                capacity,
                hidden.shape[1],
                hidden.dtype,
                indices.shape[1],
                hidden.device,
            )
            # FlashInfer gets its stream from the current CUDA device. Torch
            # tensor operations above do not change the caller's device context.
            with torch.cuda.device(hidden.device):
                if key not in self._transports:
                    self._transports[key] = sparse_expert_transport(group, *key)
                result = self._transports[key](
                    self.local,
                    padded_hidden,
                    padded_indices,
                    padded_weights,
                    owners,
                    active,
                    self.num_experts,
                )
            self._broadcast(result)
            return result[:rows].to(hidden.dtype)

        def exchange(tensor: Tensor, *, mask: bool = False) -> Tensor:
            # Routing stays on the GPU. Zero padding also makes every idle
            # source join exactly the same collectives as active sources.
            padded = tensor.new_zeros((capacity, *tensor.shape[1:]))
            if source:
                padded[:rows].copy_(tensor)
            if mask and valid_rows is not None:
                inactive = torch.arange(capacity, device=tensor.device) >= valid_rows
                padded.masked_fill_(inactive[:, None], 0)
            received = tensor.new_empty((group.size * capacity, *tensor.shape[1:]))
            dist.all_gather_into_tensor(received, padded, group=group.handle)
            return received

        received_hidden = exchange(hidden)
        received_indices = exchange(indices)
        received_weights = exchange(weights, mask=True)
        partial = self.local(received_hidden, received_indices, received_weights)
        # Keep the local kernel's partial precision until all expert owners
        # have been combined. BF16 owner sums can lose small residuals.
        result = partial.new_empty((capacity, width))
        dist.reduce_scatter_tensor(result, partial.contiguous(), group=group.handle)
        self._broadcast(result)
        return result[:rows].to(hidden.dtype)


@contextmanager
def fixed_expert_capacity(
    model: torch.nn.Module, capacity: int, *, valid_rows: Tensor | None = None
):
    """Select fixed-capacity expert transport throughout a captured model."""

    seen = set()
    with ExitStack() as contexts:
        for module in model.modules():
            for value in vars(module).values():
                if isinstance(value, ExpertParallelDispatch) and id(value) not in seen:
                    seen.add(id(value))
                    contexts.enter_context(
                        value.fixed_capacity(capacity, valid_rows=valid_rows)
                    )
        yield


def close_expert_transports(model: torch.nn.Module) -> None:
    """Release shared workspaces after all ranks have destroyed their graphs."""
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, ExpertParallelDispatch):
                value._transports.clear()
