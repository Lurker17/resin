"""Variable-size expert dispatch over explicit process groups.

Models supply their prepared local expert operation. This transport preserves
local token rows across independent attention replicas and replicated attention.
It supports CPU and CUDA tensors; kernel and quantization choices stay local.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from resin.backend.topology import RankGroup


class ExpertParallelDispatch:
    """Send each token once to each selected expert owner and return its sum.

    Sources may have different batch sizes, including zero. Every rank joins
    the same collectives; top-k indices stay global so the local expert kernel
    can filter its own contiguous expert interval. Counts are read on the host,
    making this a deliberately eager baseline rather than a graphable kernel.
    """

    def __init__(
        self,
        local: Callable[[Tensor, Tensor, Tensor], Tensor],
        *,
        num_experts: int,
        group: RankGroup,
        replication: RankGroup | None = None,
    ) -> None:
        if num_experts <= 0 or num_experts % group.size:
            raise ValueError("the number of experts must be divisible by EP")
        self.local = local
        self.group = group
        self.replication = replication
        self.experts_per_rank = num_experts // group.size

    def __call__(self, hidden: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        group, replication = self.group, self.replication
        # Attention copies execute the same request rows. One copy dispatches;
        # the resulting rows are shared again after the expert exchange.
        source = replication is None or replication.local_rank == 0
        send_hidden = hidden if source else hidden[:0]
        send_indices = indices if source else indices[:0]
        send_weights = weights if source else weights[:0]
        owners = send_indices.div(self.experts_per_rank, rounding_mode="floor")
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
        returned = hidden.new_empty((sum(send_splits), hidden.shape[1]))
        dist.all_to_all_single(
            returned,
            partial.contiguous(),
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=group.handle,
        )
        result = torch.zeros_like(hidden)
        result.index_add_(0, rows, returned)
        if replication is not None and replication.size > 1 and result.numel():
            dist.broadcast(result, src=replication.source, group=replication.handle)
        return result
