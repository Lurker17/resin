"""Bounded owner lookups across a device group, independent of model format."""

from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import Tensor

from resin.backend.collectives import LayerCollective
from resin.backend.topology import RankGroup


class ShardedEmbeddingLookup(LayerCollective):
    """NCCL prototype: gather IDs, look up owners, scatter BF16 responses.

    Buffers scale with group size and the common token capacity. This is a
    replaceable GPU transport, rather than DeepEP's sparse RDMA table fetch.
    Results alias reusable buffers until the next lookup on this collective.
    """

    def __init__(self, group: RankGroup, dimension: int):
        super().__init__()
        self.group, self.dimension = group, dimension
        self._buffers = {}

    @property
    def allocated(self):
        return bool(self._buffers)

    def close(self):
        self._buffers.clear()

    def __call__(
        self, indices: Tensor, local: Callable[[Tensor, Tensor | None], Tensor]
    ):
        shape, columns = indices.shape, indices.shape[-1]
        flat = indices.reshape(-1, columns)
        rows, group = len(flat), self.group
        capacity = self._capacity
        if capacity is None:
            counts = torch.tensor([rows], device=indices.device, dtype=torch.int64)
            received = torch.empty(group.size, device=indices.device, dtype=torch.int64)
            dist.all_gather_into_tensor(received, counts, group=group.handle)
            capacity = max(1, int(received.max().item()))
        if rows > capacity:
            raise ValueError("embedding rows exceed the collective capacity")
        key = capacity, columns, indices.device
        if key not in self._buffers:
            self._buffers[key] = (
                indices.new_empty((capacity, columns)),
                indices.new_empty((group.size * capacity, columns)),
                torch.empty(
                    (group.size * capacity, columns, self.dimension),
                    device=indices.device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    (capacity, columns, self.dimension),
                    device=indices.device,
                    dtype=torch.bfloat16,
                ),
            )
        source, gathered, partial, output = self._buffers[key]
        source.fill_(-1)
        source[:rows].copy_(flat)
        if self._valid_rows is not None:
            if self._valid_rows.device != indices.device:
                raise ValueError("valid embedding rows must be on the input device")
            source.masked_fill_(
                (torch.arange(capacity, device=indices.device) >= self._valid_rows)[
                    :, None
                ],
                -1,
            )
        dist.all_gather_into_tensor(gathered, source, group=group.handle)
        local(gathered, partial)
        # Each ID has exactly one owner, so the BF16 sum introduces no rounding.
        dist.reduce_scatter_tensor(output, partial, group=group.handle)
        return output[:rows].reshape(*shape, self.dimension)
