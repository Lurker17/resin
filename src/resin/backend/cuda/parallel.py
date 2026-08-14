"""Small CUDA process-group primitives shared by model parallel backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


@dataclass(frozen=True, slots=True)
class TensorShard:
    """One rank's contiguous interval in a globally ordered tensor axis."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class TensorParallelLayout:
    """A rank coordinate plus deterministic contiguous sharding rules."""

    rank: int = 0
    size: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise ValueError("tensor parallel size must be a positive integer")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < self.size
        ):
            raise ValueError("tensor parallel rank must be in [0, size)")

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def even_shard(self, total: int, *, name: str) -> TensorShard:
        """Split an axis evenly, rejecting geometries that cannot be sharded."""

        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if total % self.size:
            raise ValueError(f"{name}={total} must be divisible by TP={self.size}")
        width = total // self.size
        start = self.rank * width
        return TensorShard(start, start + width)

    def aligned_shard(self, total: int, *, alignment: int, name: str) -> TensorShard:
        """Balance aligned units across ranks, allowing unequal final widths."""

        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
        ):
            raise ValueError("shard alignment must be a positive integer")
        if total % alignment:
            raise ValueError(f"{name}={total} must be divisible by {alignment}")
        units = total // alignment
        if units < self.size:
            raise ValueError(
                f"{name} has only {units} aligned units for TP={self.size}"
            )
        base, extra = divmod(units, self.size)
        start_units = self.rank * base + min(self.rank, extra)
        local_units = base + (self.rank < extra)
        start = start_units * alignment
        return TensorShard(start, start + local_units * alignment)


@dataclass(frozen=True, slots=True)
class TensorParallelContext:
    """One tensor-parallel rank and its device collective group."""

    layout: TensorParallelLayout = TensorParallelLayout()
    process_group: ProcessGroup | None = None

    def __post_init__(self) -> None:
        if self.layout.enabled and self.process_group is None:
            raise ValueError("multi-rank tensor parallelism requires a process group")
        if not self.layout.enabled and self.process_group is not None:
            raise ValueError(
                "single-rank tensor parallelism cannot use a process group"
            )

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def size(self) -> int:
        return self.layout.size

    @property
    def enabled(self) -> bool:
        return self.layout.enabled

    def sum(self, tensor: Tensor) -> Tensor:
        """Sum a row-parallel partial result in place on every rank."""

        if self.process_group is not None:
            dist.all_reduce(tensor, group=self.process_group)
        return tensor

    def barrier(self) -> None:
        """Wait until every rank reaches the same host-side phase boundary."""

        if self.process_group is not None:
            dist.barrier(group=self.process_group)


__all__ = [
    "TensorParallelContext",
    "TensorParallelLayout",
    "TensorShard",
]
