"""Explicit rank layout and process-group ownership for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from resin import Parallelism


@dataclass(frozen=True)
class RankGroup:
    ranks: tuple[int, ...]
    handle: Any = field(repr=False)
    local_rank: int

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def source(self) -> int:
        return self.ranks[0]


@dataclass
class ParallelGroups:
    topology: Parallelism
    rank: int
    attn_tp: RankGroup
    ep: RankGroup
    expert_tp: RankGroup
    cp: RankGroup
    attn_dp: RankGroup
    attn_replication: RankGroup
    replica: RankGroup
    domain: RankGroup
    drivers: RankGroup | None
    distributed: Any = field(repr=False)
    _owned: list[Any] = field(repr=False)

    @property
    def is_driver(self) -> bool:
        return self.rank == self.replica.source

    @classmethod
    def create(
        cls,
        topology: Parallelism,
        *,
        device_backend: str = "nccl",
        distributed=None,
        timeout: timedelta = timedelta(minutes=10),
    ) -> ParallelGroups:
        """Collective initialization on an already initialized default world group."""
        if distributed is None:
            import torch.distributed as distributed
        if not distributed.is_initialized():
            raise ValueError("initialize the default process group first")
        world_size, rank = distributed.get_world_size(), distributed.get_rank()
        configs = [None] * world_size
        distributed.all_gather_object(configs, (topology, device_backend))
        if (
            any(config != configs[0] for config in configs)
            or topology.world_size != world_size
        ):
            raise ValueError(
                "all ranks must agree on topology and match the world size"
            )
        cache, local, owned = {}, {}, []
        world = tuple(range(world_size))
        cache[world, str(distributed.get_backend())] = distributed.group.WORLD
        for name, partitions in topology.partitions().items():
            backend = (
                "gloo" if name in ("replica", "domain", "drivers") else device_backend
            )
            for ranks in partitions:
                key = ranks, backend
                if key not in cache:
                    handle = distributed.new_group(
                        list(ranks), backend=backend, timeout=timeout
                    )
                    cache[key] = handle
                    if rank in ranks:
                        owned.append(handle)
                if rank in ranks:
                    local[name] = RankGroup(ranks, cache[key], ranks.index(rank))
        local.setdefault("drivers", None)
        return cls(
            topology,
            rank,
            **local,
            distributed=distributed,
            _owned=owned,
        )

    def close(self) -> None:
        """Destroy only groups created here; call after all workers have stopped."""
        while self._owned:
            self.distributed.destroy_process_group(self._owned.pop())


__all__ = ["ParallelGroups", "RankGroup"]
