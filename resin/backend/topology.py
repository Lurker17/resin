"""Explicit rank layout and process-group ownership for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ParallelTopology:
    """Attention ranks are [ATTN_DP, CP, REPLICATION, ATTN_TP].

    Experts use [EP, EXPERT_TP] over the same `world_size` ranks:

    - Expert tensor parallelism is derived as world_size // ep.
    - Attention tensor parallelism is explicit; remaining ranks replicate it.
      The runtime uses CP to shard prefill and replicates decode across CP ranks.
    """

    world_size: int = 1  # One rank per GPU, shared by both meshes

    attn_dp: int = 1  # Attention data parallel
    attn_tp: int = 1  # Attention tensor parallel (partition heads)
    cp: int = 1  # Context parallelism - only for prefill, replicated on decode

    ep: int = 1  # Expert parallelism

    def __post_init__(self):
        for value in (self.world_size, self.attn_dp, self.attn_tp, self.cp, self.ep):
            if type(value) is not int or value < 1:
                raise ValueError("parallel sizes must be positive integers")
        if self.world_size % self.ep:
            raise ValueError("world_size must be divisible by ep")
        if self.world_size % (self.attn_dp * self.cp * self.attn_tp):
            raise ValueError("world_size must be divisible by attn_dp * cp * attn_tp")

    @property
    def expert_tp(self) -> int:
        """Expert tensor parallelism.

        Automatically computed from the world size and expert parallelism, and
        experts are never replicated.
        """
        return self.world_size // self.ep

    @property
    def attn_replication(self) -> int:
        """Copies of each attention shard within the same request replica."""
        return self.world_size // (self.attn_dp * self.cp * self.attn_tp)

    def coordinates(self, rank: int) -> tuple[int, int, int, int]:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the topology")
        slot, attn_tp = divmod(rank, self.attn_tp)
        slot, replication = divmod(slot, self.attn_replication)
        attn_dp, cp = divmod(slot, self.cp)
        return attn_dp, cp, replication, attn_tp

    def rank(self, attn_dp: int, cp: int, replication: int, attn_tp: int) -> int:
        if any(
            type(value) is not int for value in (attn_dp, cp, replication, attn_tp)
        ) or not (
            0 <= attn_dp < self.attn_dp
            and 0 <= cp < self.cp
            and 0 <= replication < self.attn_replication
            and 0 <= attn_tp < self.attn_tp
        ):
            raise ValueError("coordinates are outside the topology")
        return (
            (attn_dp * self.cp + cp) * self.attn_replication + replication
        ) * self.attn_tp + attn_tp

    def partitions(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        replicas = tuple(
            tuple(
                self.rank(dp, cp, replication, tp)
                for cp in range(self.cp)
                for replication in range(self.attn_replication)
                for tp in range(self.attn_tp)
            )
            for dp in range(self.attn_dp)
        )
        return {
            "attn_tp": tuple(
                tuple(self.rank(dp, cp, replication, tp) for tp in range(self.attn_tp))
                for dp in range(self.attn_dp)
                for cp in range(self.cp)
                for replication in range(self.attn_replication)
            ),
            "cp": tuple(
                tuple(self.rank(dp, cp, replication, tp) for cp in range(self.cp))
                for dp in range(self.attn_dp)
                for replication in range(self.attn_replication)
                for tp in range(self.attn_tp)
            ),
            "attn_replication": tuple(
                tuple(
                    self.rank(dp, cp, replication, tp)
                    for replication in range(self.attn_replication)
                )
                for dp in range(self.attn_dp)
                for cp in range(self.cp)
                for tp in range(self.attn_tp)
            ),
            "ep": tuple(
                tuple(expert * self.expert_tp + tp for expert in range(self.ep))
                for tp in range(self.expert_tp)
            ),
            "expert_tp": tuple(
                tuple(range(start, start + self.expert_tp))
                for start in range(0, self.world_size, self.expert_tp)
            ),
            "attn_dp": tuple(
                tuple(self.rank(dp, cp, replication, tp) for dp in range(self.attn_dp))
                for cp in range(self.cp)
                for replication in range(self.attn_replication)
                for tp in range(self.attn_tp)
            ),
            "replica": replicas,
            # The complete [EP, EXPERT_TP] mesh connects every attention replica.
            "domain": (tuple(range(self.world_size)),),
            "drivers": (tuple(replica[0] for replica in replicas),),
        }


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
    topology: ParallelTopology
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
        topology: ParallelTopology,
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


__all__ = ["ParallelGroups", "ParallelTopology", "RankGroup"]
