"""Explicit rank layout and process-group ownership for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class ParallelTopology:
    """Attention ranks are [ATTN_DP, CP, ATTN_TP]; experts use [EP, EXPERT_TP].

    Both meshes partition the same ``tp`` ranks. Attention TP and expert TP
    are derived independently from the total width.
    """

    tp: int = 1  # Total ranks shared by the attention and expert meshes.
    ep: int = 1  # Expert parallelism
    cp: int = 1  # Context parallelism - only for prefill, replicated on decode
    attn_dp: int = 1  # Attention data parallel

    def __post_init__(self):
        for value in (self.tp, self.ep, self.cp, self.attn_dp):
            if type(value) is not int or value < 1:
                raise ValueError("parallel sizes must be positive integers")
        if self.tp % self.ep:
            raise ValueError("tp must be divisible by ep")
        if self.tp % (self.attn_dp * self.cp):
            raise ValueError("tp must be divisible by attn_dp * cp")

    @property
    def world_size(self) -> int:
        return self.tp

    @property
    def expert_tp(self) -> int:
        return self.tp // self.ep

    @property
    def attn_tp(self) -> int:
        return self.tp // (self.attn_dp * self.cp)

    def coordinates(self, rank: int) -> tuple[int, int, int]:
        if type(rank) is not int or not 0 <= rank < self.tp:
            raise ValueError("rank is outside the topology")
        slot, attn_tp = divmod(rank, self.attn_tp)
        attn_dp, cp = divmod(slot, self.cp)
        return attn_dp, cp, attn_tp

    def rank(self, attn_dp: int, cp: int, attn_tp: int) -> int:
        if any(type(value) is not int for value in (attn_dp, cp, attn_tp)) or not (
            0 <= attn_dp < self.attn_dp
            and 0 <= cp < self.cp
            and 0 <= attn_tp < self.attn_tp
        ):
            raise ValueError("coordinates are outside the topology")
        return (attn_dp * self.cp + cp) * self.attn_tp + attn_tp

    def partitions(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        replicas = tuple(
            tuple(
                self.rank(dp, cp, tp)
                for cp in range(self.cp)
                for tp in range(self.attn_tp)
            )
            for dp in range(self.attn_dp)
        )
        return {
            "attn_tp": tuple(
                tuple(self.rank(dp, cp, tp) for tp in range(self.attn_tp))
                for dp in range(self.attn_dp)
                for cp in range(self.cp)
            ),
            "cp": tuple(
                tuple(self.rank(dp, cp, tp) for cp in range(self.cp))
                for dp in range(self.attn_dp)
                for tp in range(self.attn_tp)
            ),
            "ep": tuple(
                tuple(expert * self.expert_tp + tp for expert in range(self.ep))
                for tp in range(self.expert_tp)
            ),
            "expert_tp": tuple(
                tuple(range(start, start + self.expert_tp))
                for start in range(0, self.tp, self.expert_tp)
            ),
            "attn_dp": tuple(
                tuple(self.rank(dp, cp, tp) for dp in range(self.attn_dp))
                for cp in range(self.cp)
                for tp in range(self.attn_tp)
            ),
            "replica": replicas,
            # The complete [EP, EXPERT_TP] mesh connects every attention replica.
            "domain": (tuple(range(self.tp)),),
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
