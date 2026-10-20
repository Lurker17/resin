"""Explicit rank layout and process-group ownership for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class ParallelTopology:
    """Attention ranks are [DP, CP, TP]; experts use [replica, EP, expert TP].

    EP is an alternative placement of the same ranks, not another multiplier.
    Expert TP is independent of attention TP. The communication groups connect
    precisely the attention-DP replicas that must step together.
    """

    tp: int = 1
    ep: int = 1
    cp: int = 1
    attention_dp: int = 1
    expert_tp: int = 1

    def __post_init__(self):
        for value in (self.tp, self.ep, self.cp, self.attention_dp, self.expert_tp):
            if type(value) is not int or value < 1:
                raise ValueError("parallel sizes must be positive integers")
        if self.world_size % (self.ep * self.expert_tp):
            raise ValueError("EP * expert_tp must divide the world size")

    @property
    def world_size(self) -> int:
        return self.attention_dp * self.cp * self.tp

    def coordinates(self, rank: int) -> tuple[int, int, int]:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the topology")
        slot, tp = divmod(rank, self.tp)
        dp, cp = divmod(slot, self.cp)
        return dp, cp, tp

    def rank(self, dp: int, cp: int, tp: int) -> int:
        if any(type(value) is not int for value in (dp, cp, tp)) or not (
            0 <= dp < self.attention_dp and 0 <= cp < self.cp and 0 <= tp < self.tp
        ):
            raise ValueError("coordinates are outside the topology")
        return (dp * self.cp + cp) * self.tp + tp

    def partitions(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        replicas = tuple(
            tuple(
                self.rank(dp, cp, tp) for cp in range(self.cp) for tp in range(self.tp)
            )
            for dp in range(self.attention_dp)
        )
        experts = tuple(
            tuple(start + expert * self.expert_tp + tp for expert in range(self.ep))
            for start in range(0, self.world_size, self.ep * self.expert_tp)
            for tp in range(self.expert_tp)
        )
        expert_tensor = tuple(
            tuple(range(start, start + self.expert_tp))
            for start in range(0, self.world_size, self.expert_tp)
        )
        # Connected components of replica ownership and expert communication.
        # DP replicas outside the same component have independent serving loops.
        owners = list(range(self.attention_dp))
        for group in (*experts, *expert_tensor):
            labels = {owners[self.coordinates(rank)[0]] for rank in group}
            target = min(labels)
            owners = [target if owner in labels else owner for owner in owners]
        domains = tuple(
            tuple(
                rank
                for dp, replica in enumerate(replicas)
                if owners[dp] == owner
                for rank in replica
            )
            for owner in sorted(set(owners))
        )
        return {
            "tp": tuple(
                tuple(self.rank(dp, cp, tp) for tp in range(self.tp))
                for dp in range(self.attention_dp)
                for cp in range(self.cp)
            ),
            "cp": tuple(
                tuple(self.rank(dp, cp, tp) for cp in range(self.cp))
                for dp in range(self.attention_dp)
                for tp in range(self.tp)
            ),
            "ep": experts,
            "expert_tp": expert_tensor,
            "attention_dp": tuple(
                tuple(self.rank(dp, cp, tp) for dp in range(self.attention_dp))
                for cp in range(self.cp)
                for tp in range(self.tp)
            ),
            "replica": replicas,
            "domain": domains,
            "drivers": tuple(
                tuple(rank for rank in domain if rank % (self.cp * self.tp) == 0)
                for domain in domains
            ),
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
    tp: RankGroup
    ep: RankGroup
    expert_tp: RankGroup
    cp: RankGroup
    attention_dp: RankGroup
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
