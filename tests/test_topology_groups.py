"""Control subgroups avoid NCCL splitting on nonmember ranks."""

from __future__ import annotations

from types import SimpleNamespace

from resin import Parallelism
from resin.backend.topology import ParallelGroups


class _RecordingDistributed:
    def __init__(self, rank, topology, backend="nccl"):
        self.rank, self.topology = rank, topology
        self.backend = backend
        self.group = SimpleNamespace(WORLD=object())
        self.calls, self.destroyed = [], []

    def is_initialized(self):
        return True

    def get_world_size(self):
        return self.topology.world_size

    def get_rank(self):
        return self.rank

    def get_backend(self):
        return self.backend

    def all_gather_object(self, configs, config):
        configs[:] = [config] * self.topology.world_size

    def new_group(self, ranks, *, backend, timeout, use_local_synchronization=False):
        self.calls.append((tuple(ranks), backend, use_local_synchronization))
        if self.backend == "nccl" and backend == "gloo" and self.rank not in ranks:
            # With a bound NCCL default world, torch's global-sync path can
            # perform a no-color NCCL split only on Gloo subgroup nonmembers.
            assert use_local_synchronization
        return object() if self.rank in ranks else None

    def destroy_process_group(self, handle):
        self.destroyed.append(handle)


def test_gloo_subgroups_use_local_sync_without_changing_device_group_order():
    topology = Parallelism(world_size=2, ep=2)
    calls = []
    for rank in range(2):
        distributed = _RecordingDistributed(rank, topology)
        groups = ParallelGroups.create(topology, distributed=distributed)
        assert groups.ep.handle is groups.attn_replication.handle
        assert groups.ep.handle is distributed.group.WORLD
        assert (groups.drivers is not None) == (rank == 0)
        assert all(
            local_sync == (backend == "gloo")
            for _, backend, local_sync in distributed.calls
        )
        assert ((0,), "gloo", True) in distributed.calls
        calls.append(distributed.calls)
        owned = groups._owned.copy()
        groups.close()
        assert distributed.destroyed == owned[::-1]
    assert calls[0] == calls[1]


def test_gloo_default_world_keeps_global_synchronization():
    topology = Parallelism(world_size=2, ep=2)
    distributed = _RecordingDistributed(1, topology, backend="gloo")
    groups = ParallelGroups.create(
        topology, device_backend="gloo", distributed=distributed
    )
    assert all(not local_sync for _, _, local_sync in distributed.calls)
    groups.close()
