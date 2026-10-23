from __future__ import annotations

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin.backend.control_plane import (
    ControlOperation,
    ControlPlane,
    ControlPlaneError,
)
from resin.backend.topology import RankGroup


def test_single_rank_control_vote_needs_no_initialized_transport():
    control = ControlPlane(None)
    group = RankGroup((7,), None, 0)
    assert control.gather(group, ControlOperation.WORK, (1, -3)) == (
        [(1, -3)],
        [None],
    )
    assert control.gather(group, ControlOperation.COMMIT, error="failed") == (
        [()],
        ["failed"],
    )
    assert not control._buffers
    with pytest.raises(ValueError, match="at most 6"):
        control.gather(group, ControlOperation.WORK, range(7))


def _control_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )

    class Transport:
        reductions = 0
        objects = 0

        def all_reduce(self, *args, **kwargs):
            self.reductions += 1
            dist.all_reduce(*args, **kwargs)

        def all_gather_object(self, *args, **kwargs):
            self.objects += 1
            dist.all_gather_object(*args, **kwargs)

    transport = Transport()
    control = ControlPlane(transport)
    group = RankGroup((0, 1), dist.group.WORLD, rank)
    saved = None
    for operation in ControlOperation:
        records, errors = control.gather(
            group, operation, (rank * 3, -rank - 1, 2**40 + rank)
        )
        assert records == [(0, -1, 2**40), (3, -2, 2**40 + 1)]
        assert errors == [None, None]
        if saved is None:
            saved = records
    assert transport.objects == 0
    records, errors = control.gather(
        group,
        ControlOperation.VALIDATE,
        (rank,),
        error="rank one failed before launch" if rank == 1 else None,
    )
    assert records == [(0,), (1,)]
    assert errors == [None, "rank one failed before launch"]
    assert transport.objects == 1

    # Different logical operations still use identical collective type/shape.
    # Both ranks detect the mismatch, rather than one entering model execution.
    with pytest.raises(ControlPlaneError, match="different orders"):
        control.gather(
            group, ControlOperation.PHASE if rank else ControlOperation.PREPARE
        )
    assert control.gather(group, ControlOperation.COMMIT) == (
        [(), ()],
        [None, None],
    )
    assert saved == [(0, -1, 2**40), (3, -2, 2**40 + 1)]
    assert len(control._buffers) == 1
    (output / f"rank{rank}.json").write_text(
        json.dumps({"reductions": transport.reductions, "objects": transport.objects})
    )
    dist.destroy_process_group()


def test_control_votes_preserve_rank_records_failure_diagnostics_and_order(tmp_path):
    mp.spawn(
        _control_worker,
        args=(tmp_path / "rendezvous", tmp_path),
        nprocs=2,
        join=True,
    )
    records = [json.loads(path.read_text()) for path in tmp_path.glob("rank*.json")]
    assert records == [{"reductions": 9, "objects": 1}] * 2
