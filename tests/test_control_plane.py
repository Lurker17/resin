from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin.backend.control_plane import (
    ControlOperation,
    ControlPlane,
    ControlPlaneError,
)
from resin.backend.distributed import DistributedExecutionError, DistributedWorker
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
        gathers = 0
        objects = 0

        def all_gather_single(self, *args, **kwargs):
            self.gathers += 1
            dist.all_gather_single(*args, **kwargs)

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
        control.gather(group, "http output" if rank else ControlOperation.PREPARE)
    assert control.gather(group, ControlOperation.COMMIT) == (
        [(), ()],
        [None, None],
    )
    assert saved == [(0, -1, 2**40), (3, -2, 2**40 + 1)]
    assert len(control._buffers) == 1

    # Exercise actual driver codecs and their opaque HTTP fallback on the same
    # group. Every successful step vote remains one tensor collective.
    worker = object.__new__(DistributedWorker)
    worker.groups = SimpleNamespace(drivers=group, distributed=transport)
    worker._control, worker.failure = control, None
    assert worker._drivers("work", (rank == 0, None)) == [
        (True, None),
        (False, None),
    ]
    assert worker._drivers(
        "phase", ("decode" if rank == 0 else "prefill", None, rank == 0, True)
    ) == [("decode", None, True, True), ("prefill", None, False, True)]
    assert worker._drivers("prepare", (rank + 1, rank + 5, None, True, False)) == [
        (1, 5, None, True, False),
        (2, 6, None, True, False),
    ]
    assert transport.objects == 1
    assert worker._drivers("commit", "failed after enqueue" if rank else None) == [
        None,
        "failed after enqueue",
    ]
    assert worker._drivers("http configuration", {"rank": rank}) == [
        {"rank": 0},
        {"rank": 1},
    ]
    with pytest.raises(DistributedExecutionError, match="different orders"):
        if rank == 0:
            worker._drivers("phase", ("decode", None, True, True))
        else:
            worker._drivers("http output", {"rank": rank})
    assert transport.objects == 3  # No opaque payload exchange after mismatch.
    (output / f"rank{rank}.json").write_text(
        json.dumps({"gathers": transport.gathers, "objects": transport.objects})
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
    assert records == [{"gathers": 15, "objects": 3}] * 2


def _inference_mode_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        group = RankGroup((0, 1), dist.group.WORLD, rank)
        for first_inference in (True, False):
            control = ControlPlane(dist)
            saved = None
            for generation, enabled in enumerate(
                (first_inference, not first_inference, first_inference)
            ):
                with torch.inference_mode(enabled):
                    records, errors = control.gather(
                        group,
                        f"inference boundary {first_inference}:{generation}",
                        (rank, generation, int(enabled)),
                    )
                assert records == [
                    (0, generation, int(enabled)),
                    (1, generation, int(enabled)),
                ]
                assert errors == [None, None]
                assert len(control._buffers) == 1
                buffers = next(iter(control._buffers.values()))
                assert all(not torch.is_inference(tensor) for tensor in buffers)
                if saved is None:
                    saved = buffers
                else:
                    assert all(
                        tensor is previous
                        for tensor, previous in zip(buffers, saved, strict=True)
                    )
                # Both send and receive remain writable after leaving the
                # creating scope, and the next real gather replaces these.
                for tensor in buffers:
                    tensor.fill_(-1)
    finally:
        dist.destroy_process_group()


def test_control_buffers_cross_inference_mode_boundaries(tmp_path):
    mp.spawn(
        _inference_mode_worker,
        args=(tmp_path / "inference-rendezvous",),
        nprocs=2,
        join=True,
    )
