"""Real NCCL expert dispatch checked against independent arithmetic."""

from __future__ import annotations

import faulthandler
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin import Parallelism
from resin.backend.topology import ParallelGroups

WORLD_SIZE = int(os.environ.get("RESIN_EP_TEST_WORLD_SIZE", "4"))

pytestmark = pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices and NCCL",
)


def _dispatch_rank(rank, rendezvous, directory, replicated):
    from resin.backend.expert_parallel import ExpertParallelDispatch

    faulthandler.dump_traceback_later(45, repeat=True)

    def progress(message):
        print(
            f"[NCCL reference rank={rank}, replicated={replicated}] {message}",
            flush=True,
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=2),
        device_id=device,
    )
    progress("world ready")
    groups = ParallelGroups.create(
        Parallelism(
            world_size=WORLD_SIZE,
            ep=WORLD_SIZE,
            attn_dp=WORLD_SIZE // 2 if replicated else WORLD_SIZE,
        )
    )
    progress("groups ready")
    try:
        local_experts = 8 // WORLD_SIZE
        start = local_experts * groups.ep.local_rank
        stop = start + local_experts
        calls = []

        def local(hidden, indices, weights):
            calls.append(int(hidden.shape[0]))
            partial = torch.zeros_like(hidden)
            for expert in range(start, stop):
                rows, slots = torch.where(indices == expert)
                partial.index_add_(
                    0,
                    rows,
                    hidden[rows] * (expert + 1) * weights[rows, slots, None],
                )
            return partial

        dispatch = ExpertParallelDispatch(
            local,
            num_experts=8,
            group=groups.ep,
            replication=groups.attn_replication if replicated else None,
        )
        owner = groups.topology.coordinates(rank)[0]
        sizes = (
            ((3,) if WORLD_SIZE == 2 else (0, 3))
            if replicated
            else (0, 5, 1, 3)[:WORLD_SIZE]
        )
        errors = []
        for phase, count in enumerate((sizes[owner], 0, sizes[owner])):
            progress(f"phase {phase}: {count} local rows")
            hidden = (
                torch.arange(count * 16, device=device, dtype=torch.float32).reshape(
                    count, 16
                )
                / 32
                + owner
            )
            rows = torch.arange(count, device=device, dtype=torch.int64)
            # Include repeated experts and multiple experts on one destination.
            indices = torch.stack((rows % 8, rows % 8, (rows * 3 + 7) % 8), dim=1)
            weights = (
                torch.tensor((0.25, 0.125, 0.625), device=device)
                .expand(count, 3)
                .contiguous()
            )
            expected = (
                hidden[:, None, :] * (indices + 1)[:, :, None] * weights[:, :, None]
            ).sum(dim=1)
            actual = dispatch(hidden, indices, weights)
            progress(f"phase {phase}: dispatch returned")
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            errors.append(0.0 if not count else float((actual - expected).abs().max()))
        torch.cuda.synchronize()
        progress("device synchronized")
        dist.barrier()
        progress("barrier complete")
        Path(directory, f"{rank}.json").write_text(
            json.dumps({"rank": rank, "max_abs_errors": errors, "received_rows": calls})
        )
    finally:
        groups.close()
        progress("groups closed")
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("replicated", [False, True])
def test_nccl_expert_dispatch_matches_reference_with_unequal_and_empty_rows(
    tmp_path, replicated
):
    context = mp.spawn(
        _dispatch_rank,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), replicated),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("NCCL expert dispatch exceeded its 150 second bound")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    records = [
        json.loads((tmp_path / f"{rank}.json").read_text())
        for rank in range(WORLD_SIZE)
    ]
    assert all(len(record["max_abs_errors"]) == 3 for record in records)
    assert any(any(record["received_rows"]) for record in records)
