"""Run the CP qualification's metadata and ragged-exchange paths on Gloo."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from deploy.v41_yoco_cp_probe import run_rank
from resin import Parallelism
from resin.backend.topology import ParallelGroups


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    groups = None
    try:
        groups = ParallelGroups.create(
            Parallelism(world_size=2, cp=2, ep=2), device_backend="gloo"
        )
        result = run_rank(groups, device="cpu", capture_graphs=False)
        assert result["ok"] and not result["captured_graphs"]
        assert result["graph_counts"] is None
        assert len(result["cases"]) == 7
        ownership = {1: (0, 1), 11: (5, 6), 140: (70, 70)}
        for case in result["cases"]:
            assert case["local_queries"] == ownership[case["retained_rows"]][rank]
            assert case["metadata_fields"] == 16
            assert case["region_shapes"] is None
        with pytest.raises(ValueError, match="requires CUDA"):
            run_rank(groups, device="cpu")
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_cp_compacted_metadata_preserves_page_indices_and_gather_order(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "yoco-cp")))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
