"""World-sharded Engram lookup against an independent full-table Torch oracle."""

from ctypes.util import find_library
from pathlib import Path

import torch
import torch.cuda.nccl
import torch.distributed as dist

from resin.backend.embedding import ShardedEmbeddingLookup
from resin.backend.topology import RankGroup
from resin_models_cuda.deepseek_v41.engram import ParallelEngramEmbedding


def rdma_inventory():
    """Report visibility only; no communicator or host-memory registration test."""
    maps = Path("/proc/self/maps")
    return {
        "nccl_version": torch.cuda.nccl.version(),
        "infiniband_devices": sorted(
            str(path) for path in Path("/sys/class/infiniband").glob("*")
        ),
        "infiniband_device_nodes": sorted(
            str(path) for path in Path("/dev/infiniband").glob("*")
        ),
        "library_visibility": {
            name: find_library(name) for name in ("mlx5", "ibverbs", "nccl-gin")
        },
        "loaded_transport_libraries": sorted(
            {
                line.split()[-1]
                for line in (maps.read_text().splitlines() if maps.exists() else ())
                if any(name in line for name in ("libnccl", "libmlx5", "libibverbs"))
            }
        ),
        "cpu_rdma_qualified": False,
    }


def table_values(rows, dimension, device):
    values = (
        (
            torch.arange(rows * dimension, device=device).reshape(rows, dimension) % 19
            - 9
        )
        / 8
    ).to(torch.float8_e4m3fn)
    exponents = (
        torch.arange(rows, device=device)[:, None]
        + torch.arange(dimension // 32, device=device)
    ) % 7 - 3
    scales = (2.0 ** exponents.float()).to(torch.float8_e8m0fnu)
    reference = (values.float() * scales.float().repeat_interleave(32, -1)).bfloat16()
    return values, scales, reference


@torch.inference_mode()
def run_probe(group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    rows, dimension, columns, capacity = 257, 256, 24, 1280
    lookup = ShardedEmbeddingLookup(
        RankGroup(tuple(dist.get_process_group_ranks(group)), group, rank), dimension
    )
    with torch.device(device):
        table = ParallelEngramEmbedding(rows, dimension, lookup)
        second = ParallelEngramEmbedding(rows, dimension, lookup)
    values, scales, reference = table_values(rows, dimension, device)
    count = min(table.stop, rows) - table.start
    table.weight.zero_()
    table.scale.fill_(1)
    table.weight[:count].copy_(values[table.start : table.start + count])
    table.scale[:count].copy_(scales[table.start : table.start + count])
    second.weight.copy_(table.weight)
    second.scale.copy_((table.scale.float() * 2).to(torch.float8_e8m0fnu))
    indices = (
        torch.arange(capacity * columns, device=device).reshape(capacity, columns) * 13
        + rank * 91
    ) % rows
    original = indices.clone()
    valid = torch.tensor(capacity, dtype=torch.int32, device=device)
    output = torch.empty(
        capacity, columns, dimension, dtype=torch.bfloat16, device=device
    )
    second_output = torch.empty_like(output)

    def call():
        with lookup.fixed_capacity(capacity, valid_rows=valid):
            hooks = lookup.prefetch(
                [(indices, table.local), (indices.flip(-1), second.local)]
            )
            # Hooks retain independent storage even when consumed out of order.
            second_output.copy_(hooks[1]())
            output.copy_(hooks[0]())

    stream, graph = torch.cuda.Stream(device), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    results = []
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for name in ("full", "partial", "idle", "changed"):
            live = (
                (capacity - 3 if rank == 0 else capacity // 2)
                if name == "partial"
                else 0
                if name == "idle" and rank == world - 1
                else capacity
            )
            with torch.cuda.stream(stream):
                valid.fill_(live)
                indices.copy_(original if name == "full" else original.flip(-1))
                expected = reference[indices]
                expected[live:] = 0
                expected_second = (reference[indices.flip(-1)].float() * 2).bfloat16()
                expected_second[live:] = 0
                call()
                eager = output.clone()
                eager_second = second_output.clone()
                for _ in range(3):
                    graph.replay()
                stream.synchronize()
                if (
                    not torch.equal(output, expected)
                    or not torch.equal(output, eager)
                    or not torch.equal(second_output, expected_second)
                    or not torch.equal(second_output, eager_second)
                ):
                    raise AssertionError(
                        f"Engram {name} differs from full-table oracle or identical eager inputs"
                    )
            results.append(
                dict(
                    case=name,
                    valid_rows=live,
                    oracle_bitwise_equal=True,
                    graph_bitwise_equal=True,
                    replays=3,
                )
            )
        for live in (5 if rank == 0 else 0, 0):
            actual = table(indices[:live])
            if not torch.equal(actual, reference[indices[:live]]):
                raise AssertionError(
                    "unequal/empty eager Engram differs from full table"
                )
        return dict(
            rank=rank,
            world_size=world,
            rows=rows,
            local_rows=table.weight.shape[0],
            dimension=dimension,
            hash_columns=columns,
            capacity=capacity,
            prefetched_layers=2,
            cases=results,
            unequal_and_empty_eager=True,
            rdma_inventory=rdma_inventory(),
        )
    finally:
        graph.reset()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        lookup.close()
        lookup.close()
