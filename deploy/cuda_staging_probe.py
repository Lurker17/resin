"""Delayed real-GPU H2D reads verify graph input staging ownership."""

import torch

from resin.backend.cuda.graphs import CudaGraphCache, CudaGraphStep
from resin.backend.cuda.staging import TensorStager, TensorStagerPool


@torch.inference_mode()
def run_probe():
    device = torch.device("cuda")
    count, width = 8, 64
    target = torch.zeros(width, device=device, dtype=torch.int64)
    snapshots = [torch.empty_like(target) for _ in range(count)]
    inputs = {"tokens": target}
    expected = [[value] * width for value in range(1, count + 1)]

    def delayed_copies(stager):
        stager.copy_into(inputs, {"tokens": [0] * width})
        for snapshot in snapshots:
            snapshot.copy_(target)
        torch.cuda.synchronize()
        # Queue all CPU writes while their stream's first GPU read is delayed.
        torch.cuda._sleep(500_000_000)
        for value, snapshot in enumerate(snapshots, 1):
            stager.copy_into(inputs, {"tokens": [value] * width})
            snapshot.copy_(target)
        torch.cuda.synchronize()
        return [snapshot.cpu().tolist() for snapshot in snapshots]

    unsafe = delayed_copies(TensorStager(device))
    safe = delayed_copies(TensorStagerPool(device))
    if unsafe == expected:
        raise AssertionError("delayed persistent staging did not reproduce the race")
    if safe != expected:
        raise AssertionError(f"pooled staging corrupted queued inputs: {safe}")

    pool = TensorStagerPool(device)
    cpu_source = torch.zeros(width, dtype=torch.int64, pin_memory=True)
    pool.copy_into(inputs, {"tokens": cpu_source})
    torch.cuda.synchronize()
    torch.cuda._sleep(500_000_000)
    for value, snapshot in enumerate(snapshots, 1):
        cpu_source.fill_(value)
        pool.copy_into(inputs, {"tokens": cpu_source})
        snapshot.copy_(target)
    cpu_source.fill_(-1)
    torch.cuda.synchronize()
    cpu_tensor = [snapshot.cpu().tolist() for snapshot in snapshots]
    if cpu_tensor != expected:
        raise AssertionError(
            f"pooled staging borrowed mutable CPU inputs: {cpu_tensor}"
        )

    output = torch.empty_like(target)

    def launch():
        output.copy_(target)

    cache = CudaGraphCache(
        device,
        ((1, width),),
        build=lambda _shape: CudaGraphStep(
            inputs=inputs,
            output=output,
            launch=launch,
            output_to_host=False,
        ),
        synchronize_failure=lambda: torch.cuda.synchronize(device),
    )
    cache.capture((1, width), {"tokens": [0] * width})
    torch.cuda._sleep(500_000_000)
    for value, snapshot in enumerate(snapshots, 1):
        result = cache.run((1, width), {"tokens": [value] * width})
        # Own each result; the captured output buffer is intentionally shared.
        snapshot.copy_(result)
    torch.cuda.synchronize()
    captured = [snapshot.cpu().tolist() for snapshot in snapshots]
    if captured != expected:
        raise AssertionError(f"graph cache corrupted queued inputs: {captured}")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "queued_batches": count,
        "unsafe_persistent_values": [row[0] for row in unsafe],
        "pooled_values": [row[0] for row in safe],
        "pooled_cpu_tensor_values": [row[0] for row in cpu_tensor],
        "graph_cache_values": [row[0] for row in captured],
        "graph_replays": cache.stats["replay_count"],
        "synchronize_during_enqueue": False,
    }
    cache.close()
    return result
