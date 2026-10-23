"""Unmeasured steady-state decode traces for Perfetto's Chrome trace importer."""

import gzip
import json
import shutil
from dataclasses import asdict
from pathlib import Path


def decode_trace(directory: str | Path, rank: int, metadata: dict | None = None):
    """Profile one extra full Engine step after a case's timing window."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def trace(case, step):
        name = f"decode-ctx{case.context_length}-b{case.batch_size}-rank{rank:03}"
        target = directory / f"{name}.json"
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as profiler:
            profiler.add_metadata("resin_rank", str(rank))
            profiler.add_metadata("resin_timing", "extra step after measurement")
            profiler.add_metadata_json(
                "resin_benchmark",
                json.dumps({"case": asdict(case), **(metadata or {})}),
            )
            with torch.profiler.record_function(f"resin.{name}"):
                step()
        profiler.export_chrome_trace(str(target))
        with (
            target.open("rb") as source,
            gzip.open(target.with_suffix(".json.gz"), "wb") as destination,
        ):
            shutil.copyfileobj(source, destination)
        data = json.loads(target.read_text())
        kernels = sum(event.get("cat") == "kernel" for event in data["traceEvents"])
        target.unlink()
        print(
            f"[bench-trace] rank={rank} case={name} CUDA kernel events={kernels}",
            flush=True,
        )
        if not kernels:
            raise RuntimeError("decode trace contains no CUDA kernel activity")

    return trace
