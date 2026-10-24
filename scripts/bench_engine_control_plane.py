"""CPU-only long-context Engine replay using the published V4.1 page geometry.

No model weights, CUDA tensors or device execution are involved. Timings split
the Engine step at entry to a fake runner, exposing native plan preparation.
Build the extension with ``maturin develop --release`` before comparing runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import resin
from resin import BatchPhase, Engine, ForwardOutput, RunnerConfig
from resin_models_cuda.deepseek_v41.cache import CacheGeometry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "models/cuda/tests/fixtures/deepseek_v41/inference.json"


class DryRunner:
    def __init__(self, layout, accepted: int, lookahead: int):
        self.runner_config = RunnerConfig(
            prefill_chunk_size=1024,
            continuation_prefill=True,
            cache_write_lookahead_tokens=lookahead,
            cache_layout=layout,
        )
        self.accepted = accepted
        self.started = None
        self.before_forward_ns = 0

    def bind_resources(self, _resources):
        pass

    def forward(self, plan):
        if self.started is not None:
            self.before_forward_ns = time.perf_counter_ns() - self.started
        width = self.accepted if plan.phase == BatchPhase.DECODE else 1
        return ForwardOutput(
            np.full(plan.num_sequences * width, 3, dtype=np.uint32),
            np.arange(plan.num_sequences + 1, dtype=np.uint32) * width,
        )


def run_case(
    batch: int,
    steps: int,
    *,
    context: int = 131072,
    shared: int = 126976,
    accepted: int = 3,
    lookahead: int = 3,
    warmup: int = 3,
    parents: int = 12000,
    config: str | Path = DEFAULT_CONFIG,
):
    """Replay real scheduler/cache transactions, returning host-only medians."""
    if min(batch, steps, context, accepted, parents) < 1 or warmup < 0:
        raise ValueError("counts must be positive and warmup must be nonnegative")
    if not 0 <= shared < context or lookahead < accepted - 1:
        raise ValueError("shared prefix or speculative lookahead is invalid")
    args = SimpleNamespace(**json.loads(Path(config).read_text()))
    geometry = CacheGeometry.build(args, prefill_chunk_size=1024, max_batch_size=batch)
    layout = geometry.layout(parents)
    runner = DryRunner(layout, accepted, lookahead)
    maximum = 1 + (steps + warmup + 2) * accepted
    engine = Engine(
        runner,
        max_running_requests=batch,
        context_length=context + maximum + 1,
        prefix_cache=True,
    )
    cached = []
    try:
        if shared:
            seed = engine.submit([2] * shared, 1, stop_token_ids=[])
            while not seed.done:
                engine.step()
        for index in range(batch):
            prompt = [2] * context
            prompt[shared] = 100 + index
            lease = engine.submit(prompt, maximum, stop_token_ids=[])
            cached.append(lease.cached_tokens)
        if cached != [shared] * batch:
            raise RuntimeError(
                "the replay did not restore the requested physical prefix"
            )
        while engine.next_phase() == BatchPhase.PREFILL:
            engine.step()
        samples = []
        for step in range(warmup + steps):
            runner.started = time.perf_counter_ns()
            events = engine.step()
            elapsed = time.perf_counter_ns() - runner.started
            if len(events) != batch * accepted:
                raise RuntimeError(
                    "dry runner did not commit the expected decode tokens"
                )
            if step >= warmup:
                samples.append(
                    {
                        "whole_ms": elapsed / 1e6,
                        "prepare_ms": runner.before_forward_ns / 1e6,
                        "callback_and_complete_ms": (elapsed - runner.before_forward_ns)
                        / 1e6,
                    }
                )
        return {
            "batch_size": batch,
            "context_length": context,
            "shared_prefix_tokens": shared,
            "accepted_tokens_per_sequence": accepted,
            "cache_write_lookahead_tokens": lookahead,
            "cached_tokens_min": min(cached),
            "cached_tokens_max": max(cached),
            "history_blocks_per_sequence": (context + 127) // 128,
            "page_packings": geometry.packings,
            "warmup_steps": warmup,
            "measure_steps": steps,
            "median_ms": {
                key: statistics.median(sample[key] for sample in samples)
                for key in samples[0]
            },
            "samples": samples,
        }
    finally:
        runner.started = None
        engine.close()


def environment():
    extension = Path(resin._resin.__file__)
    checksum = hashlib.sha256(extension.read_bytes()).hexdigest()
    candidates = [
        ROOT / "target/release/lib_resin.dylib",
        ROOT / "target/release/lib_resin.so",
    ]
    release = next((path for path in candidates if path.exists()), None)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "source_commit": commit,
        "working_tree_dirty": bool(changes.strip()),
        "extension": str(extension),
        "extension_sha256": checksum,
        "matches_local_release_artifact": None
        if release is None
        else hashlib.sha256(release.read_bytes()).hexdigest() == checksum,
        "timing_scope": "CPU-only Engine step; preparation ends at fake runner entry",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 128, 256])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--shared-prefix", type=int, default=126976)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = {
        "environment": environment(),
        "cases": [
            run_case(batch, args.steps, context=args.context, shared=args.shared_prefix)
            for batch in args.batches
        ],
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
