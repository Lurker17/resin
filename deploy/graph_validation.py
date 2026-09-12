"""Bounded correctness and timing probes for captured model prefill regions."""

from __future__ import annotations

import json
import statistics
import time


def _rank(runner) -> int:
    return getattr(getattr(runner, "tensor_parallel", None), "rank", 0)


def capture_cuda_graphs(runner, *, validate: bool = False):
    """Log both graph modes; optionally validate before admitting requests."""
    configured = {
        mode: len(stats["configured_shapes"])
        for mode, stats in runner.cuda_graph_stats.items()
    }
    if _rank(runner) == 0:
        print(f"[cuda-graphs] capturing {configured}", flush=True)
    started = time.perf_counter()
    stats = runner.capture_cuda_graphs()
    parallel = getattr(runner, "tensor_parallel", None)
    if parallel is not None:
        parallel.barrier()
    if _rank(runner) == 0:
        counts = {mode: values["capture_count"] for mode, values in stats.items()}
        print(
            f"[cuda-graphs] captured {counts} in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    if validate:
        validate_prefill_graphs(runner)
    return stats


def _difference(actual, expected):
    import torch

    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), expected.reshape(1, -1)
    ).item()
    if not actual.count_nonzero().item() and not expected.count_nonzero().item():
        cosine = 1.0
    return {
        "max_abs_error": delta.abs().max().item(),
        "relative_l2_error": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine": cosine,
    }


def _compare(runner, actual, expected, *, tolerance: float, label: str):
    import torch

    difference = _difference(actual, expected)
    error = None
    try:
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        assert difference["cosine"] >= 0.999, f"{label}: {difference}"
    except AssertionError as failure:
        error = str(failure)
    parallel = getattr(runner, "tensor_parallel", None)
    failed = torch.tensor(int(error is not None), device=runner.device)
    if parallel is not None and parallel.enabled:
        torch.distributed.all_reduce(
            failed, op=torch.distributed.ReduceOp.MAX, group=parallel.process_group
        )
    if failed.item():
        raise AssertionError(
            f"{label}: {error or 'comparison failed on another TP rank'}; "
            f"rank {_rank(runner)} metrics: "
            + ", ".join(f"{key}={value:.6g}" for key, value in difference.items())
        )
    return difference


def _paired_timings(
    eager_padded, eager_live, dispatch, *, rounds: int = 7, iterations: int = 20
):
    import torch

    for _ in range(3):
        eager_padded()
        eager_live()
        dispatch()
    timings = {"eager_padded_us": [], "eager_live_us": [], "configured_policy_us": []}
    for round_index in range(rounds):
        order = (
            ("eager_padded_us", eager_padded),
            ("eager_live_us", eager_live),
            ("configured_policy_us", dispatch),
        )
        for name, operation in order[:: -1 if round_index % 2 else 1]:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(iterations):
                operation()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000 / iterations)
    return {
        name: {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in timings.items()
    }


def validate_prefill_graphs(runner):
    """Compare first/last regions, including padding, without touching model state.

    Every TP rank runs the same seeded cases and alternating timing order.
    Timings include configured-policy staging and both padded and live eager rows.
    Reported inference replay counters also include these validation replays.
    """
    import torch

    operations, buckets = runner._prefill_operations, runner._prefill_buckets
    if not operations or not buckets:
        return []
    records = []
    generator = torch.Generator(device=runner.device).manual_seed(173)
    captured = set(runner.cuda_graph_stats["prefill"]["captured_shapes"])
    cases = sorted(
        {
            min(rows, buckets[-1])
            for rows in (3, 17, 128, 129, buckets[-1] // 2 + 1, buckets[-1])
        }
    )
    with torch.inference_mode(), runner._prefill_graphs():
        for region in sorted({0, len(operations) - 1}):
            operation = operations[region]
            for rows in cases:
                bucket = next(size for size in buckets if size >= rows)
                assert (bucket, region + 1) in captured, (
                    "prefill region is not captured"
                )
                hidden = torch.randn(
                    rows,
                    runner._prefill_hidden_size,
                    dtype=torch.bfloat16,
                    device=runner.device,
                    generator=generator,
                )
                padded = torch.zeros(
                    bucket,
                    runner._prefill_hidden_size,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                padded[:rows].copy_(hidden)
                ids = padded_ids = None
                if runner._prefill_token_ids:
                    ids = torch.randint(
                        runner.vocab_size,
                        (rows,),
                        device=runner.device,
                        generator=generator,
                    )
                    padded_ids = torch.zeros(
                        bucket, dtype=torch.int64, device=runner.device
                    )
                    padded_ids[:rows].copy_(ids)

                def eager():
                    with runner._prefill_capture_context():
                        return operation(padded, padded_ids)

                def dispatch():
                    with runner._prefill_capture_context():
                        return runner._run_prefill_region(region, hidden, ids)

                def eager_live():
                    with runner._prefill_capture_context():
                        return operation(hidden, ids)

                expected = eager().clone()[:rows]
                before = runner.cuda_graph_stats["prefill"]["replay_count"]
                actual = dispatch().clone()
                uses_graph = runner.cuda_graph_stats["prefill"]["replay_count"] > before
                assert rows != bucket or uses_graph, (
                    "exact prefill bucket did not replay"
                )
                unpadded = eager_live().clone()
                label = f"prefill region {region}, rows {rows}, bucket {bucket}"
                record = {
                    "region": region,
                    "rows": rows,
                    "bucket": bucket,
                    "uses_graph": uses_graph,
                }
                if uses_graph:
                    record["same_bucket"] = _compare(
                        runner,
                        actual,
                        expected,
                        tolerance=0.01,
                        label=f"{label}, same bucket",
                    )
                else:
                    record["padded_eager_difference"] = _difference(actual, expected)
                record["unpadded"] = _compare(
                    runner,
                    actual,
                    unpadded,
                    tolerance=0.03 if uses_graph else 0.01,
                    label=f"{label}, unpadded",
                )
                large_case = rows > 129
                timed = (region == 0) if large_case else rows in (3, 128, 129)
                if timed:
                    rounds, iterations = (3, 5) if large_case else (7, 20)
                    record.update(timing_rounds=rounds, replays_per_round=iterations)
                    record["timings"] = _paired_timings(
                        eager,
                        eager_live,
                        dispatch,
                        rounds=rounds,
                        iterations=iterations,
                    )
                records.append(record)
    if _rank(runner) == 0:
        print(
            "[prefill-graph-validation] "
            + json.dumps(
                {
                    "cases": records,
                    "prefill_replays_after_validation": runner.cuda_graph_stats[
                        "prefill"
                    ]["replay_count"],
                }
            ),
            flush=True,
        )
    return records
