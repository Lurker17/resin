"""Bounded correctness and timing probes for captured model prefill regions."""

from __future__ import annotations

import hashlib
import json
import statistics
import time


def prefill_plan_metadata(plan):
    """Keep bounded host evidence that replay receives new logical cache maps."""

    if plan is None:
        return {"contexts": [], "query_lengths": [], "cache_map": None}
    addresses = tuple(
        (
            int(group.group_index),
            tuple(map(int, group.block_slab_ids)),
            tuple(map(int, group.block_slotnums)),
            tuple(map(int, group.block_table_start_indices)),
        )
        for group in plan.cache_groups
    )
    return {
        "contexts": list(map(int, plan.context_lengths)),
        "query_lengths": [
            int(end - start)
            for start, end in zip(plan.token_indptr[:-1], plan.token_indptr[1:])
        ],
        "cache_map": hashlib.sha256(repr(addresses).encode()).hexdigest()[:16],
    }


def validate_prefill_attention(records, *, stats_key, required_operations):
    """Require projection replay and changed live inputs, beyond FFN capture."""

    operations = set()
    for record in records:
        stats = record[stats_key].get("prefill_attention")
        groups = {}
        for name in () if stats is None else stats.get("captured_operations", ()):
            layer, operation = name.rsplit(".", 1)
            groups.setdefault(layer, set()).add(operation)
        if (
            stats is None
            or stats.get("capture_scope") != "attention_projections"
            or not groups
            or any(
                not set(required_operations).issubset(names)
                for names in groups.values()
            )
            or not stats["capture_count"]
            or not stats["replay_count"]
            or set(stats["fallback_counts"]) - {"padding"}
            or stats["eager_fallback_count"] != sum(stats["fallback_counts"].values())
            or stats["poisoned"]
        ):
            raise AssertionError(
                "prefill attention projections were not captured and replayed"
            )
        operations.update(stats["captured_operations"])
    live_calls = [
        call
        for record in records
        for call in record["calls"]
        if call["phase"] == "prefill" and not call["placeholder"]
    ]
    calls = [
        call for call in live_calls if call["graph_replays"].get("prefill_attention", 0)
    ]
    contexts = sorted({value for call in calls for value in call["contexts"]})
    lengths = sorted({value for call in calls for value in call["query_lengths"]})
    all_lengths = sorted(
        {value for call in live_calls for value in call["query_lengths"]}
    )
    maps = {call["cache_map"] for call in calls}
    if min(len(contexts), len(all_lengths), len(maps)) < 2:
        raise AssertionError(
            "prefill projection replay did not vary positions, lengths and cache maps"
        )
    return {
        "capture_scope": "attention_projections",
        "captured_operations": sorted(operations),
        "replayed_context_lengths": contexts,
        "replayed_query_lengths": lengths,
        "all_query_lengths": all_lengths,
        "distinct_cache_maps": len(maps),
    }


def validate_prefill_attention_regions(runner):
    """Compare checkpoint projections eagerly and captured on identical inputs."""
    import torch

    graphs = runner._prefill_tensor_graphs["prefill_attention"]
    layers = tuple(
        dict.fromkeys(region.name.rsplit(".", 1)[0] for region in graphs.regions)
    )
    selected = {layers[0], layers[-1]}
    cases = [
        (n, offset)
        for n, offset in ((1, 3), (16, 17), (32, 31), (16, 37))
        if n in graphs.buckets
    ]
    generator = torch.Generator(device=runner.device).manual_seed(4187)
    position_limit = getattr(runner, "max_seq_len", 128)
    records, failures = [], []
    with torch.inference_mode(), runner._prefill_graphs():
        for index, region in enumerate(graphs.regions):
            if region.name.rsplit(".", 1)[0] not in selected:
                continue
            for tokens, offset in cases:
                values = []
                for layout in region.inputs:
                    shape = (tokens, *layout.shape)
                    if layout.dtype.is_floating_point:
                        value = torch.randn(
                            shape,
                            device=runner.device,
                            dtype=layout.dtype,
                            generator=generator,
                        )
                    else:
                        value = (
                            torch.arange(
                                tokens, device=runner.device, dtype=layout.dtype
                            )
                            + offset
                        ) % position_limit
                        value = (
                            value.reshape(tokens, *(1 for _ in layout.shape))
                            .expand(shape)
                            .contiguous()
                        )
                    values.append(value)
                # Finish operations may rotate their input in place. Neither
                # execution may mutate the other's reference inputs.
                with graphs.capture_context():
                    eager = region.operation(*(value.clone() for value in values))
                eager = eager if isinstance(eager, tuple) else (eager,)
                expected = tuple(value.clone() for value in eager)
                before = runner.cuda_graph_stats[graphs.mode]["replay_count"]
                actual = graphs.invoke(index, *(value.clone() for value in values))
                replayed = runner.cuda_graph_stats[graphs.mode]["replay_count"] > before
                actual = actual if isinstance(actual, tuple) else (actual,)
                exact = (
                    replayed
                    and len(actual) == len(expected)
                    and all(
                        bool(torch.isfinite(value).all())
                        and bool(torch.isfinite(reference).all())
                        and torch.equal(value, reference)
                        for value, reference in zip(actual, expected, strict=True)
                    )
                )
                record = {
                    "operation": region.name,
                    "tokens": tokens,
                    "position_offset": offset,
                    "same_bucket_exact": exact,
                }
                if tokens == 32 and region.name.rsplit(".", 1)[0] == layers[0]:
                    timing_values = tuple(value.clone() for value in values)

                    def eager_projection():
                        with graphs.capture_context():
                            return region.operation(*timing_values)

                    def replay_projection():
                        with graphs.capture_context():
                            return graphs.invoke(index, *timing_values)

                    # Exact buckets give both eager lanes the same operation.
                    # Replay includes input staging; cloning stays outside timing.
                    record.update(
                        timing_rounds=3,
                        calls_per_round=10,
                        staging_included=True,
                        timings=_paired_timings(
                            eager_projection,
                            eager_projection,
                            replay_projection,
                            rounds=3,
                            iterations=10,
                        ),
                    )
                records.append(record)
                if not exact:
                    failures.append((region.name, tokens, offset))
    failed = torch.tensor(bool(failures), device=runner.device, dtype=torch.int32)
    groups = getattr(runner, "parallel_groups", None)
    if groups is not None:
        torch.distributed.all_reduce(
            failed, op=torch.distributed.ReduceOp.MAX, group=groups.device_domain.handle
        )
    if failed.item():
        raise AssertionError(
            f"prefill projection eager/captured mismatch: {failures or 'another rank'}"
        )
    return records


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
