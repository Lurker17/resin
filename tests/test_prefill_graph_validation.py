"""Checkpoint reports distinguish captured projections from ordinary FFN graphs."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from deploy.graph_validation import (
    prefill_plan_metadata,
    validate_prefill_attention,
    validate_prefill_attention_regions,
    validate_prefill_graphs,
)
from resin import BatchPlan, CacheGroupPlan


def _record():
    return {
        "cuda_graph_stats": {
            "prefill_attention": {
                "capture_scope": "attention_projections",
                "captured_operations": [
                    "layer.0.project",
                    "layer.0.query",
                    "layer.0.finish",
                ],
                "capture_count": 3,
                "replay_count": 6,
                "eager_fallback_count": 1,
                "fallback_counts": {"padding": 1},
                "poisoned": False,
            },
        },
        "calls": [
            {
                "phase": "prefill",
                "placeholder": False,
                "contexts": [context],
                "query_lengths": [length],
                "cache_map": cache,
                "graph_replays": {"prefill_attention": 3},
            }
            for context, length, cache in ((0, 16, "first"), (32, 32, "second"))
        ],
    }


def _validate(record):
    return validate_prefill_attention(
        [record],
        stats_key="cuda_graph_stats",
        required_operations=("project", "query", "finish"),
    )


def test_projection_validation_accepts_explicit_padding_and_records_changed_inputs():
    result = _validate(_record())
    assert result["replayed_context_lengths"] == [0, 32]
    assert result["replayed_query_lengths"] == [16, 32]
    assert result["all_query_lengths"] == [16, 32]
    assert result["distinct_cache_maps"] == 2
    assert result["captured_operations"] == [
        "layer.0.finish",
        "layer.0.project",
        "layer.0.query",
    ]


def test_exact_bucket_replays_and_eager_partial_tail_cover_variable_query_lengths():
    record = _record()
    record["calls"][0]["query_lengths"] = [32]
    tail = deepcopy(record["calls"][1])
    tail.update(
        query_lengths=[18],
        contexts=[64],
        cache_map="third",
        graph_replays={"prefill_attention": 0},
    )
    record["calls"].append(tail)
    result = _validate(record)
    assert result["replayed_query_lengths"] == [32]
    assert result["all_query_lengths"] == [18, 32]
    assert result["replayed_context_lengths"] == [0, 32]
    assert result["distinct_cache_maps"] == 2


@pytest.mark.parametrize(
    "failure", ["ffn_only", "missing_query", "no_replay", "fallback", "poison"]
)
def test_projection_validation_rejects_incomplete_capture(failure):
    record = _record()
    stats = record["cuda_graph_stats"]["prefill_attention"]
    if failure == "ffn_only":
        stats["capture_scope"] = "feed_forward_regions"
    elif failure == "missing_query":
        stats["captured_operations"].remove("layer.0.query")
    elif failure == "no_replay":
        stats["replay_count"] = 0
    elif failure == "fallback":
        stats["fallback_counts"] = {"dtype": 1}
    else:
        stats["poisoned"] = True
    with pytest.raises(AssertionError, match="projections were not captured"):
        _validate(record)


@pytest.mark.parametrize("field", ["contexts", "query_lengths", "cache_map"])
def test_projection_replay_requires_changed_live_metadata(field):
    record = _record()
    record["calls"][1][field] = deepcopy(record["calls"][0][field])
    with pytest.raises(AssertionError, match="vary positions, lengths and cache maps"):
        _validate(record)


def test_plan_metadata_tracks_physical_parent_and_slot_without_dumping_page_tables():
    def plan(parent, slot):
        return BatchPlan(
            [7],
            [4],
            [2, 3],
            [0, 2],
            cache_groups=[
                CacheGroupPlan(
                    0,
                    [0, 1],
                    [parent],
                    [slot],
                    [0],
                    [parent, parent],
                    [slot, slot],
                    [4, 5],
                    blocks_per_slab=2,
                )
            ],
        )

    metadata = prefill_plan_metadata(plan(3, 0))
    assert metadata["contexts"] == [4]
    assert metadata["query_lengths"] == [2]
    assert metadata["cache_map"] == prefill_plan_metadata(plan(3, 0))["cache_map"]
    assert metadata["cache_map"] != prefill_plan_metadata(plan(4, 0))["cache_map"]
    assert metadata["cache_map"] != prefill_plan_metadata(plan(3, 1))["cache_map"]
    assert prefill_plan_metadata(None)["cache_map"] is None


def test_projection_probe_replays_changed_inputs_and_isolates_mutating_reference(
    monkeypatch,
):
    from deploy import graph_validation

    timed = []

    def timings(eager, live, replay, *, rounds, iterations):
        assert eager is live
        assert (rounds, iterations) == (3, 10)
        eager()
        replay()
        timed.append(True)
        return {"eager_padded_us": {"median": 2}, "configured_policy_us": {"median": 1}}

    monkeypatch.setattr(graph_validation, "_paired_timings", timings)

    def operation(hidden, positions):
        return hidden.add_(positions[:, None])

    region = SimpleNamespace(
        name="layer.0.finish",
        operation=operation,
        inputs=(
            SimpleNamespace(shape=(2,), dtype=torch.bfloat16),
            SimpleNamespace(shape=(), dtype=torch.int64),
        ),
    )
    stats = {"prefill_attention": {"replay_count": 0}}
    seen = []

    def invoke(index, *values):
        stats["prefill_attention"]["replay_count"] += 1
        seen.append(values[1].clone())
        return operation(*values)

    graphs = SimpleNamespace(
        regions=(region,),
        buckets=(1, 16, 32),
        capture_context=nullcontext,
        invoke=invoke,
        mode="prefill_attention",
    )
    runner = SimpleNamespace(
        device="cpu",
        max_seq_len=64,
        cuda_graph_stats=stats,
        _prefill_tensor_graphs={"prefill_attention": graphs},
        _prefill_graphs=nullcontext,
    )
    result = validate_prefill_attention_regions(runner)
    assert len(result) == 4 and all(record["same_bucket_exact"] for record in result)
    positions = [value for value in seen if len(value) == 16]
    assert not torch.equal(positions[0], positions[1])
    assert len(timed) == 1
    assert next(record for record in result if record["tokens"] == 32)[
        "staging_included"
    ]


@pytest.mark.parametrize("stale_replay", [False, True])
def test_ffn_probe_uses_graph_owned_buckets_and_capture_context(
    monkeypatch, stale_replay
):
    from deploy import graph_validation

    active, entered, timed = [], [], []

    @contextmanager
    def capture_context():
        active.append(True)
        entered.append(True)
        try:
            yield
        finally:
            active.pop()

    def operation(hidden, ids):
        assert active == [True]
        return hidden.float() * 2 + ids[:, None]

    def unused(hidden, ids):
        raise AssertionError("the bounded probe must select only first/last regions")

    def timings(eager, live, replay, *, rounds, iterations):
        assert (rounds, iterations) == (7, 20)
        expected = eager()[:3]
        torch.testing.assert_close(live(), expected, rtol=0, atol=0)
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
        timed.append(True)
        return {"configured_policy_us": {"median": 1}}

    monkeypatch.setattr(graph_validation, "_paired_timings", timings)
    graphs = SimpleNamespace(buckets=(4, 32), capture_context=capture_context)
    stats = {
        "prefill": {
            "captured_shapes": [
                (size, region) for size in (4, 32) for region in (1, 3)
            ],
            "replay_count": 0,
        }
    }
    seen, outputs = {}, {}

    def replay(region, hidden, ids):
        assert active == [True]
        rows = len(hidden)
        bucket = next(size for size in graphs.buckets if size >= rows)
        key = (region, bucket)
        seen.setdefault(key, []).append(hidden.clone())
        stats["prefill"]["replay_count"] = sum(map(len, seen.values()))
        if not stale_replay or key not in outputs:
            padded = torch.zeros(bucket, 2, dtype=hidden.dtype)
            padded[:rows].copy_(hidden)
            padded_ids = torch.zeros(bucket, dtype=ids.dtype)
            padded_ids[:rows].copy_(ids)
            outputs[key] = operation(padded, padded_ids)
        return outputs[key][:rows]

    # Only current graph-owned state is present: no legacy bucket/context fields.
    runner = SimpleNamespace(
        device="cpu",
        vocab_size=101,
        cuda_graph_stats=stats,
        _prefill_operations=(operation, unused, operation),
        _prefill_hidden_size=2,
        _prefill_token_ids=True,
        _prefill_tensor_graphs={"prefill": graphs},
        _prefill_graphs=nullcontext,
        _run_prefill_region=replay,
    )
    if stale_replay:
        with pytest.raises(AssertionError, match="same bucket"):
            validate_prefill_graphs(runner)
    else:
        result = validate_prefill_graphs(runner)
        assert [(item["region"], item["rows"], item["bucket"]) for item in result] == [
            (region, rows, bucket)
            for region in (0, 2)
            for rows, bucket in ((3, 4), (17, 32), (32, 32))
        ]
        assert all(item["uses_graph"] for item in result)
        assert all(item["same_bucket"]["max_abs_error"] == 0 for item in result)
        assert len(timed) == 2 and stats["prefill"]["replay_count"] == 8
    assert entered and not active
    first, changed = seen[(0, 32)]
    assert not torch.equal(first, changed[:17])
