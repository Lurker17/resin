"""Checkpoint reports distinguish captured projections from ordinary FFN graphs."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from deploy.graph_validation import (
    prefill_plan_metadata,
    validate_prefill_attention,
    validate_prefill_attention_regions,
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
    assert result["distinct_cache_maps"] == 2
    assert result["captured_operations"] == [
        "layer.0.finish",
        "layer.0.project",
        "layer.0.query",
    ]


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


def test_projection_probe_replays_changed_inputs_and_isolates_mutating_reference():
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
    assert not torch.equal(seen[1], seen[3])
