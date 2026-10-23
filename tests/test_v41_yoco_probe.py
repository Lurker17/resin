"""Qualification reports must fail on changed contexts, scores, or cache bytes."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from deploy.v41_yoco_probe import (
    ProbeFailure,
    ValidationFailure,
    _cache_snapshot,
    _compare,
    _public,
    _sample_positions,
    _tensor_report,
    compare_fixed_mode_reports,
    run_probe,
)


@pytest.mark.parametrize(
    "end,start", [(1, 0), (128, 0), (129, 127), (8193, 8000), (8193, 9000)]
)
def test_logical_cache_samples_are_bounded_and_cover_live_end(end, start):
    values = _sample_positions(end, start=start)
    assert values == sorted(set(values))
    assert all(start <= value < end for value in values)
    assert len(values) <= 10
    if end > start:
        assert values[0] == start and values[-1] == end - 1


def _result():
    return dict(
        requests={
            129: dict(
                tokens=[2, 3],
                top_ids=[[2, 4], [3, 4]],
                scores=[-0.5, -0.75],
                cached_tokens=0,
            )
        },
        snapshots={
            (129, 129): dict(
                prompt=129,
                endpoint=129,
                input_rows=128,
                consumer_rows=[128],
                prompt_scores=False,
                state={
                    "main.20": torch.tensor([1, 2], dtype=torch.uint8),
                    "window.40": torch.tensor([0.125, -0.5]),
                },
            )
        },
    )


@pytest.mark.parametrize("changed", ["token", "score", "bytes", "window", "context"])
def test_equivalence_checks_reject_each_material_difference(changed):
    left, right = _result(), _result()
    if changed == "token":
        right["requests"][129]["tokens"][1] = 4
    elif changed == "score":
        right["requests"][129]["scores"][1] += 0.001
    elif changed == "context":
        right["snapshots"][(129, 128)] = right["snapshots"].pop((129, 129))
    else:
        name = "main.20" if changed == "bytes" else "window.40"
        right["snapshots"][(129, 129)]["state"][name][0] += 1
    with pytest.raises(AssertionError):
        _compare(left, right)


def test_public_report_preserves_scores_and_hashes_without_tensor_payloads():
    result = _result()
    public = _public(result)
    json.dumps(public)
    assert public["requests"] == result["requests"]
    assert public["snapshots"][0]["prompt"] == 129
    assert public["snapshots"][0]["endpoint"] == 129
    assert "state" not in public["snapshots"][0]
    assert len(public["snapshots"][0]["state_sha256"]["main.20"]) == 64
    assert _compare(result, copy.deepcopy(result))["state"]


def test_snapshot_keeps_compressor_vectors_and_empty_early_consumer_ranges(monkeypatch):
    from resin_models_cuda.deepseek_v41 import attention

    reads = []

    def gather(_cache, rows):
        reads.append(rows.tolist())
        return torch.zeros(rows.numel(), 512)

    monkeypatch.setattr(attention, "gather_kv", gather)
    state = torch.arange(6 * 512, dtype=torch.float32).reshape(6, 512)
    cache = SimpleNamespace(
        storage=torch.empty(0),
        geometry=SimpleNamespace(
            kv_sources=(1,),
            ratios={1: 1},
            compressor_sources=(0,),
            attention_layers=2,
            window_size=128,
        ),
        compressed_rows=lambda _source, _ratio, logical: logical,
        compressed_cache=lambda _source: torch.zeros(1, 3, 1, 656, dtype=torch.uint8),
        index_parts=lambda _source: (
            torch.zeros(1, 3, 64, dtype=torch.int8),
            torch.zeros(1, 3, dtype=torch.int32),
        ),
        compressor_rows=lambda _source, _tail: (
            torch.tensor([[2, 4]]),
            torch.tensor([[3, 5]]),
        ),
        compressor_values=state,
        window_rows=lambda _layer, logical: logical,
        window_cache=lambda _layer: None,
    )
    runner = SimpleNamespace(_model=SimpleNamespace(prefill_boundary=1))
    snapshot = _cache_snapshot(runner, cache, end=3, replay_start=140)
    assert reads == [[[0, 1, 2]], [[]]]
    assert snapshot["window.1"].shape == (0, 512)
    assert torch.equal(snapshot["compressor.0.values"], state[torch.tensor([[2, 4]])])
    assert torch.equal(snapshot["compressor.0.scores"], state[torch.tensor([[3, 5]])])


def test_failures_preserve_context_max_error_and_json_partial_reports():
    with pytest.raises(AssertionError, match=r"fresh.*main.20.*max_abs=1.0"):
        _tensor_report(torch.tensor([2.0]), torch.tensor([1.0]), label="fresh main.20")
    partial = {"mask": {"fresh": _public(_result())}}
    failure = ProbeFailure("changed compact cache", partial)
    records = [
        dict(
            ranks=[dict(rank=0, ok=False, error=str(failure), partial_results=partial)]
        )
    ]
    wrapped = ValidationFailure(records)
    assert "changed compact cache" in str(wrapped)
    assert wrapped.validation_records == records
    json.dumps(wrapped.validation_records)


def test_cp_cannot_silently_switch_modes_only_on_drivers():
    runner = SimpleNamespace(
        parallel_groups=SimpleNamespace(
            topology=SimpleNamespace(world_size=8),
            drivers=SimpleNamespace(size=4, local_rank=0),
        ),
        _swa_replay_mode="compact",
    )
    with pytest.raises(ValueError, match="constructor-fixed"):
        run_probe(runner, object(), [2, 3])
    with pytest.raises(ValueError, match="constructor mode"):
        run_probe(runner, object(), [2, 3], modes=("mask",))


def test_separate_cp_reports_compare_scores_and_canonical_state_hashes():
    metadata = dict(
        git_commit="source",
        source_digest="digest",
        checkpoint_revision="weights",
        parallelism=dict(cp=2),
        expert_backend="flashinfer",
        quant="checkpoint",
        hardware="B300",
    )

    def report(mode):
        return [
            dict(
                metadata=metadata,
                ranks=[
                    dict(
                        rank=0,
                        ok=True,
                        modes=[mode],
                        generated=160,
                        retained_tokens=140,
                        results={mode: {"fresh": _public(_result())}},
                    )
                ],
            )
        ]

    mask, compact = report("mask"), report("compact")
    assert compare_fixed_mode_reports(mask, compact)["exact_state_hashes"] == 2
    compact[0]["ranks"][0]["results"]["compact"]["fresh"]["snapshots"][0][
        "state_sha256"
    ]["main.20"] = "changed"
    with pytest.raises(AssertionError, match="canonical cache state"):
        compare_fixed_mode_reports(mask, compact)
