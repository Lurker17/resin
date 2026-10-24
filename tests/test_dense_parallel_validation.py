"""GPU workload contracts and arithmetic diagnostics, checked without a GPU."""

from copy import deepcopy

import pytest
from test_prompt_logprobs import TransitionRunner

from deploy.dense_parallel_validation import (
    _cases,
    _numerical_comparison,
    _Observed,
    _validate_record,
    _workload,
    evaluate_records,
)
from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SchedulerConfig,
)


def test_dense_validation_workload_obeys_native_scoring_and_cache_contracts():
    runner = TransitionRunner()
    runner.runner_config = RunnerConfig(
        continuation_prefill=True,
        cache_layout=CacheLayout(128, [CacheGroupLayout(16)]),
    )
    observed = _Observed(runner)
    engine = Engine(
        observed,
        max_running_requests=2,
        context_length=512,
        eos_token_id=None,
        prefix_cache=False,
        scheduler_config=SchedulerConfig(max_batch_tokens=64, prefill_chunk_size=32),
    )
    tokens = [1, 3, 5, 2, 6, 4, 7] * 7
    prompts = {"short": tokens[:17], "long": tokens}
    try:
        output = _workload(engine, prompts)
        _validate_record(
            dict(
                output=output,
                calls=observed.calls,
                speculation=dict(draft_tokens=0, drafts=0, blocks=0),
            ),
            spec=0,
            prompt_tokens=prompts["long"],
        )
        assert output["cancelled_active"] and output["cancelled_queued"]
        assert output["free_parents"] == 128
        assert observed.released
        for suffix in ("short", "long"):
            unscored = output["outputs"][f"graph_{suffix}"]
            assert unscored["tokens"] == output["outputs"][suffix]["tokens"]
            assert all(event["score"] is None for event in unscored["events"])
    finally:
        engine.close()


@pytest.mark.parametrize(
    "fallback,batch_two_replays,error",
    [("logprobs", 2, None), ("shape_miss", 2, "fallback"), ("logprobs", 0, "two-row")],
)
def test_dense_validation_requires_graph_coverage_and_only_scoring_fallbacks(
    fallback, batch_two_replays, error
):
    record = dict(
        output=None,
        calls=[dict(phase=str(BatchPhase.DECODE), sequences=2, logprobs=False)],
        speculation=dict(draft_tokens=0, drafts=0, blocks=0),
        graphs=dict(
            prefill=dict(replay_count=1),
            decode=dict(
                replay_count=2,
                shape_stats=[dict(shape=[2, 512], replay_count=batch_two_replays)],
                fallback_counts={fallback: 8},
            ),
        ),
    )
    if error is None:
        _validate_record(record, spec=0, graphs=True)
    else:
        with pytest.raises(AssertionError, match=error):
            _validate_record(record, spec=0, graphs=True)


def _output():
    def score(token=7):
        return dict(logprob=-1.0, top=[[7, -1.0], [6, -1.0], [5, -2.0]], targets=[])

    outputs = {}
    for name in (
        "short",
        "long",
        "batch_short",
        "batch_long",
        "graph_short",
        "graph_long",
    ):
        outputs[name] = dict(
            tokens=[7] * 8,
            events=[
                dict(
                    token=7,
                    index=i,
                    score=None if name.startswith("graph") else score(),
                )
                for i in range(8)
            ],
        )
    prompt_score = score()
    prompt_score["targets"] = [[7, -1.0], [0, -4.0]]
    return dict(
        outputs=outputs,
        prompt_scores=[None, deepcopy(prompt_score), deepcopy(prompt_score)],
        cancelled_queued=True,
        cancelled_active=True,
        capacity=128,
        free_parents=128,
    )


def _record(rank=0, spec=0):
    return dict(
        rank=rank,
        output=_output(),
        calls=[dict(phase=str(BatchPhase.DECODE), sequences=2, logprobs=False)],
        speculation=dict(draft_tokens=spec, drafts=spec, blocks=bool(spec)),
        graphs=dict(
            prefill=dict(replay_count=1),
            decode=dict(
                replay_count=2,
                shape_stats=[dict(shape=[2, 512], replay_count=2)],
                fallback_counts={"logprobs": 8},
            ),
        ),
    )


def test_numerical_report_stops_at_first_token_mismatch_even_if_tokens_reconverge():
    reference = _output()
    actual = deepcopy(reference)
    generation = actual["outputs"]["short"]
    generation["tokens"][2] = 8
    for i, event in enumerate(generation["events"]):
        event["score"]["logprob"] -= 0.25 if i < 2 else 100
    report = _numerical_comparison(actual, reference)
    result = report["generation"]["short"]
    assert result["changed_positions"] == [2]
    assert result["matched_prefix_tokens"] == 2
    assert result["matched_prefix_observed_logprob_error"] == dict(
        count=2, mean=0.25, max=0.25
    )
    assert not report["baseline_equivalent"]
    assert report["generation"]["graph_short"][
        "matched_prefix_observed_logprob_error"
    ] == dict(count=0, mean=None, max=None)


def test_prompt_arithmetic_statistics_and_top1_ties_are_reported():
    reference = _output()
    actual = deepcopy(reference)
    for position, delta in ((1, 0.25), (2, 0.75)):
        score = actual["prompt_scores"][position]
        score["logprob"] -= delta
        score["targets"][0][1] -= delta
    actual["prompt_scores"][1]["top"][:2] = [[6, -1.0], [7, -1.0]]
    report = _numerical_comparison(actual, reference)
    assert report["prompt_observed_logprob_error"] == dict(count=2, mean=0.5, max=0.75)
    assert report["prompt_target_logprob_error"] == dict(count=4, mean=0.25, max=0.75)
    assert report["prompt_top1_changed_positions"] == [1]


@pytest.mark.parametrize("field", ["observed", "top", "target"])
def test_nonfinite_scores_are_execution_failures(field):
    record = _record()
    score = record["output"]["prompt_scores"][1]
    if field == "observed":
        record["output"]["outputs"]["short"]["events"][0]["score"]["logprob"] = float(
            "nan"
        )
    else:
        score["top" if field == "top" else "targets"][0][1] = float("inf")
    with pytest.raises(AssertionError, match="nonfinite"):
        _validate_record(record, spec=0)


def test_matrix_distinguishes_graph_regression_from_cross_topology_arithmetic():
    records = {}
    for case in _cases(speculative=True):
        values = [_record(rank, case["spec"]) for rank in (0, 1)]
        if case["attn_tp"] is not None:
            values[1]["output"] = None
            score = values[0]["output"]["prompt_scores"][1]
            delta = 0.5 if case["spec"] else 0.25
            score["logprob"] -= delta
            score["targets"][0][1] -= delta
        records[case["name"]] = values
    prompts = dict(long=[7, 7, 7])
    report = evaluate_records(records, prompts)
    assert report["execution_passed"] and report["same_topology_graph_exact"]
    assert not report["baseline_equivalent"]
    assert len(report["speculative_vs_non_speculative"]) == 2
    difference = report["speculative_vs_non_speculative"]["attn-tp2-ffn-tp2-spec2"]
    assert difference["prompt_observed_logprob_error"]["max"] == 0.25

    records["attn-tp2-ffn-tp2-spec0"][0]["output"]["outputs"]["graph_short"]["tokens"][
        0
    ] = 8
    records["attn-tp2-ffn-tp2-spec0"][0]["output"]["outputs"]["graph_short"]["events"][
        0
    ]["token"] = 8
    with pytest.raises(AssertionError, match="same-topology graph/eager"):
        evaluate_records(records, prompts)


def test_matrix_rejects_independent_baseline_rank_disagreement():
    records = {
        case["name"]: [_record(rank) for rank in (0, 1)] for case in _cases(False)
    }
    records["baseline"][1]["output"]["prompt_scores"][1]["top"][:2] = [
        [6, -1.0],
        [7, -1.0],
    ]
    with pytest.raises(AssertionError, match="independent eager baseline"):
        evaluate_records(records, dict(long=[7, 7, 7]))
