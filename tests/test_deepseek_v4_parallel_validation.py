"""Keep full-checkpoint numerical failures available for focused GPU retries."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from deploy import deepseek_v4_parallel_validation as validation
from resin import BatchPhase, ForwardOutput


def test_failed_case_persists_logits_and_reuses_baseline_for_retry(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(validation.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(validation, "_validate_records", lambda *args, **kwargs: {})
    case = "attention-tp-expert-tp"
    prompts = {"probe": [7]}
    expected = torch.ones(4, 3)

    def record(logits):
        return {
            "rank": 0,
            "driver": True,
            "driver_result": {"probes": [{"token_ids": [9]}]},
            "_teacher_logits": logits,
        }

    failed_logits = expected * 2
    spawn = Mock(
        side_effect=[
            [{"token_ids": [9], "events": [], "_teacher_logits": expected}],
            [record(failed_logits)],
        ]
    )
    monkeypatch.setattr(validation, "_spawn", spawn)
    failure_path = tmp_path / "failure"
    with pytest.raises(AssertionError, match="same-context model logits"):
        validation.run_composable_validation(
            tmp_path / "checkpoint",
            prompts,
            vocab_size=3,
            diagnostic_path=failure_path,
            case_names=(case,),
        )

    assert spawn.call_count == 2
    report = json.loads((failure_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failed_case"] == case
    assert report["selected_cases"] == [case]
    comparison = report["cases"][0]["teacher_forced_comparisons"][0]
    assert comparison["passed"] is False
    assert comparison["max_abs"] == 1.0
    case_report = json.loads((failure_path / f"{case}.json").read_text())
    assert case_report == report["cases"][0]
    saved_logits = torch.load(failure_path / f"{case}-logits.pt", weights_only=True)
    torch.testing.assert_close(saved_logits[0], failed_logits)
    baseline_path = failure_path / "baseline.pt"
    baseline = torch.load(baseline_path, weights_only=True)
    torch.testing.assert_close(baseline["teacher_logits"], expected)

    # A focused retry needs only the selected worker layout; it must neither
    # reload the full reference model nor overwrite the failed run's evidence.
    spawn = Mock(return_value=[record(expected.clone())])
    monkeypatch.setattr(validation, "_spawn", spawn)
    retry_path = tmp_path / "retry"
    result = validation.run_composable_validation(
        tmp_path / "checkpoint",
        prompts,
        vocab_size=3,
        diagnostic_path=retry_path,
        case_names=(case,),
        baseline_path=baseline_path,
        graphs=False,
    )
    spawn.assert_called_once()
    assert spawn.call_args.args[0] is validation._run_rank
    assert result["status"] == "passed"
    assert result["graphs"] == {"prefill": "disabled", "decode": "disabled"}
    assert spawn.call_args.args[-3] is False
    assert result["cases"][0]["teacher_forced_comparisons"][0]["passed"] is True
    assert json.loads((retry_path / "report.json").read_text())["status"] == "passed"
    assert json.loads((failure_path / "report.json").read_text())["status"] == "failed"
    assert result["cases"][0]["teacher_forced_decode"]["status"] == (
        "skipped_legacy_baseline"
    )

    # Old compact-window baselines describe different attention semantics.
    # Refuse them before starting workers even if checkpoint/prompts match.
    baseline["identity"].pop("attention_contract")
    incompatible_path = tmp_path / "old-baseline.pt"
    torch.save(baseline, incompatible_path)
    spawn.reset_mock()
    with pytest.raises(ValueError, match="attention contract"):
        validation.run_composable_validation(
            tmp_path / "checkpoint",
            prompts,
            vocab_size=3,
            case_names=(case,),
            baseline_path=incompatible_path,
        )
    spawn.assert_not_called()


def test_decode_trace_captures_genuine_graph_output_before_engine_override():
    shape = (1, 8)

    class Runner:
        def __init__(self):
            self._model = SimpleNamespace(head=torch.nn.Identity())
            self._sampler = object()
            self._graph_logits = {shape: torch.tensor([[0.0, 3.0, 1.0]])}

        def _run_cuda_graph(self, mode, graph_shape, inputs):
            return [1]

        def forward(self, plan):
            if plan.phase == BatchPhase.PREFILL:
                self._model.head(torch.tensor([[0.0, 1.0, 3.0]]))
                return [2]
            return self._run_cuda_graph("decode", shape, {})

    runner = Runner()
    observed = validation._ObservedRunner(runner)
    original_graph, original_sampler = runner._run_cuda_graph, runner._sampler

    def plan(phase):
        return SimpleNamespace(
            phase=phase,
            num_sequences=1,
            sampling_mask=(True,),
            needs_logprobs=False,
            token_ids=[7],
            positions=[0],
        )

    with validation._DecodeTrace(observed, forced_tokens=[0, 2]) as trace:
        assert observed.forward(plan(BatchPhase.PREFILL)) == [0]
        assert observed.forward(plan(BatchPhase.DECODE)) == [2]
    result = trace.result()
    assert result["sampler_token_ids"] == [2, 1]
    assert result["advanced_token_ids"] == [0, 2]
    assert result["phases"] == ["prefill", "decode"]
    assert result["graph_shapes"] == [None, shape]
    torch.testing.assert_close(
        result["logits"], torch.tensor([[0.0, 1.0, 3.0], [0.0, 3.0, 1.0]])
    )
    assert runner._run_cuda_graph == original_graph
    assert runner._sampler is original_sampler
    assert observed.decode_trace is None


def test_decode_numerical_failure_persists_genuine_and_forced_history(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(validation.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(validation, "_validate_records", lambda *args, **kwargs: {})
    expected = torch.tensor([[0.0, 1.0, 3.0], [0.0, 3.0, 1.0]])
    trace = {
        "logits": expected,
        "sampler_token_ids": [2, 1],
        "advanced_token_ids": [2, 1],
        "phases": ["prefill", "decode"],
        "input_token_ids": [[7], [2]],
        "positions": [[0], [1]],
        "graph_shapes": [None, None],
    }
    # A uniform offset leaves the distribution unchanged, but still fails the
    # existing absolute tolerance. Diagnostic KL must not relax that check.
    actual = {
        **trace,
        "logits": expected + 2,
        "sampler_token_ids": [2, 0],
        "graph_shapes": [None, (1, 8)],
    }
    case = "ep-attention-tp"
    spawn = Mock(
        side_effect=[
            [
                {
                    "token_ids": [2, 1],
                    "_teacher_logits": expected,
                    "_decode_trace": trace,
                }
            ],
            [
                {
                    "rank": 0,
                    "driver": True,
                    "driver_result": {"probes": [{"token_ids": [2, 0]}]},
                    "_teacher_logits": expected,
                    "_decode_trace": actual,
                }
            ],
        ]
    )
    monkeypatch.setattr(validation, "_spawn", spawn)
    with pytest.raises(AssertionError, match="fixed-context decode logits"):
        validation.run_composable_validation(
            tmp_path / "checkpoint",
            {"probe": [7]},
            vocab_size=3,
            diagnostic_path=tmp_path,
            case_names=(case,),
            decode_capacity=32,
        )
    report = json.loads((tmp_path / "report.json").read_text())
    decode = report["cases"][0]["teacher_forced_decode"]["ranks"][0]
    assert decode["contexts_match"]
    assert not decode["passed"]
    assert not decode["sampler_matches_baseline"]
    assert decode["sampler_token_ids"] == [2, 0]
    assert decode["advanced_token_ids"] == [2, 1]
    comparison = decode["comparisons"][1]
    assert comparison["logit_offsets"] == [2.0]
    assert comparison["centered_rms"] == 0
    assert comparison["kl_expected_actual"] == pytest.approx([0], abs=1e-7)
    saved = torch.load(tmp_path / f"{case}-decode.pt", weights_only=True)
    torch.testing.assert_close(saved[0]["logits"], actual["logits"])
    baseline = torch.load(tmp_path / "baseline.pt", weights_only=True)
    assert baseline["numerical_version"] == 2
    torch.testing.assert_close(baseline["decode_trace"]["logits"], expected)
    assert spawn.call_args.args[-2] is trace
    assert spawn.call_args.args[-1] == 32
    assert report["decode_shapes"] == [[1, 32], [2, 512]]


@pytest.mark.parametrize("protocol_error", [False, True])
def test_numerical_failures_allow_later_cases_but_protocol_errors_stop(
    monkeypatch, tmp_path, protocol_error
):
    monkeypatch.setattr(validation.torch.cuda, "device_count", lambda: 2)
    validator = (
        Mock(side_effect=RuntimeError("protocol failure"))
        if protocol_error
        else (Mock(return_value={}))
    )
    monkeypatch.setattr(validation, "_validate_records", validator)
    expected = torch.ones(4, 3)

    def record(logits):
        return {
            "rank": 0,
            "driver": True,
            "driver_result": {"probes": [{"token_ids": [9]}]},
            "_teacher_logits": logits,
        }

    cases = ("attention-tp-expert-tp", "ep-context")
    spawn = Mock(
        side_effect=[
            [{"token_ids": [9], "_teacher_logits": expected}],
            [record(expected * 2)],
            [record(expected.clone())],
        ]
    )
    monkeypatch.setattr(validation, "_spawn", spawn)
    with pytest.raises(
        RuntimeError if protocol_error else AssertionError,
        match="protocol failure" if protocol_error else "same-context model logits",
    ):
        validation.run_composable_validation(
            tmp_path / "checkpoint",
            {"probe": [7]},
            vocab_size=3,
            diagnostic_path=tmp_path,
            case_names=cases,
        )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failed_case"] == cases[0]
    assert spawn.call_count == (2 if protocol_error else 3)
    assert len(report["cases"]) == (1 if protocol_error else 2)
    if not protocol_error:
        assert report["failed_cases"] == [cases[0]]
        assert report["cases"][0]["status"] == "failed"
        assert report["cases"][1]["status"] == "passed"
        assert len(report["numerical_errors"]) == 1
        assert (tmp_path / f"{cases[1]}-logits.pt").exists()


@pytest.fixture
def speculative_record():
    events = [
        {
            "request_id": 7,
            "index": index,
            "logprob": -0.5,
            "top_logprobs": [(0, -0.5), (1, -1.0), (2, -2.0)],
        }
        for index in range(4)
    ]
    return {
        "rank": 0,
        "driver": True,
        "calls": [
            {"phase": "prefill", "placeholder": False, "rows": 1},
            {"phase": "decode", "placeholder": True, "rows": 0, "output_lengths": []},
            {"phase": "decode", "placeholder": False, "rows": 1, "output_lengths": [3]},
            {"phase": "decode", "placeholder": False, "rows": 1, "output_lengths": [1]},
        ],
        "speculative_stats": {"blocks": 2, "drafts": 4, "accepted": 2, "emitted": 4},
        "driver_result": {
            "events": events,
            "probes": [
                {
                    "token_ids": [1] * 16,
                    "events": [
                        {
                            "request_id": 8,
                            "index": index,
                            "token_id": 1,
                            "logprob": None,
                            "top_logprobs": [],
                        }
                        for index in range(16)
                    ],
                },
                {
                    "token_ids": [1] * 16,
                    "events": [
                        {
                            "request_id": 8,
                            "index": index,
                            "token_id": 1,
                            "logprob": None,
                            "top_logprobs": [],
                        }
                        for index in range(16)
                    ],
                },
            ],
        },
    }


def test_speculative_block_checks_cover_ragged_outputs_and_idle_resume(
    speculative_record,
):
    other = deepcopy(speculative_record)
    other["rank"] = 1
    result = validation._validate_speculative_records([speculative_record, other], 2)
    assert result["verified_blocks"] == 4
    assert result["accepted_draft_tokens"] == 4
    assert result["idle_replica_resumed"]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("budget", "block budget"),
        ("counters", "counters"),
        ("indices", "streamed token index"),
        ("scores", "logprob columns"),
        ("no_acceptance", "accept draft tokens"),
        ("no_resume", "did not resume"),
    ],
)
def test_speculative_checks_reject_incomplete_execution(
    speculative_record, failure, message
):
    records = [speculative_record]
    if failure == "budget":
        speculative_record["calls"][2]["output_lengths"] = [4]
    elif failure == "counters":
        speculative_record["speculative_stats"]["drafts"] = 3
    elif failure == "indices":
        speculative_record["driver_result"]["events"][2]["index"] = 1
    elif failure == "scores":
        speculative_record["driver_result"]["events"][0]["top_logprobs"].pop()
    elif failure == "no_acceptance":
        speculative_record["calls"][2]["output_lengths"] = [1]
        speculative_record["speculative_stats"].update(accepted=0, emitted=2)
    else:
        speculative_record["calls"].append(speculative_record["calls"].pop(1))
        records.append(deepcopy(speculative_record))
    with pytest.raises(AssertionError, match=message):
        validation._validate_speculative_records(records, 2)


def test_speculative_output_lengths_support_zero_copy_scored_blocks():
    output = ForwardOutput(
        np.array([1, 2, 3, 4], np.uint32),
        np.array([0, 3, 4], np.uint32),
    )
    assert validation._output_lengths(output) == [3, 1]
    assert validation._output_lengths([[1, 2, 3], [4]]) == [3, 1]
    assert validation._output_lengths(None) == []


def test_speculative_validation_uses_both_references_and_rejects_other_block_size(
    monkeypatch, tmp_path, speculative_record
):
    monkeypatch.setattr(validation.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(validation, "_validate_records", lambda *args, **kwargs: {})
    target = {"token_ids": [0] * 16}
    baseline = {"token_ids": [1] * 16}
    spawn = Mock(side_effect=[[target], [baseline], [speculative_record]])
    monkeypatch.setattr(validation, "_spawn", spawn)
    case = "ep-attention-dp"
    options = dict(
        vocab_size=3,
        case_names=(case,),
        diagnostic_path=tmp_path,
        speculative_tokens=2,
    )
    result = validation.run_composable_validation(
        tmp_path / "checkpoint", {"probe": [7]}, **options
    )
    assert result["status"] == "passed"
    assert not result["baseline_comparison"]["matches"]
    assert result["cases"][0]["greedy_comparisons"][0]["legacy_dspark"]["matches"]
    assert not result["cases"][0]["greedy_comparisons"][0]["target"]["matches"]
    for call, count in zip(spawn.call_args_list[:2], (0, 2), strict=True):
        assert call.args[0] is validation._run_tp_baseline_rank
        assert call.args[4:] == (False, count, 16)
    assert spawn.call_args.args[-1] == 2
    saved = torch.load(tmp_path / "baseline.pt", weights_only=True)
    assert saved["identity"]["speculative_tokens"] == 2
    assert saved["target_report"] == target
    assert saved["report"] == baseline
    spawn.reset_mock()
    with pytest.raises(ValueError, match="K and attention contract"):
        validation.run_composable_validation(
            tmp_path / "checkpoint",
            {"probe": [7]},
            **{
                **options,
                "speculative_tokens": 3,
                "baseline_path": tmp_path / "baseline.pt",
            },
        )
    spawn.assert_not_called()


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_composed_speculative_validation_requires_integer_block_size(
    monkeypatch, count
):
    monkeypatch.setattr(validation.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="non-negative integer"):
        validation.run_composable_validation(
            "checkpoint", {"probe": [7]}, vocab_size=3, speculative_tokens=count
        )
