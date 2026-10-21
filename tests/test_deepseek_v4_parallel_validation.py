"""Keep full-checkpoint numerical failures available for focused GPU retries."""

import json
from unittest.mock import Mock

import pytest
import torch

from deploy import deepseek_v4_parallel_validation as validation


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
    )
    spawn.assert_called_once()
    assert spawn.call_args.args[0] is validation._run_rank
    assert result["status"] == "passed"
    assert result["cases"][0]["teacher_forced_comparisons"][0]["passed"] is True
    assert json.loads((retry_path / "report.json").read_text())["status"] == "passed"
    assert json.loads((failure_path / "report.json").read_text())["status"] == "failed"
