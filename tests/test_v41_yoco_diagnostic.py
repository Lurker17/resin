"""Diagnostic hooks retain the same logical rows and restore model methods."""

import json
from types import SimpleNamespace

import pytest
import torch

from deploy.v41_yoco_diagnostic import RetainedActivations, run_diagnostic_validation
from deploy.v41_yoco_probe import ValidationFailure
from resin_models_cuda.deepseek_v41.hc import HCLazyPost


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = 0.0

    def mixes(self, hidden, name):
        return (hidden.sum(-2) + self.delta, hidden.mean(-1))

    def forward(self, hidden, pre):
        self.mixes(hidden, "attn")
        return hidden, pre


@pytest.mark.parametrize("delta", [0.0, 0.125])
def test_hooks_compare_full_and_compact_rows_without_changing_outputs(delta):
    layer = Block()
    runner = SimpleNamespace(_model=SimpleNamespace(layers=[layer]))
    original = layer.mixes
    hidden = torch.arange(64).reshape(1, 8, 2, 4).float()
    state = HCLazyPost(hidden.clone(), hidden[..., 0], hidden[..., 0], hidden[..., :2])
    with RetainedActivations(runner, layers=(0,)) as trace:
        trace.begin_row("mask", 10, 0, 8, 4)
        output = layer(hidden, state)
        trace.end_row()
        assert output[0] is hidden and output[1] is state
        layer.delta = delta
        trace.begin_row("compact", 10, 0, 8, 4)
        compact_state = HCLazyPost(*(value[:, -2:] for value in vars(state).values()))
        layer(hidden[:, -2:], compact_state)
        trace.end_row()
        report = trace.report()
        json.dumps(report)
        if delta:
            assert report["first_difference"]["field"] == "layer.0.mixes.attn.output.0"
            assert report["first_difference"]["max_abs"] == delta
            assert report["comparisons"]["(10, 8)"]["differences"].keys() == {
                "layer.0.mixes.attn.output.0"
            }
        else:
            assert report["first_difference"] is None
        assert report["snapshots"]["mask"][0]["positions"] == (6, 7)
    assert runner._yoco_diagnostic is None and layer.mixes == original
    assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_placeholder_and_partial_failure_keep_diagnostics_and_restore_hooks():
    layer = Block()
    runner = SimpleNamespace(_model=SimpleNamespace(layers=[layer]))
    original = layer.mixes
    with pytest.raises(RuntimeError, match="original failure"):
        with RetainedActivations(runner, layers=(0,)) as trace:
            trace.begin_row("mask", 100, 0, 8, 4)
            layer(torch.zeros(1, 8, 2, 4), torch.zeros(1, 8, 2))
            trace.end_row()
            trace.begin_row("compact", 100, 8, 8, 4)
            report = trace.report()
            assert report["snapshots"]["mask"][0]["positions"] == (7,)
            assert report["missing_rows"]["mask"] == [(100, 8)]
            json.dumps(report)
            raise RuntimeError("original failure")
    assert layer.mixes == original and runner._yoco_diagnostic is None


@pytest.mark.parametrize("failed", [False, True])
def test_combined_callback_reuses_load_and_preserves_passed_diagnostic(
    monkeypatch, failed
):
    from deploy import v41_yoco_diagnostic as diagnostic

    calls = []
    prefix = [dict(ranks=[dict(rank=0, ok=True)])]
    suffix = [dict(ranks=[dict(rank=0, ok=not failed, error="full failure")])]

    def validation(*args, **kwargs):
        calls.append((args, kwargs))
        if kwargs:
            return prefix
        if failed:
            raise ValidationFailure(suffix)
        return suffix

    monkeypatch.setattr(diagnostic, "run_validation", validation)
    objects = tuple(object() for _ in range(5))
    if failed:
        with pytest.raises(ValidationFailure) as caught:
            run_diagnostic_validation(*objects)
        assert caught.value.validation_records == prefix + suffix
    else:
        assert run_diagnostic_validation(*objects) == prefix + suffix
    assert len(calls) == 2 and calls[0][0] == calls[1][0] == objects
    assert calls[0][1]["probe"] is diagnostic.run_prefill_diagnostic
