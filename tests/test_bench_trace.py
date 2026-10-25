"""Trace export and CLI routing without requiring a development CUDA device."""

import gzip
import json

import pytest
import torch

from resin.bench import BenchCase, perfetto_trace


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_trace_preserves_cpu_diagnostics_but_rejects_missing_cuda(
    tmp_path, monkeypatch, overlap, phase
):
    profile = torch.profiler.profile
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda **kwargs: profile(activities=[torch.profiler.ProfilerActivity.CPU]),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    trace = perfetto_trace(
        tmp_path, 3, {"git_commit": "qualified-source", "overlap": overlap}
    )
    case = BenchCase(phase=phase, context_length=4096, batch_size=256)
    steps = []

    def step():
        steps.append(torch.ones(8).sum())

    with pytest.raises(RuntimeError, match="no CUDA kernel activity"):
        trace(case, step)
    assert len(steps) == 2
    target = tmp_path / f"{phase}-ctx4096-b256-rank003.json.gz"
    with gzip.open(target, "rt") as stream:
        data = json.load(stream)
    assert data["resin_benchmark"]["git_commit"] == "qualified-source"
    assert data["resin_benchmark"]["case"]["batch_size"] == 256
    assert any(
        event.get("name") == f"resin.{phase}-ctx4096-b256-rank003"
        for event in data["traceEvents"]
    )
    assert not target.with_suffix("").exists()
