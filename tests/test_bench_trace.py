"""Trace export and CLI routing without requiring a development CUDA device."""

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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
        tmp_path,
        3,
        {"git_commit": "qualified-source", "overlap": overlap, "escape_token": '<|"|>'},
        decode_passes=2 + int(overlap),
    )
    assert getattr(trace, "decode_passes") == 2 + int(overlap)
    case = BenchCase(phase=phase, context_length=4096, batch_size=256)
    steps = []

    def step():
        steps.append(torch.ones(8).sum())

    with pytest.raises(RuntimeError, match="no CUDA kernel activity"):
        trace(case, step)
    passes = 2 + int(overlap and phase == "decode")
    assert len(steps) == passes
    target = tmp_path / f"{phase}-ctx4096-b256-rank003.json.gz"
    with gzip.open(target, "rt") as stream:
        data = json.load(stream)
    assert data["resin_benchmark"]["git_commit"] == "qualified-source"
    assert data["resin_benchmark"]["escape_token"] == '<|"|>'
    assert data["resin_benchmark"]["case"]["batch_size"] == 256
    assert data["resin_benchmark"]["trace_passes"] == passes
    unit = "decode completions" if phase == "decode" else "prefill cohorts"
    assert data["resin_benchmark"]["trace_unit"] == unit
    assert data["resin_timing"] == f"{passes} untimed {unit} after measurement"
    assert any(
        event.get("name") == f"resin.{phase}-ctx4096-b256-rank003"
        for event in data["traceEvents"]
    )
    assert not target.with_suffix("").exists()


@pytest.mark.parametrize("passes", [0, -1, True, 1.5])
def test_trace_rejects_invalid_pass_counts(tmp_path, passes):
    with pytest.raises(ValueError, match="decode_passes must be a positive integer"):
        perfetto_trace(tmp_path, decode_passes=passes)


@pytest.mark.parametrize("outcome", ["success", "no_kernels", "export_error"])
def test_trace_stages_raw_export_locally_and_cleans_up(tmp_path, monkeypatch, outcome):
    output = tmp_path / "shared-volume"
    exported = []
    payload = {"traceEvents": [{"cat": "kernel"}] if outcome == "success" else []}

    def export(path):
        raw = Path(path)
        exported.append(raw)
        assert raw.parent != output
        assert not list(output.iterdir())
        raw.write_text(json.dumps(payload))
        if outcome == "export_error":
            raise OSError("export interrupted")

    @contextmanager
    def profile(**kwargs):
        yield SimpleNamespace(
            add_metadata=lambda name, value: None,
            add_metadata_json=lambda name, value: None,
            export_chrome_trace=export,
        )

    monkeypatch.setattr(torch.profiler, "profile", profile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    trace = perfetto_trace(output, rank=2)
    case = BenchCase(phase="decode", context_length=8192, batch_size=8)
    if outcome == "success":
        trace(case, lambda: None)
    else:
        error = OSError if outcome == "export_error" else RuntimeError
        message = "export interrupted" if outcome == "export_error" else "no CUDA"
        with pytest.raises(error, match=message):
            trace(case, lambda: None)

    assert len(exported) == 1
    assert not exported[0].parent.exists()
    if outcome == "export_error":
        assert not list(output.iterdir())
    else:
        target = output / "decode-ctx8192-b8-rank002.json.gz"
        assert list(output.iterdir()) == [target]
        with gzip.open(target, "rt") as stream:
            data = json.load(stream)
        assert data["traceEvents"] == payload["traceEvents"]
        assert data["resin_benchmark"]["case"]["phase"] == "decode"
