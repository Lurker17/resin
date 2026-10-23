"""Trace export and CLI routing without requiring a development CUDA device."""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from deploy.bench_trace import decode_trace
from resin.bench import BenchCase


def test_trace_preserves_cpu_diagnostics_but_rejects_missing_cuda(
    tmp_path, monkeypatch
):
    profile = torch.profiler.profile
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda **kwargs: profile(activities=[torch.profiler.ProfilerActivity.CPU]),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    trace = decode_trace(tmp_path, 3, {"git_commit": "qualified-source"})
    case = BenchCase(phase="decode", context_length=4096, batch_size=256)
    with pytest.raises(RuntimeError, match="no CUDA kernel activity"):
        trace(case, lambda: torch.ones(8).sum())
    target = tmp_path / "decode-ctx4096-b256-rank003.json.gz"
    with gzip.open(target, "rt") as stream:
        data = json.load(stream)
    assert data["resin_benchmark"]["git_commit"] == "qualified-source"
    assert data["resin_benchmark"]["case"]["batch_size"] == 256
    assert any(
        event.get("name") == "resin.decode-ctx4096-b256-rank003"
        for event in data["traceEvents"]
    )
    assert not target.with_suffix("").exists()


def test_trace_cli_separates_remote_storage_from_local_output(tmp_path, monkeypatch):
    from deploy import modal_bench

    seen = []
    output = tmp_path / "decode.jsonl"
    destination = tmp_path / "traces"

    def remote(model, cases, topology, options):
        assert options["trace_dir"].startswith("/bench-traces/")
        assert str(destination) not in options["trace_dir"]
        seen.append(options["trace_dir"])
        return [{"timed": True}]

    def download(remote_directory, local_directory):
        assert remote_directory == seen[0]
        assert local_directory == str(destination)
        assert json.loads(output.read_text()) == {"timed": True}
        return [str(destination / "decode-all-ranks.json.gz")]

    monkeypatch.setattr(
        modal_bench,
        "sweep",
        SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(remote=remote)),
    )
    monkeypatch.setattr(modal_bench, "download_traces", download)
    modal_bench.main(
        model="deepseek-v41",
        phases="decode",
        trace_dir=str(destination),
        output=str(output),
    )


@pytest.mark.parametrize("validate,phase", [(True, "decode"), (False, "prefill")])
def test_trace_requires_a_decode_benchmark_before_cuda(monkeypatch, validate, phase):
    from deploy.bench import run_benchmark

    def unexpected_cuda():
        raise AssertionError("invalid trace options must fail before CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", unexpected_cuda)
    with pytest.raises(ValueError, match="traces require a decode benchmark"):
        run_benchmark(
            "deepseek-v41",
            "unused",
            [{"phase": phase, "context_length": 64, "batch_size": 1}],
            validate_model=validate,
            trace_dir="unused",
        )
