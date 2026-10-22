"""Chart normalization and compatibility checks use synthetic measurements only."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.plot_bench import axes, load_records, main


def record(phase="decode", **changes):
    row = {
        "schema_version": 1,
        "phase": phase,
        "context_length": 4096,
        "batch_size": 8,
        "global_batch_size": 32,
        "gpu_count": 8,
        "attn_dp": 4,
        "prefill_tokens": 0,
        "input_tokens": 0,
        "draft_tokens": 4,
        "accepted_drafts": 1,
        "measured_accept_rate": 0.25,
        "balanced_experts": True,
        "warmup_steps": 1,
        "measure_steps": 2,
        "step_seconds": [0.1, 0.3],
        "window_seconds": 0.4,
        "committed_tokens": 128,
        "metadata": {
            "model": "fixture/model",
            "checkpoint_revision": "fixture-revision",
            "git_commit": "fixture-commit",
            "source_digest": "fixture-source",
            "hardware": "fixture GPU",
            "parallelism": {
                "world_size": 8,
                "attn_dp": 4,
                "attn_tp": 2,
                "cp": 1,
                "ep": 4,
            },
            "graphs": {"prefill": "breakable", "decode": "full"},
            "quant": {"experts": "mxfp4"},
            "expert_backend": "flashinfer",
        },
    }
    if phase == "prefill":
        row.update(input_tokens=40000, prefill_tokens=20000)
    row.update(changes)
    return row


def write(path: Path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_decode_projects_steps_without_relabeling_measured_acceptance(tmp_path):
    row = record()
    rows = load_records([write(tmp_path / "decode.jsonl", row)])
    before = deepcopy(rows)
    assert axes(rows, "decode") == {4096: [(20, 80, "8")]}
    assert axes(rows, "decode", accept_rate=0) == {4096: [(5, 20, "8")]}
    assert rows == before
    assert rows[0]["measured_accept_rate"] == 0.25
    assert rows[0]["committed_tokens"] / 0.4 / 8 == 40


def test_prefill_uses_actual_input_counts_and_measured_step_latency(tmp_path):
    rows = load_records([write(tmp_path / "prefill.jsonl", record("prefill"))])
    assert axes(rows, "prefill") == {4096: [(200, 12500, "20,000")]}


def test_context_curves_and_latest_repeated_point_are_kept_separate(tmp_path):
    original = record()
    repeated = record(window_seconds=0.8)
    other_context = record(context_length=8192)
    another_batch = record(batch_size=16, global_batch_size=64)
    path = write(
        tmp_path / "sweep.jsonl", original, other_context, repeated, another_batch
    )
    rows = load_records([path])
    assert axes(rows, "decode") == {
        4096: [(10, 40, "8"), (20, 160, "16")],
        8192: [(20, 80, "8")],
    }


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "checkpoint_revision",
        "git_commit",
        "source_digest",
        "hardware",
        "parallelism",
        "graphs",
        "quant",
        "expert_backend",
    ],
)
def test_mixed_fixed_metadata_is_rejected_before_duplicate_selection(tmp_path, field):
    first, different = record(), record()
    different["metadata"][field] = (
        {"world_size": 8, "attn_dp": 4, "cp": 2}
        if field == "parallelism"
        else "different"
    )
    path = write(tmp_path / "mixed.jsonl", first, different, first)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "changes",
    [{"draft_tokens": 3}, {"accepted_drafts": 2}, {"balanced_experts": False}],
)
def test_mixed_execution_conditions_are_not_connected(tmp_path, changes):
    path = write(tmp_path / "mixed.jsonl", record(), record(**changes))
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"step_seconds": []},
        {"step_seconds": [0.1, float("nan")]},
        {"window_seconds": 0},
        {"global_batch_size": 8},
        {"gpu_count": 4},
        {"attn_dp": 2},
        {"accepted_drafts": 5},
        {"measured_accept_rate": 1.1},
        {"balanced_experts": 1},
        {"batch_size": True},
    ],
)
def test_invalid_schema_or_measurements_have_source_locations(tmp_path, changes):
    path = write(tmp_path / "bad.jsonl", record(**changes))
    with pytest.raises(ValueError, match=r"bad.jsonl:1:"):
        load_records([path])


def test_prefill_budget_is_not_accepted_as_actual_measured_batch(tmp_path):
    path = write(tmp_path / "bad.jsonl", record("prefill", prefill_tokens=32768))
    with pytest.raises(ValueError, match="measured mean input"):
        load_records([path])


@pytest.mark.parametrize("row", [None, [], "wrong"])
def test_nonobject_json_has_source_location(tmp_path, row):
    path = write(tmp_path / "bad.jsonl", row)
    with pytest.raises(ValueError, match=r"bad.jsonl:1:.*JSON object"):
        load_records([path])


@pytest.mark.parametrize("field", ["prefill_chunk_size", "max_batch_tokens"])
def test_changed_prefill_budget_configuration_is_rejected(tmp_path, field):
    first, second = record(), record()
    first["metadata"][field] = 4096
    second["metadata"][field] = 8192
    path = write(tmp_path / "mixed.jsonl", first, second)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "options",
    [
        {"draft_tokens": 8},
        {"draft_tokens": -1},
        {"accept_rate": -0.1},
        {"accept_rate": 1.1},
        {"accept_rate": float("nan")},
    ],
)
def test_projection_options_are_validated(options):
    with pytest.raises(ValueError):
        axes([record()], "decode", **options)


def test_natural_acceptance_measurements_allow_an_explicit_projection(tmp_path):
    row = record(accepted_drafts=None, measured_accept_rate=0.6)
    rows = load_records([write(tmp_path / "natural.jsonl", row)])
    assert axes(rows, "decode") == {4096: [(20, 80, "8")]}
    assert rows[0]["measured_accept_rate"] == 0.6


def test_plotting_import_does_not_load_matplotlib_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.plot_bench; assert 'matplotlib' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_cli_renders_two_pngs_and_two_svgs(tmp_path, capsys):
    rows = [
        record(),
        record("prefill"),
        record(context_length=8192),
        record("prefill", context_length=8192),
    ]
    path = write(tmp_path / "synthetic.jsonl", *rows)
    output = tmp_path / "plots"
    main([str(path), "--output-dir", str(output)])
    assert {file.name for file in output.iterdir()} == {
        "decode.png",
        "decode.svg",
        "prefill.png",
        "prefill.svg",
    }
    assert all(file.stat().st_size > 1000 for file in output.iterdir())
    svg = (output / "decode.svg").read_text()
    assert "Projected acceptance 75%" in svg
    assert "measured acceptance 25.0%" in svg
    assert "Context 8,192" in svg
    assert "decode.png" in capsys.readouterr().out
