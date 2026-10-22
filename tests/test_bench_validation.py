"""Checkpoint validation keeps attention-DP drivers synchronized on failures."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from deploy import bench
from deploy import deepseek_v41_validation as validation
from resin import Parallelism
from resin.backend.topology import RankGroup
from resin.models.deepseek_v41 import codec


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    Tokenizer(models.WordLevel({"token": 0})).save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setattr(
        codec,
        "DeepSeekV41Codec",
        lambda tokenizer: SimpleNamespace(render_ids=lambda messages: [2, 3]),
    )
    return tmp_path


@pytest.mark.parametrize("failure", [None, "events", "comparison"])
def test_all_driver_reports_survive_local_and_coordinated_failures(
    checkpoint, monkeypatch, failure
):
    ranks = (0, 4)
    barrier = threading.Barrier(2, timeout=5)
    slots = [None, None]
    progressed = []

    def validate(raw, engine, *, coordinate, **kwargs):
        with validation._validation_stage("eager events", coordinate):
            if raw.rank == 4 and failure == "events":
                raise AssertionError("missing scores")
        progressed.append(raw.rank)
        if raw.rank == 4 and failure == "comparison":
            raise AssertionError("eager/captured tokens differ")
        return {"ok": True, "captured": {"token_ids": [raw.rank + 1]}}

    monkeypatch.setattr(validation, "validate_model", validate)

    def run(index):
        def gather(output, value, *, group):
            slots[index] = value
            barrier.wait()
            output[:] = slots
            barrier.wait()

        rank = ranks[index]
        groups = SimpleNamespace(
            rank=rank,
            drivers=RankGroup(ranks, object(), index),
            distributed=SimpleNamespace(all_gather_object=gather),
        )
        return bench._checkpoint_validation(
            SimpleNamespace(rank=rank),
            object(),
            checkpoint,
            {"model": "fixture"},
            groups,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        reports = [future.result(timeout=10) for future in futures]
    assert reports[0] == reports[1]
    result = reports[0][0]
    assert result["ok"] == (failure is None)
    assert [driver["rank"] for driver in result["drivers"]] == list(ranks)
    if failure == "events":
        assert not progressed  # Neither driver enters another generation step.
        assert all(
            "rank 4: AssertionError: missing scores" in row["error"]
            for row in result["drivers"]
        )
    elif failure == "comparison":
        assert result["drivers"][0]["ok"]
        assert "eager/captured tokens differ" in result["drivers"][1]["error"]
    else:
        assert sorted(progressed) == list(ranks)


def test_single_driver_success_keeps_the_existing_report_shape(checkpoint, monkeypatch):
    monkeypatch.setattr(
        validation,
        "validate_model",
        lambda *args, **kwargs: {"ok": True, "captured": {"token_ids": [7]}},
    )
    assert bench._checkpoint_validation(
        object(), object(), checkpoint, {"model": "fixture"}, None
    ) == [
        {"ok": True, "captured": {"token_ids": [7]}, "metadata": {"model": "fixture"}}
    ]


@pytest.mark.parametrize(
    "model,topology,message",
    [
        ("deepseek-v4", Parallelism(), "currently requires V4.1"),
        ("gpt-oss-20b", Parallelism(), "currently requires V4.1"),
        ("deepseek-v41", Parallelism(), "at least two ranks"),
        (
            "deepseek-v41",
            Parallelism(world_size=8, attn_dp=8, ep=4),
            "ADP=EP=world_size",
        ),
        (
            "deepseek-v41",
            Parallelism(world_size=8, cp=2, attn_dp=4, ep=8),
            "ADP=EP=world_size",
        ),
    ],
)
def test_deepgemm_rejects_wrong_models_and_meshes_before_cuda(
    monkeypatch, model, topology, message
):
    def cannot_inspect_cuda():
        raise AssertionError("invalid options must fail before CUDA inspection")

    monkeypatch.setattr(bench.torch.cuda, "device_count", cannot_inspect_cuda)
    with pytest.raises(ValueError, match=message):
        bench.run_benchmark(
            model, "unused", [], parallelism=topology, expert_backend="deepgemm"
        )


def test_adp8_validation_uses_natural_routing_and_acceptance(monkeypatch):
    monkeypatch.setattr(bench.torch.cuda, "device_count", lambda: 8)
    seen = []

    def spawn(worker, *, args, nprocs, join):
        _, output, model, checkpoint, cases, topology, options = args
        assert topology == Parallelism(world_size=8, attn_dp=8, ep=8)
        assert nprocs == 8 and join
        seen.append(options)
        Path(output).write_text(
            json.dumps(
                [
                    {
                        "ok": True,
                        "drivers": [{"rank": rank, "ok": True} for rank in range(8)],
                    }
                ]
            )
        )

    monkeypatch.setattr(bench.torch.multiprocessing, "spawn", spawn)
    report = bench.run_benchmark(
        "deepseek-v41",
        "unused",
        [{"phase": "decode", "context_length": 64, "batch_size": 1}],
        parallelism=Parallelism(world_size=8, attn_dp=8, ep=8),
        validate_model=True,
        expert_backend="deepgemm",
    )
    assert len(report[0]["drivers"]) == 8
    assert seen[0]["accepted_drafts"] is None
    assert not seen[0]["balanced_experts"]
    assert seen[0]["expert_backend"] == "deepgemm"


def test_validation_cli_saves_all_driver_failures_before_raising(tmp_path, monkeypatch):
    from deploy import modal_bench

    report = [
        {
            "ok": False,
            "drivers": [
                {"rank": 0, "ok": True},
                {"rank": 1, "ok": False, "error": "greedy token mismatch"},
            ],
        }
    ]
    monkeypatch.setattr(
        modal_bench,
        "sweep",
        SimpleNamespace(
            with_options=lambda **kwargs: SimpleNamespace(remote=lambda *args: report)
        ),
    )
    output = tmp_path / "validation.json"
    with pytest.raises(RuntimeError, match="reports saved"):
        modal_bench.main(model="deepseek-v41", validate_model=True, output=str(output))
    assert json.loads(output.read_text()) == report
