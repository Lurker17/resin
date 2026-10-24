"""Checkpoint validation observes graph replay without forcing captured tokens."""

from types import SimpleNamespace

import pytest
import torch
from output_helpers import forward_output

from deploy.gpt_oss_parallel_validation import (
    _Capture,
    _expert_comparisons,
    _ObservedRunner,
    _selected_validation_cases,
    _spawn,
    _validation_cases,
)
from resin import BatchPhase, Parallelism
from resin.backend.sampling import TorchSampler


class _GraphRunner:
    def __init__(self, token=2):
        self._model = SimpleNamespace(lm_head=torch.nn.Identity())
        self._sampler = TorchSampler()
        self.token = token
        self.cuda_graph_stats = {
            "prefill": {"replay_count": 0},
            "decode": {"replay_count": 0},
        }

    def forward_distributed(self, plan, context):
        self.cuda_graph_stats["decode"]["replay_count"] += 1
        self._last_graph_logits = torch.tensor([[0.0, 1.0, 3.0], [9.0, 8.0, 7.0]])
        return forward_output([self.token])


def _plan():
    return SimpleNamespace(num_sequences=1, sampling_mask=(True,), sequence_ids=(42,))


def _context():
    return SimpleNamespace(
        step_id=3,
        phase=BatchPhase.DECODE,
        placeholder=False,
        num_sequences=1,
        num_tokens=1,
        replica_sizes=(1, 0),
    )


def test_validation_records_graph_logits_after_prefill_sampler_history():
    runner = _GraphRunner()
    capture = _Capture(runner)
    try:
        capture.begin_history([1, 2], force=False)
        # The prefill sampler runs in Python; decode's sampler is already
        # captured, so its logits are observed only after graph replay.
        tokens = runner._sampler(torch.tensor([[0.0, 3.0, 1.0]]), None)
        capture.record(torch.tensor([[0.0, 3.0, 1.0]]), _plan(), tokens)
        observed = _ObservedRunner(runner, capture)
        result = observed.forward_distributed(_plan(), _context())
        assert result.token_ids.tolist() == [2]
        history = capture.end_history()
        assert len(history) == 2
        torch.testing.assert_close(history[1], torch.tensor([[0.0, 1.0, 3.0]]))
        assert observed.calls[0]["graph_replays"] == {"prefill": 0, "decode": 1}
    finally:
        capture.close()


def test_validation_rejects_graph_tokens_that_change_baseline_context():
    runner = _GraphRunner(token=1)
    capture = _Capture(runner)
    try:
        capture.begin_history([2], force=False)
        with pytest.raises(AssertionError, match="changed baseline context"):
            _ObservedRunner(runner, capture).forward_distributed(_plan(), _context())
    finally:
        capture.close()


def test_validation_still_forces_only_the_eager_numerical_probe():
    runner = _GraphRunner()
    capture = _Capture(runner)
    plan = SimpleNamespace(
        num_sequences=1,
        sampling_mask=(True,),
        sampling_params=(SimpleNamespace(temperature=0, top_p=1, top_k=None, seed=0),),
        token_indptr=(0, 1),
        context_lengths=(1,),
        token_ids=(0,),
    )
    original = runner._sampler
    try:
        capture.begin_history([1])
        tokens = runner._sampler(torch.tensor([[0.0, 1.0, 3.0]]), plan)
        assert tokens.tolist() == [1]
        assert len(capture.end_history()) == 1
        assert runner._sampler is original
    finally:
        capture.close()


def test_composable_validation_matrix_covers_independent_meshes():
    cases = _validation_cases(4, composable=True)
    assert len(cases) == 5
    axes = {
        tuple(case["parallelism"][axis] for axis in ("attn_dp", "attn_tp", "cp", "ep"))
        for case in cases
    }
    assert axes == {
        (1, 2, 1, 2),
        (2, 2, 1, 2),
        (1, 2, 2, 2),
        (2, 1, 2, 2),
        (4, 1, 1, 4),
    }
    for case in cases:
        topology = Parallelism(**case["parallelism"])
        assert topology.expert_tp == (2 if topology.ep == 2 else 1)
    all_axes = _validation_cases(8, composable=True)[-1]
    assert all_axes["name"] == "all-axes"
    topology = Parallelism(**all_axes["parallelism"])
    assert (topology.attn_dp, topology.attn_tp, topology.cp, topology.ep) == (
        2,
        2,
        2,
        2,
    )
    assert topology.expert_tp == 4
    assert [case["parallelism"]["attn_dp"] for case in _validation_cases()] == [1, 2]


def test_two_gpu_composable_matrix_exercises_each_attention_and_expert_pair():
    cases = _validation_cases(2, composable=True)
    assert len(cases) == 6
    axes = {
        tuple(case["parallelism"][axis] for axis in ("attn_dp", "attn_tp", "cp", "ep"))
        for case in cases
    }
    assert axes == {
        (1, 2, 1, 1),
        (1, 1, 2, 1),
        (2, 1, 1, 1),
        (1, 2, 1, 2),
        (1, 1, 2, 2),
        (2, 1, 1, 2),
    }
    assert [Parallelism(**case["parallelism"]).expert_tp for case in cases] == [
        2,
        2,
        2,
        1,
        1,
        1,
    ]


def test_selected_validation_cases_preserve_requested_order_and_default_catalog():
    assert _selected_validation_cases(8, composable=True) == _validation_cases(
        8, composable=True
    )
    cases = _selected_validation_cases(
        8, composable=True, case_names=("all-axes", "ep2-attn-tp2")
    )
    assert [case["name"] for case in cases] == ["all-axes", "ep2-attn-tp2"]
    assert cases[0]["parallelism"] == dict(
        world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2
    )


@pytest.mark.parametrize("names", [(), ("unknown",), ("all-axes", "all-axes")])
def test_selected_validation_cases_reject_empty_unknown_and_duplicate_names(names):
    with pytest.raises(ValueError, match="unique choices"):
        _selected_validation_cases(8, composable=True, case_names=names)


@pytest.mark.parametrize("world_size,composable", [(4, False), (1, True), (3, True)])
def test_validation_rejects_unsupported_allocation(world_size, composable):
    with pytest.raises(ValueError, match="four/eight"):
        _validation_cases(world_size, composable=composable)


def test_validation_passes_world_size_as_spawned_child_argument(monkeypatch, tmp_path):
    from deploy import gpt_oss_parallel_validation as validation

    calls = []

    def spawn(target, **kwargs):
        calls.append((target, kwargs))
        return SimpleNamespace(join=lambda **kwargs: True, processes=[])

    monkeypatch.setattr(validation.torch.multiprocessing, "spawn", spawn)
    target = object()
    case = _validation_cases(4, composable=True)[0]
    _spawn(target, str(tmp_path), "/model", {"short": [1]}, 4, case, True)
    assert calls[0][0] is target
    assert calls[0][1]["nprocs"] == 4
    assert calls[0][1]["args"][4:] == (4, case, True)
    assert calls[0][1]["join"] is False


@pytest.mark.parametrize("rank,expected_rows", [(3, [3, 0]), (7, [3, 5])])
def test_expert_reference_rows_follow_dp_coordinates_not_gpu_rank(rank, expected_rows):
    topology = Parallelism(world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2)

    class Expert:
        def __init__(self):
            self._dispatch = SimpleNamespace(local=lambda hidden, *_: hidden.float())

        def __call__(self, hidden, indices, weights, logits):
            return self._dispatch.local(hidden, indices, weights)

    expert = Expert()
    runner = SimpleNamespace(
        device="cpu",
        _model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mlp=SimpleNamespace(experts=expert))]
            )
        ),
    )
    hidden = torch.arange(6 * 4).reshape(6, 4).float()
    reference = {
        "hidden": hidden,
        "output": hidden,
        "indices": torch.zeros(6, 2, dtype=torch.int32),
        "weights": torch.ones(6, 2),
        "logits": torch.zeros(6, 8),
    }
    records = _expert_comparisons(
        runner, SimpleNamespace(topology=topology, rank=rank), reference
    )
    assert [record["rows"] for record in records] == expected_rows
    assert all(record["passed"] for record in records)
