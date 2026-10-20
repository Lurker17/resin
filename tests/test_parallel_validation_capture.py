"""Checkpoint validation observes graph replay without forcing captured tokens."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deploy.gpt_oss_parallel_validation import _Capture, _ObservedRunner
from resin import BatchPhase
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
        return SimpleNamespace(token_ids=np.array([self.token], dtype=np.uint32))


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
