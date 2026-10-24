from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from resin import BatchPhase, BatchPlan, ForwardOutput
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.speculative import SpeculativeStats
from resin_models_cuda.gemma4 import cuda as gemma_runtime
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
from resin_models_cuda.inkling.cuda import InklingCudaRunner
from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner


class _DeviceTokens:
    """A replay result that requires one bulk host transfer before iteration."""

    def __init__(self, values, copies):
        self.values, self.copies = values, copies

    def __getitem__(self, index):
        return _DeviceTokens(self.values[index], self.copies)

    def __iter__(self):
        raise AssertionError("do not iterate device tokens")

    def cpu(self):
        self.copies.append(tuple(self.values.shape))
        return self.values.clone()


@pytest.mark.parametrize(
    "runner_type", [Gemma4CudaRunner, Qwen3_5CudaRunner, InklingCudaRunner]
)
@pytest.mark.parametrize("draft_tokens", [0, 2])
def test_graph_output_preserves_acceptance_layout_and_owns_tokens(
    monkeypatch, runner_type, draft_tokens
):
    runner: Any = object.__new__(runner_type)
    runner.device = torch.device("cpu")
    runner.speculative_tokens = draft_tokens
    runner._speculation = SpeculativeStats(draft_tokens)
    runner._sampler = FlashInferSampler
    runner.vocab_size = 10
    runner.config = SimpleNamespace(vocab_size=10, hidden_size=2)
    shape = (3, 128)
    runner._select_cuda_graph = lambda _mode, *, match: shape if match(shape) else None
    runner._cuda_graph_mode_enabled = lambda _mode: True
    runner._graph_device_caches = {shape: object()}
    runner._cache = SimpleNamespace(graph_plan_inputs=lambda *_args: {})
    runner._graph_attention = {shape: SimpleNamespace(plan=lambda *_args: {})}
    runner._host_rope_deltas = lambda plan: np.zeros(plan.num_sequences, np.int64)
    runner._graph_page_inputs = runner._spec_graph_page_inputs = (
        lambda *_args, **_kwargs: {}
    )
    runner._prepare_graph_attention = lambda *_args: None
    runner._target_hidden = {7: torch.zeros(2), 8: torch.zeros(2)}
    hidden = torch.arange(18).reshape(3, 3, 2)
    runner._graph_verify_hidden = {shape: hidden}
    monkeypatch.setattr(gemma_runtime, "gemma4_assistant_context", lambda batch: batch)
    monkeypatch.setattr(
        gemma_runtime, "expand_gemma4_decode_batch", lambda batch, _: batch
    )
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    pages = SimpleNamespace(kv_indptr=np.array([0, 1, 2]))
    batch = SimpleNamespace(
        token_ids=plan.token_ids, positions=plan.positions, local=pages, global_=pages
    )
    qwen = runner_type is Qwen3_5CudaRunner
    if draft_tokens:
        values = [[2, 6, 2, 4, 7], [3, 4, 3, 4, 5]]
        expected = [[2, 4], [3, 4, 5]]
        if qwen:
            values = [[1, *values[0]], [6, *values[1]]]
            expected = [[1, *expected[0]], [6, *expected[1]]]
    else:
        values, expected = [[2], [3]], [[2], [3]]
    storage = torch.tensor([*values, [9] * len(values[0])])
    copies = []
    replay = _DeviceTokens(storage, copies) if draft_tokens else storage
    runner._run_cuda_graph = lambda *_args: replay
    if runner_type is Gemma4CudaRunner and draft_tokens:
        output = runner._speculative_graph_forward(plan, batch)
        torch.testing.assert_close(runner._target_hidden[7], hidden[0, 1])
        torch.testing.assert_close(runner._target_hidden[8], hidden[1, 2])
    else:
        output = runner._graph_forward(plan, batch)
    assert isinstance(output, ForwardOutput)
    assert output.token_ids.tolist() == [token for row in expected for token in row]
    assert output.token_indptr.tolist() == [
        0,
        len(expected[0]),
        sum(map(len, expected)),
    ]
    storage.fill_(0)
    assert output.token_ids.tolist() == [token for row in expected for token in row]
    if draft_tokens:
        assert copies == [(2, len(values[0]))]
        assert runner._speculation.accepted == 3
        assert runner._speculation.drafts == 4
