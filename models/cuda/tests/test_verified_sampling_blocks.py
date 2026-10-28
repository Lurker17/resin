"""Verification samples one complete block without copying vocabulary slices."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.sampling import FusedSampler
from resin.backend.speculative import SpeculativeStats
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.gemma4 import cuda as gemma_runtime
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
from resin_models_cuda.inkling import cuda as inkling_runtime
from resin_models_cuda.inkling.cuda import InklingCudaRunner
from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner


class _BlockSampler:
    graph_inputs = staticmethod(FusedSampler.graph_inputs)

    def __init__(self, expected, targets):
        self.expected = expected
        self.targets = targets
        self.calls = []

    def empty(self, rows, device):
        return object.__new__(FusedSampler).empty(rows, device)

    def __call__(self, logits, plan):
        self._record(logits)
        assert plan.num_sequences == logits.shape[0]
        return self.targets.clone()

    def sample_block(self, logits, sampling, *, out=None, start_step=0):
        self._record(logits)
        assert sampling.seeds.shape == logits.shape[:1]
        assert out is None and start_step == 0
        return self.targets.clone()

    def sample_tensors(self, *_args, **_kwargs):
        raise AssertionError("verification must use one block sampler call")

    def _record(self, logits):
        assert logits.shape == self.expected.shape
        assert logits.stride() == self.expected.stride()
        assert logits.data_ptr() == self.expected.data_ptr()
        self.calls.append(tuple(logits.shape))


def _fixture(runner_type):
    runner: Any = object.__new__(runner_type)
    runner.device = torch.device("cpu")
    runner.speculative_tokens = 2
    runner.vocab_size = 10
    runner.config = SimpleNamespace(vocab_size=10, hidden_size=2, num_hidden_layers=1)
    runner._speculation = SpeculativeStats(2)
    runner._eager_phase_timing = lambda *_args: None
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    logits = torch.arange(2 * 3 * 20, dtype=torch.float32).view(2, 3, 20)[..., ::2]
    drafts = torch.tensor([[2, 6], [3, 4]])
    targets = torch.tensor([[2, 4, 7], [3, 4, 5]])
    runner._sampler = _BlockSampler(logits, targets)
    return runner, plan, logits, drafts, targets


@pytest.mark.parametrize(
    "runner_type", [DeepSeekV41CudaRunner, Gemma4CudaRunner, InklingCudaRunner]
)
def test_eager_verification_samples_entire_strided_block_once(monkeypatch, runner_type):
    runner, plan, logits, drafts, _ = _fixture(runner_type)
    positions = torch.tensor(plan.positions, dtype=torch.int64)
    anchors = torch.tensor(plan.token_ids, dtype=torch.int64)
    broadcasts = []
    if runner_type is DeepSeekV41CudaRunner:
        runner._broadcast_tokens = lambda value: (
            broadcasts.append(tuple(value.shape)) or value
        )
        runner._model = SimpleNamespace(
            draft=lambda *_args, **_kwargs: drafts,
            forward_verify=lambda *_args, **_kwargs: (logits, object()),
            commit_dspark_main_block=lambda *_args, **_kwargs: None,
            clear_rejected_speculative_state=lambda *_args, **_kwargs: None,
        )
        output = runner._speculative_decode(plan, object(), anchors, positions)
        assert broadcasts == [(2, 2), (2, 3)]
    elif runner_type is Gemma4CudaRunner:
        draft_steps = iter(drafts.unbind(1))

        def assistant(*_args, **_kwargs):
            token_ids = next(draft_steps)
            return torch.nn.functional.one_hot(token_ids, 10).float(), torch.zeros(
                2, 2, dtype=torch.bfloat16
            )

        runner._assistant = SimpleNamespace(forward_step=assistant)
        runner._target_hidden, runner._hidden_bank = {}, None
        runner._hidden_free_slots, runner._hidden_slot_generation = [], 0
        runner._hidden_index_cache, runner._hidden_input_buffers = {}, {}
        runner._set_target_hidden([7, 8], torch.zeros(2, 2, dtype=torch.bfloat16))
        runner._cache = SimpleNamespace(
            local_keys=[None],
            local_values=[None],
            global_keys=[None],
            global_values=[None],
        )
        runner._device_plan = lambda value: value
        runner._plan_attention = lambda *_args, **_kwargs: (None, None)
        runner._model = SimpleNamespace(
            embed_tokens=lambda _ids: torch.zeros(2, 2, dtype=torch.bfloat16),
            logits=lambda _hidden: logits.view(6, 10),
        )
        runner._eager_transformer = lambda *_args: torch.zeros(
            6, 2, dtype=torch.bfloat16
        )
        batch = SimpleNamespace(
            token_ids=plan.token_ids.copy(),
            positions=plan.positions.copy(),
            local=None,
            global_=None,
        )
        monkeypatch.setattr(
            gemma_runtime, "gemma4_assistant_context", lambda value, **_kwargs: value
        )
        monkeypatch.setattr(
            gemma_runtime, "expand_gemma4_decode_batch", lambda value, _drafts: value
        )
        output = runner._speculative_decode(plan, batch)
    else:
        runner._draft = SimpleNamespace(
            propose=lambda *_args: (drafts, None),
            inject_target=lambda *_args: None,
        )
        runner._model = SimpleNamespace(
            target=lambda *_args, **_kwargs: (logits, object(), object(), None),
        )
        runner._initial_state = lambda *_args: object()
        runner._store_state = lambda *_args: None
        monkeypatch.setattr(
            inkling_runtime, "select_convolution_state", lambda *_args, **_kwargs: None
        )
        output = runner._speculative_decode(plan, object(), anchors, positions)
    assert runner._sampler.calls == [(2, 3, 10)]
    assert output.token_ids.tolist() == [2, 4, 3, 4, 5]
    assert output.token_indptr.tolist() == [0, 2, 5]
    assert runner._speculation.accepted == 3


def test_deepseek_graph_verification_samples_and_broadcasts_one_block():
    runner, plan, logits, drafts, targets = _fixture(DeepSeekV41CudaRunner)
    runner.parallel_groups = None
    runner.profile_graph_phases = False
    runner._graph_device_caches = {}
    runner._graph_logits = {}
    runner._cache = SimpleNamespace(
        graph_cache=lambda *_args: object(),
        graph_device_inputs=lambda _cache: {},
    )
    runner._new_phase_timing = lambda *_args: None
    runner._parallel_graph_capacity = lambda *_args, **_kwargs: nullcontext()
    runner._reset_model_state = lambda: None
    runner._model = SimpleNamespace(
        draft=lambda *_args, **_kwargs: drafts,
        forward_verify=lambda *_args, **_kwargs: (logits, object()),
        commit_dspark_main_block=lambda *_args, **_kwargs: None,
    )
    broadcasts = []
    runner._broadcast_tokens = lambda value: (
        broadcasts.append(tuple(value.shape)) or value
    )
    # The fake uses the real CPU metadata allocation without loading CUDA kernels.
    step = runner._new_dspark_graph_step((2, 128))
    step.inputs["token_ids"].copy_(
        torch.tensor(plan.token_ids, dtype=torch.int64).view(2, 1)
    )
    step.inputs["positions"].copy_(torch.tensor(plan.positions, dtype=torch.int64))
    step.launch()
    assert runner._sampler.calls == [(2, 3, 10)]
    assert broadcasts == [(2, 2), (2, 3)]
    torch.testing.assert_close(step.output, torch.cat((drafts, targets), dim=1))


@pytest.mark.parametrize("graph", [False, True])
def test_qwen_batches_verification_after_sequential_drafts_and_commits_prefix(graph):
    runner: Any = object.__new__(Qwen3_5CudaRunner)
    runner.device = torch.device("cpu")
    runner.speculative_tokens = 2
    runner._speculation = SpeculativeStats(2)
    runner._eager_phase_timing = runner._new_phase_timing = lambda *_args: None
    plan = BatchPlan(
        sequence_ids=[7, 8, 9],
        context_lengths=[14, 25, 36],
        token_ids=[5, 6, 7],
        token_indptr=[0, 1, 2, 3],
        phase=BatchPhase.DECODE,
    )
    positions = torch.tensor(plan.positions, dtype=torch.int64)
    anchors = torch.tensor(plan.token_ids, dtype=torch.int64)
    # Reject the first draft, reject the second, and accept the complete block.
    targets = torch.tensor([[9, 9, 9], [2, 9, 9], [2, 3, 4]])
    first = torch.ones(3, dtype=torch.int64)
    drafts = torch.tensor([[2, 3], [2, 3], [2, 3]])
    logits = torch.arange(3 * 3 * 20, dtype=torch.float32).view(3, 3, 20)[..., ::2]
    calls = []

    class Sampler:
        graph_inputs = staticmethod(FusedSampler.graph_inputs)

        def empty(self, rows, device):
            values = object.__new__(FusedSampler).empty(rows, device)
            for value in self.graph_inputs(values).values():
                assert isinstance(value, torch.Tensor)
                value.zero_()
            return values

        def single(self, value, offset):
            assert value.shape == (3, 10)
            step = len(calls)
            assert offset == step * 32
            calls.append(("single", offset))
            return first if step == 0 else drafts[:, step - 1]

        def block(self, value, offset):
            assert value.shape == (3, 3, 10)
            assert value.data_ptr() == logits.data_ptr()
            assert value.stride() == logits.stride()
            assert offset == 32
            calls.append(("block", offset))
            return targets.clone()

        def __call__(self, value, _plan, *, offset_delta=0):
            return (self.block if value.ndim == 3 else self.single)(value, offset_delta)

        def sample_tensors(self, value, sampling):
            assert torch.all(sampling.offsets == sampling.offsets[0])
            return self.single(value, int(sampling.offsets[0]))

        def sample_block(self, value, sampling, *, out=None, start_step=0):
            assert out is None and torch.all(sampling.offsets == 0)
            return self.block(value, start_step * 32)

    runner._sampler = Sampler()
    runner._model = SimpleNamespace(
        logits=lambda hidden: logits if hidden.ndim == 3 else torch.zeros(3, 10),
    )
    runner._rope_deltas = lambda _plan: torch.zeros(3, dtype=torch.int64)
    runner._initial_state = lambda *_args: (
        torch.zeros(3, 1, 2),
        torch.zeros(3, 1, 2),
        None,
    )
    cache = SimpleNamespace(row=lambda index: SimpleNamespace(row_index=index))

    def target(_cache, ids, token_positions, recurrent, convolution, **_kwargs):
        hidden = token_positions.float()[..., None].expand(-1, -1, 2).clone()
        return hidden, recurrent + ids.shape[1], convolution + ids.shape[1] * 10

    draft_inputs = []

    def mtp(_cache, ids, _previous_hidden, token_positions, **_kwargs):
        draft_inputs.append(ids.clone())
        return token_positions.float()[..., None].expand(-1, -1, 2).clone()

    runner._target_forward = target
    runner._mtp_forward = mtp
    commits = []
    runner._store_state = lambda _cache, pos, rec, conv, hidden: commits.append(
        (pos.clone(), rec.clone(), conv.clone(), hidden.clone())
    )
    if graph:
        attention = SimpleNamespace(reset=lambda: None, graph_inputs=lambda: {})
        runner._graph_attention_for = lambda _shape: attention
        runner._graph_device_caches = {(3, 128): cache}
        runner._cache = SimpleNamespace(
            graph_device_inputs=lambda _cache: {}, reset_null=lambda: None
        )
        step = runner._new_decode_graph_step((3, 128))
        step.inputs["token_ids"].copy_(anchors)
        step.inputs["positions"].copy_(positions)
        step.inputs["rope_deltas"].zero_()
        step.launch()
        torch.testing.assert_close(
            step.output, torch.cat((first[:, None], drafts, targets), 1)
        )
        assert len(commits) == 1
        committed = commits[0]
    else:
        output = runner._speculative_decode(plan, cache, anchors, positions)
        assert output.token_ids.tolist() == [1, 9, 1, 2, 9, 1, 2, 3, 4]
        assert output.token_indptr.tolist() == [0, 2, 5, 9]
        assert len(commits) == 3
        committed = tuple(
            torch.cat([item[index] for item in commits]) for index in range(4)
        )
    assert calls == [("single", 0), ("single", 32), ("single", 64), ("block", 32)]
    torch.testing.assert_close(draft_inputs[0], first[:, None])
    torch.testing.assert_close(draft_inputs[1], drafts[:, :1])
    processed = torch.tensor([1, 2, 3])
    torch.testing.assert_close(committed[0], positions + processed)
    torch.testing.assert_close(committed[1][:, 0, 0], processed.float() + 1)
    torch.testing.assert_close(committed[2][:, 0, 0], (processed.float() + 1) * 10)
    torch.testing.assert_close(committed[3][:, 0], (positions + processed).float())
