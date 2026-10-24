from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from output_helpers import forward_output

from resin import BatchPhase, BatchPlan, SamplingParams
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import RankGroup
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner


def _runner(block_size: int = 2):
    runner = DeepSeekV4CudaRunner.__new__(DeepSeekV4CudaRunner)
    runner.device = torch.device("cpu")
    runner.vocab_size = 8
    runner.speculative_tokens = block_size
    runner.profile_graph_phases = False
    runner._dspark_phase_events = {}
    runner._graph_device_caches = {}
    runner._graph_logits = {}
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            graph_cache=lambda *args: "graph-cache",
            graph_device_inputs=lambda cache: {"cache": cache},
        ),
    )
    runner._sampler = FlashInferSampler.__new__(FlashInferSampler)
    setattr(
        runner._sampler, "sample_tensors", lambda logits, sampling: logits.argmax(-1)
    )
    broadcasts = []
    replica = RankGroup((2, 3), "original-replica", 1)
    setattr(
        runner,
        "parallel_groups",
        SimpleNamespace(
            device_replica=replica,
            distributed=SimpleNamespace(
                broadcast=lambda values, *, src, group: broadcasts.append(
                    (src, group, values.tolist())
                )
            ),
        ),
    )
    return runner, broadcasts


@pytest.mark.parametrize("block_size", [1, 2, 5])
def test_speculative_graph_has_distinct_expert_capacities_and_replay_masks(block_size):
    runner, broadcasts = _runner(block_size)
    scopes, calls = [], []
    current_scope = None

    @contextmanager
    def capacity(rows, *, valid_rows):
        nonlocal current_scope
        assert current_scope is None
        assert valid_rows.dtype == torch.int32
        current_scope = (rows, int(valid_rows))
        scopes.append(current_scope)
        try:
            yield
        finally:
            current_scope = None

    class Model:
        def draft(self, anchors, positions, *, cache):
            assert current_scope == (3 * block_size, live_rows * block_size)
            calls.append("draft")
            return (anchors + torch.arange(1, block_size + 1)) % runner.vocab_size

        def forward_verify(self, ids, positions, graph_seq_len, *, cache):
            assert current_scope == (
                3 * (block_size + 1),
                live_rows * (block_size + 1),
            )
            assert ids.shape == (3, block_size + 1)
            assert graph_seq_len == 128
            calls.append("verify")
            targets = (ids + 1) % runner.vocab_size
            return torch.nn.functional.one_hot(targets, runner.vocab_size).float(), ids

        def commit_dspark_main_block(self, hidden, positions, *, cache):
            assert current_scope is None
            calls.append("commit")

    runner._parallel_graph_capacity = capacity
    runner._model = Model()
    step = runner._new_dspark_graph_step((3, 128))
    step.inputs["token_ids"].copy_(torch.tensor([[1], [3], [5]]))
    step.inputs["positions"].fill_(7)

    # A common graph bucket remains fixed while one attention-DP replica
    # shrinks, becomes idle, and resumes with a different number of live rows.
    for live_rows in (3, 1, 0, 2):
        step.inputs["ep_valid_rows"].fill_(live_rows)
        step.launch()
        assert scopes[-2:] == [
            (3 * block_size, live_rows * block_size),
            (3 * (block_size + 1), live_rows * (block_size + 1)),
        ]
        assert calls[-3:] == ["draft", "verify", "commit"]
        expected_drafts = (
            step.inputs["token_ids"] + torch.arange(1, block_size + 1)
        ) % runner.vocab_size
        expected_targets = (
            torch.cat((step.inputs["token_ids"], expected_drafts), dim=1) + 1
        ) % runner.vocab_size
        torch.testing.assert_close(step.output[:, :block_size], expected_drafts)
        torch.testing.assert_close(step.output[:, block_size:], expected_targets)
        torch.testing.assert_close(
            runner._graph_logits[(3, 128)],
            torch.nn.functional.one_hot(expected_targets, runner.vocab_size).float(),
        )

    assert len(broadcasts) == 4 * (block_size + 2)
    assert all(
        (source, group) == (2, "original-replica") for source, group, _ in broadcasts
    )


@pytest.mark.parametrize("needs_logprobs", [False, True])
def test_distributed_speculative_graph_selection_includes_verification_lookahead(
    needs_logprobs,
):
    runner, _ = _runner()
    selected, replayed = [], []
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
        sampling_params=[SamplingParams(logprobs=needs_logprobs)] * 2,
    )

    def select(mode, context, **kwargs):
        selected.append((mode, context, kwargs))
        return (4, 128)

    runner._select_distributed_cuda_graph = select
    graph_output = forward_output([[1], [2, 3]])
    runner._graph_forward = lambda *args, **kwargs: (
        replayed.append((args, kwargs)) or graph_output
    )

    used_graph, output = runner._distributed_decode_graph(plan, "context", "cache-plan")

    assert selected == [
        (
            "decode",
            "context",
            {"required_shape": (2, 28)},
        )
    ]
    assert used_graph
    assert output is graph_output
    assert len(replayed) == 1
    assert replayed[0][0] == (plan, "cache-plan")


def test_idle_speculative_replica_votes_before_replaying_scratch_graph():
    runner, _ = _runner()
    runner._decode_graph_capture_inputs = lambda shape: {
        "scratch_shape": shape,
        "ep_valid_rows": shape[0],
    }
    replayed = []
    runner._run_cuda_graph = lambda *args: replayed.append(args)

    def select(mode, context, **kwargs):
        assert kwargs == {"required_shape": (0, 0)}
        return (4, 128)

    runner._select_distributed_cuda_graph = select

    assert runner._distributed_decode_graph(None, "context", None) == (True, None)
    assert replayed == [
        ("decode", (4, 128), {"scratch_shape": (4, 128), "ep_valid_rows": 0})
    ]


@pytest.mark.parametrize("logprobs", [False, True])
def test_speculative_graph_scores_only_accepted_verification_positions(logprobs):
    runner, _ = _runner()
    shape = (4, 128)
    runner._graph_device_caches[shape] = "graph-cache"
    logits = torch.arange(4 * 3 * 8, dtype=torch.float32).view(4, 3, 8)
    runner._graph_logits[shape] = logits
    synchronized = []

    class Event:
        def synchronize(self):
            synchronized.append(self)

        def elapsed_time(self, _other):
            assert synchronized == [events[-1]]
            return 1.0

    events = tuple(Event() for _ in range(4))
    runner._dspark_phase_events[shape] = events
    runner._cache.graph_plan_inputs = lambda cache, plan: {}
    runner._speculation = SpeculativeStats(2)
    cleared = []
    runner._model = SimpleNamespace(
        clear_rejected_speculative_state=lambda positions, counts, block_length, **kwargs: (
            cleared.append((positions.tolist(), counts.tolist(), block_length, kwargs))
        )
    )
    replayed = []

    def replay(mode, shape_arg, inputs):
        replayed.append((mode, shape_arg, inputs))
        return torch.tensor([[2, 6, 2, 4, 7], [3, 4, 3, 4, 5], [0] * 5, [0] * 5])

    runner._run_cuda_graph = replay
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
        sampling_params=[SamplingParams(logprobs=logprobs, seed=1)] * 2,
    )

    output = runner._graph_forward(plan, "cache-plan", shape=shape)

    assert output.token_ids.tolist() == [2, 4, 3, 4, 5]
    assert output.token_indptr.tolist() == [0, 2, 5]
    scores = logits.log_softmax(-1)
    if logprobs:
        assert output.logprobs.tolist() == pytest.approx(
            [
                scores[0, 0, 2],
                scores[0, 1, 4],
                scores[1, 0, 3],
                scores[1, 1, 4],
                scores[1, 2, 5],
            ]
        )
    else:
        assert output.logprobs is None
    assert replayed[0][2]["ep_valid_rows"] == 2
    assert cleared == [([14, 25], [2, 3], 3, {"cache": "graph-cache"})]
