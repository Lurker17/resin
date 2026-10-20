import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ForwardOutput,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.parallel import batch_plan_from_payload, batch_plan_payload
from resin.backend.sampling import TorchSampler, format_output


def scored_output(rows, scores, **kwargs):
    return ForwardOutput(
        np.asarray([token for row in rows for token in row], np.uint32),
        np.asarray([0, *np.cumsum(list(map(len, rows)))], np.uint32),
        logprobs=np.asarray([value for row in scores for value in row], np.float32),
        logprob_indptr=np.asarray([0, *np.cumsum(list(map(len, scores)))], np.uint32),
        **kwargs,
    )


def test_scores_use_model_distribution_and_cover_only_accepted_positions():
    logits = torch.tensor(
        [
            [[1.0, 3.0, 0.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0]],
            [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    plan = BatchPlan(
        [1, 2],
        [0, 0],
        [0, 0],
        [0, 1, 2],
        sampling_params=[
            SamplingParams(temperature=0.1, top_k=1, top_logprobs=2),
            SamplingParams(logprobs=True),
        ],
        sampling_mask=[True, False],
    )
    output = format_output(logits, [[1, 2], [2]], plan)
    expected = logits.log_softmax(-1)
    np.testing.assert_array_equal(output.token_ids, [1, 2, 2])
    np.testing.assert_array_equal(output.token_indptr, [0, 2, 3])
    np.testing.assert_array_equal(output.logprob_indptr, [0, 2, 2])
    for column, token in enumerate([1, 2]):
        assert output.logprobs[column] == pytest.approx(
            expected[0, column, token].item()
        )
        start, end = output.top_indptr[column : column + 2]
        np.testing.assert_array_equal(
            output.top_token_ids[start:end], expected[0, column].topk(2).indices.numpy()
        )
        np.testing.assert_allclose(
            output.top_logprobs[start:end], expected[0, column].topk(2).values.numpy()
        )


def test_logprob_options_survive_broadcast_and_do_not_change_sampling():
    params = SamplingParams(seed=7, top_k=2, top_logprobs=3)
    assert params.logprobs
    plan = BatchPlan([1], [0], [0], [0, 1], sampling_params=[params])
    restored = batch_plan_from_payload(batch_plan_payload(plan))
    assert restored.sampling_params[0].top_logprobs == 3
    assert restored.needs_logprobs
    assert not BatchPlan(
        [1], [0], [0], [0, 1], sampling_params=[params], sampling_mask=[False]
    ).needs_logprobs
    logits = torch.tensor([[2.0, 4.0, 1.0]])
    tokens = TorchSampler()(logits, plan)
    output = format_output(logits, tokens, plan)
    np.testing.assert_array_equal(output.token_ids, tokens.numpy())
    assert len(output.top_logprobs) == 3
    with pytest.raises(ValueError, match="top_logprobs"):
        SamplingParams(top_logprobs=1001)


def test_mixed_top_k_requests_use_packed_columns():
    logits = torch.tensor([[2.0, 4.0, 1.0], [0.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    plan = BatchPlan(
        [1, 2, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 2, 3],
        sampling_params=[
            SamplingParams(top_logprobs=1),
            SamplingParams(logprobs=True),
            SamplingParams(top_logprobs=9),
        ],
    )
    output = format_output(logits, torch.tensor([1, 2, 0]), plan)
    np.testing.assert_array_equal(output.top_indptr, [0, 1, 1, 4])
    np.testing.assert_allclose(
        output.logprobs, logits.log_softmax(-1)[range(3), [1, 2, 0]].numpy()
    )


class ScoredRunner:
    def __init__(self):
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(4, [CacheGroupLayout(2)]),
            cache_write_lookahead_tokens=2,
        )
        self.calls = 0
        self.released = []

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        self.released.extend(ids)

    def forward(self, plan):
        self.calls += 1
        if self.calls == 1:
            return scored_output([[2]], [[-0.1]])
        return scored_output([[7, 99, 8]], [[-0.2, -0.3, -0.4]])


def test_native_events_keep_speculative_scores_aligned_through_eos():
    runner = ScoredRunner()
    engine = Engine(runner)
    request = engine.submit(
        [1],
        max_new_tokens=5,
        stop_token_ids=[99],
        sampling_params=SamplingParams(logprobs=True),
    )
    events = engine.step() + engine.step()
    assert [event.token_id for event in events] == [2, 7, 99]
    assert [event.logprobs.logprob for event in events] == pytest.approx(
        [-0.1, -0.2, -0.3]
    )
    assert [event.index for event in events] == [0, 1, 2]
    assert request.take_result() == [2, 7, 99]
    assert runner.released == [request]
    assert engine.cache_free_parent_count == 4


@pytest.mark.parametrize(
    "scores", [[[-0.2]], [[-0.2, -0.3, float("nan")]], [[-0.2, -0.3, 0.1]], []]
)
def test_invalid_scores_after_stop_still_quarantine_the_whole_batch(scores):
    runner = ScoredRunner()
    original = runner.forward

    def forward(plan):
        return scored_output([[7, 99, 8]], scores) if runner.calls else original(plan)

    runner.forward = forward
    engine = Engine(runner)
    request = engine.submit(
        [1],
        max_new_tokens=5,
        stop_token_ids=[99],
        sampling_params=SamplingParams(logprobs=True),
    )
    engine.step()
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned and not request.done
    assert runner.released == []
    assert engine.cache_occupied_block_count > 0


@pytest.mark.parametrize("offsets", [[1, 3], [0, 4], [0, 2], [0, 3, 2]])
def test_invalid_output_offsets_fail_before_commit(offsets):
    output = ForwardOutput(np.array([3, 4, 5], np.uint32), np.array(offsets, np.uint32))
    engine = Engine(SimpleNamespace(forward=lambda plan: output))
    request = engine.submit([1], max_new_tokens=1)
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned and not request.done


def test_array_handoff_retains_storage_and_events_own_their_scores():
    values = np.array([-0.125], np.float32)
    ids = np.array([3], np.uint32)
    output = ForwardOutput(ids, np.array([0, 1], np.uint32), logprobs=values)
    assert output.token_ids is ids and output.logprobs is values
    assert np.shares_memory(output.logprobs, values)
    sources = []

    def forward(plan):
        local = np.array([-0.5], np.float32)
        sources.append(weakref.ref(local))
        return ForwardOutput(ids, np.array([0, 1], np.uint32), logprobs=local)

    def release(sequence_ids):
        # The ForwardOutput temporary is gone, but Rust still retains its borrows.
        assert sources[0]() is not None

    engine = Engine(SimpleNamespace(forward=forward, release=release))
    engine.submit([1], max_new_tokens=1, sampling_params=SamplingParams(logprobs=True))
    (event,) = engine.step()
    assert sources[0]() is None
    assert event.logprobs.logprob == -0.5


def test_exact_array_dtypes_and_contiguity_are_required_without_coercion():
    with pytest.raises(TypeError):
        ForwardOutput(np.array([3], np.int64), np.array([0, 1], np.uint32))
    output = ForwardOutput(
        np.array([3], np.uint32),
        np.array([0, 1], np.uint32),
        logprobs=np.zeros(4, np.float32)[::2],
    )
    engine = Engine(SimpleNamespace(forward=lambda plan: output))
    engine.submit([1], max_new_tokens=1)
    with pytest.raises(ValueError, match="contiguous"):
        engine.step()
    assert engine.poisoned


def test_top_alternatives_are_read_from_arrays_and_validated_even_past_eos():
    good = scored_output(
        [[3]],
        [[-0.1]],
        top_indptr=np.array([0, 2], np.uint32),
        top_token_ids=np.array([3, 4], np.uint32),
        top_logprobs=np.array([-0.1, -2.0], np.float32),
    )
    engine = Engine(SimpleNamespace(forward=lambda plan: good))
    engine.submit([1], max_new_tokens=1, sampling_params=SamplingParams(top_logprobs=2))
    (event,) = engine.step()
    assert np.array(event.logprobs.top_logprobs) == pytest.approx(
        np.array([[3, -0.1], [4, -2.0]])
    )

    bad = scored_output(
        [[3]],
        [[-0.1]],
        top_indptr=np.array([0, 2], np.uint32),
        top_token_ids=np.array([3], np.uint32),
        top_logprobs=np.array([-0.1, float("nan")], np.float32),
    )
    engine = Engine(SimpleNamespace(forward=lambda plan: bad))
    engine.submit([1], max_new_tokens=1, sampling_params=SamplingParams(top_logprobs=2))
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned


def test_requested_scores_cannot_silently_disappear():
    engine = Engine(SimpleNamespace(forward=lambda plan: [3]))
    engine.submit([1], max_new_tokens=1, sampling_params=SamplingParams(logprobs=True))
    with pytest.raises(RuntimeError, match="requested token logprobs"):
        engine.step()
    assert engine.poisoned


def test_plain_token_outputs_keep_the_original_contract():
    engine = Engine(SimpleNamespace(forward=lambda plan: [3]))
    request = engine.submit([1], max_new_tokens=1)
    (event,) = engine.step()
    assert event.logprobs is None and request.take_result() == [3]
