"""Input scoring shares prefill computation, including chunk-boundary logits."""

import numpy as np
import pytest
import torch

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.parallel import batch_plan_from_payload, batch_plan_payload
from resin.backend.sampling import format_output, prompt_scores


class TransitionRunner:
    def __init__(self, *, cached=False, malformed=False):
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=CacheLayout(32, [CacheGroupLayout(2)]) if cached else None,
        )
        self.calls = []
        self.malformed = malformed
        self.table = torch.arange(64, dtype=torch.float32).view(8, 8).sin()

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        pass

    def forward(self, plan):
        self.calls.append(plan)
        hidden = self.table[torch.as_tensor(plan.token_ids.astype(np.int64))]
        last = torch.as_tensor(plan.token_indptr[1:].astype(np.int64) - 1)
        logits = hidden[last]
        if self.malformed:
            return logits.argmax(-1).tolist()
        return format_output(
            logits, logits.argmax(-1), plan, prompt_logprobs=prompt_scores(plan, hidden)
        )


@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
def test_prompt_scores_cover_chunk_successor_and_do_not_change_generation(chunk):
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk))
    tokens = [1, 3, 5, 2, 6]
    request = engine.submit(
        tokens,
        max_new_tokens=2,
        sampling_params=SamplingParams(
            prompt_logprobs=True, logprobs=True, temperature=0
        ),
    )
    while engine.has_work():
        engine.step()
    scores = request.prompt_logprobs
    assert scores is not None and scores[0] is None
    expected = runner.table[tokens[:-1]].log_softmax(-1)[
        range(len(tokens) - 1), tokens[1:]
    ]
    assert [item.logprob for item in scores[1:] if item is not None] == pytest.approx(
        expected.tolist(), abs=1e-6
    )
    assert len(runner.calls) == (len(tokens) + chunk - 1) // chunk + 1
    generated = request.take_result()
    assert generated == [
        runner.table[tokens[-1]].argmax().item(),
        runner.table[generated[0]].argmax().item(),
    ]


def test_prompt_cache_reuse_is_bypassed_only_when_scores_are_requested():
    runner = TransitionRunner(cached=True)
    engine = Engine(runner, prefix_cache=True)
    for scoring in (False, False, True, False):
        request = engine.submit(
            [1, 2, 3, 4, 5],
            max_new_tokens=1,
            sampling_params=SamplingParams(prompt_logprobs=scoring, temperature=0),
        )
        if scoring:
            assert request.cached_tokens == 0
        elif len(runner.calls):
            assert request.cached_tokens > 0
        while engine.has_work():
            engine.step()
        scores = request.prompt_logprobs
        assert (scores is not None) == scoring
        if scores is not None:
            assert scores[0] is None and all(value is not None for value in scores[1:])


def test_single_token_prompt_has_no_distribution_to_score():
    runner = TransitionRunner()
    engine = Engine(runner)
    request = engine.submit(
        [1], max_new_tokens=1, sampling_params=SamplingParams(prompt_logprobs=True)
    )
    engine.step()
    assert request.prompt_logprobs == [None]


@pytest.mark.parametrize("tokens", [[1], [1, 3, 2, 4]])
def test_score_only_request_retires_without_generated_events(tokens):
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    request = engine.submit(
        tokens, max_new_tokens=0, sampling_params=SamplingParams(prompt_logprobs=True)
    )
    while engine.has_work():
        assert engine.step() == []
    assert request.done and request.take_result() == []
    assert request.prompt_logprobs is not None
    assert request.prompt_logprobs[0] is None
    assert all(score is not None for score in request.prompt_logprobs[1:])


def test_score_only_full_prompt_does_not_require_continuation_support():
    runner = TransitionRunner()
    runner.runner_config = RunnerConfig()
    engine = Engine(runner)
    request = engine.submit(
        [1, 2, 3],
        max_new_tokens=0,
        sampling_params=SamplingParams(prompt_logprobs=True),
    )
    assert engine.step() == []
    assert request.done and len(runner.calls) == 1
    assert request.prompt_logprobs is not None
    assert request.prompt_logprobs[1] is not None


def test_missing_requested_prompt_scores_poison_without_partial_progress():
    engine = Engine(TransitionRunner(malformed=True))
    request = engine.submit(
        [1, 2], max_new_tokens=1, sampling_params=SamplingParams(prompt_logprobs=True)
    )
    with pytest.raises(RuntimeError, match="requested prompt logprobs"):
        engine.step()
    assert engine.poisoned and not request.done
    assert request.prompt_logprobs == [None, None]


def test_prompt_metadata_survives_follower_transport():
    plan = BatchPlan(
        [1],
        [0],
        [1, 2],
        [0, 2],
        sampling_params=[SamplingParams(prompt_logprobs=True)],
        prompt_score_indices=[0, 1],
        prompt_score_token_ids=[2, 3],
        prompt_score_indptr=[0, 2],
    )
    restored = batch_plan_from_payload(batch_plan_payload(plan))
    assert restored.sampling_params[0].prompt_logprobs
    assert restored.needs_prompt_logprobs
    np.testing.assert_array_equal(restored.prompt_score_token_ids, [2, 3])
    np.testing.assert_array_equal(restored.prompt_score_indices, [0, 1])


def test_mixed_rows_keep_input_and_output_scores_separate():
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    scored = engine.submit(
        [1, 3, 5],
        max_new_tokens=2,
        sampling_params=SamplingParams(prompt_logprobs=True, logprobs=True),
    )
    ordinary = engine.submit([2, 4], max_new_tokens=2)
    while engine.has_work():
        engine.step()
    scores = scored.prompt_logprobs
    assert scores is not None and scores[0] is None
    expected = runner.table[[1, 3]].log_softmax(-1)[[0, 1], [3, 5]]
    assert [
        score.logprob for score in scores[1:] if score is not None
    ] == pytest.approx(expected.tolist(), abs=1e-6)
    assert ordinary.prompt_logprobs is None


@pytest.mark.parametrize(
    "bad_scores",
    [
        np.array([], np.float32),
        np.array([0.1], np.float32),
        np.array([float("nan")], np.float32),
    ],
)
def test_invalid_prompt_columns_leave_scores_unpublished(bad_scores, monkeypatch):
    from resin import ForwardOutput

    runner = TransitionRunner()
    monkeypatch.setattr(
        runner,
        "forward",
        lambda plan: ForwardOutput(
            np.array([3], np.uint32),
            np.array([0, 1], np.uint32),
            prompt_logprobs=bad_scores,
            prompt_logprob_indptr=np.array([0, len(bad_scores)], np.uint32),
        ),
    )
    engine = Engine(runner)
    request = engine.submit(
        [1, 2], max_new_tokens=1, sampling_params=SamplingParams(prompt_logprobs=True)
    )
    with pytest.raises(RuntimeError, match="prompt"):
        engine.step()
    assert engine.poisoned and not request.done
    assert request.prompt_logprobs == [None, None]


def test_projection_memory_is_bounded_by_chunk_size():
    count = 513
    plan = BatchPlan(
        [1],
        [0],
        [0] * count,
        [0, count],
        prompt_score_indices=list(range(count)),
        prompt_score_token_ids=[1] * count,
        prompt_score_indptr=[0, count],
    )
    rows = []
    hidden = torch.randn(count, 4)
    weight = torch.randn(4, 7)

    def project(values):
        rows.append(len(values))
        return values @ weight

    actual = prompt_scores(plan, hidden, project)
    assert max(rows) == 128 and sum(rows) == count
    torch.testing.assert_close(actual, (hidden @ weight).log_softmax(-1)[:, 1])
