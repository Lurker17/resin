from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

import pytest

from resin import CacheGroupLayout, CacheLayout, Engine, RunnerConfig


@dataclass
class Feedback:
    tokens: list[int]
    positions: list[int]


class Completion:
    def __init__(self, rows, feedback, *, prior_harvest=None, submitted=None):
        self.feedback = feedback
        self.rows = rows
        self.prior_harvest = prior_harvest
        self.submitted = submitted
        self.future: Future | None = None
        self.error: BaseException | None = None

    def result(self):
        if self.error is not None:
            raise self.error
        if self.submitted is not None:
            assert self.submitted.wait(2), (
                "next GPU submission must precede previous CPU harvest"
            )
        if self.prior_harvest is not None:
            self.prior_harvest.set()
        return self.future.result(timeout=2) if self.future else self.rows


class AsyncRunner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(16, [CacheGroupLayout(128)]),
        cache_write_lookahead_tokens=2,
    )

    def __init__(self):
        self.plans = []
        self.released = []
        self.pending = []
        self.completed_before_release = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.next_submitted = Event()
        self.prior_harvest = Event()
        self.require_overlap = False
        self.fail = False

    def bind_resources(self, binding):
        self.binding = binding

    def forward(self, plan):
        self.plans.append(plan)
        return [
            [int(plan.token_ids[int(end) - 1]) + 1] for end in plan.token_indptr[1:]
        ]

    def enqueue(self, plan, *, feedback=None):
        self.plans.append(plan)
        tokens = plan.token_ids.tolist() if feedback is None else feedback.tokens
        positions = plan.positions.tolist() if feedback is None else feedback.positions
        rows = [[int(token) + 1, int(token) + 2] for token in tokens]
        completion = Completion(
            rows,
            Feedback(
                [row[-1] for row in rows], [position + 2 for position in positions]
            ),
        )
        if self.require_overlap and not self.pending:
            completion.submitted = self.next_submitted
            completion.prior_harvest = self.prior_harvest
        if self.require_overlap and feedback is not None:

            def device_work():
                self.next_submitted.set()
                assert self.prior_harvest.wait(2), (
                    "GPU work and previous CPU harvest must overlap"
                )
                return rows

            completion.future = self.executor.submit(device_work)
        if self.fail:
            completion.error = RuntimeError("asynchronous device failure")
        self.pending.append(completion)
        return completion

    def release(self, ids):
        self.completed_before_release.append(
            all(value.future is None or value.future.done() for value in self.pending)
        )
        self.released.extend(ids)

    def shutdown(self):
        self.executor.shutdown(wait=True)


def test_next_device_submission_precedes_previous_cpu_harvest():
    runner = AsyncRunner()
    runner.require_overlap = True
    engine = Engine(runner)
    request = engine.submit([1], max_new_tokens=24)
    assert [event.token_id for event in engine.step()] == [2]
    assert engine.step() == []
    assert engine.overlap_pending
    assert [event.token_id for event in engine.step()] == [3, 4]
    assert runner.next_submitted.is_set() and runner.prior_harvest.is_set()
    assert engine.overlap_stats == (2, 1)
    engine.close()
    assert request.cancelled
    assert request.take_result() == [2, 3, 4, 5, 6]
    assert all(runner.completed_before_release)
    runner.shutdown()


@pytest.mark.parametrize("overlap", [True, False])
def test_reference_switch_preserves_single_step_for_legacy_runners(overlap):
    class Legacy:
        def forward(self, plan):
            return [int(token) + 1 for token in plan.token_ids]

    engine = Engine(Legacy(), overlap=overlap)
    request = engine.submit([1], max_new_tokens=8)
    assert [event.token_id for event in engine.step()] == [2]
    assert not engine.overlap_pending
    while engine.has_work():
        assert len(engine.step()) == 1
    assert request.take_result() == list(range(2, 10))


def test_cancel_drains_accepted_work_and_preserves_other_request_events():
    runner = AsyncRunner()
    engine = Engine(runner)
    first = engine.submit([1], max_new_tokens=24)
    second = engine.submit([10], max_new_tokens=24)
    engine.step()
    assert engine.step() == []
    assert engine.cancel(first)
    assert first.cancelled and first.take_result() == [2, 3, 4]
    assert not engine.overlap_pending
    events = engine.step()
    assert [(event.request_id, event.token_id) for event in events[:2]] == [
        (second.id, 12),
        (second.id, 13),
    ]
    engine.close()
    assert all(runner.completed_before_release)
    runner.shutdown()


def test_async_failure_quarantines_the_original_writer_lease():
    runner = AsyncRunner()
    engine = Engine(runner)
    request = engine.submit([1], max_new_tokens=24)
    engine.step()
    runner.fail = True
    engine.step()
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        engine.step(allow_overlap=False)
    assert engine.poisoned and not request.done
    assert runner.released == []
    assert engine.cache_occupied_block_count == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        engine.close()
    runner.shutdown()


def test_stop_conditions_and_short_budgets_use_synchronous_reference():
    for stop, limit in (([999], 24), (None, 4)):
        runner = AsyncRunner()
        engine = Engine(runner)
        request = engine.submit([1], max_new_tokens=limit, stop_token_ids=stop)
        engine.step()
        assert engine.step()
        assert not runner.pending and not engine.overlap_pending
        engine.close()
        assert request.done
        runner.shutdown()


def test_gpu_feedback_matches_reference_across_page_and_budget_drains():
    outputs = []
    for overlap in (False, True):
        runner = AsyncRunner()
        engine = Engine(runner, overlap=overlap, context_length=1024)
        request = engine.submit([1] * 123, max_new_tokens=270)
        events = []
        while engine.has_work():
            events.extend(engine.step())
        outputs.append([event.token_id for event in events])
        assert request.take_result() == outputs[-1]
        assert engine.cache_free_parent_count == 16
        assert (
            engine.overlap_stats[1] > 0 if overlap else engine.overlap_stats == (0, 0)
        )
        engine.close()
        runner.shutdown()
    assert outputs[0] == outputs[1] == list(range(2, 272))


@pytest.mark.parametrize("second_enqueue", [False, True])
def test_completion_inspection_failure_quarantines_accepted_work(
    monkeypatch, second_enqueue
):
    import gc
    import weakref

    class InvalidCompletion:
        @property
        def result(self):
            raise RuntimeError("completion descriptor failed")

    runner = AsyncRunner()
    original = runner.enqueue
    references = []
    calls = 0

    def enqueue(plan, *, feedback=None):
        nonlocal calls
        calls += 1
        if calls == (2 if second_enqueue else 1):
            result = InvalidCompletion()
            references.append(weakref.ref(result))
            return result
        return original(plan, feedback=feedback)

    monkeypatch.setattr(runner, "enqueue", enqueue)
    engine = Engine(runner)
    request = engine.submit([1], max_new_tokens=24)
    engine.step()
    if second_enqueue:
        assert engine.step() == []
    with pytest.raises(RuntimeError, match="completion descriptor failed"):
        engine.step()
    gc.collect()
    assert references[0]() is not None  # Its GPU buffer owner stays quarantined.
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    runner.shutdown()


def test_feedback_descriptor_failure_cannot_leave_a_healthy_pending_engine(monkeypatch):
    class InvalidFeedback:
        @property
        def feedback(self):
            raise RuntimeError("feedback descriptor failed")

        def result(self):
            return [[3, 4]]

    runner = AsyncRunner()
    monkeypatch.setattr(runner, "enqueue", lambda plan: InvalidFeedback())
    engine = Engine(runner)
    request = engine.submit([1], max_new_tokens=24)
    engine.step()
    assert engine.step() == []
    with pytest.raises(RuntimeError, match="feedback descriptor failed"):
        engine.step()
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    runner.shutdown()
