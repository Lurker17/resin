from types import SimpleNamespace

import pytest
import torch

from resin import CacheGroupLayout, CacheLayout, Engine, RunnerConfig
from resin.backend.cuda.execution import CudaStep
from resin.backend.cuda.failures import CudaFailureGuard, PostAcceptCudaError
from resin.backend.parallel import TensorParallelRunner


class DeferredRunner:
    def __init__(self):
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(2, [CacheGroupLayout(2)])
        )
        self.plans = []
        self.completed = []
        self.released = []
        self.error = None

    def bind_resources(self, binding):
        self.binding = binding

    def forward(self, plan):
        raise AssertionError("Engine should use enqueue when available")

    def enqueue(self, plan):
        self.plans.append(plan)

        def result():
            self.completed.append(plan.batch_id)
            if self.error is not None:
                raise self.error
            return [int(plan.token_ids[-1]) + 1]

        return SimpleNamespace(result=result)

    def release(self, ids):
        self.released.extend(ids)


def test_enqueue_keeps_resources_and_step_collects_the_existing_batch():
    runner = DeferredRunner()
    engine = Engine(runner)
    request = engine.submit([1, 2], max_new_tokens=2)

    assert engine.enqueue_step()
    assert engine.step_in_flight and engine.has_work()
    assert engine.active_request_count == 1 and engine.ready_request_count == 0
    assert runner.completed == runner.released == []
    assert engine.cache_free_parent_count == 1
    with pytest.raises(ValueError, match="already in flight"):
        engine.enqueue_step()
    with pytest.raises(ValueError, match="in-flight"):
        engine.cancel(request)

    events = engine.step()
    assert [event.token_id for event in events] == [3]
    assert len(runner.plans) == 1
    assert not engine.step_in_flight
    engine.step()
    assert request.take_result() == [3, 4]
    assert runner.released == [request]
    assert engine.cache_free_parent_count == 2
    assert not engine.enqueue_step()


@pytest.mark.parametrize("malformed", [False, True])
def test_completion_failure_is_post_accept_even_without_a_marker(malformed):
    runner = DeferredRunner()
    if malformed:
        runner.enqueue = lambda plan: object()
    else:
        runner.error = ValueError("completion failed")
    engine = Engine(runner)
    request = engine.submit([1, 2], max_new_tokens=1)
    assert engine.enqueue_step()
    with pytest.raises((AttributeError, ValueError)):
        engine.complete_step()
    assert engine.poisoned
    assert engine.cache_free_parent_count == 1
    assert not request.done
    assert runner.released == []


def test_close_drains_pending_work_before_releasing_requests():
    runner = DeferredRunner()
    engine = Engine(runner)
    request = engine.submit([1, 2], max_new_tokens=2)
    assert engine.enqueue_step()
    engine.close()
    assert engine.closed and request.done
    assert runner.completed == [1]
    assert runner.released == [request]
    assert request.take_result() == [3]
    assert engine.cache_free_parent_count == 2


def test_tensor_parallel_enqueue_broadcasts_before_launching_the_leader():
    commands = []
    distributed = SimpleNamespace(
        broadcast_object_list=lambda objects, **kw: commands.extend(objects)
    )
    local = DeferredRunner()
    runner = TensorParallelRunner(local, "control", distributed)
    engine = Engine(runner)
    engine.submit([1, 2], max_new_tokens=1)
    assert engine.enqueue_step()
    assert [command["operation"] for command in commands] == [
        "bind_resources",
        "forward",
    ]
    assert local.completed == []
    assert [event.token_id for event in engine.complete_step()] == [3]
    assert commands[-1]["operation"] == "release"


def test_enqueue_rejection_restores_the_cohort_before_retry():
    runner = DeferredRunner()
    original = runner.enqueue
    attempts = []

    def reject(plan):
        attempts.append(plan.batch_id)
        if len(attempts) == 1:
            raise ValueError("before launch")
        return original(plan)

    runner.enqueue = reject
    engine = Engine(runner)
    request = engine.submit([1, 2], max_new_tokens=1)
    with pytest.raises(ValueError, match="before launch"):
        engine.enqueue_step()
    assert not engine.poisoned and not engine.step_in_flight
    assert engine.ready_request_count == 1
    assert engine.enqueue_step()
    assert [event.token_id for event in engine.complete_step()] == [3]
    assert attempts == [1, 2] and request.done


def test_cuda_step_waits_for_the_result_copy_and_collects_once(monkeypatch):
    calls = []
    output = SimpleNamespace(is_cuda=True, device="cuda:0")
    event = SimpleNamespace(
        record=lambda stream: calls.append(("record", stream)),
        query=lambda: False,
        synchronize=lambda: calls.append("wait"),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "execution")
    staging = SimpleNamespace(output=lambda name, source: torch.tensor([7, 8]))
    guard = CudaFailureGuard(lambda: None, label="test")
    plan = object()
    step = CudaStep(plan, output, staging, guard)
    assert not step.ready() and not step.done
    assert step.plan is plan and step.output is output and step.staging is staging
    assert step.result() == step.result() == [7, 8]
    assert step.done
    assert calls == [("record", "execution"), "wait"]

    step.completion.synchronize = lambda: (_ for _ in ()).throw(
        ValueError("device error")
    )
    failed = CudaStep(plan, output, staging, guard)
    with pytest.raises(PostAcceptCudaError, match="cannot be retried"):
        failed.result()
    assert not failed.done
