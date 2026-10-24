from __future__ import annotations

import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.cuda.failures import (
    CudaFailureGuard,
    GraphInputError,
    PostAcceptCudaError,
)


def test_cuda_failure_guard_can_synchronize_failed_graph_work() -> None:
    synchronized = []
    guard = CudaFailureGuard(lambda: synchronized.append(True), label="runner")

    guard()

    assert synchronized == [True]


@pytest.mark.parametrize(
    "error", [ValueError("failed"), KeyboardInterrupt(), SystemExit()]
)
def test_cuda_failure_guard_quiesces_only_when_the_guarded_step_fails(error) -> None:
    synchronized = []
    guard = CudaFailureGuard(lambda: synchronized.append(True), label="runner")

    with guard:
        pass
    with pytest.raises(PostAcceptCudaError, match="cannot be retried") as caught:
        with guard:
            raise error

    assert synchronized == [True]
    assert caught.value.__cause__ is error


def test_host_graph_input_rejection_stays_retryable() -> None:
    synchronized = []
    guard = CudaFailureGuard(lambda: synchronized.append(True), label="runner")

    with pytest.raises(GraphInputError, match="wrong shape"):
        with guard:
            raise GraphInputError("wrong shape")

    assert synchronized == []


def test_engine_quarantines_state_advanced_before_sampling_failure() -> None:
    synchronized = []

    class StatefulRunner:
        runner_config = RunnerConfig(
            cache_layout=CacheLayout(
                4, [CacheGroupLayout(8, family=CacheGroupFamily.STATE)]
            )
        )

        def __init__(self):
            self.state = 0
            self.calls = 0
            self.guard = CudaFailureGuard(
                lambda: synchronized.append(True), label="stateful runner"
            )

        def bind_resources(self, _binding):
            pass

        def forward(self, plan):
            self.calls += 1
            with self.guard:
                # This models the runner's in-place checkpoint update before
                # logits/sampling, within one recurrent checkpoint interval.
                self.state += plan.num_tokens
                if plan.phase == BatchPhase.DECODE:
                    raise ValueError("sampling failed after state update")
                return forward_output([3])

    runner = StatefulRunner()
    engine = Engine(runner)
    engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=2))
    engine.step()
    with pytest.raises(PostAcceptCudaError, match="sampling failed"):
        engine.step()

    assert synchronized == [True]
    assert engine.poisoned
    assert runner.state == 3
    with pytest.raises(RuntimeError, match="resources are quarantined"):
        engine.step()
    assert runner.calls == 2
    assert runner.state == 3


def test_cuda_failure_guard_marks_an_unquiesced_launch_post_accept() -> None:
    def fail() -> None:
        raise RuntimeError("synchronize failed")

    guard = CudaFailureGuard(fail, label="runner")
    with pytest.raises(
        PostAcceptCudaError, match="resources must remain quarantined"
    ) as caught:
        guard()

    assert caught.value._resin_post_accept is True
    assert isinstance(caught.value.__cause__, RuntimeError)
