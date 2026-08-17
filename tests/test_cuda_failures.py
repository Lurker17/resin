from __future__ import annotations

import pytest

from resin.backend.cuda.failures import CudaFailureGuard, PostAcceptCudaError


def test_cuda_failure_guard_allows_retry_after_successful_quiescence() -> None:
    synchronized = []
    guard = CudaFailureGuard(lambda: synchronized.append(True), label="runner")

    guard()

    assert synchronized == [True]


def test_cuda_failure_guard_quiesces_only_when_the_guarded_step_fails() -> None:
    synchronized = []
    guard = CudaFailureGuard(lambda: synchronized.append(True), label="runner")

    with guard:
        pass
    with pytest.raises(ValueError, match="forward failed"):
        with guard:
            raise ValueError("forward failed")

    assert synchronized == [True]


def test_cuda_failure_guard_marks_an_unquiesced_launch_post_accept() -> None:
    def fail() -> None:
        raise RuntimeError("synchronize failed")

    guard = CudaFailureGuard(fail, label="runner")
    with pytest.raises(PostAcceptCudaError, match="resources must remain quarantined") as caught:
        guard()

    assert caught.value._resin_post_accept is True
    assert isinstance(caught.value.__cause__, RuntimeError)
