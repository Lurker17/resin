"""Failure containment after CUDA execution may have changed model state."""

from __future__ import annotations

from collections.abc import Callable


class GraphInputError(ValueError):
    """Host inputs were rejected before graph copies or execution began."""


class PostAcceptCudaError(RuntimeError):
    """Signal that accepted device work is not safely retryable."""

    _resin_post_accept = True


class CudaFailureGuard:
    """Quarantine failed execution even when its CUDA work has finished.

    Enter after validating the plan and before model work. Synchronization can
    establish memory quiescence, but cannot undo an in-place recurrent-state or
    sliding-window update. Only graph input validation is safe to retry here.
    """

    def __init__(self, synchronize: Callable[[], None], *, label: str) -> None:
        self._synchronize = synchronize
        self._label = label

    def __call__(self) -> None:
        try:
            self._synchronize()
        except BaseException as synchronization_error:
            raise PostAcceptCudaError(
                f"{self._label} failed after launch and CUDA quiescence could "
                "not be established; resources must remain quarantined"
            ) from synchronization_error

    def __enter__(self) -> CudaFailureGuard:
        return self

    def __exit__(self, _type, error, _traceback) -> bool:
        if error is None or isinstance(error, GraphInputError):
            return False
        if isinstance(error, PostAcceptCudaError):
            return False
        self()
        raise PostAcceptCudaError(
            f"{self._label} failed after execution began; persistent state may "
            f"have changed and cannot be retried: {error}"
        ) from error
