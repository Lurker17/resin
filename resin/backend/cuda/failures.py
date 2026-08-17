"""Conservative CUDA quiescence after a synchronous runner failure."""

from __future__ import annotations

from collections.abc import Callable


class PostAcceptCudaError(RuntimeError):
    """Signal that accepted device work is not safely retryable."""

    _resin_post_accept = True


class CudaFailureGuard:
    """Synchronize failed CUDA work or mark its resources for quarantine."""

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
        if error is not None:
            self()
        return False
