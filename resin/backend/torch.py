"""Small PyTorch scopes shared by concrete model backends."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def use_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def benchmark_expert_indices(
    scores: torch.Tensor, top_k: int, replica: int = 0
) -> torch.Tensor:
    """Synthetic round-robin routes for explicitly labeled throughput benchmarks.

    Consecutive assignments cover experts evenly and each token uses distinct
    experts. Keep this shape-only operation capturable with ordinary routing.
    """

    return (
        (
            torch.arange(scores.shape[0] * top_k, device=scores.device)
            + replica * scores.shape[0] * top_k
        )
        % scores.shape[-1]
    ).reshape(-1, top_k)


__all__ = ["benchmark_expert_indices", "use_default_dtype"]
