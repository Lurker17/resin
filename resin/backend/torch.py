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
    scores: torch.Tensor, top_k: int, replica: int = 0, expert_parallel_size: int = 1
) -> torch.Tensor:
    """Synthetic round-robin routes for explicitly labeled throughput benchmarks.

    Assign consecutive routes across contiguous expert owners first, then across
    their experts, so small batches also balance communication. This permutation
    retains distinct token routes and evenly covers individual experts.
    """

    experts_per_owner = scores.shape[-1] // expert_parallel_size
    ordinal = (
        torch.arange(scores.shape[0] * top_k, device=scores.device)
        + replica * scores.shape[0] * top_k
    )
    return (
        (ordinal % expert_parallel_size) * experts_per_owner
        + (ordinal // expert_parallel_size) % experts_per_owner
    ).reshape(-1, top_k)


__all__ = ["benchmark_expert_indices", "use_default_dtype"]
