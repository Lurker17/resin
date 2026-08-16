"""Static runner protocols accepted by :class:`resin.Engine`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from resin._resin import BatchPlan, ResourceBinding, RunnerConfig


RunnerOutput: TypeAlias = Sequence[int] | Sequence[Sequence[int]]


class SyncRunner(Protocol):
    """Runner which executes one Engine-owned immutable plan synchronously."""

    def forward(self, plan: BatchPlan) -> RunnerOutput: ...


class AsyncRunner(Protocol):
    """Runner which schedules kernels and exposes an opaque completion."""

    def dispatch(self, plan: BatchPlan) -> object: ...

    def poll(self, completion: object) -> RunnerOutput | None: ...


class ResourceRunner(Protocol):
    """Optional persistent-resource declaration shared by either runner form."""

    runner_config: RunnerConfig

    def bind_resources(self, binding: ResourceBinding) -> int | None: ...


Runner: TypeAlias = SyncRunner | AsyncRunner

__all__ = [
    "AsyncRunner",
    "ResourceRunner",
    "Runner",
    "RunnerOutput",
    "SyncRunner",
]
