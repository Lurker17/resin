"""Static runner protocols accepted by :class:`resin.Engine`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from resin._resin import BatchPlan


class SyncRunner(Protocol):
    """Runner which completes one immutable plan synchronously."""

    def forward(self, plan: BatchPlan) -> Sequence[int]: ...


class AsyncRunner(Protocol):
    """Runner which returns an opaque completion and is later polled."""

    def dispatch(self, plan: BatchPlan) -> object: ...

    def poll(self, completion: object) -> Sequence[int] | None: ...


Runner: TypeAlias = SyncRunner | AsyncRunner

__all__ = ["AsyncRunner", "Runner", "SyncRunner"]
