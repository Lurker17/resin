"""Structural protocols accepted by Resin's native control plane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from resin._resin import BatchPlan, ResourceBinding, RunnerConfig


RunnerOutput: TypeAlias = Sequence[int] | Sequence[Sequence[int]]


class RenderedPrompt(TypedDict):
    """Tokenized conversation returned by a server codec."""

    token_ids: list[int]
    stop_token_ids: list[int]


class ParserEvent(TypedDict, total=False):
    """One incremental semantic event produced by a response parser."""

    type: str
    delta: str
    index: int
    id: str
    name: str


class ResponseParser(Protocol):
    """Incrementally turn generated token IDs into wire-level deltas."""

    def feed(self, token_ids: Sequence[int]) -> list[ParserEvent]: ...


class ConversationCodec(Protocol):
    """Render requests and construct their incremental response parsers."""

    def render(self, request: dict[str, Any]) -> RenderedPrompt: ...

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> ResponseParser: ...


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
    "ConversationCodec",
    "ParserEvent",
    "RenderedPrompt",
    "ResourceRunner",
    "ResponseParser",
    "Runner",
    "RunnerOutput",
    "SyncRunner",
]
