"""Structural protocols accepted by Resin's native control plane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypedDict,
    runtime_checkable,
)

if TYPE_CHECKING:
    from resin._resin import BatchPlan, ForwardOutput, ResourceBinding, RunnerConfig


class MediaTokenSpan(NamedTuple):
    """A half-open media range described by its prompt position and length."""

    position: int
    length: int


@runtime_checkable
class MediaInput(Protocol):
    """Model-owned prompt payload understood by the generic Engine.

    ``cache_key`` is a token-aligned, hashable identity. It normally equals
    ``token_ids`` outside media spans and is digest-salted inside them, allowing
    the Engine's ordinary prefix tree to distinguish equal placeholders backed
    by different media. ``token_spans`` marks ranges that must remain intact in
    one prefill row when a model uses bidirectional attention within media.
    """

    @property
    def token_ids(self) -> Sequence[int]: ...

    @property
    def cache_key(self) -> tuple[int, ...]: ...

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]: ...


class RenderedPrompt(TypedDict):
    """Tokenized conversation returned by a server codec."""

    token_ids: list[int]
    stop_token_ids: list[int]
    media: NotRequired[MediaInput]


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


class FinalizingResponseParser(ResponseParser, Protocol):
    """Flush buffered output before a response ends normally or at its limit."""

    def finish(self, reason: Literal["stop", "length"]) -> list[ParserEvent]: ...


class ConversationCodec(Protocol):
    """Render requests and construct their incremental response parsers."""

    def render(self, request: dict[str, Any]) -> RenderedPrompt: ...

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> ResponseParser: ...


class Runner(Protocol):
    """Execute one Engine-owned immutable plan synchronously."""

    def forward(
        self, plan: BatchPlan, /
    ) -> ForwardOutput | Sequence[int] | Sequence[Sequence[int]]: ...


class ManagedRunner(Runner, Protocol):
    """Declare cache resources and bind their arena once during construction.

    An optional ``release(sequence_ids)`` callback cleans up model-local state
    on completion, cancellation, or retraction. Physical page ownership remains
    with the Engine; release must never independently return cache pages.
    """

    runner_config: RunnerConfig

    def bind_resources(self, resources: ResourceBinding) -> None: ...


__all__ = [
    "ConversationCodec",
    "FinalizingResponseParser",
    "ManagedRunner",
    "MediaInput",
    "MediaTokenSpan",
    "ParserEvent",
    "RenderedPrompt",
    "ResponseParser",
    "Runner",
]
