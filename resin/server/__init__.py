"""HTTP serving for a token-native Resin engine.

The networking, scheduling bridge, wire protocols, and streaming adapters run
in Rust. Stock model formats run in resin-recipe; custom codecs implement the same
structural Python protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from resin._resin import Engine, _HttpServer
from resin.protocols import (
    ConversationCodec,
    ParserEvent,
    RenderedPrompt,
    ResponseParser,
)


def encode_stop_tokens(
    tokenizer: Any, defaults: Sequence[int], stop_strings: Sequence[str]
) -> list[int]:
    """Combine model stops with tokenized user stops, preserving their order."""

    token_ids = list(defaults)
    for stop in stop_strings:
        encoded = list(tokenizer.encode(stop, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                "multi-token stop sequences are not supported yet"
            )
        token_ids.append(int(encoded[0]))
    return list(dict.fromkeys(token_ids))


class HttpServer:
    """Serve one Resin Engine through OpenAI, Anthropic, and demo routes."""

    def __init__(
        self,
        engine: Engine,
        *,
        codec: ConversationCodec,
        model: str | None = None,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
    ) -> None:
        if model is None:
            model = getattr(getattr(codec, "tokenizer", None), "name_or_path", None)
        if not model:
            raise ValueError(
                "model is required when it cannot be inferred from the codec"
            )
        self._native = _HttpServer(
            engine,
            codec,
            model=model,
            enable_openai=enable_openai,
            enable_anthropic=enable_anthropic,
            enable_demo=enable_demo,
        )

    @property
    def model(self) -> str:
        return self._native.model

    @property
    def running(self) -> bool:
        return self._native.running

    def serve(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Block while serving, until Ctrl-C or another thread calls shutdown."""

        self._native.serve(host=host, port=port)

    def shutdown(self) -> None:
        """Request shutdown after the current inference step finishes.

        This does not block the caller. Join the thread running ``serve()`` to
        wait for shutdown; active requests are cancelled and the Engine closes.
        """

        self._native.shutdown()


__all__ = [
    "ConversationCodec",
    "encode_stop_tokens",
    "HttpServer",
    "ParserEvent",
    "RenderedPrompt",
    "ResponseParser",
]
