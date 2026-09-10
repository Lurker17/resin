"""OpenAI-wire adapter for canonical Gemma 4 media arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from resin.models.gemma4.media import Gemma4Media
from resin.server.media import normalize_image_messages


class Gemma4Codec:
    """Render HTTP conversations into one opaque ``Gemma4Media`` request."""

    def __init__(self, processor: Any, *, eos_token_ids: Sequence[int]) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.eos_token_ids = tuple(int(token) for token in eos_token_ids)

    def render(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = normalize_image_messages(request["messages"])
        media = Gemma4Media.from_messages(
            self.processor,
            messages,
            tools=request.get("tools") or None,
            processor_kwargs={"do_rescale": False},
        )
        stop_token_ids = list(self.eos_token_ids)
        for stop in request.get("stop") or []:
            encoded = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"stop sequence {stop!r} encodes to {len(encoded)} tokens; "
                    "multi-token stop sequences are not supported yet"
                )
            stop_token_ids.append(int(encoded[0]))
        return {
            "token_ids": list(media.token_ids),
            "stop_token_ids": list(dict.fromkeys(stop_token_ids)),
            "media": media,
        }

    def create_parser(
        self, request: dict[str, Any], prompt_ids: Sequence[int]
    ) -> Gemma4ResponseParser:
        del request, prompt_ids
        return Gemma4ResponseParser(self.tokenizer)


class Gemma4ResponseParser:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.text = ""

    def feed(self, token_ids: Sequence[int]) -> list[dict[str, str]]:
        self.token_ids.extend(int(token) for token in token_ids)
        text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        text = text.rstrip("\N{REPLACEMENT CHARACTER}")
        if not text.startswith(self.text):
            raise ValueError("Gemma 4 decoded output is not prefix-stable")
        delta = text[len(self.text) :]
        self.text = text
        return [] if not delta else [{"type": "text_delta", "delta": delta}]


__all__ = ["Gemma4Codec", "Gemma4ResponseParser"]
