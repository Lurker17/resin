from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from resin.protocols import FinalizingResponseParser, ParserEvent
from resin.recipe import InklingResponseParser, Tokenizer

MARKERS = (
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
)


def _byte_tokenizer() -> HuggingFaceTokenizer:
    vocab = {
        character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))
    }
    tokenizer = HuggingFaceTokenizer(BPE(vocab, []))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(list(MARKERS))
    return tokenizer


class _FastTokenizer:
    """Expose the same native tokenizer entry point as Transformers."""

    def __init__(self, backend_tokenizer: HuggingFaceTokenizer) -> None:
        self.backend_tokenizer = backend_tokenizer

    def decode(self, *_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("fast tokenizers must decode in Rust")


def _token_ids(tokenizer: HuggingFaceTokenizer) -> dict[str, int]:
    result = {}
    for marker in MARKERS:
        token_id = tokenizer.token_to_id(marker)
        assert token_id is not None
        result[marker] = token_id
    return result


def _coalesce(events: list[ParserEvent]) -> list[ParserEvent]:
    """Ignore event chunk boundaries when comparing semantic output."""
    result: list[ParserEvent] = []
    for event in events:
        if (
            result
            and event["type"] in ("text_delta", "reasoning_delta")
            and result[-1]["type"] == event["type"]
        ):
            result[-1]["delta"] += event["delta"]
        else:
            result.append(event.copy())
    return result


def test_native_decode_is_equivalent_for_every_chunk_boundary() -> None:
    backend = _byte_tokenizer()
    tokenizer = Tokenizer(_FastTokenizer(backend))
    markers = _token_ids(backend)
    text = (
        "<|content_thinking|>想一想 🧠"
        "<|end_message|><|message_model|><|content_text|>Hi café! \ufffd"
        "<|end_message|><|message_model|><|content_invoke_tool_json|>"
        '{"name":"weather","args":{"city":"東京"}}'
        "<|end_message|><|content_model_end_sampling|>"
    )
    ids = backend.encode(text, add_special_tokens=False).ids

    def parse(chunks: Sequence[Sequence[int]]) -> list[ParserEvent]:
        parser: FinalizingResponseParser = InklingResponseParser(
            tokenizer, markers, prompt_ids=[markers["<|message_model|>"]]
        )
        events: list[ParserEvent] = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish("stop"))
        return _coalesce(events)

    expected: list[ParserEvent] = [
        {"type": "reasoning_delta", "delta": "想一想 🧠"},
        {"type": "text_delta", "delta": "Hi café! \ufffd"},
        {"type": "tool_call_start", "index": 0, "id": "call_0", "name": "weather"},
        {"type": "tool_call_arguments_delta", "index": 0, "delta": '{"city":"東京"}'},
    ]
    assert parse([ids]) == expected
    assert parse([[token_id] for token_id in ids]) == expected
    for split in range(len(ids) + 1):
        assert parse([ids[:split], ids[split:]]) == expected


def test_native_decoder_buffers_partial_utf8_between_feeds() -> None:
    backend = _byte_tokenizer()
    ids = backend.encode("🦀", add_special_tokens=False).ids
    assert len(ids) == 4
    markers = _token_ids(backend)
    parser = InklingResponseParser(
        Tokenizer(_FastTokenizer(backend)),
        markers,
        prompt_ids=[markers["<|content_text|>"]],
    )

    for token_id in ids[:-1]:
        assert parser.feed([token_id]) == []
    assert parser.feed(ids[-1:]) == [{"type": "text_delta", "delta": "🦀"}]
    assert parser.finish("length") == []


def test_native_decoder_flushes_incomplete_utf8_at_finish() -> None:
    backend = _byte_tokenizer()
    markers = _token_ids(backend)
    parser = InklingResponseParser(
        Tokenizer(_FastTokenizer(backend)),
        markers,
        prompt_ids=[markers["<|content_text|>"]],
    )
    incomplete = backend.encode("🦀", add_special_tokens=False).ids[:2]

    assert parser.feed(incomplete) == []
    assert parser.finish("length") == [{"type": "text_delta", "delta": "\ufffd"}]
    assert parser.finish("length") == []


def test_native_tokenizer_is_a_snapshot_and_parsers_have_independent_state() -> None:
    backend = _byte_tokenizer()
    markers = _token_ids(backend)
    tokenizer = Tokenizer(_FastTokenizer(backend))
    backend.decoder = None
    parsers = [
        InklingResponseParser(
            tokenizer, markers, prompt_ids=[markers["<|content_text|>"]]
        )
        for _ in range(2)
    ]
    ids = backend.encode("café", add_special_tokens=False).ids

    assert parsers[0].feed(ids) == [{"type": "text_delta", "delta": "café"}]
    assert parsers[1].feed(ids) == [{"type": "text_delta", "delta": "café"}]
    assert parsers[0].finish("stop") == []
    assert parsers[1].feed(ids) == [{"type": "text_delta", "delta": "café"}]


def test_custom_tokenizer_with_no_native_backend_uses_decode() -> None:
    calls: list[list[int]] = []

    class CustomTokenizer:
        backend_tokenizer = None

        @staticmethod
        def decode(ids: list[int], *, skip_special_tokens: bool) -> str:
            assert not skip_special_tokens
            calls.append(list(ids))
            return "".join(chr(token_id) for token_id in ids)

    tokenizer = Tokenizer(CustomTokenizer())
    assert not tokenizer.is_native
    markers = _token_ids(_byte_tokenizer())
    parser = InklingResponseParser(
        tokenizer, markers, prompt_ids=[markers["<|content_text|>"]]
    )

    assert parser.feed([ord("x")]) == [{"type": "text_delta", "delta": "x"}]
    assert calls == [[ord("x")]]
    assert parser.finish("stop") == []
