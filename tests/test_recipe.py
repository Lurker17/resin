from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from resin.protocols import FinalizingResponseParser, ParserEvent
from resin.recipe import InklingResponseParser, Recipe, Tokenizer

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


def test_complete_response_uses_actual_generation_prefix() -> None:
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
        ]
    )
    recipe = Recipe(backend, model="gpt_oss", options={"use_system_prompt": False})
    request = {"messages": [{"role": "user", "content": "hi"}]}
    prompt_ids = recipe.render(request)["token_ids"]
    output = backend.encode(
        "think<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>",
        add_special_tokens=False,
    ).ids
    expected = {"content": "answer", "reasoning_content": "think", "tool_calls": []}
    assert recipe.parse_response(output, request=request) == expected
    assert recipe.parse_response(output, prompt_ids=prompt_ids) == expected
    # Supplied tokens describe the real continuation even when re-rendering
    # the caller's request would produce a different generation prefix.
    assert (
        recipe.parse_response(
            output,
            request={**request, "add_generation_prompt": False},
            prompt_ids=prompt_ids,
        )
        == expected
    )


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


class _CallbackTokenizer:
    """A tokenizer with explicit ordinary encoding and reversible markers."""

    markers = {
        marker: index + 1000
        for index, marker in enumerate(
            (*MARKERS, "<|message_user|>", "<|message_system|>", "<|message_tool|>")
        )
    }

    def __init__(self):
        self.raw_encodes = []

    def encode(self, text, *, add_special_tokens=False):
        self.raw_encodes.append(text)
        return [self.markers[text]] if text in self.markers else list(text.encode())

    def encode_ordinary(self, text):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return self.markers.get(token, 0)

    def convert_ids_to_tokens(self, token_id):
        return next(
            (token for token, value in self.markers.items() if value == token_id),
            "<unk>",
        )

    def decode(self, ids, *, skip_special_tokens=False):
        return bytes(ids).decode(errors="replace")


def test_callback_tokenizer_keeps_ordinary_encoding_and_serving_stops_separate():
    tokenizer = _CallbackTokenizer()
    recipe = Recipe(tokenizer, model="inkling")
    marker = "<|content_model_end_sampling|>"
    request = {"messages": [{"role": "user", "content": marker}]}
    rendered = recipe.render(request)
    assert tokenizer.markers[marker] not in rendered["token_ids"]
    assert tokenizer.raw_encodes == []
    assert recipe.render({**request, "stop": [marker]})["stop_token_ids"] == [
        tokenizer.markers[marker]
    ]
    assert tokenizer.raw_encodes == []


@pytest.mark.parametrize("invalid_id", [0, True])
def test_callback_tokenizer_rejects_unknown_or_boolean_control_ids(
    invalid_id, monkeypatch
):
    tokenizer = _CallbackTokenizer()
    monkeypatch.setattr(tokenizer, "convert_tokens_to_ids", lambda _: invalid_id)
    recipe = Recipe(tokenizer, model="inkling")
    with pytest.raises(ValueError, match="control token"):
        recipe.get_stop_token_ids()


def test_callback_decoder_is_captured_once_instead_of_refetched():
    class TokenizerWithProperty(_CallbackTokenizer):
        reads = 0

        @property
        def decode(self):
            self.reads += 1
            if self.reads > 1:
                raise ValueError("decode accessor cannot be reread")
            return super().decode

    tokenizer = TokenizerWithProperty()
    recipe = Recipe(tokenizer, model="inkling")
    parser = recipe.create_parser({}, [tokenizer.markers["<|content_text|>"]])
    assert parser.feed([ord("x")]) == [{"type": "text_delta", "delta": "x"}]
    assert parser.finish("stop") == []
    assert tokenizer.reads == 1


def test_deepseek_constructor_effort_matches_per_request_effort():
    backend = _byte_tokenizer()
    request = {"messages": [{"role": "user", "content": "hi"}]}
    configured = Recipe(
        backend, model="deepseek_v4", options={"reasoning_effort": "high"}
    )
    per_request = Recipe(backend, model="deepseek_v4")
    expected = per_request.render_text({**request, "reasoning_effort": "high"})
    assert configured.render_text(request) == expected
    assert configured.render_text({**request, "reasoning_effort": None}) == expected
    assert configured.render_text(
        {**request, "reasoning_effort": "low"}
    ) == per_request.render_text({**request, "reasoning_effort": "low"})


def test_inkling_tool_float_preserves_original_value() -> None:
    import json

    backend = _byte_tokenizer()
    parser = InklingResponseParser(Tokenizer(backend), _token_ids(backend))
    # TML 0.2.0 rounds this to 0.9999899864196776; the generated JSON and
    # native parser must retain the exact Python value.
    ids = backend.encode(
        '<|content_invoke_tool_json|>{"name":"weather","args":{"":0.9999899864196777}}<|end_message|>'
    ).ids
    events = parser.feed(ids)
    assert json.loads(events[-1]["delta"]) == {"": 0.9999899864196777}
