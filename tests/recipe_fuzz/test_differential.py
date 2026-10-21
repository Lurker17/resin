"""Differential grammar fuzzing and arbitrary-byte decoder fuzzing."""

from __future__ import annotations

import json
import os

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from tml_renderers import chat, tokenizers, v0
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from recipe_fuzz.harness import Case, Part, assert_streams, normalize
from resin.recipe import InklingResponseParser, Tokenizer

MARKERS = (
    "message_model",
    "content_text",
    "content_thinking",
    "content_invoke_tool_json",
    "content_invoke_tool_text",
    "content_tool_error",
    "content_model_end_sampling",
    "end_message",
)
LIMIT = int(os.environ.get("RESIN_FUZZ_EXAMPLES", "50"))
FUZZ = settings(max_examples=LIMIT, deadline=None, derandomize=True)
TEXT = st.one_of(
    st.text(min_size=1, max_size=48),
    st.sampled_from(("🦀 café 東京", "\ufffd", "<|end_message|>", "\x00\n\t")),
)
JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**53), 2**53)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=24),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=8,
)
TOOLS = st.builds(
    lambda name, args: Part("tool", {"name": name, "args": args}),
    st.sampled_from(("weather", "lookup", "f", "a_b")),
    st.dictionaries(st.text(max_size=8), JSON_VALUES, max_size=3),
)
PARTS = st.one_of(st.builds(Part, st.sampled_from(("text", "reasoning")), TEXT), TOOLS)
CUTS = st.lists(st.integers(0, 512), max_size=12)


class InklingAdapter:
    """Encode the same grammar independently for HF and TML tokenizers."""

    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens([f"<|{name}|>" for name in MARKERS])
        self.ids = {
            f"<|{name}|>": backend.token_to_id(f"<|{name}|>") for name in MARKERS
        }
        assert all(token_id is not None for token_id in self.ids.values())
        self.tokenizer = Tokenizer(backend)
        assert self.tokenizer.is_native
        self.reference = tokenizers.O200kBaseChatTokenizer()
        _, self.reference_parser = v0.Renderer(self.reference).render_for_completion(
            chat.OpenAIMessage.from_oss_messages([{"role": "user", "content": "hi"}])
        )
        self.oracles = (("tml-v0", self.oracle),)

    @staticmethod
    def fragments(case):
        for part in case.parts:
            yield True, "message_model"
            yield (
                True,
                {
                    "text": "content_text",
                    "reasoning": "content_thinking",
                    "tool": "content_invoke_tool_json",
                }[part.kind],
            )
            payload = (
                json.dumps(part.value, ensure_ascii=False)
                if part.kind == "tool"
                else part.value
            )
            yield False, payload
            yield True, "end_message"

    def encode(self, case, *, terminal=True):
        result = []
        for special, text in self.fragments(case):
            if special:
                result.append(self.ids[f"<|{text}|>"])
            else:
                # No added markers on this tokenizer: literal marker text
                # stays ordinary content rather than changing parser state.
                result.extend(self.ordinary.encode(text, add_special_tokens=False).ids)
        if terminal:
            result.append(self.ids["<|content_model_end_sampling|>"])
        return result

    def oracle(self, case):
        ids = []
        for special, text in self.fragments(case):
            ids.extend(
                [self.reference.encode_special(text)]
                if special
                else self.reference.encode_ordinary(text)
            )
        ids.append(self.reference.encode_special("content_model_end_sampling"))
        parts = []
        for message in self.reference_parser.parse_tokens(ids):
            content = message.content
            if isinstance(content, chat.Text):
                parts.append(Part("text", content.text))
            elif isinstance(content, chat.Thinking):
                parts.append(Part("reasoning", content.text))
            elif isinstance(content, chat.InvokeTool):
                parts.append(
                    Part(
                        "tool",
                        {
                            "name": content.name,
                            "args": {
                                arg.name: json.loads(arg.value) for arg in content.args
                            },
                        },
                    )
                )
            else:
                assert isinstance(content, chat.ModelEndSampling)
        return normalize(parts)

    def parser(self):
        return InklingResponseParser(self.tokenizer, self.ids)

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        ids = self.encode(case, terminal=False)
        payload_start = len(self.encode(prefix, terminal=False)) + 2
        return ids, payload_start, len(ids) - 1

    def text_parser(self):
        return InklingResponseParser(
            self.tokenizer, self.ids, prompt_ids=[self.ids["<|content_text|>"]]
        )


# Each additional format supplies encoding, its real oracle, and parser setup;
# the case generator, semantic normalization, and chunk driver stay shared.
ADAPTERS = {"inkling": InklingAdapter}


@pytest.fixture(scope="module", params=ADAPTERS)
def adapter(request):
    return ADAPTERS[request.param]()


@FUZZ
@given(st.lists(PARTS, min_size=1, max_size=5), TOOLS, CUTS, st.data())
def test_reference_semantics_and_truncated_tools(adapter, parts, tool, cuts, data):
    case = Case(tuple([*parts, tool]))
    # A final tool may be cut at any byte of its JSON, including complete JSON
    # with its closing control token still absent. It must remain uncommitted.
    prefix = Case(tuple(parts))
    ids, first, last = adapter.tool_truncation(case)
    cut = data.draw(st.integers(first, last), label="tool truncation")
    for name, oracle in adapter.oracles:
        note(f"reference oracle: {name}")
        assert_streams(adapter.parser, adapter.encode(case), oracle(case), cuts)
        assert_streams(
            adapter.parser,
            ids[:cut],
            oracle(prefix),
            cuts,
            reason="length",
        )


@FUZZ
@given(st.lists(st.integers(0, 255), max_size=192), CUTS)
def test_native_decoder_matches_full_reference_for_arbitrary_bytes(adapter, ids, cuts):
    # Every ordinary token represents one byte; generated inputs include
    # malformed UTF-8, incomplete suffixes, and literal replacement characters.
    expected = adapter.ordinary.decode(ids, skip_special_tokens=False)
    assert_streams(
        adapter.text_parser,
        ids,
        [Part("text", expected)],
        cuts,
        reason="length",
    )
