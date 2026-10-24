"""Checkpoint image slots match the released Rust recipe, including token trust."""

from __future__ import annotations

import os

import pytest
from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    SpecialTokens,
)
from deepseek_recipe import Tokenizer as DeepSeekTokenizer
from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from recipe_fuzz.deepseek import MARKERS
from resin.recipe import Recipe

IMAGE = "<｜deepseek_image｜>"
BLOCKS = st.one_of(
    st.none(),
    st.text(max_size=32),
    st.sampled_from((f"literal {IMAGE}", "🦀 café 東京", "\n\n", "")),
)


@pytest.fixture(scope="module")
def encoders():
    vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
    tokenizer = Tokenizer(BPE(vocab, []))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens([*MARKERS, IMAGE])
    specials = SpecialTokens(image=IMAGE)
    reference = (
        DeepseekV41Encoding()
        .with_special_tokens(specials)
        .with_tokenizer(DeepSeekTokenizer.from_str(tokenizer.to_str()))
    )
    return (
        Recipe(tokenizer, model="deepseek_v41"),
        reference,
        ConversionOptions().with_special_tokens(specials),
    )


@settings(
    max_examples=int(os.environ.get("RESIN_FUZZ_EXAMPLES", "40")),
    deadline=None,
    derandomize=True,
)
@given(
    st.lists(st.lists(BLOCKS, min_size=1, max_size=5), min_size=1, max_size=3),
    st.booleans(),
)
def test_v41_image_blocks_match_reference_text_and_token_ids(encoders, turns, thinking):
    native, reference, options = encoders
    messages, oracle_messages = [], []
    for turn in turns:
        blocks, oracle_blocks = [], []
        for block in turn:
            blocks.append(
                {"type": "image"} if block is None else {"type": "text", "text": block}
            )
            oracle_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }
                if block is None
                else {"type": "text", "text": block}
            )
        if messages:
            assistant = {
                "role": "assistant",
                "content": "previous answer",
                "reasoning_content": "previous reasoning",
            }
            messages.append(assistant)
            oracle_messages.append(assistant)
        messages.append({"role": "user", "content": blocks})
        oracle_messages.append({"role": "user", "content": oracle_blocks})
    converted = ChatCompletionRequest(
        {
            "model": "DeepSeek-V4.1-Flash",
            "messages": oracle_messages,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
    ).convert(options)
    request = {
        "messages": messages,
        "extra": {"thinking_mode": "thinking" if thinking else "chat"},
    }
    expected = reference.render_conversation(converted.conversation)
    assert native.render_text(request) == expected.prompt
    assert native.render(request)["token_ids"] == reference.encode(
        converted.conversation
    )
