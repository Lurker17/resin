"""Gemma's pinned HF template is its rendering/value serialization oracle."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers.utils.chat_template_utils import _compile_jinja_template

import resin
from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import Gemma4Recipe

MARKERS = (
    "<bos>",
    "<eos>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|tool>",
    "<tool|>",
    "<|think|>",
    "<|image>",
    "<|image|>",
    "<image|>",
    "<|video|>",
)


def legal_keys(value):
    # Gemma uses unquoted object keys at every nesting level. Keep arbitrary
    # JSON values while projecting generated keys onto that grammar.
    if isinstance(value, dict):
        return {
            f"p{index}": legal_keys(item) for index, item in enumerate(value.values())
        }
    if isinstance(value, list):
        return list(map(legal_keys, value))
    return value


def value_fragments(value):
    if isinstance(value, str):
        yield True, '<|"|>'
        yield False, value
        yield True, '<|"|>'
    elif isinstance(value, dict):
        yield False, "{"
        for index, (key, item) in enumerate(sorted(value.items())):
            yield False, ("," if index else "") + key + ":"
            yield from value_fragments(item)
        yield False, "}"
    elif isinstance(value, list):
        yield False, "["
        for index, item in enumerate(value):
            if index:
                yield False, ","
            yield from value_fragments(item)
        yield False, "]"
    elif value is None:
        yield False, "None"
    elif isinstance(value, bool):
        yield False, "true" if value else "false"
    else:
        yield False, repr(value)


class GemmaAdapter:
    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens(list(MARKERS))
        self.ids = {marker: backend.token_to_id(marker) for marker in MARKERS}
        self.codec = Gemma4Recipe(backend)
        self.template = _compile_jinja_template(
            (Path(__file__).with_name("templates") / "gemma4.jinja").read_text()
        )
        module = self.template.make_module(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "bos_token": "<bos>",
                "add_generation_prompt": False,
            }
        )
        self.value_oracle = module.format_argument
        # No public reference library currently parses Gemma's output grammar.
        # Its official serializer checks each value; source semantic objects
        # then provide a round-trip oracle, independently of the native parser.
        self.oracles = (("hf-template-values", self.oracle),)

    @staticmethod
    def case(case):
        return Case(
            tuple(
                Part(
                    "tool",
                    {
                        "name": part.value["name"],
                        "args": legal_keys(part.value["args"]),
                    },
                )
                if part.kind == "tool"
                else part
                for part in case.parts
            )
        )

    @staticmethod
    def fragments(case, *, terminal=True):
        for part in case.parts:
            if part.kind == "reasoning":
                yield True, "<|channel>"
                yield False, "thought\n"
                yield False, part.value
                yield True, "<channel|>"
            elif part.kind == "text":
                yield False, part.value
            else:
                yield True, "<|tool_call>"
                yield False, "call:" + part.value["name"]
                yield from value_fragments(part.value["args"])
                yield True, "<tool_call|>"
        if terminal:
            yield True, "<turn|>"

    def encode(self, case, *, terminal=True):
        ids = []
        for special, text in self.fragments(case, terminal=terminal):
            ids.extend([self.ids[text]] if special else self.ordinary.encode(text).ids)
        return ids

    def oracle(self, case):
        for part in case.parts:
            if part.kind == "tool":
                rendered = "".join(
                    text for _, text in value_fragments(part.value["args"])
                )
                assert rendered == self.value_oracle(
                    part.value["args"], escape_keys=False
                )
        return normalize(case.parts)

    def parser(self):
        return self.codec.create_parser(
            self.codec.render(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER, [resin.recipe.Text("hi")]
                    )
                ]
            )
        )

    def text_parser(self):
        return self.parser()

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        ids = self.encode(case, terminal=False)
        first = (
            len(self.encode(prefix, terminal=False))
            + 1
            + len(
                self.ordinary.encode("call:" + case.parts[-1].value["name"] + "{").ids
            )
        )
        return ids, first, len(ids) - 1

    @staticmethod
    def prompt_case(case):
        calls = [
            {
                "id": f"c{index}",
                "function": {
                    "name": part.value["name"],
                    "arguments": part.value["args"],
                },
            }
            for index, part in enumerate(case.parts)
            if part.kind == "tool"
        ]
        text = "".join(part.value for part in case.parts if part.kind == "text")
        reasoning = "".join(
            part.value for part in case.parts if part.kind == "reasoning"
        )
        return {
            "messages": [
                {"role": "system", "content": "Helpful."},
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": reasoning,
                    "tool_calls": calls,
                },
                *(
                    {"role": "tool", "tool_call_id": call["id"], "content": "result"}
                    for call in calls
                ),
                {"role": "user", "content": "Continue."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": text,
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                        },
                    },
                }
            ],
            "enable_thinking": len(case.parts) % 2 == 0,
        }

    def prompt_oracle(self, request):
        return self.template.render(
            **request, bos_token="<bos>", add_generation_prompt=True
        )
