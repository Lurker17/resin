"""Python's literal AST is the oracle for LFM2's safe function-call grammar."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

import resin
from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import Lfm2Recipe

MARKERS = (
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
)


class Lfm2Adapter:
    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens(list(MARKERS))
        self.ids = {marker: backend.token_to_id(marker) for marker in MARKERS}
        self.codec = Lfm2Recipe(backend)
        self.oracles = (("python-literal-ast", self.oracle),)

    @staticmethod
    def tool_text(tool):
        # Python keyword names have a narrower grammar than JSON object keys.
        # Keep the shared generator's arbitrary nested values and give each
        # generated parameter a legal, deterministic keyword name.
        args = ", ".join(
            f"arg_{index}={value!r}"
            for index, value in enumerate(tool["args"].values())
        )
        return f"[{tool['name']}({args})]"

    @classmethod
    def fragments(cls, case, *, terminal=True):
        for part in case.parts:
            if part.kind == "reasoning":
                yield True, "<think>"
                yield False, part.value
                yield True, "</think>"
            elif part.kind == "text":
                yield False, part.value
            else:
                yield True, "<|tool_call_start|>"
                yield False, cls.tool_text(part.value)
                yield True, "<|tool_call_end|>"
        if terminal:
            yield True, "<|im_end|>"

    def encode(self, case, *, terminal=True):
        ids = []
        for special, text in self.fragments(case, terminal=terminal):
            ids.extend(
                [self.ids[text]]
                if special
                else self.ordinary.encode(text, add_special_tokens=False).ids
            )
        return ids

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

    @staticmethod
    def prompt_case(case):
        messages = [{"role": "system", "content": "Helpful."}]
        for part in case.parts:
            if part.kind == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Calling.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": part.value["name"],
                                    "arguments": part.value["args"],
                                }
                            }
                        ],
                    }
                )
                messages.append({"role": "tool", "content": "result"})
            else:
                messages.append({"role": "user", "content": part.value})
                messages.append(
                    {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": part.value,
                    }
                )
        messages.append({"role": "user", "content": "Continue."})
        return {
            "messages": messages,
            # LFM prints declarations verbatim. Match the canonical function
            # envelope represented by a typed Tool on both sides of the oracle.
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
            "extra": {"preserve_thinking": len(case.parts) % 2 == 0},
        }

    @staticmethod
    def prompt_oracle(request):
        return checkpoint_template().render(
            messages=request["messages"],
            tools=request.get("tools"),
            bos_token="<|startoftext|>",
            add_generation_prompt=request.get("add_generation_prompt", True),
            preserve_thinking=request.get("extra", {}).get("preserve_thinking", False),
        )

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        ids = self.encode(case, terminal=False)
        return ids, len(self.encode(prefix, terminal=False)) + 1, len(ids) - 1

    def oracle(self, case):
        parts = []
        for part in case.parts:
            if part.kind != "tool":
                parts.append(part)
                continue
            tree = ast.parse(self.tool_text(part.value), mode="eval")
            assert isinstance(tree.body, ast.List)
            for call in tree.body.elts:
                assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                parts.append(
                    Part(
                        "tool",
                        {
                            "name": call.func.id,
                            "args": {
                                keyword.arg: ast.literal_eval(keyword.value)
                                for keyword in call.keywords
                            },
                        },
                    )
                )
        return normalize(parts)


@cache
def checkpoint_template():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    path = Path(__file__).with_name("references") / "lfm2.jinja"
    return _compile_jinja_template(path.read_text())
