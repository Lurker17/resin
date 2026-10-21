"""DeepSeek's official Rust parser is the primary DSML semantic oracle."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import cache
from pathlib import Path

from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    InferenceChunk,
    InferenceFinishReason,
    ParsingOptions,
    StreamProcessor,
    ThinkingInitialStage,
)
from deepseek_recipe import (
    Tokenizer as DeepSeekTokenizer,
)
from tml_renderers import chat, renderers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import Recipe

MARKERS = (
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
)


class DeepSeekAdapter:
    commits_tools_at_close = False

    @staticmethod
    def case(case):
        # The official DSML stream parser writes attribute names directly into
        # JSON; its public oracle grammar requires identifier parameter names.
        # Native regressions separately cover quoted/unusual parameter names.
        return Case(
            tuple(
                Part(
                    "tool",
                    {
                        "name": part.value["name"],
                        "args": {
                            f"arg_{index}": value
                            for index, value in enumerate(part.value["args"].values())
                        },
                    },
                )
                if part.kind == "tool"
                else part
                for part in case.parts
            )
        )

    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens(list(MARKERS))
        self.ids = {marker: backend.token_to_id(marker) for marker in MARKERS}
        self.codec = Recipe(backend, model="deepseek_v4")
        self.reference = DeepSeekTokenizer.from_str(backend.to_str())
        self.tml = renderers.DeepSeekV4()
        _, self.tml_parser = self.tml.render_for_completion(
            chat.OpenAIMessage.from_oss_messages([{"role": "user", "content": "hi"}])
        )
        self.oracles = (
            ("deepseek-recipe", self.oracle),
            ("tml-deepseek-v4", self.tml_oracle),
        )

    @staticmethod
    def fragments(case, *, terminal=True):
        # One DeepSeek assistant turn has reasoning, then answer, then calls.
        # Reorder the common grammar cases into that model's legal sequence.
        # Non-newline guards avoid the newer official parser's newline trimming;
        # the pinned checkpoint's preserved whitespace has separate regressions.
        yield False, "R:"
        for part in case.parts:
            if part.kind == "reasoning":
                yield False, part.value
        yield False, "."
        yield True, "</think>"
        yield False, "C:"
        for part in case.parts:
            if part.kind == "text":
                yield False, part.value
        yield False, "."
        tools = [part.value for part in case.parts if part.kind == "tool"]
        if tools:
            yield False, "\n\n<"
            yield True, "｜DSML｜"
            yield False, "tool_calls>\n"
            for tool in tools:
                yield False, "<"
                yield True, "｜DSML｜"
                yield False, f'invoke name="{tool["name"]}">\n'
                if not tool["args"]:
                    # The pinned template has one newline on either side of
                    # the arguments, including an empty argument list.
                    yield False, "\n"
                for name, value in tool["args"].items():
                    yield False, "<"
                    yield True, "｜DSML｜"
                    string = isinstance(value, str)
                    yield (
                        False,
                        f'parameter name="{name}" string="{str(string).lower()}">',
                    )
                    yield (
                        False,
                        value if string else json.dumps(value, ensure_ascii=False),
                    )
                    yield False, "</"
                    yield True, "｜DSML｜"
                    yield False, "parameter>\n"
                yield False, "</"
                yield True, "｜DSML｜"
                yield False, "invoke>\n"
            yield False, "</"
            yield True, "｜DSML｜"
            yield False, "tool_calls>"
        if terminal:
            yield True, "<｜end▁of▁sentence｜>"

    def encode(self, case, *, terminal=True):
        result = []
        for special, text in self.fragments(case, terminal=terminal):
            result.extend(
                [self.ids[text]]
                if special
                else self.ordinary.encode(text, add_special_tokens=False).ids
            )
        return result

    def parser(self):
        return self.codec.create_parser({"extra": {"thinking_mode": "thinking"}}, [])

    def text_parser(self):
        return self.codec.create_parser({}, [])

    @staticmethod
    def prompt_case(case):
        messages = [{"role": "system", "content": "Answer carefully."}]
        for index, part in enumerate(case.parts):
            if part.kind == "tool":
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": "Calling.",
                            "tool_calls": [
                                {
                                    "id": f"call_{index}",
                                    "function": {
                                        "name": part.value["name"],
                                        "arguments": json.dumps(
                                            part.value["args"], ensure_ascii=False
                                        ),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{index}",
                            "content": "result",
                        },
                    ]
                )
            else:
                messages.append({"role": "user", "content": part.value})
                messages.append(
                    {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": part.value,
                    }
                )
        messages.append({"role": "user", "content": "continue"})
        request = {
            "messages": messages,
            "reasoning_effort": ("low", "medium", "xhigh")[len(case.parts) % 3],
            "extra": {"thinking_mode": "thinking" if len(case.parts) % 2 else "chat"},
        }
        if len(case.parts) % 2:
            request["tools"] = [
                {
                    "name": "weather",
                    "description": "Weather.",
                    "parameters": {"type": "object"},
                }
            ]
        return request

    @staticmethod
    def prompt_oracle(request):
        messages = deepcopy(request["messages"])
        if tools := request.get("tools"):
            messages[0]["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
        mode = request["extra"]["thinking_mode"]
        effort = {"low": "low", "medium": "high", "xhigh": "max"}[
            request["reasoning_effort"]
        ]
        return checkpoint_encoder()(
            messages, thinking_mode=mode, reasoning_effort=effort
        )

    def tool_truncation(self, case):
        ids = self.encode(case, terminal=False)
        return ids, 0, len(ids)

    def oracle(self, case):
        converted = ChatCompletionRequest(
            {
                "model": "DeepSeek-V4-Pro",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).convert(ConversionOptions())
        generator = ChatCompletionRequest.chunk_generator(converted, "fuzz", "deepseek")
        processor = StreamProcessor(
            generator,
            ParsingOptions(
                thinking_initial_stage=ThinkingInitialStage.ReasoningContinuation
            ),
            self.reference,
        )
        chunks = processor.push(InferenceChunk.ready())
        chunks.extend(processor.push(InferenceChunk.tokens(self.encode(case)[:-1])))
        chunks.extend(processor.push(InferenceChunk.finish(InferenceFinishReason.Stop)))
        chunks.extend(processor.finish())
        reasoning, text = "", ""
        tools = {}
        for chunk in chunks:
            for choice in json.loads(chunk.to_json()).get("choices", []):
                delta = choice.get("delta") or {}
                reasoning += delta.get("reasoning_content") or ""
                text += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    tool = tools.setdefault(call["index"], {"name": "", "args": ""})
                    function = call.get("function") or {}
                    tool["name"] += function.get("name") or ""
                    tool["args"] += function.get("arguments") or ""
        return normalize(
            [Part("reasoning", reasoning), Part("text", text)]
            + [
                Part("tool", {"name": tool["name"], "args": json.loads(tool["args"])})
                for tool in tools.values()
            ]
        )

    def tml_oracle(self, case):
        ids = []
        for special, text in self.fragments(case):
            ids.extend(
                [self.tml.tokenizer.encode_special(text)]
                if special
                else self.tml.tokenizer.encode_ordinary(text)
            )
        parts = []
        for message in self.tml_parser.parse_tokens(ids):
            match message.content:
                case chat.Text(text=text):
                    parts.append(Part("text", text))
                case chat.Thinking(text=text):
                    parts.append(Part("reasoning", text))
                case chat.InvokeTool(name=name, args=args):
                    parts.append(
                        Part(
                            "tool",
                            {
                                "name": name,
                                "args": {
                                    arg.name: json.loads(arg.value) for arg in args
                                },
                            },
                        )
                    )
                case chat.ModelEndSampling():
                    pass
                case other:
                    raise AssertionError(f"unexpected DeepSeek content {other}")
        return normalize(parts)


@cache
def checkpoint_encoder():
    path = Path(__file__).with_name("references") / "deepseek_v4_encoding.py.txt"
    namespace = {"__name__": "_resin_test_deepseek_v4_reference"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["encode_messages"]
