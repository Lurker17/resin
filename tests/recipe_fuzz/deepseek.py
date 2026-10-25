"""DeepSeek's official Rust parser is the primary DSML semantic oracle."""

from __future__ import annotations

import json
from copy import deepcopy

from deepseek_recipe import (
    ChatCompletionRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    InferenceChunk,
    InferenceFinishReason,
    ParsingOptions,
    StreamProcessor,
    ThinkingInitialStage,
)
from deepseek_recipe import (
    Tokenizer as DeepSeekTokenizer,
)
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

import resin
from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import DeepSeekV41Recipe

MARKERS = (
    "<｜User｜>",
    "<｜System｜>",
    "<｜Assistant｜>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
)


class DeepSeekV41Adapter:
    commits_tools_at_close = False
    model = "deepseek_v41"
    reference_model = "DeepSeek-V4.1-Flash"
    tag_names = (" calls", " invoke", " parameter")

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
        self.recipe = DeepSeekV41Recipe(backend)
        self.reference = DeepSeekTokenizer.from_str(backend.to_str())
        self.oracles = (("deepseek-recipe", self.oracle),)

    @classmethod
    def fragments(cls, case, *, terminal=True):
        calls, invoke, parameter = cls.tag_names
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
            yield False, f"{calls}>\n"
            for tool in tools:
                yield False, "<"
                yield True, "｜DSML｜"
                yield False, f'{invoke} name="{tool["name"]}">\n'
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
                        f'{parameter} name="{name}" string="{str(string).lower()}">',
                    )
                    yield (
                        False,
                        value if string else json.dumps(value, ensure_ascii=False),
                    )
                    yield False, "</"
                    yield True, "｜DSML｜"
                    yield False, f"{parameter}>\n"
                yield False, "</"
                yield True, "｜DSML｜"
                yield False, f"{invoke}>\n"
            yield False, "</"
            yield True, "｜DSML｜"
            yield False, f"{calls}>"
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
        return self.recipe.create_parser(
            self.recipe.render(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER, [resin.recipe.Text("hi")]
                    )
                ],
                options=resin.recipe.RenderOptions(thinking_mode="thinking"),
            )
        )

    def text_parser(self):
        return self.recipe.create_parser(
            self.recipe.render(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER, [resin.recipe.Text("hi")]
                    )
                ],
                options=resin.recipe.RenderOptions(thinking_mode="chat"),
            )
        )

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
        request["messages"].insert(1, {"role": "user", "content": "Begin."})
        messages = []
        for message in request["messages"]:
            if message["role"] == "assistant" and messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": "Continue."})
            messages.append(message)
        request["messages"] = messages
        request["reasoning_effort"] = {"low": "low", "medium": "high", "xhigh": "max"}[
            request["reasoning_effort"]
        ]
        request["messages"].insert(-1, {"role": "system", "content": "Be precise."})
        return request

    @staticmethod
    def prompt_oracle(request):
        body = {
            "model": "DeepSeek-V4.1-Flash",
            "messages": deepcopy(request["messages"]),
            "thinking": {
                "type": "enabled"
                if request["extra"]["thinking_mode"] == "thinking"
                else "disabled"
            },
        }
        for message in body["messages"]:
            for call in message.get("tool_calls", []):
                call.setdefault("type", "function")
        if "reasoning_effort" in request:
            body["reasoning_effort"] = request["reasoning_effort"]
        if tools := request.get("tools"):
            body["tools"] = [
                tool if "function" in tool else {"type": "function", "function": tool}
                for tool in tools
            ]
        converted = ChatCompletionRequest(body).convert(ConversionOptions())
        return DeepseekV41Encoding().render_conversation(converted.conversation).prompt

    def tool_truncation(self, case):
        ids = self.encode(case, terminal=False)
        return ids, 0, len(ids)

    def oracle(self, case):
        converted = ChatCompletionRequest(
            {
                "model": self.reference_model,
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
