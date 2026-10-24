"""Harmony's official implementation is the GPT-OSS semantic oracle."""

from __future__ import annotations

import json

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)
from tml_renderers import chat, renderers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

import resin
from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import GptOssRecipe

MARKERS = (
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|channel|>",
    "<|message|>",
    "<|constrain|>",
)


class GptOssAdapter:
    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens(list(MARKERS))
        self.ids = {marker: backend.token_to_id(marker) for marker in MARKERS}
        self.codec = GptOssRecipe(backend)
        self.reference = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.tml = renderers.GptOss()
        _, self.tml_parser = self.tml.render_for_completion(
            chat.OpenAIMessage.from_oss_messages([{"role": "user", "content": "hi"}])
        )
        self.oracles = (
            ("openai-harmony", self.oracle),
            ("tml-harmony", self.tml_oracle),
        )

    @staticmethod
    def fragments(case, *, terminal=True):
        for index, part in enumerate(case.parts):
            yield True, "<|start|>"
            author = "assistant"
            if part.kind == "tool":
                author += " to=functions." + part.value["name"]
            yield False, author
            yield True, "<|channel|>"
            yield (
                False,
                {"text": "final", "reasoning": "analysis", "tool": "commentary"}[
                    part.kind
                ],
            )
            yield True, "<|message|>"
            yield (
                False,
                json.dumps(part.value["args"], ensure_ascii=False)
                if part.kind == "tool"
                else part.value,
            )
            yield (
                True,
                "<|call|>"
                if part.kind == "tool"
                else "<|return|>"
                if terminal and index == len(case.parts) - 1
                else "<|end|>",
            )

    def encode(self, case, *, terminal=True):
        ids = []
        for special, value in self.fragments(case, terminal=terminal):
            ids.extend(
                [self.ids[value]]
                if special
                else self.ordinary.encode(value, add_special_tokens=False).ids
            )
        return ids

    def reference_ids(self, case):
        ids = []
        for special, value in self.fragments(case):
            ids.extend(
                self.reference.encode(value, allowed_special="all")
                if special
                else self.reference.encode(value, disallowed_special=())
            )
        return ids

    def oracle(self, case):
        parts = []
        for message in self.reference.parse_messages_from_completion_tokens(
            self.reference_ids(case)
        ):
            text_parts = []
            for content in message.content:
                assert isinstance(content, TextContent)
                text_parts.append(content.text)
            text = "".join(text_parts)
            if message.recipient:
                parts.append(
                    Part(
                        "tool",
                        {
                            "name": message.recipient.removeprefix("functions."),
                            "args": json.loads(text),
                        },
                    )
                )
            else:
                parts.append(
                    Part("reasoning" if message.channel == "analysis" else "text", text)
                )
        return normalize(parts)

    def tml_oracle(self, case):
        parts = []
        for message in self.tml_parser.parse_tokens(self.reference_ids(case)):
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
                    raise AssertionError(f"unexpected Harmony content {other}")
        return normalize(parts)

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

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        prefix_length = len(self.encode(prefix, terminal=False))
        # Everything through the final tool's content-start marker is header.
        tail = list(self.fragments(Case(case.parts[-1:]), terminal=False))
        header_length = 0
        for special, value in tail:
            header_length += 1 if special else len(self.ordinary.encode(value).ids)
            if special and value == "<|message|>":
                break
        ids = self.encode(case, terminal=False)
        return ids, prefix_length + header_length, len(ids) - 1

    def text_parser(self):
        parser = self.parser()
        # Complete the generated channel header before testing arbitrary text.
        prefix = [self.ids["<|end|>"], self.ids["<|start|>"]]
        prefix.extend(self.ordinary.encode("assistant").ids)
        prefix.append(self.ids["<|channel|>"])
        prefix.extend(self.ordinary.encode("final").ids)
        prefix.append(self.ids["<|message|>"])
        assert parser.feed(prefix) == []
        return parser

    @staticmethod
    def prompt_case(case):
        calls = [
            {"function": {"name": part.value["name"], "arguments": part.value["args"]}}
            for part in case.parts
            if part.kind == "tool"
        ]
        messages = [
            {
                "role": "user",
                "content": "".join(
                    part.value for part in case.parts if part.kind != "tool"
                ),
            }
        ]
        if calls:
            messages.extend(
                [
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {
                        "role": "tool",
                        "name": calls[-1]["function"]["name"],
                        "content": "done",
                    },
                ]
            )
        return {"messages": messages, "conversation_start_date": "2026-10-21"}

    def prompt_oracle(self, request):
        system = SystemContent.new().with_conversation_start_date(
            request["conversation_start_date"]
        )
        prefix = [Message.from_role_and_content(Role.SYSTEM, system)]
        tools = request.get("tools") or []
        if tools:
            developer = DeveloperContent.new().with_function_tools(
                [
                    ToolDescription.new(
                        tool["name"],
                        tool.get("description", ""),
                        tool.get("parameters") or {},
                    )
                    for tool in tools
                ]
            )
            prefix.append(Message.from_role_and_content(Role.DEVELOPER, developer))
        ids = self.reference.render_conversation(Conversation.from_messages(prefix))
        for item in request["messages"]:
            if item["role"] == "user":
                ids.extend(
                    self.reference.render(
                        Message.from_role_and_content(Role.USER, item["content"])
                    )
                )
            elif item["role"] == "assistant":
                for call in item["tool_calls"]:
                    function = call["function"]
                    message = (
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            json.dumps(function["arguments"], ensure_ascii=False),
                        )
                        .with_recipient("functions." + function["name"])
                        .with_channel("commentary")
                    )
                    ids.extend(self.reference.render(message))
            elif item["role"] == "tool":
                message = (
                    Message.from_author_and_content(
                        Author(role=Role.TOOL, name="functions." + item["name"]),
                        item["content"],
                    )
                    .with_recipient("assistant")
                    .with_channel("commentary")
                )
                ids.extend(self.reference.render(message))
        ids.extend(
            self.reference.encode(
                "<|start|>assistant<|channel|>analysis<|message|>",
                allowed_special="all",
            )
        )
        return self.reference.decode(ids)
