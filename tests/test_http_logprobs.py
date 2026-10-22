"""Scores cross the real HTTP actor without losing token identity or bytes."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest
from test_http_lifecycle import running_server
from test_recipe import _byte_tokenizer

from resin import ForwardOutput, RunnerConfig
from resin.recipe import Recipe


def request(base, path, body):
    return urllib.request.urlopen(
        urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=5,
    )


def events(response):
    return [
        json.loads(line[6:])
        for line in response.read().decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("metaspace", [False, True])
def test_byte_fallback_scores_preserve_raw_bytes_and_literal_metaspace(metaspace):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteFallback, Fuse, Replace, Sequence
    from tokenizers.models import WordLevel

    backend = Tokenizer(
        WordLevel({"▁": 0, "<0xE2>": 1, "<0x82>": 2, "<0xAC>": 3, "<0x0000>": 4})
    )
    backend.decoder = Sequence(
        ([Replace("▁", " ")] if metaspace else []) + [ByteFallback(), Fuse()]
    )

    class Codec:
        tokenizer = backend

        def render(self, _request):
            return {"token_ids": [0], "stop_token_ids": []}

        def create_parser(self, _request, _prompt_ids):
            class Parser:
                def feed(self, _ids):
                    return []

            return Parser()

    class Runner:
        offset = 0

        def forward(self, _plan):
            row = [0] if self.offset == 0 else [1, 2, 3, 4]
            self.offset += len(row)
            return ForwardOutput(
                np.asarray(row, np.uint32),
                np.asarray([0, len(row)], np.uint32),
                logprobs=np.asarray([-np.inf] * len(row), np.float32),
            )

    with running_server(Runner(), Codec()) as (_, _, _, errors, base, _):
        with request(
            base,
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "logprobs": True,
            },
        ) as response:
            scores = json.load(response)["resin"]["generation_logprobs"]
        assert errors == []
    assert [score["bytes"] for score in scores] == [
        list((" " if metaspace else "▁").encode()),
        [0xE2],
        [0x82],
        [0xAC],
        list(b"<0x0000>"),
    ]
    assert all(score["logprob"] is None for score in scores)


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
def test_generation_scores_preserve_utf8_reasoning_tools_and_stream_parity(api):
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|message_system|>",
            "<|message_user|>",
            "<|message_tool|>",
            "<|content_xml|>",
        ]
    )
    codec = Recipe(backend, model="inkling")
    content = "café 🧠"
    output = (
        "<|content_thinking|>plan<|end_message|><|message_model|>"
        f"<|content_text|>{content}<|end_message|><|message_model|>"
        '<|content_invoke_tool_json|>{"name":"weather","args":{"n":1}}'
        "<|end_message|><|content_model_end_sampling|>"
    )
    ids = backend.encode(output, add_special_tokens=False).ids
    alternative = backend.encode("z", add_special_tokens=False).ids[0]

    class Runner:
        offset = 0

        def forward(self, plan):
            params = plan.sampling_params[0]
            assert params.logprobs and params.top_logprobs == 2
            # A large verified block also exercises event coalescing: bytewise
            # parser feeding must not fill the client's bounded event queue.
            row = ids[:1] if self.offset == 0 else ids[1:]
            scores = np.asarray(
                [
                    -(index + 1) / 100
                    for index in range(self.offset, self.offset + len(row))
                ],
                dtype=np.float32,
            )
            self.offset += len(row)
            return ForwardOutput(
                np.asarray(row, np.uint32),
                np.asarray([0, len(row)], np.uint32),
                logprobs=scores,
                logprob_indptr=np.asarray([0, len(row)], np.uint32),
                top_indptr=np.arange(0, len(row) * 2 + 1, 2, dtype=np.uint32),
                top_token_ids=np.asarray(
                    [token for actual in row for token in (actual, alternative)],
                    np.uint32,
                ),
                top_logprobs=np.asarray(
                    [score for actual in scores for score in (actual, -9)],
                    np.float32,
                ),
            )

    body = {
        "model": "test",
        "top_logprobs": 2,
        "max_completion_tokens": len(ids),
        "max_output_tokens": len(ids),
        "messages": [{"role": "user", "content": "hi"}],
        "input": "hi",
    }
    if api == "chat/completions":
        body["logprobs"] = True
    else:
        body["include"] = ["message.output_text.logprobs"]
    results = []
    for stream in (False, True):
        with running_server(Runner(), codec) as (_, engine, _, errors, base, _):
            with request(base, "/v1/" + api, {**body, "stream": stream}) as response:
                if not stream:
                    result = json.load(response)
                    raw = result["resin"]["generation_logprobs"]
                    if api == "chat/completions":
                        choice = result["choices"][0]
                        assert choice["message"]["content"] == content
                        canonical = choice["logprobs"]["content"]
                    else:
                        message = next(
                            item
                            for item in result["output"]
                            if item["type"] == "message"
                        )
                        assert message["content"][0]["text"] == content
                        canonical = message["content"][0]["logprobs"]
                else:
                    chunks = events(response)
                    if api == "chat/completions":
                        raw = [
                            token
                            for chunk in chunks
                            for token in chunk.get("resin", {}).get(
                                "generation_logprobs", []
                            )
                        ]
                        canonical = [
                            token
                            for chunk in chunks
                            for token in (
                                chunk.get("choices", [{}])[0].get("logprobs") or {}
                            ).get("content", [])
                        ]
                    else:
                        raw = [
                            token
                            for chunk in chunks
                            if chunk["type"] == "resin.generation_logprobs.delta"
                            for token in chunk["logprobs"]
                        ]
                        completed = next(
                            chunk["response"]
                            for chunk in chunks
                            if chunk["type"] == "response.completed"
                        )
                        message = next(
                            item
                            for item in completed["output"]
                            if item["type"] == "message"
                        )
                        canonical = message["content"][0]["logprobs"]
            assert not engine.poisoned and errors == []
        assert [token["token_id"] for token in raw] == ids
        assert [token["token_index"] for token in raw] == list(range(len(ids)))
        assert [token["logprob"] for token in raw] == pytest.approx(
            [-(index + 1) / 100 for index in range(len(ids))]
        )
        assert (
            bytes(byte for token in canonical for byte in token["bytes"]).decode()
            == content
        )
        assert {token["channel"] for token in raw} >= {
            "control",
            "reasoning",
            "content",
            "tool",
        }
        for token in raw:
            assert token["top_logprobs"][0]["token_id"] == token["token_id"]
            assert token["top_logprobs"][1]["bytes"] == [ord("z")]
        # Individual tokens can be incomplete UTF-8. Their raw bytes, rather
        # than the replacement-character display spelling, reconstruct output.
        assert any(token["token"] == "�" for token in canonical)
        results.append((raw, canonical))
    assert results[0] == results[1]


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_prompt_scores_allow_zero_output_tokens(api, stream):
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|message_system|>",
            "<|message_user|>",
            "<|message_tool|>",
            "<|content_xml|>",
        ]
    )
    codec = Recipe(backend, model="inkling")
    request_input = {"messages": [{"role": "user", "content": "score this"}]}
    prompt_ids = codec.render(request_input)["token_ids"]

    class Runner:
        runner_config = RunnerConfig(continuation_prefill=True)

        def forward(self, plan):
            assert plan.sampling_params[0].prompt_logprobs
            np.testing.assert_array_equal(plan.prompt_score_token_ids, prompt_ids[1:])
            assert not any(plan.sampling_mask)
            return ForwardOutput(
                # Non-sampling rows retain the runner's ordinary dummy token.
                np.asarray([0], np.uint32),
                np.asarray([0, 1], np.uint32),
                prompt_logprobs=np.asarray(
                    [-index / 100 for index in range(1, len(prompt_ids))], np.float32
                ),
                prompt_logprob_indptr=np.asarray([0, len(prompt_ids) - 1], np.uint32),
            )

    body = {
        **request_input,
        "model": "test",
        "input": "score this",
        "max_completion_tokens": 0,
        "max_output_tokens": 0,
        "prompt_logprobs": True,
        "stream": stream,
    }
    with running_server(Runner(), codec) as (_, engine, _, errors, base, _):
        with request(base, "/v1/" + api, body) as response:
            if stream:
                chunks = events(response)
                if api == "chat/completions":
                    scores = next(
                        chunk["resin"]["prompt_logprobs"]
                        for chunk in chunks
                        if "prompt_logprobs" in chunk.get("resin", {})
                    )
                else:
                    scores = next(
                        chunk["logprobs"]
                        for chunk in chunks
                        if chunk["type"] == "resin.prompt_logprobs"
                    )
            else:
                scores = json.load(response)["resin"]["prompt_logprobs"]
        assert errors == [] and not engine.poisoned
    assert scores[0] is None
    assert [score["token_id"] for score in scores[1:]] == prompt_ids[1:]
    assert [score["logprob"] for score in scores[1:]] == pytest.approx(
        [-index / 100 for index in range(1, len(prompt_ids))]
    )
