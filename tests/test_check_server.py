"""Public server checks distinguish protocol, cache, latency, and model accuracy."""

import asyncio
import json

import httpx
import pytest

from scripts import check_server as checks
from scripts.stress_http import Completion


def options(tmp_path, *extra):
    return checks.parse_args(
        [
            "http://test",
            "--output",
            str(tmp_path / "report.json"),
            "--context-lengths",
            "4",
            "8",
            "--max-tokens",
            "4",
            *extra,
        ]
    )


def payload(body, *, content="A", cached=0, tokens=3):
    value = {
        "id": "completion",
        "choices": [
            {
                "index": 0,
                "delta" if body.get("stream") else "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": 1,
            "total_tokens": tokens + 1,
        },
    }
    if cached is not None:
        value["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
    if body.get("logprobs"):
        value["resin"] = {
            "generation_logprobs": [
                {
                    "token_index": 0,
                    "token_id": 10,
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token_id": 10 + i, "logprob": -0.1 - i}
                        for i in range(body["top_logprobs"])
                    ],
                }
            ]
        }
    return value


class Server:
    def __init__(self, *, cache=True, tokenize=False, counter=None, answer="A"):
        self.cache, self.tokenize, self.counter, self.answer = (
            cache,
            tokenize,
            counter,
            answer,
        )
        self.seen = set()
        self.requests = []
        self.active = self.peak = 0

    async def __call__(self, request):
        self.requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture"}]})
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=""
                if self.counter is None
                else f'compile_total{{rank="0"}} {self.counter}\n',
            )
        if request.url.path == "/tokenize" and not self.tokenize:
            return httpx.Response(404)
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        prompt = content if isinstance(content, str) else content[0]["text"]
        tokens = len(prompt.split()) + 2
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": tokens})
        assert request.url.path == "/v1/chat/completions"
        if prompt.startswith("x ") and tokens > 32:
            return httpx.Response(
                400, json={"error": {"message": "prompt exceeds context limit"}}
            )
        self.active += 1
        self.peak = max(self.active, self.peak)
        await asyncio.sleep(0)
        self.active -= 1
        cached = tokens - 1 if prompt in self.seen else 0
        self.seen.add(prompt)
        value = payload(
            body,
            content=self.answer,
            cached=cached if self.cache else None,
            tokens=tokens,
        )
        if body.get("stream"):
            return httpx.Response(
                200,
                text="data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=value)


def run(args, server):
    async def execute():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(server)
        ) as client:
            return await checks.check_server(client, args)

    return asyncio.run(execute())


def by_name(report):
    return {check["name"]: check for check in report["checks"]}


def test_checks_real_protocol_cache_context_media_reasoning_and_evaluation(tmp_path):
    source = tmp_path / "questions.jsonl"
    source.write_text(
        json.dumps({"question": "Which is one?", "choices": ["1", "2"], "answer": 0})
        + "\n"
    )
    args = options(
        tmp_path,
        "--image",
        "--reasoning-effort",
        "low",
        "high",
        "--context-length",
        "32",
        "--compilation-metric",
        "compile_total",
        "--eval-file",
        str(source),
        "--min-accuracy",
        "1",
    )
    server = Server(tokenize=True, counter=0)
    report = run(args, server)
    assert report["ok"], report
    rows = by_name(report)
    assert rows["context_boundary"]["tokenizer_matches_usage"]
    assert (
        rows["context_boundary"]["inside_tokens"]
        < 32
        < rows["context_boundary"]["outside_tokens"]
    )
    assert rows["prefix_cache"]["results"][1]["cached_prompt_tokens"] > 0
    assert rows["kernel_compilation"]["status"] == "passed"
    assert rows["evaluation"]["accuracy"] == 1
    assert rows["context_ttft_4"]["requested_words"] == 4
    assert rows["context_ttft_4"]["results"][0]["prompt_tokens"] > 4
    assert server.peak == args.concurrency
    bodies = [
        json.loads(r.content)
        for r in server.requests
        if r.url.path == "/v1/chat/completions"
    ]
    assert {b["reasoning_effort"] for b in bodies if "reasoning_effort" in b} == {
        "low",
        "high",
    }
    image = next(
        b["messages"][0]["content"][1]["image_url"]["url"]
        for b in bodies
        if isinstance(b["messages"][0]["content"], list)
    )
    assert image.startswith("data:image/png;base64,")


def test_unavailable_telemetry_is_skipped_not_inferred_from_fast_responses(tmp_path):
    report = run(options(tmp_path, "--context-length", "32"), Server(cache=False))
    assert report["ok"]
    rows = by_name(report)
    assert rows["prefix_cache"]["status"] == "skipped"
    assert rows["context_boundary"]["status"] == "skipped"
    assert rows["kernel_compilation"]["status"] == "skipped"
    assert rows["context_ttft_4"]["status"] == "passed"
    assert report["counts"]["skipped"] >= 3


def test_zero_cache_hits_fail_when_telemetry_is_present(tmp_path):
    class NoCache(Server):
        async def __call__(self, request):
            self.seen.clear()
            return await super().__call__(request)

    report = run(options(tmp_path, "--context-lengths"), NoCache())
    assert not report["ok"]
    assert by_name(report)["prefix_cache"]["status"] == "failed"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("(B)", "B"),
        ("Answer: C\nExplanation", "C"),
        ("\\boxed{D}", "D"),
        ("A good explanation is not a letter answer.", None),
        ("Answer: A\nAnswer: B", None),
        ("E", None),
        ("", None),
    ],
)
def test_answer_extraction_does_not_grade_incidental_letters(text, expected):
    assert checks.extract_answer(text, "ABCD") == expected


def test_missing_eval_answers_fail_even_with_zero_minimum_accuracy(tmp_path):
    source = tmp_path / "questions.jsonl"
    source.write_text(
        json.dumps(
            {"question": "Choose", "choices": ["first", "second"], "answer": "A"}
        )
    )
    report = run(
        options(
            tmp_path,
            "--context-lengths",
            "--eval-file",
            str(source),
            "--min-accuracy",
            "0",
        ),
        Server(answer="I cannot choose."),
    )
    assert not report["ok"]
    row = by_name(report)["evaluation"]
    assert row["missing_answers"] == 1 and row["status"] == "failed"


def test_context_first_pass_latency_regression_is_not_called_recompilation(
    tmp_path, monkeypatch
):
    original = checks.run_case

    async def measured(client, case, timeout):
        result = await original(client, case, timeout)
        if case.name.startswith("context-"):
            result["ttft_ms"] = 4000 if case.name.endswith("pass-0") else 10
        return result

    monkeypatch.setattr(checks, "run_case", measured)
    report = run(options(tmp_path), Server())
    rows = by_name(report)
    assert not report["ok"] and rows["context_ttft_4"]["first_pass_spike"]
    assert rows["kernel_compilation"]["status"] == "skipped"


def test_compilation_counter_change_fails_even_with_fast_requests(tmp_path):
    class Compiling(Server):
        async def __call__(self, request):
            response = await super().__call__(request)
            if request.url.path == "/metrics":
                assert self.counter is not None
                self.counter += 1
            return response

    report = run(
        options(tmp_path, "--compilation-metric", "compile_total"), Compiling(counter=0)
    )
    assert not report["ok"]
    assert by_name(report)["kernel_compilation"]["status"] == "failed"


def test_stream_tracker_retains_final_answer_separately_from_reasoning():
    tracker = Completion(scored=False, top_k=0)
    first = payload({"stream": True}, content="", cached=2)
    first["choices"][0]["finish_reason"] = None
    first["choices"][0]["delta"]["reasoning_content"] = "Maybe B."
    tracker.add(first, stream=True)
    tracker.add(payload({"stream": True}, content="A", cached=2), stream=True)
    result = tracker.finish(2)
    assert result["content"] == "A" and result["reasoning_content"] == "Maybe B."
    assert result["cached_prompt_tokens"] == 2


@pytest.mark.parametrize("value", [-1, 4, True])
def test_invalid_cache_token_counts_fail(value):
    tracker = Completion(scored=False, top_k=0)
    tracker.add(payload({}, cached=value), stream=False)
    with pytest.raises(ValueError, match="cached prompt"):
        tracker.finish(2)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--timeout", "nan"),
        ("--max-ttft-ms", "inf"),
        ("--repeats", "0"),
        ("--min-accuracy", "nan"),
    ],
)
def test_invalid_cli_limits_fail(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        options(tmp_path, flag, value)


def test_standard_mmlu_csv_and_jsonl_produce_same_questions(tmp_path):
    source = tmp_path / "mmlu_subject.csv"
    source.write_text(
        '"Which answer, precisely?",first,second,third,fourth,B\nAnother?,one,two,three,four,A\n'
    )
    rows = checks.load_evaluation(source, 1)
    assert rows == [
        {
            "question": "Which answer, precisely?",
            "choices": ["first", "second", "third", "fourth"],
            "answer": "B",
        }
    ]
    report = run(
        options(
            tmp_path,
            "--context-lengths",
            "--eval-file",
            str(source),
            "--eval-limit",
            "1",
        ),
        Server(answer="B"),
    )
    assert report["ok"] and by_name(report)["evaluation"]["accuracy"] == 1


def test_entrypoint_against_native_http_server_writes_report(tmp_path):
    from test_http_lifecycle import Recipe, Runner, running_server

    args = options(
        tmp_path,
        "--context-lengths",
        "--skip-prefix-cache",
        "--skip-logprobs",
        "--timeout",
        "5",
    )
    with running_server(Runner(), Recipe()) as (_, _, _, errors, base, _):
        args.url = base
        assert asyncio.run(checks.main(args))
        assert not errors
    report = json.loads(args.output.read_text())
    assert report["ok"] and report["model"] == "test"
    assert by_name(report)["concurrent"]["requests"] == 4


@pytest.mark.parametrize("mismatch", [False, True])
def test_prefix_check_rejects_static_template_hits_and_changed_prompt_counts(
    tmp_path, mismatch
):
    class TemplateOnly(Server):
        async def __call__(self, request):
            response = await super().__call__(request)
            if request.url.path == "/v1/chat/completions":
                body = json.loads(request.content)
                text = body["messages"][0]["content"]
                if isinstance(text, str) and text.startswith("Cache check"):
                    data = response.json()
                    cached = data["usage"]["prompt_tokens_details"]["cached_tokens"]
                    data["usage"]["prompt_tokens_details"]["cached_tokens"] = (
                        cached if mismatch else 2
                    )
                    if mismatch and cached:
                        data["usage"]["prompt_tokens"] += 1
                        data["usage"]["total_tokens"] += 1
                    return httpx.Response(200, json=data)
            return response

    report = run(options(tmp_path, "--context-lengths"), TemplateOnly())
    assert not report["ok"]
    assert by_name(report)["prefix_cache"]["status"] == "failed"


def test_missing_auth_environment_still_writes_failure_report(tmp_path, monkeypatch):
    monkeypatch.delenv("RESIN_TEST_MISSING_AUTH", raising=False)
    args = options(tmp_path, "--api-key-env", "RESIN_TEST_MISSING_AUTH")
    assert not asyncio.run(checks.main(args))
    report = json.loads(args.output.read_text())
    assert report["counts"]["failed"] == 1
    assert report["checks"][0]["name"] == "startup"


@pytest.mark.parametrize(
    "answer",
    [
        "Answer: A\n\\boxed{B}",
        "Answer: A or B",
        "Answer: A because it is correct",
        "Answer: A\nB",
        "A\n\\boxed{B}",
        "\\boxed{A} or B",
        "\\boxed{A} \\boxed{B}",
        "Answer: A\nFinal answer: B",
        "Answer: A\nAnswer: unknown",
    ],
)
def test_conflicting_or_ambiguous_answer_forms_are_not_graded(answer):
    assert checks.extract_answer(answer, "ABCD") is None


@pytest.mark.parametrize(
    "answer",
    [
        "Answer: A\n\\boxed{A}",
        "Final answer: (A).\nExplanation on a separate line.",
        "A\nAnswer: A\n\\boxed{A}.",
        "Answer: \\boxed{A}.",
    ],
)
def test_agreeing_answer_forms_and_separate_explanations_remain_valid(answer):
    assert checks.extract_answer(answer, "ABCD") == "A"


@pytest.mark.parametrize("completion_tokens", [0, 1])
def test_empty_image_response_fails_positive_check_while_text_stays_healthy(
    tmp_path, completion_tokens
):
    class EmptyImage(Server):
        async def __call__(self, request):
            response = await super().__call__(request)
            if request.url.path == "/v1/chat/completions":
                body = json.loads(request.content)
                if isinstance(body["messages"][0]["content"], list):
                    data = response.json()
                    data["choices"][0]["message"]["content"] = ""
                    usage = data["usage"]
                    usage["completion_tokens"] = completion_tokens
                    usage["total_tokens"] = usage["prompt_tokens"] + completion_tokens
                    return httpx.Response(200, json=data)
            return response

    report = run(options(tmp_path, "--context-lengths", "--image"), EmptyImage())
    rows = by_name(report)
    assert not report["ok"] and report["counts"]["failed"] == 1
    assert rows["chat"]["status"] == rows["concurrent"]["status"] == "passed"
    image = rows["image"]
    assert image["status"] == "failed" and image["passed"] == 0
    assert image["completed_requests_per_s"] == image["completion_tokens"] == 0
    assert image["results"][0]["ok"] is False
    assert "no meaningful output" in image["results"][0]["error"]


def test_generic_completion_parser_still_allows_protocol_valid_empty_eos():
    value = payload({}, content="")
    value["usage"]["completion_tokens"] = 0
    value["usage"]["total_tokens"] = value["usage"]["prompt_tokens"]
    tracker = Completion(scored=False, top_k=0)
    tracker.add(value, stream=False)
    assert tracker.finish(1)["completion_tokens"] == 0
