# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Check a deployed Resin server through its public chat-completions API.

uv run scripts/check_server.py URL --output artifacts/server-check.json
Add --image, --reasoning-effort low high, --stress, or --eval-file mmlu_subject.csv.
Context sweeps report actual server token usage; TTFT is not a compilation counter.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import math
import mimetypes
import os
import re
import statistics
import struct
import sys
import uuid
import zlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Direct script execution and module/pytest imports both use the shared client.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stress_http import Case, chat_case, robustness_cases, run_case, run_stage


def request_body(model, content, max_tokens=32, *, stream=False):
    body = dict(
        model=model,
        messages=[dict(role="user", content=content)],
        max_tokens=max_tokens,
        temperature=0,
        stream=stream,
    )
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def image_url(source):
    if source.startswith(("https://", "http://", "data:image/")):
        return source
    if source == "builtin":
        # A small red RGB PNG keeps this smoke check self-contained.
        def chunk(kind, payload):
            return (
                struct.pack("!I", len(payload))
                + kind
                + payload
                + struct.pack("!I", zlib.crc32(kind + payload))
            )

        pixels = (b"\0" + b"\xff\0\0" * 64) * 64
        data = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels))
            + chunk(b"IEND", b"")
        )
        mime = "image/png"
    else:
        path = Path(source)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError("image must be PNG, JPEG, WebP, or GIF")
        data = path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def load_evaluation(path: Path, limit: int):
    """Read canonical MMLU CSV or JSONL with indexed/letter choice answers."""
    rows = []
    with path.open() as stream:
        if path.suffix.lower() == ".csv":
            records = (
                dict(question=row[0], choices=row[1:-1], answer=row[-1])
                for row in csv.reader(stream)
                if row
            )
        else:
            records = (json.loads(line) for line in stream if line.strip())
        for number, row in enumerate(records, 1):
            question, choices, answer = (
                row.get("question"),
                row.get("choices"),
                row.get("answer"),
            )
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(choices, list)
                or not 2 <= len(choices) <= 26
                or not all(
                    isinstance(choice, str) and choice.strip() for choice in choices
                )
            ):
                raise ValueError(f"invalid evaluation question at line {number}")
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
            if type(answer) is int and 0 <= answer < len(choices):
                answer = labels[answer]
            if not isinstance(answer, str) or answer not in labels or len(answer) != 1:
                raise ValueError(f"invalid evaluation answer at line {number}")
            rows.append(dict(question=question, choices=choices, answer=answer))
            if len(rows) == limit:
                break
    if not rows:
        raise ValueError("evaluation file contains no questions")
    return rows


def extract_answer(text: str, labels: str) -> str | None:
    """Require all explicit answers to agree, rejecting ambiguous declarations."""

    def unbold(value):
        value = value.strip()
        # Do not turn prose such as "**A** or **B**" into a declaration.
        match = re.fullmatch(r"\*\*([^*]+)\*\*", value)
        return match[1].strip() if match else value

    def letter(value):
        match = re.fullmatch(
            r"(?:([A-Z])|\(([A-Z])\)|\[([A-Z])\])[.,;:!?]*", value.strip()
        )
        return next(group for group in match.groups() if group) if match else None

    answers = set()
    boxed = r"\\boxed\{([^{}]*)\}"
    for line in text.splitlines():
        if re.match(r"^\s*\*\*[A-Z]\*\*\s+(?i:or|and)\b", line):
            return None
        plain = unbold(line)
        bold_line = plain != line.strip()
        declaration = re.match(
            r"^\s*(?:final\s+answer|answer)\s*[:=](.*)$", plain, re.IGNORECASE
        )
        value = unbold(declaration[1]) if declaration else plain
        selected = re.fullmatch(r"\*\*([A-Z])\*\*\s*[—–:-]\s*(\S.*)", value)
        if selected:
            # A bold selected letter followed by an explanation is an explicit
            # declaration; apply the same ambiguity checks as a bold option.
            value = f"{selected[1]}. {selected[2]}"
            bold_line = True
        if answer := letter(value):
            answers.add(answer)
        elif (declaration or bold_line) and (
            option := re.fullmatch(r"([A-Z])\.\s+(\S.*)", value)
        ):
            # A selected option may include its text. Limit this to explicit
            # answer declarations or a complete bold line, so incidental prose
            # and an ordinary A./B. option list are not scored as answers.
            suffix = option[2]
            if (
                re.search(r"\b(?i:or|and)\s+(?:(?i:option)\s+)?[A-Z]\b", suffix)
                or re.search(r"(?:^|\s)[A-Z][.)]\s+", suffix)
                or re.search(r"\b[A-Z]\s*[/,]\s*[A-Z]\b", suffix)
                or letter(suffix)
                or "\\boxed" in suffix
                or re.search(r"\b(?:final\s+answer|answer)\s*[:=]", suffix, re.I)
            ):
                return None
            answers.add(option[1])
        elif "\\boxed" in value:
            boxes = re.findall(boxed, value)
            # A box may be repeated, but surrounding alternatives or prose on
            # the same answer line make the declaration ambiguous.
            if not boxes or re.sub(boxed, "", value).strip(" \t.,;:!?"):
                return None
            for box in boxes:
                if (answer := letter(box)) is None:
                    return None
                answers.add(answer)
        elif declaration:
            return None
    return next(iter(answers)) if len(answers) == 1 and answers <= set(labels) else None


def require_generated_output(result):
    """Positive smoke prompts must produce visible content or reasoning."""
    if result.get("completion_tokens", 0) <= 0 or result.get("visible_bytes", 0) <= 0:
        raise ValueError("positive correctness request produced no meaningful output")


async def compilation_counter(client, path, name, timeout):
    if name is None:
        return None
    async with asyncio.timeout(timeout):
        response = await client.get(
            client.base_url.copy_with(path=path, query=None),
            headers={"Accept": "text/plain"},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
    values = []
    for line in response.text.splitlines():
        match = re.fullmatch(r"([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+\d+)?", line)
        if match and match[1] == name:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "compilation metric must be a finite nonnegative counter"
                )
            values.append(value)
    return sum(values) if values else None


async def context_boundary(client, model, length, path, timeout):
    """Use server chat tokenization to find prompts just inside/outside its limit."""

    async def count(repeats):
        if repeats > max(4 * length, 4096):
            raise ValueError("tokenizer count did not grow with bounded prompt size")
        body = request_body(model, "x " * repeats, 1)
        async with asyncio.timeout(timeout):
            response = await client.post(
                client.base_url.copy_with(path=path, query=None),
                json={
                    "model": model,
                    "messages": body["messages"],
                    "add_generation_prompt": True,
                },
            )
            if response.status_code in (404, 405):
                return None, body
            response.raise_for_status()
        data = response.json()
        value = data.get("count")
        if value is None and isinstance(data.get("tokens"), list):
            value = len(data["tokens"])
        if type(value) is not int or value < 1:
            raise ValueError("tokenize endpoint omitted a positive chat token count")
        return value, body

    first, _ = await count(1)
    if first is None:
        return dict(
            status="skipped",
            reason="server has no chat tokenization endpoint; exact context boundary is unverified",
        )
    if first >= length:
        raise ValueError("context length is smaller than the rendered minimal prompt")
    lower, upper = 1, length
    for _ in range(24):
        high_count, _ = await count(upper)
        if high_count > length:
            break
        upper *= 2
    else:
        raise ValueError("could not bound the tokenizer within 24 attempts")
    while upper - lower > 1:
        middle = (lower + upper) // 2
        tokens, _ = await count(middle)
        if tokens < length:
            lower = middle
        else:
            upper = middle
    inside_tokens, inside = await count(lower)
    outside_tokens, outside = await count(upper + 1)
    if not inside_tokens < length < outside_tokens:
        raise ValueError("tokenizer did not provide monotonic context boundary counts")
    accepted = await run_case(client, Case(name="inside-context", body=inside), timeout)
    rejected = await run_case(
        client,
        Case(name="outside-context", body=outside, expected_status=(400,)),
        timeout,
    )
    agrees = accepted.get("prompt_tokens") == inside_tokens
    return dict(
        status="passed" if accepted["ok"] and rejected["ok"] and agrees else "failed",
        configured_context_length=length,
        inside_tokens=inside_tokens,
        outside_tokens=outside_tokens,
        tokenizer_matches_usage=agrees,
        results=[accepted, rejected],
    )


async def check_server(client, args, *, on_update=None):
    report: dict[str, Any] = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        url=str(client.base_url),
        checks=[],
        notes=[
            "Checks exercise the public HTTP protocol; they do not establish model quality beyond the supplied evaluation.",
            "Context sweeps use requested word counts and report actual prompt tokens. TTFT includes network, queueing, rendering and prefill.",
        ],
    )

    def record(name, status, **details):
        report["checks"].append(dict(name=name, status=status, **details))
        report["counts"] = {
            state: sum(check["status"] == state for check in report["checks"])
            for state in ("passed", "failed", "skipped")
        }
        report["ok"] = report["counts"]["failed"] == 0
        if on_update:
            on_update(report)

    async def stage(name, cases, concurrency=1, *, require_output=True):
        result = await run_stage(
            client,
            name,
            cases,
            concurrency,
            args.timeout,
            validate_result=require_generated_output if require_output else None,
        )
        record(
            name,
            "passed" if result["passed"] == result["requests"] else "failed",
            **{key: value for key, value in result.items() if key != "name"},
        )
        return result

    try:
        model = args.model
        if model is None:
            async with asyncio.timeout(args.timeout):
                response = await client.get("models")
                response.raise_for_status()
                models = response.json()["data"]
            if len(models) != 1:
                raise ValueError(
                    "model discovery requires exactly one model; supply --model"
                )
            model = models[0]["id"]
            if not isinstance(model, str) or not model:
                raise ValueError("model discovery returned an invalid model ID")
        report["model"] = model
        await stage(
            "chat",
            [
                Case(
                    name="json",
                    body=request_body(model, "Say hello briefly.", args.max_tokens),
                ),
                Case(
                    name="stream",
                    body=request_body(
                        model, "Count from one to three.", args.max_tokens, stream=True
                    ),
                ),
            ],
        )
        if args.skip_logprobs:
            record("logprobs", "skipped", reason="disabled by --skip-logprobs")
        else:
            await stage(
                "logprobs",
                [
                    Case(
                        name="scored-stream",
                        body={
                            **request_body(
                                model, "Say hello.", args.max_tokens, stream=True
                            ),
                            "logprobs": True,
                            "top_logprobs": 3,
                        },
                    )
                ],
            )
        await stage(
            "concurrent",
            [
                Case(
                    name=f"concurrent-{i}",
                    body=request_body(
                        model,
                        f"Say hello to reader {i}.",
                        args.max_tokens,
                        stream=bool(i % 2),
                    ),
                )
                for i in range(args.concurrency)
            ],
            args.concurrency,
        )
        if args.image is None:
            record("image", "skipped", reason="enable --image for a vision model")
        else:
            await stage(
                "image",
                [
                    Case(
                        name="image",
                        body=request_body(
                            model,
                            [
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url(args.image)},
                                },
                            ],
                            args.max_tokens,
                        ),
                    )
                ],
            )
        if not args.reasoning_effort:
            record(
                "reasoning_effort",
                "skipped",
                reason="no supported reasoning efforts configured",
            )
        else:
            await stage(
                "reasoning_effort",
                [
                    Case(
                        name=f"reasoning-{effort}",
                        body={
                            **request_body(
                                model,
                                "What is 17 plus 25? Give the answer.",
                                args.max_tokens,
                                stream=True,
                            ),
                            "reasoning_effort": effort,
                        },
                    )
                    for effort in args.reasoning_effort
                ],
            )

        if args.skip_prefix_cache:
            record("prefix_cache", "skipped", reason="disabled by --skip-prefix-cache")
        else:
            body = {
                **request_body(
                    model,
                    f"Cache check {uuid.uuid4().hex}. "
                    + "Remember this passage. " * args.prefix_cache_repeats
                    + "Reply OK.",
                    1,
                ),
                "routed_dp_rank": 0,
            }
            first = await run_case(
                client, Case(name="prefix-first", body=body), args.timeout
            )
            second = await run_case(
                client, Case(name="prefix-repeat", body=body), args.timeout
            )
            if not first["ok"] or not second["ok"]:
                record(
                    "prefix_cache",
                    "failed",
                    prompt_repetitions=args.prefix_cache_repeats,
                    results=[first, second],
                )
            elif any(
                "cached_prompt_tokens" not in result for result in (first, second)
            ):
                record(
                    "prefix_cache",
                    "skipped",
                    reason="response has no per-request cached-token telemetry",
                    prompt_repetitions=args.prefix_cache_repeats,
                    results=[first, second],
                )
            else:
                record(
                    "prefix_cache",
                    "passed"
                    if second["prompt_tokens"] == first["prompt_tokens"]
                    and second["cached_prompt_tokens"] > first["cached_prompt_tokens"]
                    else "failed",
                    prompt_repetitions=args.prefix_cache_repeats,
                    results=[first, second],
                )

        if args.context_length is None:
            record(
                "context_boundary",
                "skipped",
                reason="supply --context-length to check a configured limit",
            )
        else:
            boundary = await context_boundary(
                client, model, args.context_length, args.tokenize_path, args.timeout
            )
            record("context_boundary", **boundary)

        before = await compilation_counter(
            client, args.metrics_path, args.compilation_metric, args.timeout
        )
        for length in args.context_lengths:
            results = []
            for repeat in range(args.repeats + 1):
                # New early prefixes keep cache reuse from hiding the prefill work.
                body = {
                    **request_body(
                        model,
                        f"Request {uuid.uuid4().hex}. "
                        + "context " * length
                        + "Reply OK.",
                        args.max_tokens,
                        stream=True,
                    ),
                    "routed_dp_rank": 0,
                }
                result = await run_case(
                    client,
                    Case(name=f"context-{length}-pass-{repeat}", body=body),
                    args.timeout,
                )
                results.append(result)
            timings = [
                result["ttft_ms"]
                for result in results
                if result["ok"] and result["ttft_ms"] is not None
            ]
            passed = len(timings) == len(results)
            steady = (
                statistics.median(timings[1:]) if len(timings) == len(results) else None
            )
            spike = bool(
                steady is not None
                and timings[0] > steady * args.ttft_ratio
                and timings[0] - steady > args.ttft_slack_ms
            )
            slow = any(value > args.max_ttft_ms for value in timings)
            record(
                f"context_ttft_{length}",
                "passed" if passed and not spike and not slow else "failed",
                requested_words=length,
                first_pass_spike=spike,
                exceeds_ttft_limit=slow,
                steady_ttft_ms=steady,
                results=results,
            )
        after = await compilation_counter(
            client, args.metrics_path, args.compilation_metric, args.timeout
        )
        if not args.context_lengths or before is None or after is None:
            record(
                "kernel_compilation",
                "skipped",
                reason="no compilation counter available; TTFT results alone cannot establish recompilation",
            )
        else:
            record(
                "kernel_compilation",
                "passed" if after == before else "failed",
                metric=args.compilation_metric,
                before=before,
                after=after,
                note="Counter changes can include other server traffic; use an idle deployment.",
            )

        if args.eval_file is None:
            record(
                "evaluation",
                "skipped",
                reason="supply --eval-file with MMLU CSV or JSONL questions",
            )
        else:
            rows = load_evaluation(args.eval_file, args.eval_limit)
            cases = []
            for index, row in enumerate(rows):
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(row["choices"])]
                prompt = (
                    row["question"]
                    + "\n"
                    + "\n".join(
                        f"{label}. {choice}"
                        for label, choice in zip(labels, row["choices"], strict=True)
                    )
                    + "\nRespond with only the letter of the correct answer."
                )
                cases.append(
                    Case(
                        name=f"eval-{index}",
                        body={
                            **request_body(model, prompt, args.eval_max_tokens),
                            **(
                                {"reasoning_effort": args.eval_reasoning_effort}
                                if args.eval_reasoning_effort is not None
                                else {}
                            ),
                        },
                    )
                )
            evaluation = await run_stage(
                client, "evaluation", cases, args.eval_concurrency, args.timeout
            )
            results = []
            for row, result in zip(rows, evaluation["results"], strict=True):
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(row["choices"])]
                predicted = (
                    extract_answer(result.get("content", ""), labels)
                    if result["ok"]
                    else None
                )
                truncated = result.get("finish_reason") == "length"
                results.append(
                    {
                        **result,
                        "answer": row["answer"],
                        "predicted": predicted,
                        "truncated": truncated,
                        "correct": predicted == row["answer"] and not truncated,
                    }
                )
            correct = sum(1 for row in results if row["correct"])
            missing = sum(row["predicted"] is None for row in results)
            truncated = sum(1 for row in results if row["truncated"])
            accuracy = correct / len(results)
            record(
                "evaluation",
                "passed"
                if not missing and not truncated and accuracy >= args.min_accuracy
                else "failed",
                source=str(args.eval_file),
                concurrency=args.eval_concurrency,
                elapsed_s=evaluation["elapsed_s"],
                requested_questions=args.eval_limit,
                reasoning_effort=args.eval_reasoning_effort,
                max_tokens=args.eval_max_tokens,
                questions=len(results),
                correct=correct,
                missing_answers=missing,
                truncated_outputs=truncated,
                accuracy=accuracy,
                minimum_accuracy=args.min_accuracy,
                results=results,
            )

        if args.stress:
            for concurrency in args.stress_concurrency:
                cases = [
                    chat_case(model, i, args.dp_size, args.max_tokens)
                    for i in range(args.stress_requests)
                ]
                if args.skip_logprobs:
                    cases = [
                        replace(
                            case,
                            body={
                                key: value
                                for key, value in case.body.items()
                                if key not in {"logprobs", "top_logprobs"}
                            },
                        )
                        for case in cases
                    ]
                await stage(
                    f"stress_c{concurrency}", cases, concurrency, require_output=False
                )
            robustness = robustness_cases(model, args.dp_size)
            if args.skip_logprobs:
                robustness = [
                    replace(
                        case,
                        body={
                            key: value
                            for key, value in case.body.items()
                            if key not in {"logprobs", "top_logprobs"}
                        },
                    )
                    for case in robustness
                    if case.name != "bad-top-logprobs"
                ]
            await stage(
                "robustness",
                robustness,
                min(8, max(args.stress_concurrency)),
                require_output=False,
            )
            await stage(
                "recovery",
                [
                    Case(
                        name=f"recover-{rank}",
                        body={
                            **request_body(model, "Say hello.", args.max_tokens),
                            "routed_dp_rank": rank,
                        },
                    )
                    for rank in range(args.dp_size)
                ],
                args.dp_size,
            )
        else:
            record(
                "stress",
                "skipped",
                reason="enable --stress for load, invalid-input, disconnect, and recovery checks",
            )
    except Exception as error:
        record("execution", "failed", error=f"{type(error).__name__}: {error}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-logprobs", action="store_true")
    parser.add_argument(
        "--image",
        nargs="?",
        const="builtin",
        help="Test vision with a built-in PNG, local image, or URL",
    )
    parser.add_argument("--reasoning-effort", nargs="+")
    parser.add_argument("--skip-prefix-cache", action="store_true")
    parser.add_argument(
        "--prefix-cache-repeats",
        type=int,
        default=256,
        help=(
            "Repeated phrases in the cache probe (default: 256); choose a prompt "
            "longer than the model's state-checkpoint interval and below its context "
            "limit. Actual prompt and cached token counts are reported."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Configured context limit; exact boundary checking needs chat tokenization",
    )
    parser.add_argument("--tokenize-path", default="/tokenize")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="*",
        default=[128, 512, 2048],
        help="Approximate prompt sizes in repeated words; actual token counts are reported",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Measured repeats per context size after its first pass",
    )
    parser.add_argument("--max-ttft-ms", type=float, default=10000)
    parser.add_argument("--ttft-ratio", type=float, default=3)
    parser.add_argument("--ttft-slack-ms", type=float, default=2000)
    parser.add_argument(
        "--compilation-metric",
        help="Optional Prometheus compilation counter to require unchanged across the context sweep",
    )
    parser.add_argument("--metrics-path", default="/metrics")
    parser.add_argument(
        "--eval-file",
        type=Path,
        help="Local MMLU CSV (question,A,B,C,D,answer; no header) or JSONL {question,choices,answer}",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=100,
        help="Maximum evaluation questions (default: 100)",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=32,
        help="Maximum in-flight evaluation requests (default: 32)",
    )
    parser.add_argument("--eval-max-tokens", type=int, default=256)
    parser.add_argument(
        "--eval-reasoning-effort",
        help="Reasoning effort for evaluation requests only, separate from --reasoning-effort checks",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.5)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-requests", type=int, default=32)
    parser.add_argument("--stress-concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--dp-size", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.tokenize_path.startswith("/") or not args.metrics_path.startswith("/"):
        parser.error("tokenize and metrics paths must start with / on the same server")
    positive = [
        args.timeout,
        args.max_tokens,
        args.concurrency,
        args.prefix_cache_repeats,
        args.repeats,
        args.max_ttft_ms,
        args.ttft_ratio,
        args.eval_limit,
        args.eval_concurrency,
        args.eval_max_tokens,
        args.stress_requests,
        args.dp_size,
        *args.context_lengths,
        *args.stress_concurrency,
    ]
    if args.context_length is not None:
        positive.append(args.context_length)
    if (
        any(not math.isfinite(value) or value <= 0 for value in positive)
        or not math.isfinite(args.ttft_slack_ms)
        or args.ttft_slack_ms < 0
        or not 0 <= args.min_accuracy <= 1
    ):
        parser.error(
            "sizes, counts, timeouts and thresholds must be finite and positive; accuracy must be in [0, 1]"
        )
    return args


async def main(args):
    base = args.url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        pending = args.output.with_suffix(args.output.suffix + ".tmp")
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(args.output)

    try:
        headers = (
            {"Authorization": f"Bearer {os.environ[args.api_key_env]}"}
            if args.api_key_env
            else {}
        )
        async with httpx.AsyncClient(
            base_url=base + "/",
            headers=headers,
            timeout=args.timeout,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=max(
                    args.concurrency, args.eval_concurrency, *args.stress_concurrency
                )
            ),
        ) as client:
            report = await check_server(client, args, on_update=save)
    except Exception as error:
        report = dict(
            ok=False,
            url=base,
            counts=dict(passed=0, failed=1, skipped=0),
            checks=[
                dict(
                    name="startup",
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            ],
        )
        save(report)
    print(
        json.dumps(
            dict(ok=report["ok"], counts=report["counts"], output=str(args.output))
        )
    )
    return report["ok"]


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main(parse_args())) else 1)
