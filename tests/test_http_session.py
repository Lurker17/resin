"""The routed session retains the ordinary actor's bounded-stream lifecycle."""

import json

from resin import BatchPhase, Engine
from resin._resin import _HttpSession


def test_session_overflow_terminates_client_and_keeps_other_requests_usable():
    class Runner:
        def forward(self, plan):
            return [1] if plan.phase == BatchPhase.PREFILL else [[2] * 40]

    class Codec:
        def render(self, request):
            return {"token_ids": [0], "stop_token_ids": []}

        def create_parser(self, request, prompt_ids):
            class Parser:
                def feed(self, ids):
                    return [
                        {"type": "reasoning_delta", "delta": "a"},
                        {"type": "text_delta", "delta": "b"},
                    ]

            return Parser()

    def admission(request_id, limit):
        return json.dumps(
            [
                {
                    "id": request_id,
                    "model": "test",
                    "messages": [],
                    "tools": [],
                    "max_output_tokens": limit,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": None,
                    "seed": None,
                    "logprobs": False,
                    "top_logprobs": 0,
                    "prompt_logprobs": False,
                    "routed_dp_rank": None,
                    "stop": [],
                    "reasoning_effort": None,
                    "extra": {},
                }
            ]
        )

    engine = Engine(Runner())
    session = _HttpSession(engine, Codec())
    session.admit(admission("overflow", 100))
    first = json.loads(session.process(engine.step()))
    assert [event["type"] for _, event in first] == [
        "started",
        "reasoning_delta",
        "text_delta",
    ]
    # Forty verified tokens produce eighty noncoalescible semantic events.
    overflow = json.loads(session.process(engine.step()))
    assert len(overflow) == 65  # 64 buffered events plus the terminal error.
    assert all(request_id == "overflow" for request_id, _ in overflow)
    assert overflow[-1][1] == {
        "type": "error",
        "data": {"message": "the inference stream ended before completion"},
    }
    assert session.active_count == 0
    assert not engine.has_work() and not engine.closed and not engine.poisoned
    assert session.process([]) == "[]"

    session.admit(admission("ordinary", 1))
    ordinary = json.loads(session.process(engine.step()))
    assert [event["type"] for _, event in ordinary] == [
        "started",
        "reasoning_delta",
        "text_delta",
        "finished",
    ]
    assert all(request_id == "ordinary" for request_id, _ in ordinary)
    assert ordinary[-1][1]["data"]["reason"] == "length"
    assert session.active_count == 0 and session.process([]) == "[]"
    session.close()
    assert not engine.closed
    engine.close()
