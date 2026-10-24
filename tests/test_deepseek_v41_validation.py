"""Checkpoint validation honors EOS and diagnoses shared-prefix mismatches."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from deploy.deepseek_v41_validation import validate_model
from resin import BatchPhase


@pytest.fixture
def validation(tmp_path):
    Tokenizer(models.WordLevel({str(i): i for i in range(16)})).save(
        str(tmp_path / "tokenizer.json")
    )
    runner = SimpleNamespace(
        model_path=tmp_path,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        speculative_tokens=0,
        eos_token_id=1,
        prefill_chunk_size=32,
        cuda_graph_stats={
            "prefill": {"replay_count": 0},
            "decode": {"replay_count": 0},
        },
        _model=SimpleNamespace(modules=lambda: []),
        _speculation=SimpleNamespace(
            benchmark_accepted_drafts=None,
            snapshot=lambda: dict(blocks=0, drafts=0, accepted=0, emitted=0),
        ),
    )

    class Engine:
        context_length = 128

        def __init__(self, eager, captured=None):
            self.outputs = [eager, eager if captured is None else captured]
            self.request = None
            self.submissions = []

        def clear_prefix_cache(self):
            pass

        def submit(self, prompt, **kwargs):
            self.submissions.append(kwargs)
            self.tokens, self.index = self.outputs.pop(0), 0
            self.request = SimpleNamespace(
                id=1,
                done=False,
                cached_tokens=0,
                take_result=lambda: self.tokens,
                prompt_logprobs=[None]
                + [SimpleNamespace(logprob=-0.5) for _ in prompt[1:]],
            )
            return self.request

        def has_work(self):
            return self.request is not None and not self.request.done

        def next_phase(self):
            return BatchPhase.PREFILL if not self.index else BatchPhase.DECODE

        def step(self):
            assert self.request is not None
            mode = "prefill" if not self.index else "decode"
            if getattr(runner, f"cuda_graph_{mode}") != "disabled":
                runner.cuda_graph_stats[mode]["replay_count"] += 1
            token = self.tokens[self.index]
            event = SimpleNamespace(
                request_id=1,
                index=self.index,
                token_id=token,
                logprobs=SimpleNamespace(
                    logprob=-0.2,
                    top_logprobs=[(token, -0.2), (15, -0.3), (14, -2)],
                ),
            )
            self.index += 1
            self.request.done = self.index == len(self.tokens)
            return [event]

        def cancel(self, request):
            request.done = True

    def run(eager, captured=None, *, coordinate=None):
        engine = Engine(eager, captured)
        try:
            result = validate_model(
                runner,
                engine,
                prompt_tokens=[2, 3],
                synchronize=lambda: None,
                coordinate=coordinate,
            )
        finally:
            assert runner.cuda_graph_prefill == "breakable"
            assert runner.cuda_graph_decode == "full"
            if engine.request is not None:
                assert engine.request.done
        assert all(
            row["sampling_params"].stop_token_ids == (1,) for row in engine.submissions
        )
        return result

    return run


def test_completed_answer_honors_eos_and_keeps_reproducible_prompt(validation):
    result = validation([5, 6, 7, 1])
    assert result["captured"]["finished_by"] == "eos"
    assert result["captured"]["graph_replays"]["decode"] == 3
    assert result["prompt_token_ids"] == [2, 3]
    assert result["prompt_text"] == "2 3"


def test_early_eos_cannot_skip_required_decode_replays(validation):
    with pytest.raises(AssertionError, match="EOS before three decode steps"):
        validation([5, 1])


def test_first_difference_reports_same_context_score_margins(validation):
    with pytest.raises(AssertionError, match='"index": 3') as error:
        validation([5, 6, 7, 8, 1], [5, 6, 7, 9, 1])
    assert '"token_id": 8' in str(error.value)
    assert '"token_id": 9' in str(error.value)
    assert '"top1_top2_margin"' in str(error.value)


def test_remote_event_failure_cancels_before_the_next_step_and_restores_policies(
    validation,
):
    stages = []

    def coordinate(stage, error):
        assert error is None
        stages.append(stage)
        if stage == "eager events":
            raise RuntimeError("rank 7 omitted selected scores")

    with pytest.raises(RuntimeError, match="rank 7"):
        validation([5, 6, 7, 1], coordinate=coordinate)
    assert stages == [
        "setup",
        "eager admission",
        "eager phase",
        "eager events",
        "eager cleanup",
    ]
