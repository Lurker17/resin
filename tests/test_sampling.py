from __future__ import annotations

import pytest
import torch

import resin
from resin import BatchPlan, Engine, SamplingParams
from resin.backend.cuda.sampling import FlashInferSampler, SamplingTensors
from resin.backend.sampling import TorchSampler, batch_sampling_inputs
from resin.backend.speculative import SpeculativeStats, accept_speculative_tokens


class RecordingRunner:
    def __init__(self) -> None:
        self.plans = []

    def forward(self, plan):
        self.plans.append(plan)
        return [0] * plan.num_sequences


class TestSamplingContract:
    def test_speculative_acceptance_returns_prefix_plus_one_bonus(self) -> None:
        acceptance = accept_speculative_tokens(
            [[10, 11, 12], [20, 21, 22]],
            [[10, 11, 13, 14], [31, 30, 29, 28]],
        )

        assert acceptance.token_rows == [[10, 11, 13], [31]]
        assert acceptance.accepted_counts == [2, 0]
        assert acceptance.processed_counts == [3, 1]

    def test_speculative_stats_own_shared_runner_bookkeeping(self) -> None:
        stats = SpeculativeStats(3)
        acceptance = stats.accept([[10, 11, 12]], [[10, 21, 22, 23]])

        assert acceptance.token_rows == [[10, 21]]
        assert stats.snapshot(include_draft_tokens=True) == {
            "draft_tokens": 3,
            "blocks": 1,
            "drafts": 3,
            "accepted": 1,
            "emitted": 2,
            "acceptance_rate": 1 / 3,
        }

    def test_sampling_params_validate_defaults_and_preserve_extra_identity(
        self,
    ) -> None:
        marker = object()
        params = SamplingParams(extra=marker)

        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k is None
        assert params.seed is None
        assert params.extra is marker

        for kwargs in (
            {"temperature": -1.0},
            {"temperature": float("inf")},
            {"top_p": 0.0},
            {"top_p": 1.1},
            {"top_k": 0},
            {"seed": -1},
        ):
            with pytest.raises(ValueError):
                SamplingParams(**kwargs)

    def test_engine_freezes_mixed_row_annotations_and_advances_offsets(self) -> None:
        first_extra = {"processor": "first"}
        second_extra = object()
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit(
            [1],
            2,
            sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=32,
                seed=11,
                extra=first_extra,
            ),
        )
        second = engine.submit(
            [2],
            2,
            sampling_params=SamplingParams(
                temperature=0.0,
                seed=22,
                extra=second_extra,
            ),
        )

        engine.step()
        initial = runner.plans[0]
        assert [params.temperature for params in initial.sampling_params] == [0.7, 0.0]
        assert [params.top_p for params in initial.sampling_params] == [0.9, 1.0]
        assert [params.top_k for params in initial.sampling_params] == [32, None]
        assert initial.sampling_params[0].extra is first_extra
        assert initial.sampling_params[1].extra is second_extra
        assert [params.seed for params in initial.sampling_params] == [11, 22]
        assert batch_sampling_inputs(initial, 128)["offsets"] == (32, 32)

        engine.step()
        decode = runner.plans[1]
        assert batch_sampling_inputs(decode, 128)["offsets"] == (64, 64)
        assert first.take_result() == [0, 0]
        assert second.take_result() == [0, 0]

    def test_verified_block_advances_sampling_offset_per_committed_token(self) -> None:
        class BlockRunner:
            def __init__(self) -> None:
                self.plans = []

            def forward(self, plan):
                self.plans.append(plan)
                if plan.phase == resin.BatchPhase.PREFILL:
                    return [1]
                if len(self.plans) == 2:
                    return [[2, 3, 4]]
                return [[5]]

        runner = BlockRunner()
        engine = Engine(runner)
        request = engine.submit([0], 5, sampling_params=SamplingParams(seed=7))

        while engine.has_work():
            engine.step()

        assert [
            batch_sampling_inputs(plan, 128)["offsets"] for plan in runner.plans
        ] == [
            (32,),
            (64,),
            (160,),
        ]
        assert request.take_result() == [1, 2, 3, 4, 5]

    def test_batch_plan_requires_sequence_aligned_sampling_annotations(self) -> None:
        params = SamplingParams()
        with pytest.raises(ValueError, match="one item per sequence"):
            BatchPlan(
                [7, 8],
                [0, 0],
                [1, 2],
                [0, 1, 2],
                sampling_params=[params],
            )


class TestSharedSampler:
    @staticmethod
    def _plan(params, *, sampling_mask=None):
        rows = len(params)
        return BatchPlan(
            list(range(10, 10 + rows)),
            [0] * rows,
            list(range(1, rows + 1)),
            list(range(rows + 1)),
            sampling_params=params,
            sampling_mask=sampling_mask,
        )

    def test_torch_sampler_unifies_greedy_top_k_and_unsampled_rows(self) -> None:
        sampler = TorchSampler()
        logits = torch.tensor([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
        plan = self._plan(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=1.0, top_k=1),
                SamplingParams(temperature=1.0),
            ],
            sampling_mask=[True, True, False],
        )

        assert sampler(logits, plan).tolist() == [1, 1, 1]

    def test_flashinfer_sampler_receives_per_row_filter_and_rng_tensors(self) -> None:
        calls = {}
        sampler = FlashInferSampler.__new__(FlashInferSampler)

        def softmax(logits, *, temperature):
            calls["temperatures"] = temperature.clone()
            return torch.softmax(logits / temperature[:, None], dim=-1)

        def sample(probabilities, top_k, top_p, **kwargs):
            calls["top_k"] = top_k.clone()
            calls["top_p"] = top_p.clone()
            calls["seed"] = kwargs["seed"].clone()
            calls["offset"] = kwargs["offset"].clone()
            calls["return_valid"] = kwargs["return_valid"]
            return (
                torch.tensor([2, 0], dtype=torch.int32),
                torch.tensor([True, True]),
            )

        sampler._softmax = softmax
        sampler._sample = sample
        plan = BatchPlan(
            [10, 11],
            [1, 2],
            [1, 2],
            [0, 1, 2],
            sampling_params=[
                SamplingParams(temperature=0.5, top_p=0.8, top_k=2, seed=31),
                SamplingParams(temperature=0.0, seed=32),
            ],
        )
        logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 1.0]])

        assert sampler(logits, plan).tolist()[0] in {1, 2}
        assert sampler(logits, plan).tolist()[1] == 1
        torch.testing.assert_close(calls["temperatures"], torch.tensor([0.5, 1.0]))
        assert calls["top_k"].tolist() == [2, 3]
        torch.testing.assert_close(calls["top_p"], torch.tensor([0.8, 1.0]))
        assert calls["seed"].tolist() == [31, 32]
        assert calls["offset"].tolist() == [64, 96]
        assert calls["return_valid"] is True

        shifted = sampler.host_inputs(plan, 3, offset_delta=64)
        assert shifted["sampling_offsets"] == (128, 160)

    def test_flashinfer_sampler_rejects_invalid_or_out_of_range_samples(self) -> None:
        sampler = FlashInferSampler.__new__(FlashInferSampler)
        sampler.torch = torch
        sampler._softmax = lambda logits, *, temperature: torch.softmax(
            logits / temperature[:, None], dim=-1
        )
        sampler._sample = lambda *args, **kwargs: (
            torch.tensor([-1, 99, 0], dtype=torch.int32),
            torch.tensor([False, True, True]),
        )
        tensors = SamplingTensors(
            temperatures=torch.ones(3),
            top_ps=torch.ones(3),
            top_ks=torch.full((3,), 100, dtype=torch.int32),
            seeds=torch.zeros(3, dtype=torch.int64),
            offsets=torch.zeros(3, dtype=torch.int64),
            stochastic_mask=torch.ones(3, dtype=torch.bool),
        )
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [0.0, 1.0, 2.0]])

        assert sampler.sample_tensors(logits, tensors).tolist() == [1, 0, 0]
