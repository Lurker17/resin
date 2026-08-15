from __future__ import annotations

import pytest
import torch

import resin
from resin import BatchPlan, Engine, SamplingParams
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.sampling import TorchSampler


class RecordingRunner:
    def __init__(self) -> None:
        self.plans = []

    def forward(self, plan):
        self.plans.append(plan)
        return [0] * plan.num_sequences


class TestSamplingContract:
    def test_state_mapping_arena_mint_is_public_backend_api(self) -> None:
        assert callable(resin.mint_state_mapping_arena_id)
        assert "mint_state_mapping_arena_id" in resin.__all__

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
        assert initial.sampling_offsets == [0, 0]

        engine.step()
        decode = runner.plans[1]
        assert decode.sampling_offsets == [32, 0]
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

        assert [plan.sampling_offsets for plan in runner.plans] == [[0], [32], [128]]
        assert request.take_result() == [1, 2, 3, 4, 5]

    def test_batch_plan_requires_row_aligned_sampling_annotations(self) -> None:
        params = SamplingParams()
        with pytest.raises(ValueError, match="one item per sequence"):
            BatchPlan(
                [1, 2],
                [0, 0],
                [7, 8],
                [0, 1, 2],
                [0, 0],
                sampling_params=[params],
            )
        with pytest.raises(ValueError, match="sampling_offsets"):
            BatchPlan(
                [1, 2],
                [0, 0],
                [7, 8],
                [0, 1, 2],
                [0, 0],
                sampling_offsets=[1],
            )


class TestSharedSampler:
    @staticmethod
    def _plan(params, *, sample_sequence_indices=None):
        rows = len(params)
        return BatchPlan(
            list(range(1, rows + 1)),
            [0] * rows,
            list(range(10, 10 + rows)),
            list(range(rows + 1)),
            [0] * rows,
            sampling_params=params,
            sampling_offsets=[0] * rows,
            sample_sequence_indices=sample_sequence_indices,
        )

    def test_torch_sampler_unifies_greedy_top_k_and_unsampled_rows(self) -> None:
        sampler = TorchSampler(torch)
        logits = torch.tensor([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
        plan = self._plan(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=1.0, top_k=1),
                SamplingParams(temperature=1.0),
            ],
            sample_sequence_indices=[0, 1],
        )

        assert sampler(logits, plan).tolist() == [1, 1, 1]

    def test_flashinfer_sampler_receives_per_row_filter_and_rng_tensors(self) -> None:
        calls = {}
        sampler = FlashInferSampler.__new__(FlashInferSampler)
        sampler.torch = torch

        def softmax(logits, *, temperature):
            calls["temperatures"] = temperature.clone()
            return torch.softmax(logits / temperature[:, None], dim=-1)

        def sample(probabilities, top_k, top_p, **kwargs):
            calls["top_k"] = top_k.clone()
            calls["top_p"] = top_p.clone()
            calls["seed"] = kwargs["seed"].clone()
            calls["offset"] = kwargs["offset"].clone()
            return torch.tensor([2, 0], dtype=torch.int32)

        sampler._softmax = softmax
        sampler._sample = sample
        plan = BatchPlan(
            [1, 2],
            [0, 0],
            [10, 11],
            [0, 1, 2],
            [0, 0],
            sampling_params=[
                SamplingParams(temperature=0.5, top_p=0.8, top_k=2, seed=31),
                SamplingParams(temperature=0.0, seed=32),
            ],
            sampling_offsets=[64, 96],
        )
        logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 1.0]])

        assert sampler(logits, plan).tolist() == [2, 1]
        torch.testing.assert_close(calls["temperatures"], torch.tensor([0.5, 1.0]))
        assert calls["top_k"].tolist() == [2, 3]
        torch.testing.assert_close(calls["top_p"], torch.tensor([0.8, 1.0]))
        assert calls["seed"].tolist() == [31, 32]
        assert calls["offset"].tolist() == [64, 96]

        shifted = sampler.host_inputs(plan, 3, offset_delta=64)
        assert shifted["sampling_offsets"] == (128, 160)
