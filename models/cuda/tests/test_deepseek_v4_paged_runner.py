from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from resin import BatchPhase, BatchPlan
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner


def _validation_runner() -> DeepSeekV4CudaRunner:
    runner = DeepSeekV4CudaRunner.__new__(DeepSeekV4CudaRunner)
    runner.vocab_size = 64
    runner.max_seq_len = 256
    runner.max_batch_size = 4
    return runner


def test_managed_pages_allow_continuation_and_arbitrary_decode_rows() -> None:
    runner = _validation_runner()
    continuation = BatchPlan(
        [70],
        [128],
        [4, 5],
        [0, 2],
        phase=BatchPhase.PREFILL,
    )
    decode = BatchPlan(
        [70, 99],
        [130, 7],
        [9, 10],
        [0, 1, 2],
        phase=BatchPhase.DECODE,
    )

    assert runner._validate_plan(continuation) == ((0, 2),)
    assert runner._validate_plan(decode) == ((0, 1), (1, 2))


def test_decode_graph_pads_to_next_batch_shape_and_stages_cache_tables() -> None:
    runner = DeepSeekV4CudaRunner.__new__(DeepSeekV4CudaRunner)
    runner.vocab_size = 32
    runner.speculative_tokens = 0
    runner._dspark_phase_events = {}
    graph_cache = object()
    runner._graph_device_caches = {(4, 128): graph_cache}

    class Graphs:
        shapes = ((2, 128), (4, 128))

        def __init__(self) -> None:
            self.calls = []

        def match(self, predicate):
            return next((shape for shape in self.shapes if predicate(shape)), None)

        def fallback(self, reason):
            raise AssertionError(reason)

        def run(self, shape, inputs):
            self.calls.append((shape, inputs))
            return [11, 12, 13, 99]

    class Sampler:
        @staticmethod
        def host_inputs(plan, vocab_size):
            return {
                "sampling_temperatures": (1.0,) * plan.num_sequences,
                "sampling_top_ps": (1.0,) * plan.num_sequences,
                "sampling_top_ks": (vocab_size,) * plan.num_sequences,
                "sampling_seeds": tuple(range(plan.num_sequences)),
                "sampling_offsets": (0,) * plan.num_sequences,
                "sampling_stochastic_mask": (False,) * plan.num_sequences,
            }

    class Cache:
        @staticmethod
        def graph_plan_inputs(cache, plan):
            assert cache is graph_cache
            assert plan == "cache-plan"
            return {
                "cache.0.parent_ids": (7, 8, 9, 0),
                "cache.0.child_slots": (0, 1, 0, 0),
            }

    runner._graphs = Graphs()
    runner._sampler = Sampler()
    runner._cache = Cache()
    plan = SimpleNamespace(
        num_sequences=3,
        sequence_ids=[1, 2, 3],
        context_lengths=[11, 12, 13],
        token_ids=[5, 6, 7],
        token_indptr=[0, 1, 2, 3],
        positions=np.array([11, 12, 13]),
    )

    assert runner._graph_forward(plan, "cache-plan") == [11, 12, 13]
    shape, inputs = runner._graphs.calls[0]
    assert shape == (4, 128)
    assert inputs["token_ids"].tolist() == [5, 6, 7, 0]
    assert inputs["positions"].tolist() == [11, 12, 13, 1]
    assert inputs["cache.0.parent_ids"] == (7, 8, 9, 0)
    assert inputs["cache.0.child_slots"] == (0, 1, 0, 0)
    assert inputs["sampling_top_ks"] == (32, 32, 32, 32)


def test_prefill_uses_fresh_prompt_then_replays_only_continuation_suffix() -> None:
    runner = _validation_runner()
    runner.device = torch.device("cpu")
    runner.speculative_tokens = 0

    class Cache:
        def __init__(self) -> None:
            self.rows = []

        def row(self, index):
            value = f"cache-row-{index}"
            self.rows.append(value)
            return value

    class Model:
        def __init__(self) -> None:
            self.calls = []

        def forward_target(self, input_ids, *, cache):
            self.calls.append(("fresh", input_ids.tolist(), cache))
            logits = torch.nn.functional.one_hot(
                input_ids[:, -1], num_classes=64
            ).float()
            return logits, None

        def forward_prefill(self, input_ids, start_position, *, cache):
            self.calls.append(
                (
                    "continue",
                    input_ids.tolist(),
                    start_position,
                    cache,
                )
            )
            logits = torch.nn.functional.one_hot(
                input_ids[:, -1], num_classes=64
            ).float()
            return logits, None

    runner._model = Model()
    cache = Cache()
    plan = BatchPlan(
        [7, 8],
        [0, 128],
        [2, 3, 8, 9],
        [0, 2, 4],
        phase=BatchPhase.PREFILL,
    )

    logits = runner._prefill(
        plan,
        ((0, 2), (2, 4)),
        cache,
        torch.tensor(plan.token_ids, dtype=torch.long),
        torch.tensor(plan.positions, dtype=torch.long),
    )

    assert logits.argmax(dim=-1).tolist() == [3, 9]
    assert runner._model.calls == [
        ("fresh", [[2, 3]], "cache-row-0"),
        ("continue", [[8, 9]], 128, "cache-row-1"),
    ]
