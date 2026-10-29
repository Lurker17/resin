from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_qwen3_5_media import _media

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.runner import CudaRunner
from resin_models_cuda.qwen3_5.checkpoint import DEFAULT_MODEL
from resin_models_cuda.qwen3_5.cuda import (
    Qwen3_5CudaRunner,
    _accepted_verification_state,
)


@pytest.mark.parametrize("state_shape", [(2, 3, 4), (2, 3, 4, 5)])
def test_batched_verification_commits_each_rows_accepted_state(state_shape):
    accepted = torch.tensor([0, 1, 3])
    states = [
        torch.arange(3 * int(np.prod(state_shape))).reshape(3, *state_shape)
        + 1000 * step
        for step in range(4)
    ]
    snapshots = [state.clone() for state in states]
    selected = _accepted_verification_state(states, accepted)
    expected = torch.stack((states[0][0], states[1][1], states[3][2]))
    torch.testing.assert_close(selected, expected)
    for state, snapshot in zip(states, snapshots, strict=True):
        torch.testing.assert_close(state, snapshot)


def test_graph_rope_deltas_stay_on_host_and_match_eager_positions() -> None:
    runner = object.__new__(Qwen3_5CudaRunner)
    # No CUDA operations are needed to prepare the graph's host metadata.
    runner.device = torch.device("cpu")
    plan = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[6, 9],
        token_ids=[1, 2],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
        media=[_media(), None],
    )
    deltas = runner._host_rope_deltas(plan)
    assert isinstance(deltas, np.ndarray)
    np.testing.assert_array_equal(deltas, [-2, 0])
    adjusted = runner._decode_rotary_positions(plan, torch.tensor([6, 9]))
    torch.testing.assert_close(adjusted, torch.tensor([[[4], [9]]] * 3))


def test_qwen38_is_runner_default_and_public_name_is_consistent() -> None:
    parameters = inspect.signature(Qwen3_5CudaRunner.__init__).parameters

    assert parameters["model"].default == DEFAULT_MODEL == "RadixArk/Qwen3.8-27B-NVFP4"
    assert parameters["speculative_tokens"].default == 0
    assert parameters["kv_cache_dtype"].default == "fp8"
    assert parameters["nvfp4_backend"].default == "auto"
    assert parameters["cuda_graph_decode_shapes"].default is None
    assert Qwen3_5CudaRunner.__name__ == "Qwen3_5CudaRunner"


def test_qwen_runner_uses_optional_bundled_mtp_and_engine_cache() -> None:
    source = inspect.getsource(Qwen3_5CudaRunner)

    assert "include_mtp=speculative_tokens > 0" in source
    assert "CacheGeometry.from_config" in source
    assert "cache_write_lookahead_tokens=lookahead" in source
    assert "self._model.mtp" in source
    assert 'torch.device("cuda", torch.cuda.current_device())' in inspect.getsource(
        CudaRunner
    )


def test_qwen_runner_captures_both_ordinary_and_mtp_decode() -> None:
    source = inspect.getsource(Qwen3_5CudaRunner)

    assert issubclass(Qwen3_5CudaRunner, CudaRunner)
    assert "self._configure_cuda_graphs(" in source
    assert "GraphPagedAttentionRuntime" in source
    assert "def capture_cuda_graphs" not in source
    assert "def _new_decode_graph_step" in source
    assert "self._mtp_forward(" in source
    assert "self._run_cuda_graph(" in source


def test_speculative_decode_capacity_includes_mtp_verification_lookahead() -> None:
    runner = object.__new__(Qwen3_5CudaRunner)
    runner.max_batch_size = 4
    runner.max_seq_len = 32
    runner.speculative_tokens = 3
    valid = BatchPlan(
        sequence_ids=[1],
        context_lengths=[26],
        token_ids=[7],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
    )
    runner._validate_plan(valid)

    too_long = BatchPlan(
        sequence_ids=[1],
        context_lengths=[28],
        token_ids=[7],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
    )
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        runner._validate_plan(too_long)


@pytest.mark.parametrize(
    "forced,accepted,wanted",
    [
        (None, [0, 1], [[9], [7, 13]]),
        (0, [0, 0], [[9], [7]]),
        (1, [1, 1], [[5, 11], [7, 13]]),
    ],
)
def test_fast_graph_commit_depth_mtp_repair_stats_and_timing_on_cpu(
    monkeypatch, forced, accepted, wanted
):
    """Execute the real graph closure with CPU tensor operations and fake model kernels."""
    import sys
    from dataclasses import dataclass
    from types import SimpleNamespace

    import resin_models_cuda.qwen3_5.cuda as runtime
    from resin.backend.speculative import SpeculativeStats

    @dataclass
    class Pool:
        destination: torch.Tensor
        pending: torch.Tensor | None = None

    class Timing:
        def __init__(self, categories):
            self.categories, self.marks = categories, []

        def mark(self, index):
            self.marks.append(index)

    seen = {}
    windows = torch.arange(10).reshape(2, 1, 1, 5)
    hidden = torch.tensor([[[10.0], [11.0]], [[20.0], [21.0]]])
    targets = torch.tensor([[9, 11], [7, 13]])

    def commit(pool, layers, stored, destination, counts, **kwargs):
        seen["commit_positions"], seen["accepted"] = destination.clone(), counts.clone()

    monkeypatch.setitem(
        sys.modules,
        "resin_models_cuda.qwen3_5.gdn_decode",
        SimpleNamespace(gdn_commit=commit),
    )
    monkeypatch.setattr(runtime, "graph_sample_block", lambda *args, **kwargs: targets)
    monkeypatch.setattr(
        runtime, "graph_sample", lambda *args, **kwargs: torch.tensor([17, 19])
    )

    def store(positions, convolution, selected):
        seen["stored"] = positions.clone(), convolution.clone(), selected.clone()

    def mtp(cache, inputs, previous_hidden, positions, **kwargs):
        seen["mtp_inputs"] = inputs.clone()
        return hidden

    # This fixture deliberately replaces CUDA-only collaborators with CPU stubs.
    runner = cast(Any, object.__new__(Qwen3_5CudaRunner))
    runner.device, runner.speculative_tokens = torch.device("cpu"), 1
    runner._speculation = SpeculativeStats(1)
    runner._speculation.accept([[30]], [[30, 31]])  # Prior eager-path accounting.
    runner._speculation.benchmark_accepted_drafts = forced
    runner._drafts, runner._media_features = {}, {}
    runner._pending = torch.empty(1, 2, 1, 1, 4)
    runner._pair_offsets, runner._window_offsets = (
        torch.arange(2)[None],
        torch.arange(3),
    )
    runner.attention_config = SimpleNamespace(
        linear_layer_count=1,
        linear_num_value_heads=1,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
    )
    runner._sampler = SimpleNamespace(
        empty=lambda *args: None, graph_inputs=lambda *args: {}
    )
    runner._model = SimpleNamespace(logits=lambda value: value)
    runner._cache = SimpleNamespace(
        graph_device_inputs=lambda *args: {}, reset_null=lambda: None
    )
    cache = SimpleNamespace(
        recurrent_pool=lambda *args: Pool(torch.tensor([100, 200])),
        convolution_at=lambda *args: torch.empty(2, 1, 1, 3),
        state_offsets=lambda positions: positions,
        store_convolution=store,
    )
    runner._graph_device_caches = {(2, 8192): cache}
    runner._graph_attention_for = lambda shape: SimpleNamespace(
        reset=lambda: None, refresh=lambda positions: None, graph_inputs=lambda: {}
    )
    runner._new_phase_timing = lambda *categories: Timing(categories)
    runner._target_forward = lambda *args, **kwargs: (hidden, None, windows)
    runner._mtp_forward = mtp
    step = runner._new_decode_graph_step((2, 8192))
    step.inputs["token_ids"].copy_(torch.tensor([3, 4]))
    step.inputs["positions"].copy_(torch.tensor([4095, 4096]))
    step.inputs["rope_deltas"].zero_()
    step.inputs["draft_ids"].copy_(torch.tensor([5, 7]))
    step.launch()
    output = runner._fast_speculative_output([101, 202], step.output.tolist())
    rows = [
        output.token_ids[start:end].tolist()
        for start, end in zip(output.token_indptr, output.token_indptr[1:])
    ]
    assert rows == wanted
    torch.testing.assert_close(seen["accepted"], torch.tensor(accepted))
    positions = torch.tensor([4095, 4096]) + torch.tensor(accepted)
    torch.testing.assert_close(seen["commit_positions"], positions)
    stored_positions, convolution, selected = seen["stored"]
    torch.testing.assert_close(stored_positions, positions)
    for row, count in enumerate(accepted):
        torch.testing.assert_close(
            convolution[row], windows[row, ..., count + 1 : count + 4]
        )
        torch.testing.assert_close(selected[row], hidden[row, count])
    torch.testing.assert_close(
        seen["mtp_inputs"][:, 0], torch.tensor([row[0] for row in wanted])
    )
    assert runner.speculative_stats == {
        "draft_tokens": 1,
        "blocks": 3,
        "drafts": 3,
        "accepted": 1 + sum(accepted),
        "emitted": 2 + sum(map(len, wanted)),
        "acceptance_rate": (1 + sum(accepted)) / 3,
    }
    assert runner._drafts == {101: 17, 202: 19}
    runner.release([101])
    assert runner._drafts == {202: 19}
    timing = step.phase_timing
    assert isinstance(timing, Timing)
    assert timing.categories == ("target_verify", "decode", "draft")
    assert timing.marks == [0, 1, 2, 3]


@pytest.fixture
def _cpu_completion_events(monkeypatch):
    from types import SimpleNamespace

    from resin.backend.cuda.staging import TensorStager

    def host_buffer(self, name, shape, dtype):
        value = self.host_tensors.get(name)
        if value is None or value.shape != shape or value.dtype != dtype:
            with torch.inference_mode(False):
                value = torch.empty(shape, dtype=dtype, device="cpu")
            self.host_tensors[name] = value
        return value

    # Exercise real staging/copy semantics without CUDA's pinned allocator.
    monkeypatch.setattr(TensorStager, "_host_buffer", host_buffer)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *args, **kwargs: SimpleNamespace(
            record=lambda *args: None, synchronize=lambda: None
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: None)


@pytest.mark.parametrize(
    "forced,wanted",
    [(None, [[9], [7, 13]]), (0, [[9], [7]]), (1, [[5, 11], [7, 13]])],
)
def test_async_completion_preserves_outputs_drafts_and_shared_stats(
    forced, wanted, _cpu_completion_events
):
    from types import SimpleNamespace

    from resin.backend.speculative import SpeculativeStats

    runner = cast(Any, object.__new__(Qwen3_5CudaRunner))
    runner.device, runner.speculative_tokens = torch.device("cpu"), 1
    runner.runner_config = SimpleNamespace(cache_write_lookahead_tokens=2)
    runner._speculation = SpeculativeStats(1, benchmark_accepted_drafts=forced)
    runner._drafts, runner._media_features = {}, {}
    graph_output = torch.tensor(
        [[5, 9, 11, 17], [7, 7, 13, 19], [0, 0, 0, 21], [0, 0, 0, 22]]
    )
    runner._run_cuda_graph = lambda *args: graph_output
    plan = BatchPlan(
        sequence_ids=[101, 202],
        context_lengths=[127, 254],
        token_ids=[3, 4],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    completion = runner._enqueue_graph_output(
        plan, (4, 8192), {"positions": np.asarray([127, 254, 1, 1])}
    )
    feedback = completion.feedback
    assert feedback.sequence_ids == (101, 202) and feedback.shape == (4, 8192)
    torch.testing.assert_close(
        feedback.positions[:2],
        torch.tensor([127 + len(wanted[0]), 254 + len(wanted[1])]),
    )
    torch.testing.assert_close(feedback.positions[2:], torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(feedback.draft_ids, torch.tensor([17, 19, 0, 0]))
    assert runner.speculative_stats["blocks"] == 0  # Counters advance at harvest.
    graph_output.zero_()  # A later replay may reuse every graph output buffer.
    output = completion.result()
    actual = [
        output.token_ids[start:end].tolist()
        for start, end in zip(output.token_indptr, output.token_indptr[1:])
    ]
    assert actual == wanted
    assert runner.speculative_stats == {
        "draft_tokens": 1,
        "blocks": 2,
        "drafts": 2,
        "accepted": sum(map(len, wanted)) - 2,
        "emitted": sum(map(len, wanted)),
        "acceptance_rate": (sum(map(len, wanted)) - 2) / 2,
    }
    assert runner._drafts == {101: 17, 202: 19}
    runner.release([101])
    assert completion.result() is output
    assert runner._drafts == {202: 19}  # Repeated harvest cannot resurrect release.


@pytest.mark.parametrize(
    "start,shapes,drain",
    [
        (124, [(1, 126), (2, 256)], True),
        (124, [(1, 126)], True),
        (127, [(2, 8192)], False),
    ],
)
def test_owned_feedback_crosses_kv_page_and_drains_graph_limits(
    monkeypatch, start, shapes, drain, _cpu_completion_events
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import resin_models_cuda.qwen3_5.cuda as runtime
    from resin.backend.cuda.staging import TensorStager
    from resin.backend.sampling import token_output
    from resin.backend.speculative import SpeculativeStats

    seen: dict[str, Any] = {"stages": 0, "positions": [], "fallback": 0}
    runner = cast(Any, object.__new__(Qwen3_5CudaRunner))
    runner.device, runner.speculative_tokens = torch.device("cpu"), 0
    runner.config = SimpleNamespace(vocab_size=32)
    runner.vocab_size = 32
    runner.runner_config = SimpleNamespace(
        cache_write_lookahead_tokens=0, cache_write_reservation_tokens=136
    )
    runner.cache_layout = None
    runner._speculation, runner._drafts = SpeculativeStats(0), {}
    runner._cuda_failure = nullcontext()
    runner._check_cuda_graphs = lambda phase: None
    runner._validate_plan = lambda plan: None
    runner._host_rope_deltas = lambda plan: np.zeros(1, dtype=np.int64)
    runner._sampler = None
    runner._select_cuda_graph = lambda mode, match: next(
        (shape for shape in shapes if match(shape)), None
    )
    runner._graph_device_caches = dict.fromkeys(shapes)
    runner._graph_attention = {
        shape: SimpleNamespace(plan=lambda *args: {}) for shape in shapes
    }
    runner._feedback_graph_inputs = {
        shape: {
            name: torch.zeros(shape[0], dtype=torch.long)
            for name in ("token_ids", "positions", "rope_deltas", "sampling_offsets")
        }
        for shape in shapes
    }
    outputs = {shape: torch.empty(shape[0], 1, dtype=torch.long) for shape in shapes}

    def stage(*args):
        seen["stages"] += 1
        return {}

    def run(mode, shape, inputs):
        buffers = runner._feedback_graph_inputs[shape]
        TensorStager("cpu").copy_into(buffers, inputs)
        seen["positions"].append(int(buffers["positions"][0]))
        outputs[shape][:, 0].copy_(buffers["token_ids"] + 1)
        return outputs[shape]

    def forward(plan):
        seen["fallback"] += 1
        return token_output([[int(plan.token_ids[0]) + 1]])

    runner._cache = SimpleNamespace(graph_plan_inputs=stage)
    runner._run_cuda_graph, runner.forward = run, forward
    cache_plan = SimpleNamespace(
        groups=(
            SimpleNamespace(block_indptr=np.asarray([0, 1])),
            SimpleNamespace(block_indptr=np.asarray([0, 1])),
        )
    )
    monkeypatch.setattr(
        runtime.PagedCachePlan, "from_plan", lambda *args, **kwargs: cache_plan
    )
    monkeypatch.setattr(
        runtime,
        "pad_sampling_inputs",
        lambda sampler, plan, vocab, padding: {
            "sampling_offsets": np.zeros(1 + padding, dtype=np.int64)
        },
    )

    def plan(position, token):
        return BatchPlan(
            sequence_ids=[101],
            context_lengths=[position],
            token_ids=[token],
            token_indptr=[0, 1],
            phase=BatchPhase.DECODE,
        )

    first = runner._graph_forward(plan(start, 7), cache_plan, enqueue=True)
    second = runner.enqueue(plan(start + 1, 7), feedback=first.feedback)
    assert seen["stages"] == 1  # Continuations retain their original dense tables.
    assert runner.benchmark_counters == {
        "async.enqueues": 2,
        "async.continuations": 1,
        "host.attention_plans": 1,
    }
    assert first.result().token_ids.tolist() == [8]
    assert second.result().token_ids.tolist() == [9]
    assert seen["positions"] == [start, start + 1]
    if drain:
        assert second.feedback is None
        third = runner.enqueue(plan(start + 2, 9))
        output = third.result() if callable(getattr(third, "result", None)) else third
        assert output.token_ids.tolist() == [10]
        assert seen["fallback"] == (len(shapes) == 1)
    else:
        assert first.feedback.positions[0] == 128  # Crossed a128-token KV page.
        assert second.feedback is not None
