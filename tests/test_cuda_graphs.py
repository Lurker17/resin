from __future__ import annotations

import numpy as np
import pytest

from resin.backend.cuda import graphs as graph_runtime
from resin.backend.cuda import staging
from resin.backend.cuda.graphs import (
    DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_SEQS,
    CudaGraphCache,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphRegion,
    CudaGraphRunner,
    CudaGraphStep,
    GraphInputError,
    default_decode_graph_shapes,
    graph_shapes,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Tensor:
    def __init__(self, values, *, dtype="int"):
        self.values = np.array(values, copy=True)
        self.dtype = dtype
        self.shape = self.values.shape
        self.copy_error = None

    def clone(self):
        return _Tensor(self.values, dtype=self.dtype)

    def view(self, shape):
        size = 1
        for dimension in shape:
            size *= dimension
        if size != len(self.values):
            raise ValueError("shape mismatch")
        self.shape = shape
        return self

    def copy_(self, source, *, non_blocking=False):
        del non_blocking
        if self.copy_error is not None:
            raise self.copy_error
        self.values[...] = source.values

    def numpy(self):
        return self.values.reshape(self.shape)

    def tolist(self):
        return self.values.tolist()


class _Stream:
    def wait_stream(self, _other):
        pass


class _Graph:
    def __init__(self):
        self.error = None
        self.replay_count = 0

    def replay(self):
        if self.error is not None:
            raise self.error
        self.replay_count += 1


class _Cuda:
    def __init__(self):
        self.graphs = []

    def memory_allocated(self, _device):
        return 100

    def current_stream(self, _device):
        return _Stream()

    def Stream(self, *, device):
        del device
        return _Stream()

    def stream(self, _stream):
        return _Context()

    def synchronize(self, _device):
        pass

    def device(self, _device):
        return _Context()

    def CUDAGraph(self):
        graph = _Graph()
        self.graphs.append(graph)
        return graph

    def graph(self, _graph):
        return _Context()


class _Torch:
    Tensor = _Tensor

    def __init__(self):
        self.cuda = _Cuda()
        self.host_allocations = 0

    def inference_mode(self):
        return _Context()

    def tensor(self, values, *, dtype, device, pin_memory=False):
        assert device == "cpu"
        assert pin_memory
        return _Tensor(values, dtype=dtype)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        assert device == "cpu"
        assert pin_memory
        self.host_allocations += 1
        return _Tensor(np.empty(shape), dtype=dtype)

    @staticmethod
    def equal(left, right):
        return np.array_equal(left.values, right.values)


def _cache(shapes, build, synchronize_failure=lambda: None):
    return CudaGraphCache(
        "cuda:0",
        graph_shapes(shapes, dimensions=2),
        build=build,
        synchronize_failure=synchronize_failure,
    )


class TestCudaGraph:
    def test_runner_reuses_one_mechanism_across_execution_modes(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        events = []

        def build_prefill(_shape):
            state = _Tensor([0])
            return CudaGraphProgram(
                inputs={"value": state},
                output=state,
                regions=(
                    CudaGraphRegion("attention", lambda: None),
                    CudaGraphRegion("collective", lambda: None, capture=False),
                    CudaGraphRegion("moe", lambda: None),
                ),
            )

        def build_decode(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = CudaGraphRunner(
            "cuda:0",
            {
                "prefill": CudaGraphMode(
                    ((4, 128),), build_prefill, lambda _shape: {"value": [1]}
                ),
                "decode": CudaGraphMode(
                    ((4, 4096),), build_decode, lambda _shape: {"value": [2]}
                ),
            },
            synchronize_failure=lambda: None,
        )

        stats = graphs.capture_all(
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )

        assert events == ["before", "after"]
        assert stats["prefill"]["graph_region_count"] == 2
        assert stats["prefill"]["graph_break_count"] == 1
        assert stats["decode"]["graph_region_count"] == 1
        assert graphs.mode("prefill").run((4, 128), {"value": [3]}) == [3]
        assert graphs.mode("decode").run((4, 4096), {"value": [4]}) == [4]

    def test_shared_decode_defaults_use_one_history_capacity(self) -> None:
        assert DEFAULT_MAX_NUM_SEQS == 128
        assert DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE == 1024
        assert default_decode_graph_shapes(4096, speculative=True) == tuple(
            (batch_size, 4096)
            for batch_size in (
                *range(1, 9),
                *range(12, 33, 4),
                *range(40, 65, 8),
                *range(96, 257, 32),
                *range(384, 1025, 128),
            )
        )
        assert default_decode_graph_shapes(4096, speculative=False) == tuple(
            (batch_size, 4096)
            for batch_size in (
                1,
                2,
                4,
                8,
                *range(12, 17, 4),
                *range(24, 33, 8),
                *range(64, 129, 32),
                *range(256, 1025, 128),
            )
        )

    def test_shared_decode_defaults_respect_lower_sequence_limit(self) -> None:
        assert default_decode_graph_shapes(
            128, speculative=True, max_batch_size=16
        ) == tuple((batch_size, 128) for batch_size in (*range(1, 9), 12, 16))
        for name, value in (("width", 0), ("max_batch_size", True)):
            arguments = {"width": 8, "max_batch_size": 128, name: value}
            with pytest.raises(ValueError, match=name):
                default_decode_graph_shapes(speculative=True, **arguments)

    def test_shapes_are_frozen_sorted_and_validated(self) -> None:
        assert graph_shapes(None, dimensions=2) == ()
        assert graph_shapes({2: 8, 1: 4}, dimensions=2) == ((1, 4), (2, 8))
        assert graph_shapes(((2, 8), (1, 8)), dimensions=2) == ((1, 8), (2, 8))
        for invalid in ("1x4", ((0, 4),), ((True, 4),), ((1,),)):
            with (
                pytest.raises((TypeError, ValueError)),
            ):
                graph_shapes(invalid, dimensions=2)
        with pytest.raises(ValueError, match="duplicate"):
            graph_shapes(((1, 4), (1, 4)), dimensions=2)

    def test_matches_the_first_compatible_physical_shape(self) -> None:
        graphs = _cache(
            ((2, 8), (1, 4), (2, 4)),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        assert graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 3) == (2, 4)
        assert graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 5) == (2, 8)
        assert graphs.match(lambda shape: shape[0] == 1 and shape[1] >= 5) is None

    def test_explicit_capture_precedes_replay_of_the_same_static_step(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        staged = [7]
        allocations = []

        def build(_shape):
            state = _Tensor([0])
            allocations.append(state)
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)

        assert graphs.capture((1, 4), {"value": staged}) == [7]
        staged[:] = [8]
        assert graphs.run((1, 4), {"value": staged}) == [8]
        assert len(allocations) == 1
        assert graphs.stats["capture_count"] == 1
        assert graphs.stats["replay_count"] == 1
        assert graphs.stats["captured_shapes"] == ((1, 4),)

    def test_capture_all_owns_shape_iteration_and_model_state_hooks(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        events = []

        def build(shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4), (2, 4)), build)

        stats = graphs.capture_all(
            lambda shape: events.append(("inputs", shape)) or {"value": [1]},
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )
        assert stats["captured_shapes"] == ((1, 4), (2, 4))
        assert events == [
            "before",
            ("inputs", (1, 4)),
            ("inputs", (2, 4)),
            "after",
        ]

        events.clear()
        graphs.capture_all(
            lambda _shape: pytest.fail("captured shapes must not rebuild inputs"),
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )
        assert events == ["before", "after"]

    def test_replay_reuses_persistent_pinned_staging(self, monkeypatch) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        graphs.capture((1, 4), {"value": [1]})
        allocations_after_capture = torch.host_allocations

        graphs.run((1, 4), {"value": [2]})
        graphs.run((1, 4), {"value": [3]})

        assert torch.host_allocations == allocations_after_capture

    def test_program_replays_captured_regions_around_an_eager_break(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        events = []

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphProgram(
                inputs={"value": state},
                output=state,
                regions=(
                    CudaGraphRegion("before", lambda: events.append("before")),
                    CudaGraphRegion(
                        "unsupported",
                        lambda: events.append("break"),
                        capture=False,
                    ),
                    CudaGraphRegion("after", lambda: events.append("after")),
                ),
            )

        graphs = _cache(((1, 4),), build)
        assert graphs.capture((1, 4), {"value": [7]}) == [7]
        assert len(torch.cuda.graphs) == 2
        events.clear()

        assert graphs.run((1, 4), {"value": [8]}) == [8]
        assert events == ["break"]
        assert [graph.replay_count for graph in torch.cuda.graphs] == [2, 2]
        assert graphs.stats["graph_region_count"] == 2
        assert graphs.stats["graph_break_count"] == 1

    def test_program_requires_at_least_one_captured_region(self) -> None:
        with pytest.raises(ValueError, match="capture at least one"):
            CudaGraphProgram(
                inputs={},
                output=_Tensor([0]),
                regions=(CudaGraphRegion("eager", lambda: None, capture=False),),
            )

    def test_capture_can_prepare_mutable_state_before_each_parity_run(
        self, monkeypatch
    ) -> None:
        prepared = []

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep(
                {"value": state},
                state,
                lambda: None,
                prepare=lambda: prepared.append(True),
            )

        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        graphs = _cache(((1, 4),), build)

        graphs.capture((1, 4), {"value": [7]})

        assert len(prepared) == 5

    def test_graph_inputs_can_copy_directly_from_device_tensors(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        source = _Tensor([9])

        assert graphs.capture((1, 4), {"value": source}) == [9]

    def test_run_never_captures_dynamically(self) -> None:
        graphs = _cache(
            ((1, 4),),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        with pytest.raises(RuntimeError, match="has not been captured"):
            graphs.run((1, 4), {})
        assert graphs.stats["capture_count"] == 0

    def test_host_shape_error_is_retryable_and_drops_fresh_step(
        self, monkeypatch
    ) -> None:
        builds = []

        def build(_shape):
            builds.append(True)
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        graphs = _cache(((1, 4),), build)
        with pytest.raises(GraphInputError):
            graphs.capture((1, 4), {})
        assert not graphs.stats["poisoned"]
        assert graphs.capture((1, 4), {"value": [1]}) == [1]
        assert len(builds) == 2

    def test_capture_mismatch_poisons_before_future_work(self, monkeypatch) -> None:
        synchronized = []

        def build(_shape):
            state = _Tensor([0])

            def launch():
                state.values[0] += 1

            return CudaGraphStep({"value": state}, state, launch)

        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        graphs = _cache(((1, 4),), build, lambda: synchronized.append(True))
        with pytest.raises(RuntimeError, match="differs"):
            graphs.capture((1, 4), {"value": [0]})
        assert synchronized == [True]
        assert graphs.stats["poisoned"]
        with pytest.raises(RuntimeError, match="poisoned"):
            graphs.check()

    def test_device_copy_failure_poisons_before_capture(self, monkeypatch) -> None:
        state = _Tensor([0])
        state.copy_error = RuntimeError("copy failed")
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)
        graphs = _cache(
            ((1, 4),),
            lambda _shape: CudaGraphStep({"value": state}, state, lambda: None),
        )

        with pytest.raises(RuntimeError, match="copy failed"):
            graphs.capture((1, 4), {"value": [1]})
        assert graphs.stats["capture_count"] == 0
        assert "graph input copy" in graphs.stats["poison_error"]

    def test_replay_failure_poisons_without_counting_replay(self, monkeypatch) -> None:
        torch = _Torch()
        monkeypatch.setattr(staging, "torch", torch)
        monkeypatch.setattr(graph_runtime, "torch", torch)

        def build(_shape):
            state = _Tensor([1])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        graphs.capture((1, 4), {"value": [1]})
        torch.cuda.graphs[0].error = RuntimeError("illegal replay")
        with pytest.raises(RuntimeError, match="illegal replay"):
            graphs.run((1, 4), {"value": [1]})
        assert graphs.stats["replay_count"] == 0
        assert "graph replay" in graphs.stats["poison_error"]
