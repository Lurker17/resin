from __future__ import annotations

import unittest

from resin.backend.cuda.graphs import (
    CudaGraphCache,
    CudaGraphStep,
    GraphInputError,
    graph_shapes,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Tensor:
    def __init__(self, values, *, dtype="int"):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
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
        self.values[:] = source.values

    def tolist(self):
        return list(self.values)


class _Stream:
    def wait_stream(self, _other):
        pass


class _Graph:
    def __init__(self):
        self.error = None

    def replay(self):
        if self.error is not None:
            raise self.error


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
    def __init__(self):
        self.cuda = _Cuda()

    def inference_mode(self):
        return _Context()

    def tensor(self, values, *, dtype, device, pin_memory=False):
        assert device == "cpu"
        assert pin_memory
        return _Tensor(values, dtype=dtype)

    @staticmethod
    def equal(left, right):
        return left.values == right.values


def _cache(torch, shapes, build, synchronize_failure=lambda: None):
    return CudaGraphCache(
        torch,
        "cuda:0",
        graph_shapes(shapes, dimensions=2),
        build=build,
        synchronize_failure=synchronize_failure,
    )


class CudaGraphTests(unittest.TestCase):
    def test_shapes_are_frozen_sorted_and_validated(self) -> None:
        self.assertEqual(graph_shapes(None, dimensions=2), ())
        self.assertEqual(graph_shapes({2: 8, 1: 4}, dimensions=2), ((1, 4), (2, 8)))
        self.assertEqual(graph_shapes(((2, 8), (1, 8)), dimensions=2), ((1, 8), (2, 8)))
        for invalid in ("1x4", ((0, 4),), ((True, 4),), ((1,),)):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                graph_shapes(invalid, dimensions=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            graph_shapes(((1, 4), (1, 4)), dimensions=2)

    def test_matches_the_first_compatible_physical_shape(self) -> None:
        graphs = _cache(
            _Torch(),
            ((2, 8), (1, 4), (2, 4)),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        self.assertEqual(
            graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 3), (2, 4)
        )
        self.assertEqual(
            graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 5), (2, 8)
        )
        self.assertIsNone(graphs.match(lambda shape: shape[0] == 1 and shape[1] >= 5))

    def test_explicit_capture_precedes_replay_of_the_same_static_step(self) -> None:
        torch = _Torch()
        staged = [7]
        allocations = []

        def build(_shape):
            state = _Tensor([0])
            allocations.append(state)
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(torch, ((1, 4),), build)

        self.assertEqual(graphs.capture((1, 4), {"value": staged}), [7])
        staged[:] = [8]
        self.assertEqual(graphs.run((1, 4), {"value": staged}), [8])
        self.assertEqual(len(allocations), 1)
        self.assertEqual(graphs.stats["capture_count"], 1)
        self.assertEqual(graphs.stats["replay_count"], 1)
        self.assertEqual(graphs.stats["captured_shapes"], ((1, 4),))

    def test_run_never_captures_dynamically(self) -> None:
        graphs = _cache(
            _Torch(),
            ((1, 4),),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        with self.assertRaisesRegex(RuntimeError, "has not been captured"):
            graphs.run((1, 4), {})
        self.assertEqual(graphs.stats["capture_count"], 0)

    def test_host_shape_error_is_retryable_and_drops_fresh_step(self) -> None:
        builds = []

        def build(_shape):
            builds.append(True)
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(_Torch(), ((1, 4),), build)
        with self.assertRaises(GraphInputError):
            graphs.capture((1, 4), {})
        self.assertFalse(graphs.stats["poisoned"])
        self.assertEqual(graphs.capture((1, 4), {"value": [1]}), [1])
        self.assertEqual(len(builds), 2)

    def test_capture_mismatch_poisons_before_future_work(self) -> None:
        synchronized = []

        def build(_shape):
            state = _Tensor([0])

            def launch():
                state.values[0] += 1

            return CudaGraphStep({"value": state}, state, launch)

        graphs = _cache(_Torch(), ((1, 4),), build, lambda: synchronized.append(True))
        with self.assertRaisesRegex(RuntimeError, "differs"):
            graphs.capture((1, 4), {"value": [0]})
        self.assertEqual(synchronized, [True])
        self.assertTrue(graphs.stats["poisoned"])
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            graphs.check()

    def test_device_copy_failure_poisons_before_capture(self) -> None:
        state = _Tensor([0])
        state.copy_error = RuntimeError("copy failed")
        graphs = _cache(
            _Torch(),
            ((1, 4),),
            lambda _shape: CudaGraphStep({"value": state}, state, lambda: None),
        )

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            graphs.capture((1, 4), {"value": [1]})
        self.assertEqual(graphs.stats["capture_count"], 0)
        self.assertIn("graph input copy", graphs.stats["poison_error"])

    def test_replay_failure_poisons_without_counting_replay(self) -> None:
        torch = _Torch()

        def build(_shape):
            state = _Tensor([1])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(torch, ((1, 4),), build)
        graphs.capture((1, 4), {"value": [1]})
        torch.cuda.graphs[0].error = RuntimeError("illegal replay")
        with self.assertRaisesRegex(RuntimeError, "illegal replay"):
            graphs.run((1, 4), {"value": [1]})
        self.assertEqual(graphs.stats["replay_count"], 0)
        self.assertIn("graph replay", graphs.stats["poison_error"])


if __name__ == "__main__":
    unittest.main()
