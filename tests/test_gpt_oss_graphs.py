from __future__ import annotations

import unittest

from resin.backends.gpt_oss.graphs import DecodeGraphs, GraphInputError, graph_shapes


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Tensor:
    def __init__(self, values):
        self.values = list(values)

    def clone(self):
        return _Tensor(self.values)

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

    @staticmethod
    def equal(left, right):
        return left.values == right.values


class DecodeGraphTests(unittest.TestCase):
    def test_shapes_are_frozen_sorted_and_validated(self) -> None:
        self.assertEqual(graph_shapes(None), ())
        self.assertEqual(graph_shapes({2: 8, 1: 4}), ((1, 4), (2, 8)))
        self.assertEqual(graph_shapes(((2, 8), (1, 8))), ((1, 8), (2, 8)))
        for invalid in ("1x4", ((0, 4),), ((True, 4),), ((1,),)):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                graph_shapes(invalid)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            graph_shapes(((1, 4), (1, 4)))

    def test_selects_the_smallest_width_for_the_exact_batch_size(self) -> None:
        graphs = DecodeGraphs(_Torch(), "cuda:0", ((2, 8), (1, 4), (2, 4)), capacity=8)
        self.assertEqual(graphs.select(2, 3), (2, 4))
        self.assertEqual(graphs.select(2, 5), (2, 8))
        self.assertIsNone(graphs.select(1, 5))

    def test_captures_once_then_replays_the_same_state(self) -> None:
        torch = _Torch()
        graphs = DecodeGraphs(torch, "cuda:0", ((1, 4),), capacity=4)
        staged = [7]
        allocations = []

        def allocate(_shape):
            state = _Tensor([0])
            allocations.append(state)
            return state

        def stage(state):
            state.values[:] = staged

        def run():
            return graphs.run(
                (1, 4),
                allocate=allocate,
                stage=stage,
                body=lambda _state: None,
                output=lambda state: state,
                synchronize_failure=lambda: None,
            )

        self.assertEqual(run(), [7])
        staged[:] = [8]
        self.assertEqual(run(), [8])
        self.assertEqual(len(allocations), 1)
        self.assertEqual(graphs.stats["capture_count"], 1)
        self.assertEqual(graphs.stats["replay_count"], 1)
        self.assertEqual(graphs.stats["captured_shapes"], ((1, 4),))

    def test_host_shape_error_is_retryable_and_drops_fresh_state(self) -> None:
        graphs = DecodeGraphs(_Torch(), "cuda:0", ((1, 4),), capacity=4)
        with self.assertRaises(GraphInputError):
            graphs.run(
                (1, 4),
                allocate=lambda _shape: object(),
                stage=lambda _state: (_ for _ in ()).throw(GraphInputError("bad")),
                body=lambda _state: None,
                output=lambda _state: _Tensor([0]),
                synchronize_failure=lambda: None,
            )
        self.assertIsNone(graphs.entries[(1, 4)].state)
        self.assertFalse(graphs.stats["poisoned"])

    def test_capture_mismatch_poisons_before_future_work(self) -> None:
        graphs = DecodeGraphs(_Torch(), "cuda:0", ((1, 4),), capacity=4)
        synchronized = []

        def body(state):
            state.values[0] += 1

        with self.assertRaisesRegex(RuntimeError, "differs"):
            graphs.run(
                (1, 4),
                allocate=lambda _shape: _Tensor([0]),
                stage=lambda _state: None,
                body=body,
                output=lambda state: state,
                synchronize_failure=lambda: synchronized.append(True),
            )
        self.assertEqual(synchronized, [True])
        self.assertTrue(graphs.stats["poisoned"])
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            graphs.check()

    def test_replay_failure_poisons_without_counting_replay(self) -> None:
        torch = _Torch()
        graphs = DecodeGraphs(torch, "cuda:0", ((1, 4),), capacity=4)
        arguments = dict(
            allocate=lambda _shape: _Tensor([1]),
            stage=lambda _state: None,
            body=lambda _state: None,
            output=lambda state: state,
            synchronize_failure=lambda: None,
        )
        graphs.run((1, 4), **arguments)
        torch.cuda.graphs[0].error = RuntimeError("illegal replay")
        with self.assertRaisesRegex(RuntimeError, "illegal replay"):
            graphs.run((1, 4), **arguments)
        self.assertEqual(graphs.stats["replay_count"], 0)
        self.assertIn("graph replay", graphs.stats["poison_error"])


if __name__ == "__main__":
    unittest.main()
