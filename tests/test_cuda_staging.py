"""Deterministic delayed H2D reads exercise source lifetime without a GPU."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from resin.backend.cuda import graphs, staging


class _Tensor:
    def __init__(self, values, runtime, *, device="cpu", dtype="int64"):
        self.values = np.array(values, copy=True, dtype=dtype)
        self.shape = self.values.shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.runtime = runtime
        self.copy_error = None

    def numpy(self):
        return self.values

    def view(self, shape):
        if self.shape != shape:
            raise ValueError("shape mismatch")
        return self

    def copy_(self, source, *, non_blocking=False):
        if self.copy_error is not None:
            raise self.copy_error
        if self.device.type == "cpu":
            np.copyto(self.values, source.values)
            return
        assert non_blocking
        # Read the pinned source when the device gets here, not at enqueue.
        self.runtime.queue.append(lambda: np.copyto(self.values, source.values))


class _Event:
    def __init__(self, runtime):
        self.runtime = runtime
        self.complete = False

    def record(self, stream):
        assert stream is self.runtime
        if self.runtime.event_error is not None:
            # An unrecorded CUDA event may report complete. It is not a fence.
            self.complete = True
            raise self.runtime.event_error
        self.runtime.queue.append(lambda: setattr(self, "complete", True))

    def query(self):
        return self.complete


class _Runtime:
    Tensor = _Tensor

    def __init__(self):
        self.queue = []
        self.host_allocations = 0
        self.event_error = None
        self.cuda = SimpleNamespace(
            Event=lambda: _Event(self),
            current_stream=lambda _device: self,
            memory_allocated=lambda _device: 0,
            device=lambda _device: nullcontext(),
        )

    def empty(self, shape, *, dtype, device, pin_memory):
        assert device == "cpu" and pin_memory
        self.host_allocations += 1
        return _Tensor(np.empty(shape), self, dtype=dtype)

    def inference_mode(self):
        return nullcontext()

    def drain(self, count=None):
        for _ in range(len(self.queue) if count is None else count):
            self.queue.pop(0)()

    def target(self):
        return _Tensor([0], self, device="cuda")

    def observe(self, target, observations):
        self.queue.append(lambda: observations.append(target.values.copy().tolist()))


@pytest.fixture
def runtime(monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(staging, "torch", runtime)
    monkeypatch.setattr(graphs, "torch", runtime)
    return runtime


def test_queued_batches_keep_distinct_pinned_sources_at_arbitrary_depth(runtime):
    pool = staging.TensorStagerPool("cuda")
    target = runtime.target()
    observed = []
    for value in range(8):
        pool.copy_into({"tokens": target}, {"tokens": [value]})
        runtime.observe(target, observed)
    assert runtime.host_allocations == 8
    runtime.drain()
    assert observed == [[value] for value in range(8)]

    pool.copy_into({"tokens": target}, {"tokens": [8]})
    runtime.observe(target, observed)
    runtime.drain()
    assert observed[-1] == [8]
    assert runtime.host_allocations == 8


def test_reuse_requires_copy_completion_but_not_graph_completion(runtime):
    pool = staging.TensorStagerPool("cuda")
    target = runtime.target()
    observed = []
    pool.copy_into({"tokens": target}, {"tokens": [1]})
    runtime.observe(target, observed)
    pool.copy_into({"tokens": target}, {"tokens": [2]})
    runtime.observe(target, observed)
    runtime.drain(2)  # First H2D + fence, with its graph still queued.
    pool.copy_into({"tokens": target}, {"tokens": [3]})
    runtime.observe(target, observed)
    assert runtime.host_allocations == 2
    runtime.drain()
    assert observed == [[1], [2], [3]]


def test_partial_copy_failure_preserves_already_queued_sources(runtime):
    pool = staging.TensorStagerPool("cuda")
    first, failing = runtime.target(), runtime.target()
    failing.copy_error = RuntimeError("copy failed")
    with pytest.raises(RuntimeError, match="copy failed"):
        pool.copy_into({"a": first, "b": failing}, {"a": [1], "b": [2]})
    pool.copy_into({"a": runtime.target()}, {"a": [3]})
    runtime.drain()
    assert first.values.tolist() == [1]


def test_cpu_tensor_inputs_are_owned_before_caller_can_mutate(runtime):
    pool = staging.TensorStagerPool("cuda")
    target = runtime.target()
    source = _Tensor([1], runtime)
    observed = []
    for value in range(1, 9):
        source.values[...] = value
        pool.copy_into({"tokens": target}, {"tokens": source})
        runtime.observe(target, observed)
    source.values[...] = -1
    runtime.drain()
    assert observed == [[value] for value in range(1, 9)]


def test_failed_event_record_quarantines_unfenced_generation(runtime):
    pool = staging.TensorStagerPool("cuda")
    first = runtime.target()
    runtime.event_error = RuntimeError("record failed")
    with pytest.raises(RuntimeError, match="record failed"):
        pool.copy_into({"tokens": first}, {"tokens": [1]})
    runtime.event_error = None
    pool.copy_into({"tokens": runtime.target()}, {"tokens": [2]})
    runtime.drain()
    assert first.values.tolist() == [1]
    assert runtime.host_allocations == 2
    assert pool._pending[0].event is None


def test_alias_and_device_only_inputs_do_not_allocate_pinned_buffers(runtime):
    pool = staging.TensorStagerPool("cuda")
    target, source = runtime.target(), runtime.target()
    inputs = {"tokens": target}
    pool.copy_into(inputs, inputs)
    assert not runtime.queue
    pool.copy_into(inputs, {"tokens": source})
    assert len(runtime.queue) == 1  # No event needed for pool-owned host memory.
    assert runtime.host_allocations == 0


def test_graph_cache_queued_replays_preserve_each_host_input(runtime, monkeypatch):
    target = runtime.target()
    observed = []

    def capture(self, entry, _allocation_before):
        entry.graphs = (
            SimpleNamespace(replay=lambda: runtime.observe(target, observed)),
        )
        entry.captured = True
        runtime.drain()

    monkeypatch.setattr(graphs.CudaGraphCache, "_capture", capture)
    cache = graphs.CudaGraphCache(
        "cuda",
        ((1, 1),),
        build=lambda _shape: graphs.CudaGraphStep(
            inputs={"tokens": target},
            output=target,
            launch=lambda: None,
            output_to_host=False,
        ),
        synchronize_failure=lambda: None,
    )
    cache.capture((1, 1), {"tokens": [0]})
    for value in range(1, 9):
        cache.run((1, 1), {"tokens": [value]})
    runtime.drain()
    assert observed == [[value] for value in range(1, 9)]
    assert cache.stats["replay_count"] == 8
