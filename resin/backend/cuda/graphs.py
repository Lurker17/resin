"""Reusable, current-batch-only CUDA graph execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypedDict

from resin.backend.cuda.staging import StagingError, TensorStager

GraphShape = tuple[int, ...]


class CudaGraphShapeStats(TypedDict):
    shape: GraphShape
    captured: bool
    capture_count: int
    replay_count: int
    warmup_count: int
    capture_ms: float | None
    capture_memory_bytes: int | None


class CudaGraphStats(TypedDict):
    enabled: bool
    configured_shapes: tuple[GraphShape, ...]
    captured_shapes: tuple[GraphShape, ...]
    capture_count: int
    replay_count: int
    eager_fallback_count: int
    fallback_counts: dict[str, int]
    shape_stats: tuple[CudaGraphShapeStats, ...]
    poisoned: bool
    poison_error: str | None


class GraphInputError(ValueError):
    """Host inputs do not fit a graph's fixed device buffers."""


def graph_shapes(value: object, *, dimensions: int) -> tuple[GraphShape, ...]:
    """Freeze positive integer CUDA graph shapes in deterministic order."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw = tuple(value.items())
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raise TypeError("CUDA graph shapes must be an iterable of tuples")
    shapes: set[GraphShape] = set()
    for item in raw:
        if not isinstance(item, tuple) or len(item) != dimensions:
            raise TypeError(
                f"each CUDA graph shape must be a {dimensions}-integer tuple"
            )
        if any(
            isinstance(part, bool) or not isinstance(part, int) or part <= 0
            for part in item
        ):
            raise ValueError("CUDA graph shape dimensions must be positive integers")
        if item in shapes:
            raise ValueError(f"duplicate CUDA graph shape: {item}")
        shapes.add(item)
    return tuple(sorted(shapes))


@dataclass(frozen=True, slots=True)
class CudaGraphStep:
    """One fixed-address device operation that can be captured automatically."""

    inputs: Mapping[str, Any]
    output: Any
    launch: Callable[[], None]
    prepare: Callable[[], None] | None = None


@dataclass(slots=True)
class _GraphEntry:
    shape: GraphShape
    step: CudaGraphStep | None = None
    staging: TensorStager | None = None
    graph: Any = None
    captures: int = 0
    replays: int = 0
    warmups: int = 0
    capture_ms: float | None = None
    memory_bytes: int | None = None


class CudaGraphCache:
    """Capture and replay reusable static CUDA steps.

    The backend describes each physical shape once with :class:`CudaGraphStep`.
    This cache owns input staging, explicit warmup/capture, parity validation,
    replay, failure poisoning, and metrics. It only executes the batch passed
    to :meth:`run`; it never plans future work or implements overlap scheduling.
    """

    def __init__(
        self,
        torch,
        device,
        shapes: tuple[GraphShape, ...],
        *,
        build: Callable[[GraphShape], CudaGraphStep],
        synchronize_failure: Callable[[], None],
    ) -> None:
        self.torch = torch
        self.device = device
        self.shapes = shapes
        self._build = build
        self._synchronize_failure = synchronize_failure
        self._entries = {shape: _GraphEntry(shape) for shape in shapes}
        self._fallbacks: Counter[str] = Counter()
        self._poison: str | None = None

    def match(self, predicate: Callable[[GraphShape], bool]) -> GraphShape | None:
        """Return the first configured shape accepted by ``predicate``."""

        return next((shape for shape in self.shapes if predicate(shape)), None)

    def fallback(self, reason: str) -> None:
        if self.shapes:
            self._fallbacks[reason] += 1

    def check(self) -> None:
        if self._poison is not None:
            raise RuntimeError(
                "this runner is poisoned after a CUDA graph failure; "
                f"construct a fresh runner ({self._poison})"
            )

    def _fail(self, stage: str, error: BaseException) -> None:
        if self._poison is None:
            self._poison = f"{stage}: {type(error).__name__}: {error}"

    def capture(self, shape: GraphShape, inputs: Mapping[str, object]) -> Any:
        """Build and capture one configured shape before request execution."""

        self.check()
        try:
            entry = self._entries[shape]
        except KeyError as error:
            raise ValueError(f"unconfigured CUDA graph shape: {shape}") from error
        if entry.graph is not None:
            raise RuntimeError(f"CUDA graph shape is already captured: {shape}")
        before = self.torch.cuda.memory_allocated(self.device)
        entry.step = self._build(shape)
        assert entry.step is not None
        try:
            entry.staging = TensorStager(self.torch, self.device)
            entry.staging.copy_into(entry.step.inputs, inputs)
        except StagingError as error:
            entry.step = None
            raise GraphInputError(str(error)) from error
        except BaseException as error:
            self._fail("graph input copy", error)
            raise
        try:
            self._capture(entry, before)
        except BaseException as error:
            self._fail("graph capture", error)
            self._synchronize_failure()
            raise
        return entry.step.output.tolist()

    def run(self, shape: GraphShape, inputs: Mapping[str, object]) -> Any:
        """Stage and replay one current batch using a pre-captured graph."""

        self.check()
        try:
            entry = self._entries[shape]
        except KeyError as error:
            raise ValueError(f"unconfigured CUDA graph shape: {shape}") from error
        if entry.graph is None or entry.step is None:
            raise RuntimeError(
                f"CUDA graph shape {shape} has not been captured; "
                "capture configured graphs during runner initialization"
            )
        try:
            entry.staging = TensorStager(self.torch, self.device)
            entry.staging.copy_into(entry.step.inputs, inputs)
        except StagingError as error:
            raise GraphInputError(str(error)) from error
        except BaseException as error:
            self._fail("graph input copy", error)
            raise

        try:
            with self.torch.cuda.device(self.device):
                entry.graph.replay()
            result = entry.step.output.tolist()
        except BaseException as error:
            self._fail("graph replay", error)
            self._synchronize_failure()
            raise
        entry.replays += 1
        return result

    def _capture(self, entry: _GraphEntry, allocation_before: int) -> None:
        torch, step = self.torch, entry.step
        assert step is not None
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                if step.prepare is not None:
                    step.prepare()
                step.launch()
            expected = step.output.clone()
        current.wait_stream(warmup)
        entry.warmups = 3
        torch.cuda.synchronize(self.device)

        started = perf_counter()
        with torch.cuda.device(self.device):
            if step.prepare is not None:
                step.prepare()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                step.launch()
            if step.prepare is not None:
                step.prepare()
            graph.replay()
        torch.cuda.synchronize(self.device)
        if not torch.equal(expected, step.output):
            raise RuntimeError("CUDA graph replay differs from eager warmup")
        entry.graph = graph
        entry.captures = 1
        entry.capture_ms = (perf_counter() - started) * 1000
        entry.memory_bytes = max(
            0, torch.cuda.memory_allocated(self.device) - allocation_before
        )

    @property
    def stats(self) -> CudaGraphStats:
        rows: tuple[CudaGraphShapeStats, ...] = tuple(
            {
                "shape": entry.shape,
                "captured": entry.graph is not None,
                "capture_count": entry.captures,
                "replay_count": entry.replays,
                "warmup_count": entry.warmups,
                "capture_ms": entry.capture_ms,
                "capture_memory_bytes": entry.memory_bytes,
            }
            for entry in self._entries.values()
        )
        return {
            "enabled": bool(self.shapes),
            "configured_shapes": self.shapes,
            "captured_shapes": tuple(row["shape"] for row in rows if row["captured"]),
            "capture_count": sum(entry.captures for entry in self._entries.values()),
            "replay_count": sum(entry.replays for entry in self._entries.values()),
            "eager_fallback_count": sum(self._fallbacks.values()),
            "fallback_counts": dict(self._fallbacks),
            "shape_stats": rows,
            "poisoned": self._poison is not None,
            "poison_error": self._poison,
        }
