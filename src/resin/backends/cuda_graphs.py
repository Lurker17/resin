"""Reusable, current-batch-only CUDA graph execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

GraphShape = tuple[int, ...]


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


@dataclass(slots=True)
class _GraphEntry:
    shape: GraphShape
    step: CudaGraphStep | None = None
    graph: Any = None
    captures: int = 0
    replays: int = 0
    warmups: int = 0
    capture_ms: float | None = None
    memory_bytes: int | None = None


class CudaGraphCache:
    """Capture and replay reusable static CUDA steps.

    The backend describes each physical shape once with :class:`CudaGraphStep`.
    This cache owns input staging, warmup, capture, parity validation, replay,
    failure poisoning, and metrics. It only executes the batch passed to
    :meth:`run`; it never plans future work or implements overlap scheduling.
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

    def run(self, shape: GraphShape, inputs: Mapping[str, object]) -> list[int]:
        """Stage and execute one current batch, capturing on first use."""

        self.check()
        entry = self._entries[shape]
        fresh = entry.step is None
        if fresh:
            before = self.torch.cuda.memory_allocated(self.device)
            entry.step = self._build(shape)
        assert entry.step is not None
        try:
            self._stage(entry.step, inputs)
        except GraphInputError:
            if fresh:
                entry.step = None
            raise
        except BaseException as error:
            self._fail("graph input copy", error)
            raise

        if entry.graph is None:
            try:
                self._capture(entry, before)
            except BaseException as error:
                self._fail("graph capture", error)
                self._synchronize_failure()
                raise
            return entry.step.output.tolist()

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

    def _stage(self, step: CudaGraphStep, values: Mapping[str, object]) -> None:
        if step.inputs.keys() != values.keys():
            missing = step.inputs.keys() - values.keys()
            extra = values.keys() - step.inputs.keys()
            raise GraphInputError(
                f"CUDA graph inputs do not match buffers (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )
        staged = []
        try:
            for name, target in step.inputs.items():
                source = self.torch.as_tensor(
                    values[name], dtype=target.dtype, device="cpu"
                ).view(target.shape)
                staged.append((target, source))
        except Exception as error:
            raise GraphInputError("inputs do not fit CUDA graph buffers") from error
        for target, source in staged:
            target.copy_(source)

    def _capture(self, entry: _GraphEntry, allocation_before: int) -> None:
        torch, step = self.torch, entry.step
        assert step is not None
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                step.launch()
            expected = step.output.clone()
        current.wait_stream(warmup)
        entry.warmups = 3
        torch.cuda.synchronize(self.device)

        started = perf_counter()
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                step.launch()
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
    def stats(self) -> dict[str, object]:
        rows = tuple(
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


__all__ = [
    "CudaGraphCache",
    "CudaGraphStep",
    "GraphInputError",
    "GraphShape",
    "graph_shapes",
]
