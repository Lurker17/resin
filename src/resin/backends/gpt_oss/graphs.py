"""Small, current-batch-only CUDA graph cache for GPT-OSS decode."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

GraphShape = tuple[int, int]  # (batch size, full-attention page-table width)


class GraphInputError(ValueError):
    """Host metadata does not fit a configured graph shape; no GPU work ran."""


def graph_shapes(value: object) -> tuple[GraphShape, ...]:
    """Freeze and validate the configured decode graph shapes."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw = tuple(value.items())
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raise TypeError("cuda_graph_decode_shapes must contain (batch, width) pairs")
    shapes: set[GraphShape] = set()
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each CUDA graph shape must be a (batch, width) tuple")
        if any(
            isinstance(part, bool) or not isinstance(part, int) or part <= 0
            for part in item
        ):
            raise ValueError("CUDA graph batch size and table width must be positive")
        if item in shapes:
            raise ValueError(f"duplicate CUDA graph shape: {item}")
        shapes.add(item)
    return tuple(sorted(shapes))


@dataclass(slots=True)
class GraphEntry:
    shape: GraphShape
    state: Any = None
    graph: Any = None
    captures: int = 0
    replays: int = 0
    warmups: int = 0
    capture_ms: float | None = None
    memory_bytes: int | None = None


class DecodeGraphs:
    """Capture one graph per configured physical decode shape.

    Graphs optimize only the batch already selected by Rust. They do not plan a
    future batch, reserve future state, or introduce overlap scheduling.
    """

    def __init__(self, torch, device, shapes: object, *, capacity: int) -> None:
        self.torch = torch
        self.device = device
        self.shapes = graph_shapes(shapes)
        if any(width > capacity for _, width in self.shapes):
            raise ValueError("CUDA graph table width exceeds cache capacity")
        self.entries = {shape: GraphEntry(shape) for shape in self.shapes}
        self.fallbacks = {"prefill": 0, "shape": 0}
        self.poison: str | None = None

    def select(self, batch_size: int, minimum_width: int) -> GraphShape | None:
        return next(
            (
                shape
                for shape in self.shapes
                if shape[0] == batch_size and shape[1] >= minimum_width
            ),
            None,
        )

    def fallback(self, kind: str) -> None:
        if self.shapes:
            self.fallbacks[kind] += 1

    def check(self) -> None:
        if self.poison is not None:
            raise RuntimeError(
                "this runner is poisoned after a CUDA graph failure; "
                f"construct a fresh runner ({self.poison})"
            )

    def _fail(self, stage: str, error: BaseException) -> None:
        if self.poison is None:
            self.poison = f"{stage}: {type(error).__name__}: {error}"

    def run(
        self,
        shape: GraphShape,
        *,
        allocate: Callable[[GraphShape], Any],
        stage: Callable[[Any], None],
        body: Callable[[Any], None],
        output: Callable[[Any], Any],
        synchronize_failure: Callable[[], None],
    ) -> list[int]:
        """Stage and execute one current decode batch, capturing on first use."""

        self.check()
        entry = self.entries[shape]
        fresh = entry.state is None
        if fresh:
            before = self.torch.cuda.memory_allocated(self.device)
            entry.state = allocate(shape)
        try:
            stage(entry.state)
        except GraphInputError:
            if fresh:
                entry.state = None
            raise
        except BaseException as error:
            self._fail("graph input copy", error)
            raise

        if entry.graph is None:
            try:
                self._capture(entry, body, output, before)
            except BaseException as error:
                self._fail("graph capture", error)
                synchronize_failure()
                raise
            return output(entry.state).tolist()

        try:
            with self.torch.cuda.device(self.device):
                entry.graph.replay()
            result = output(entry.state).tolist()
        except BaseException as error:
            self._fail("graph replay", error)
            synchronize_failure()
            raise
        entry.replays += 1
        return result

    def _capture(self, entry, body, output, allocation_before: int) -> None:
        torch, state = self.torch, entry.state
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                body(state)
            expected = output(state).clone()
        current.wait_stream(warmup)
        entry.warmups = 3
        torch.cuda.synchronize(self.device)

        started = perf_counter()
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                body(state)
            graph.replay()
        torch.cuda.synchronize(self.device)
        if not torch.equal(expected, output(state)):
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
            for entry in self.entries.values()
        )
        return {
            "enabled": bool(self.shapes),
            "configured_shapes": self.shapes,
            "captured_shapes": tuple(row["shape"] for row in rows if row["captured"]),
            "capture_count": sum(entry.captures for entry in self.entries.values()),
            "replay_count": sum(entry.replays for entry in self.entries.values()),
            "eager_fallback_count": sum(self.fallbacks.values()),
            "prefill_fallback_count": self.fallbacks["prefill"],
            "shape_fallback_count": self.fallbacks["shape"],
            "shape_stats": rows,
            "poisoned": self.poison is not None,
            "poison_error": self.poison,
        }
