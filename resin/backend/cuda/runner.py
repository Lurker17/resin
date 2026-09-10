"""Model-neutral host mechanics shared by CUDA runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from resin import BatchPhase, ResourceBinding
from resin.backend.cuda.failures import CudaFailureGuard
from resin.backend.cuda.graphs import (
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphStats,
    GraphShape,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
    FlashInferSampler,
    SamplingTensors,
)
from resin.backend.loading import default_device
from resin.backend.resources import CachePoolBinding
from resin.protocols import Runner

_SAMPLING_DEFAULTS = (1.0, 1.0, None, 0, 0, False)


def resolve_kv_cache_dtype(value: str) -> torch.dtype:
    """Resolve the cache precision shared by CUDA model runners."""

    if value == "fp8":
        return torch.float8_e4m3fn
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError("kv_cache_dtype must be 'fp8' or 'bfloat16'")


class CudaRunner(Runner):
    """Common device, cache-precision, failure, and graph runtime boundary.

    Model implementations configure semantic modes and supply their graph-step
    builders. This base owns device normalization, KV precision naming, failure
    containment, capture, poisoning checks, mode-scoped statistics, physical
    shape selection, fallback accounting, and replay without knowing anything
    about transformer layers, cache geometry, kernels, or quantization.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None,
        label: str,
        kv_cache_dtype: str,
    ) -> None:
        cache_dtype = resolve_kv_cache_dtype(kv_cache_dtype)
        if not torch.cuda.is_available():
            raise RuntimeError(f"{label} requires CUDA")
        self.device = torch.device(device or default_device(require_cuda=True))
        if self.device.type != "cuda":
            raise RuntimeError(f"{label} requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = cache_dtype
        self._cuda_failure = CudaFailureGuard(
            lambda: torch.cuda.synchronize(self.device),
            label=f"{label} eager execution",
        )
        self._sampler = FlashInferSampler()
        self._cache_binding: CachePoolBinding | None = None
        self._cuda_graph_executor: CudaGraphExecutor | None = None

    def _configure_cache_binding(self, binding: CachePoolBinding) -> None:
        """Attach the model-owned physical cache to the Engine resource seam."""

        if not isinstance(binding, CachePoolBinding):
            raise TypeError("CUDA cache binding must be a CachePoolBinding")
        if self._cache_binding is not None:
            raise RuntimeError("CUDA cache binding is already configured")
        self._cache_binding = binding

    def bind_resources(self, binding: ResourceBinding) -> None:
        """Bind the model cache to its Engine-owned physical allocation."""

        if self._cache_binding is None:
            raise RuntimeError("CUDA cache binding has not been configured")
        self._cache_binding.bind_resources(binding)

    def _configure_cuda_graphs(
        self,
        modes: Mapping[str, CudaGraphMode],
    ) -> None:
        if not modes:
            raise ValueError("CUDA graph modes must not be empty")
        if self._cuda_graph_executor is not None:
            raise RuntimeError("CUDA graph modes are already configured")
        self._cuda_graph_executor = CudaGraphExecutor(
            self.device,
            modes,
            synchronize_failure=self._cuda_failure,
        )

    @property
    def cuda_graph_stats(self) -> dict[str, CudaGraphStats]:
        """Return statistics keyed by semantic graph mode."""

        return self._require_cuda_graph_executor().stats

    def _before_cuda_graph_capture(self) -> None:
        """Prepare model-owned state immediately before graph capture."""

    def _after_cuda_graph_capture(self) -> None:
        """Restore model-owned state immediately after graph capture."""

    def capture_cuda_graphs(self) -> dict[str, CudaGraphStats]:
        """Capture every configured mode and return mode-scoped statistics."""

        return self._require_cuda_graph_executor().capture_all(
            before_capture=self._before_cuda_graph_capture,
            after_capture=self._after_cuda_graph_capture,
        )

    def capture_cuda_graph(
        self, mode: str, shape: GraphShape
    ) -> CudaGraphStats:
        """Capture one mode shape while honoring model-state hooks."""

        executor = self._require_cuda_graph_executor()
        self._before_cuda_graph_capture()
        try:
            return executor.capture(mode, shape)
        finally:
            self._after_cuda_graph_capture()

    def _check_cuda_graphs(self) -> None:
        """Fail before model work when any configured graph mode is poisoned."""

        self._require_cuda_graph_executor().check()

    def _select_cuda_graph(
        self,
        mode: str,
        *,
        match: Callable[[GraphShape], bool],
        fallback: str = "shape",
    ) -> GraphShape | None:
        """Select one physical shape and account for an eager fallback."""

        cache = self._require_cuda_graph_executor().mode(mode)
        if not cache.shapes:
            return None
        shape = cache.match(match)
        if shape is None:
            cache.fallback(fallback)
            return None
        return shape

    def _run_cuda_graph(
        self,
        mode: str,
        shape: GraphShape,
        inputs: Mapping[str, object],
    ):
        """Stage model-owned inputs and replay one selected physical shape."""

        return self._require_cuda_graph_executor().mode(mode).run(shape, inputs)

    def _cuda_graph_fallback(self, mode: str, reason: str) -> None:
        """Record a model-level reason that a configured mode ran eagerly."""

        self._require_cuda_graph_executor().mode(mode).fallback(reason)

    def _cuda_graph_mode_enabled(self, mode: str) -> bool:
        """Return whether a semantic mode has any configured physical shapes."""

        return bool(self._require_cuda_graph_executor().mode(mode).shapes)

    def _require_cuda_graph_executor(self) -> CudaGraphExecutor:
        executor = self._cuda_graph_executor
        if executor is None:
            raise RuntimeError("CUDA graph modes have not been configured")
        return executor


def capture_sampling_inputs(
    rows: int, vocab_size: int, *, stochastic: bool = False, unique_seeds: bool = False
) -> dict[str, tuple]:
    zeros = (0,) * rows
    return {
        "sampling_temperatures": (1.0,) * rows,
        "sampling_top_ps": (1.0,) * rows,
        "sampling_top_ks": (vocab_size,) * rows,
        "sampling_seeds": tuple(range(rows)) if unique_seeds else zeros,
        "sampling_offsets": zeros,
        "sampling_stochastic_mask": (stochastic,) * rows,
    }


def pad_sampling_inputs(sampler, plan, vocab_size: int, padding: int):
    inputs = sampler.host_inputs(plan, vocab_size)
    defaults = (*_SAMPLING_DEFAULTS[:2], vocab_size, *_SAMPLING_DEFAULTS[3:])
    for (name, values), default in zip(inputs.items(), defaults, strict=True):
        inputs[name] = (*values, *((default,) * padding))
    return inputs


def graph_sample(sampler, logits, sampling: SamplingTensors, step: int):
    return sampler.sample_tensors(
        logits.float(),
        SamplingTensors(
            sampling.temperatures,
            sampling.top_ps,
            sampling.top_ks,
            sampling.seeds,
            sampling.offsets + step * SAMPLING_OFFSET_STRIDE,
            sampling.stochastic_mask,
        ),
    )


def validate_hybrid_plan(
    plan, *, label: str, max_batch_size: int, max_seq_len: int, decode_lookahead: int
):
    rows = plan.num_sequences
    indptr = np.asarray(plan.token_indptr)
    if rows == 0 or rows > max_batch_size:
        raise ValueError(f"{label} batch size is outside runner capacity")
    if (
        len(indptr) != rows + 1
        or indptr[0] != 0
        or indptr[-1] != len(plan.token_ids)
        or np.any(indptr[1:] <= indptr[:-1])
    ):
        raise ValueError(f"{label} sequence metadata is malformed")
    if len(plan.context_lengths) != rows or len(plan.sampling_mask) != rows:
        raise ValueError(f"{label} row metadata is malformed")
    slices = tuple(map(tuple, np.stack((indptr[:-1], indptr[1:]), axis=1).tolist()))
    for row, (start, end) in enumerate(slices):
        context = int(plan.context_lengths[row])
        if not np.array_equal(
            plan.positions[start:end], np.arange(context, context + end - start)
        ):
            raise ValueError(f"{label} positions must continue their context")
        extra = decode_lookahead if plan.phase == BatchPhase.DECODE else 0
        if context + end - start + extra > max_seq_len:
            raise ValueError(f"{label} sequence exceeds max_seq_len")
    if plan.phase == BatchPhase.DECODE:
        if any(end - start != 1 for start, end in slices):
            raise ValueError(f"{label} decode rows must contain one anchor token")
        if any(int(context) == 0 for context in plan.context_lengths):
            raise ValueError(f"{label} decode requires a cached prefix")
    elif plan.phase != BatchPhase.PREFILL:
        raise ValueError(f"{label} plan phase must be prefill or decode")
    return slices


__all__ = [
    "CudaRunner",
    "capture_sampling_inputs",
    "graph_sample",
    "pad_sampling_inputs",
    "resolve_kv_cache_dtype",
    "validate_hybrid_plan",
]
