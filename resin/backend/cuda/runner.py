"""Model-neutral host mechanics shared by CUDA runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import numpy as np
import torch

from resin import BatchPhase, ResourceBinding
from resin.backend.cuda.execution import CudaStep
from resin.backend.cuda.failures import CudaFailureGuard, GraphInputError
from resin.backend.cuda.graphs import (
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphStats,
    CudaGraphStep,
    GraphShape,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
    FlashInferSampler,
    SamplingTensors,
)
from resin.backend.cuda.staging import TensorStager
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
        device: str | torch.device,
        label: str,
        kv_cache_dtype: str,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
    ) -> None:
        if cuda_graph_prefill not in {"disabled", "breakable"}:
            raise ValueError("cuda_graph_prefill must be 'disabled' or 'breakable'")
        if cuda_graph_decode not in {"disabled", "full", "breakable"}:
            raise ValueError(
                "cuda_graph_decode must be 'disabled', 'full', or 'breakable'"
            )
        self.cuda_graph_prefill = cuda_graph_prefill
        self.cuda_graph_decode = cuda_graph_decode
        self._prefill_graph_active = False
        self._prefill_operations = ()
        self._prefill_buffers = {}
        cache_dtype = resolve_kv_cache_dtype(kv_cache_dtype)
        if not torch.cuda.is_available():
            raise RuntimeError(f"{label} requires CUDA")
        self.device = torch.device(device)
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
        self._pending_step: CudaStep | None = None
        self._step_staging = TensorStager(self.device)

    def _check_pending_step(self) -> None:
        """Protect borrowed graph buffers and shared workspaces until collection."""

        pending = getattr(self, "_pending_step", None)
        if pending is not None and not pending.done:
            raise GraphInputError("collect the pending CUDA step before another launch")

    def _retain_step(self, plan, output, staging) -> CudaStep:
        self._pending_step = CudaStep(plan, output, staging, self._cuda_failure)
        return self._pending_step

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
            {
                name: replace(mode, build=self._full_graph_builder(mode.build))
                if name == "decode" and self.cuda_graph_decode == "full"
                else mode
                for name, mode in modes.items()
            },
            synchronize_failure=self._cuda_failure,
        )

    @staticmethod
    def _full_graph_builder(build):
        def checked(shape):
            program = build(shape)
            if isinstance(program, CudaGraphProgram) and any(
                not region.capture for region in program.regions
            ):
                raise ValueError("full CUDA graph policy cannot contain eager regions")
            return program

        return checked

    def _configure_prefill_graphs(
        self,
        operations: tuple[Callable, ...],
        *,
        hidden_size: int,
        max_tokens: int,
        token_ids: bool = False,
        pad_up_to: int = 128,
        capture_context: Callable = nullcontext,
    ) -> tuple[Callable, ...]:
        """Capture row-independent compute regions used by the existing forward.

        Operations accept flattened BF16 hidden states and optional token IDs.
        Only these stateless regions are padded; attention, recurrent updates,
        and sampling keep their actual request lengths and execute eagerly.
        Large partial chunks stay eager so padding cannot double their FFN work.
        Set ``pad_up_to=0`` when a model requires exact token buckets.
        """

        if self._prefill_operations:
            raise RuntimeError("prefill graph operations are already configured")
        self._prefill_operations = tuple(operations)
        self._prefill_hidden_size = hidden_size
        self._prefill_token_ids = token_ids
        self._prefill_pad_up_to = pad_up_to
        self._prefill_capture_context = capture_context
        self._prefill_buckets = (
            tuple(
                sorted(
                    {
                        min(2**power, max_tokens)
                        for power in range(max_tokens.bit_length() + 1)
                    }
                )
            )
            if self.cuda_graph_prefill == "breakable"
            else ()
        )
        self._require_cuda_graph_executor().add_mode(
            "prefill",
            CudaGraphMode(
                tuple(
                    (tokens, region + 1)
                    for region in range(len(operations))
                    for tokens in self._prefill_buckets
                ),
                self._new_prefill_graph_step,
                lambda shape: self._prefill_graph_buffers(shape[0])[0],
                share_pool=True,
            ),
        )
        return tuple(
            lambda hidden, ids=None, region=region: self._run_prefill_region(
                region, hidden, ids
            )
            for region in range(len(operations))
        )

    def _prefill_graph_buffers(self, tokens: int):
        # Regions execute serially and consume each result before the next
        # region. Share their external buffers as well as their capture pool.
        if tokens not in self._prefill_buffers:
            hidden = torch.zeros(
                (tokens, self._prefill_hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            inputs = {"hidden": hidden}
            if self._prefill_token_ids:
                inputs["token_ids"] = torch.zeros(
                    tokens, dtype=torch.int64, device=self.device
                )
            self._prefill_buffers[tokens] = (inputs, torch.empty_like(hidden))
        return self._prefill_buffers[tokens]

    def _new_prefill_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        tokens, region = shape
        inputs, output = self._prefill_graph_buffers(tokens)
        operation = self._prefill_operations[region - 1]

        def launch():
            with self._prefill_capture_context():
                output.copy_(operation(inputs["hidden"], inputs.get("token_ids")))

        return CudaGraphStep(
            inputs=inputs,
            output=output,
            launch=launch,
            output_to_host=False,
        )

    @contextmanager
    def _prefill_graphs(self):
        active = self._prefill_graph_active
        self._prefill_graph_active = self.cuda_graph_prefill == "breakable"
        try:
            yield
        finally:
            self._prefill_graph_active = active

    def _run_prefill_region(self, region: int, hidden, token_ids=None):
        flat_input = hidden.ndim == 2
        flattened = hidden if flat_input else hidden.reshape(-1, hidden.shape[-1])
        ids = (
            token_ids
            if token_ids is None or token_ids.ndim == 1
            else token_ids.reshape(-1)
        )
        operation = self._prefill_operations[region]
        if not self._prefill_graph_active:
            output = operation(flattened, ids)
            return output if flat_input else output.view_as(hidden)
        tokens = flattened.shape[0]
        bucket = next((size for size in self._prefill_buckets if size >= tokens), None)
        fallback = None
        if bucket is None or flattened.dtype != torch.bfloat16:
            fallback = "tokens_or_dtype"
        elif tokens > self._prefill_pad_up_to and tokens != bucket:
            fallback = "padding"
        if fallback is not None:
            self._cuda_graph_fallback("prefill", fallback)
            output = operation(flattened, ids)
        else:
            inputs, _ = self._prefill_graph_buffers(bucket)
            if tokens == bucket:
                inputs["hidden"].copy_(flattened)
                if self._prefill_token_ids:
                    inputs["token_ids"].copy_(ids)
            else:
                inputs["hidden"][:tokens].copy_(flattened)
                inputs["hidden"][tokens:].zero_()
                if self._prefill_token_ids:
                    inputs["token_ids"][:tokens].copy_(ids)
                    inputs["token_ids"][tokens:].zero_()
            output = self._run_cuda_graph("prefill", (bucket, region + 1), inputs)
            if tokens != bucket:
                output = output[:tokens]
        return output if flat_input else output.view_as(hidden)

    @property
    def cuda_graph_stats(self) -> dict[str, CudaGraphStats]:
        """Return statistics keyed by semantic graph mode."""

        stats = self._require_cuda_graph_executor().stats
        if self._prefill_operations:
            stats["prefill"]["capture_scope"] = "feed_forward_regions"
        return stats

    def _before_cuda_graph_capture(self) -> None:
        """Prepare model-owned state immediately before graph capture."""

    def _after_cuda_graph_capture(self) -> None:
        """Restore model-owned state immediately after graph capture."""

    def capture_cuda_graphs(self) -> dict[str, CudaGraphStats]:
        """Capture every configured mode and return mode-scoped statistics."""

        self._check_pending_step()
        self._require_cuda_graph_executor().capture_all(
            before_capture=self._before_cuda_graph_capture,
            after_capture=self._after_cuda_graph_capture,
        )
        return self.cuda_graph_stats

    def capture_cuda_graph(self, mode: str, shape: GraphShape) -> CudaGraphStats:
        """Capture one mode shape while honoring model-state hooks."""

        self._check_pending_step()
        executor = self._require_cuda_graph_executor()
        self._before_cuda_graph_capture()
        try:
            executor.capture(mode, shape)
        finally:
            self._after_cuda_graph_capture()
        return self.cuda_graph_stats[mode]

    def _check_cuda_graphs(self, phase: BatchPhase | None = None) -> None:
        """Reject poisoned graphs and missing prefill capture before model work."""

        executor = self._require_cuda_graph_executor()
        executor.check()
        if phase == BatchPhase.PREFILL:
            executor.mode("prefill").check_ready()

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
