"""Model-neutral host mechanics shared by CUDA runners."""

from __future__ import annotations

import numpy as np

from resin import BatchPhase
from resin.backend.cuda.sampling import SAMPLING_OFFSET_STRIDE, SamplingTensors

_SAMPLING_DEFAULTS = (1.0, 1.0, None, 0, 0, False)


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
    "capture_sampling_inputs",
    "graph_sample",
    "pad_sampling_inputs",
    "validate_hybrid_plan",
]
