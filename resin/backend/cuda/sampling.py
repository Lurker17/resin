"""Shared FlashInfer logits processing for CUDA model runners."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, batch_sampling_inputs


@dataclass(frozen=True, slots=True)
class SamplingTensors:
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    offsets: torch.Tensor
    stochastic_mask: torch.Tensor


class FlashInferSampler:
    """Temperature, top-k, and top-p sampling over row-aligned batch plans.

    Sampling uses per-row seed and offset tensors, so a request's random
    stream is independent of batch composition. Greedy and unsampled rows still
    enter the fixed-shape CUDA kernels when mixed with sampled rows, but their
    random result is discarded and their native offset is not advanced.
    """

    def __init__(self) -> None:
        from flashinfer.sampling import (
            softmax,
            top_k_mask_logits,
            top_p_renorm_probs,
        )

        from resin.backend.cuda.categorical import sample_categorical

        self._mask_logits = top_k_mask_logits
        self._softmax = softmax
        self._renorm = top_p_renorm_probs
        self._sample = sample_categorical

    @staticmethod
    def host_inputs(
        plan, vocab_size: int, *, offset_delta: int = 0
    ) -> dict[str, tuple]:
        if offset_delta < 0:
            raise ValueError("sampling offset delta cannot be negative")
        values = batch_sampling_inputs(plan, vocab_size)
        return {
            "sampling_temperatures": tuple(
                temperature if temperature > 0.0 else 1.0
                for temperature in values["temperatures"]
            ),
            "sampling_top_ps": values["top_ps"],
            "sampling_top_ks": values["top_ks"],
            "sampling_seeds": values["seeds"],
            "sampling_offsets": tuple(
                offset + offset_delta for offset in values["offsets"]
            ),
            "sampling_stochastic_mask": tuple(
                should_sample and temperature > 0.0
                for temperature, should_sample in zip(
                    values["temperatures"], values["sampling_mask"], strict=True
                )
            ),
        }

    def empty(self, batch_size: int, device) -> SamplingTensors:
        return SamplingTensors(
            temperatures=torch.empty(batch_size, dtype=torch.float32, device=device),
            top_ps=torch.empty(batch_size, dtype=torch.float32, device=device),
            top_ks=torch.empty(batch_size, dtype=torch.int32, device=device),
            seeds=torch.empty(batch_size, dtype=torch.int64, device=device),
            offsets=torch.empty(batch_size, dtype=torch.int64, device=device),
            stochastic_mask=torch.empty(batch_size, dtype=torch.bool, device=device),
        )

    @staticmethod
    def graph_inputs(tensors: SamplingTensors) -> dict[str, object]:
        return {
            "sampling_temperatures": tensors.temperatures,
            "sampling_top_ps": tensors.top_ps,
            "sampling_top_ks": tensors.top_ks,
            "sampling_seeds": tensors.seeds,
            "sampling_offsets": tensors.offsets,
            "sampling_stochastic_mask": tensors.stochastic_mask,
        }

    def sample_tensors(self, logits, tensors: SamplingTensors, *, out=None):
        greedy = logits.argmax(dim=-1)
        # Mask before softmax to normalize the top-k conditional distribution.
        # FlashInfer's probabilities-first renormalizer uses unordered FP32
        # atomics across CTAs, even when the later sampler is deterministic.
        masked_logits = self._mask_logits(logits, tensors.top_ks)
        probabilities = self._softmax(masked_logits, temperature=tensors.temperatures)
        probabilities = self._renorm(
            probabilities,
            tensors.top_ps,
            is_deterministic=True,
        )
        # FlashInfer 0.6.15 reads only seed[0]/offset[0] when drawing samples.
        # Own the draw so requests keep their random stream when batches change.
        sampled, valid = self._sample(probabilities, tensors.seeds, tensors.offsets)
        sampled = sampled.to(dtype=greedy.dtype)
        in_range = (sampled >= 0) & (sampled < logits.shape[-1])
        safe_sampled = sampled.clamp(0, logits.shape[-1] - 1)
        sampled = torch.where(valid & in_range, safe_sampled, greedy)
        result = torch.where(tensors.stochastic_mask, sampled, greedy)
        if out is None:
            return result
        out.copy_(result)
        return out

    def __call__(self, logits, plan=None, *, out=None, offset_delta: int = 0):
        if plan is None:
            result = logits.argmax(dim=-1)
            if out is None:
                return result
            out.copy_(result)
            return out
        if logits.ndim != 2:
            raise ValueError("sampling logits must have shape [batch, vocab]")
        values = batch_sampling_inputs(plan, logits.shape[-1])
        if logits.shape[0] != len(values["temperatures"]):
            raise ValueError("sampling logits must contain one row per plan sequence")

        if not any(
            should_sample and temperature > 0.0
            for temperature, should_sample in zip(
                values["temperatures"], values["sampling_mask"], strict=True
            )
        ):
            result = logits.argmax(dim=-1)
        else:
            device = logits.device
            host = self.host_inputs(plan, logits.shape[-1], offset_delta=offset_delta)
            tensors = SamplingTensors(
                temperatures=torch.tensor(
                    host["sampling_temperatures"],
                    dtype=torch.float32,
                    device=device,
                ),
                top_ps=torch.tensor(
                    host["sampling_top_ps"], dtype=torch.float32, device=device
                ),
                top_ks=torch.tensor(
                    host["sampling_top_ks"], dtype=torch.int32, device=device
                ),
                seeds=torch.tensor(
                    host["sampling_seeds"], dtype=torch.int64, device=device
                ),
                offsets=torch.tensor(
                    host["sampling_offsets"], dtype=torch.int64, device=device
                ),
                stochastic_mask=torch.tensor(
                    host["sampling_stochastic_mask"],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            result = self.sample_tensors(logits, tensors)

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = ["FlashInferSampler", "SAMPLING_OFFSET_STRIDE", "SamplingTensors"]
