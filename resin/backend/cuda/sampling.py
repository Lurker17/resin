"""Shared FlashInfer logits processing for CUDA model runners."""

from __future__ import annotations

from dataclasses import dataclass

from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, batch_sampling_inputs


@dataclass(frozen=True, slots=True)
class SamplingTensors:
    temperatures: object
    top_ps: object
    top_ks: object
    seeds: object
    offsets: object
    stochastic_mask: object


class FlashInferSampler:
    """Temperature, top-k, and top-p sampling over row-aligned batch plans.

    FlashInfer receives per-row seed and offset tensors, so a request's random
    stream is independent of batch composition. Greedy and unsampled rows still
    enter the fixed-shape CUDA kernels when mixed with sampled rows, but their
    random result is discarded and their native offset is not advanced.
    """

    def __init__(self, torch) -> None:
        from flashinfer.sampling import softmax, top_k_top_p_sampling_from_probs

        self.torch = torch
        self._softmax = softmax
        self._sample = top_k_top_p_sampling_from_probs

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
        torch = self.torch
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
        probabilities = self._softmax(logits, temperature=tensors.temperatures)
        sampled, valid = self._sample(
            probabilities,
            tensors.top_ks,
            tensors.top_ps,
            filter_apply_order="top_k_first",
            deterministic=True,
            seed=tensors.seeds,
            offset=tensors.offsets,
            return_valid=True,
        )
        sampled = sampled.to(dtype=greedy.dtype)
        in_range = (sampled >= 0) & (sampled < logits.shape[-1])
        safe_sampled = sampled.clamp(0, logits.shape[-1] - 1)
        sampled = self.torch.where(valid & in_range, safe_sampled, greedy)
        result = self.torch.where(tensors.stochastic_mask, sampled, greedy)
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

        greedy = logits.argmax(dim=-1)
        if not any(
            should_sample and temperature > 0.0
            for temperature, should_sample in zip(
                values["temperatures"], values["sampling_mask"], strict=True
            )
        ):
            result = greedy
        else:
            device = logits.device
            host = self.host_inputs(plan, logits.shape[-1], offset_delta=offset_delta)
            tensors = SamplingTensors(
                temperatures=self.torch.tensor(
                    host["sampling_temperatures"],
                    dtype=self.torch.float32,
                    device=device,
                ),
                top_ps=self.torch.tensor(
                    host["sampling_top_ps"], dtype=self.torch.float32, device=device
                ),
                top_ks=self.torch.tensor(
                    host["sampling_top_ks"], dtype=self.torch.int32, device=device
                ),
                seeds=self.torch.tensor(
                    host["sampling_seeds"], dtype=self.torch.int64, device=device
                ),
                offsets=self.torch.tensor(
                    host["sampling_offsets"], dtype=self.torch.int64, device=device
                ),
                stochastic_mask=self.torch.tensor(
                    host["sampling_stochastic_mask"],
                    dtype=self.torch.bool,
                    device=device,
                ),
            )
            result = self.sample_tensors(logits, tensors)

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = ["FlashInferSampler", "SAMPLING_OFFSET_STRIDE", "SamplingTensors"]
