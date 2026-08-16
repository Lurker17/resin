"""Framework-injected sampling operations shared by model runners."""

from __future__ import annotations

SAMPLING_OFFSET_STRIDE = 32


def batch_sampling_inputs(plan, vocab_size: int) -> dict[str, tuple]:
    """Extract backend-friendly primitive arrays from one immutable plan."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    params = tuple(plan.sampling_params)
    rows = plan.num_sequences
    indptr = plan.token_indptr
    context_lengths = plan.context_lengths
    if len(params) != rows:
        raise ValueError("sampling params must contain one item per sequence")
    if len(context_lengths) != rows:
        raise ValueError("context_lengths must contain one item per sequence")
    if len(indptr) != rows + 1 or indptr[0] != 0 or indptr[-1] != len(plan.token_ids):
        raise ValueError("token_indptr does not partition token_ids")
    offsets = tuple(
        (int(context_lengths[index]) + int(indptr[index + 1] - indptr[index]))
        * SAMPLING_OFFSET_STRIDE
        for index in range(rows)
    )
    seeds = tuple(item.seed for item in params)
    if any(seed is None for seed in seeds):
        raise ValueError("sampling params must contain effective seeds")
    sample_rows = set(plan.sample_sequence_indices)
    return {
        "temperatures": tuple(float(item.temperature) for item in params),
        "top_ps": tuple(float(item.top_p) for item in params),
        "top_ks": tuple(
            min(vocab_size, vocab_size if item.top_k is None else int(item.top_k))
            for item in params
        ),
        "seeds": seeds,
        "offsets": offsets,
        "sample_mask": tuple(index in sample_rows for index in range(rows)),
    }


class TorchSampler:
    """Readable reference sampler with row-stable stateless randomness."""

    def __init__(self, torch) -> None:
        self.torch = torch

    @staticmethod
    def _mixed_seed(seed: int, offset: int) -> int:
        value = (seed + offset * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return (value ^ (value >> 31)) & ((1 << 63) - 1)

    def __call__(self, logits, plan=None, *, out=None):
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

        result = logits.argmax(dim=-1)
        for row, (temperature, should_sample) in enumerate(
            zip(values["temperatures"], values["sample_mask"], strict=True)
        ):
            if not should_sample or temperature == 0.0:
                continue
            scores = logits[row].float() / temperature
            top_k = values["top_ks"][row]
            if top_k < scores.numel():
                scores, token_ids = self.torch.topk(scores, top_k, sorted=True)
            else:
                scores, token_ids = self.torch.sort(scores, descending=True)
            probabilities = self.torch.softmax(scores, dim=-1)
            top_p = values["top_ps"][row]
            if top_p < 1.0:
                cumulative = probabilities.cumsum(dim=-1)
                keep = cumulative - probabilities < top_p
                probabilities = probabilities * keep
                probabilities = probabilities / probabilities.sum()
            generator = self.torch.Generator(device=logits.device)
            generator.manual_seed(
                self._mixed_seed(values["seeds"][row], values["offsets"][row])
            )
            selected = self.torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(0)
            result[row] = token_ids[selected]

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = [
    "SAMPLING_OFFSET_STRIDE",
    "TorchSampler",
    "batch_sampling_inputs",
]
