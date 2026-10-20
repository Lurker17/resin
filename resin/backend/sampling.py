"""PyTorch sampling operations shared by model runners."""

from __future__ import annotations

import numpy as np
import torch

SAMPLING_OFFSET_STRIDE = 32


def format_output(logits, token_ids, plan):
    """Attach raw-model logprobs to generated tokens, including verified blocks.

    Logits have shape [rows, vocab] or [rows, verification positions, vocab].
    Scores use the untempered model distribution, before top-k/top-p filtering.
    Only accepted output positions are returned; unsampled rows carry no scores.
    The ordinary token-only path does no scoring or extra device transfer.
    """

    from resin import ForwardOutput

    if not plan.needs_logprobs:
        return token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids

    if isinstance(token_ids, torch.Tensor):
        ids = token_ids.detach().to(device="cpu", dtype=torch.uint32).numpy()
        if ids.ndim not in (1, 2):
            raise ValueError(
                "output tokens must have shape [rows] or [rows, positions]"
            )
        lengths = np.full(ids.shape[0], 1 if ids.ndim == 1 else ids.shape[1], np.uint32)
        ids = np.ascontiguousarray(ids.reshape(-1))
    else:
        rows = (
            [[token] for token in token_ids]
            if token_ids and isinstance(token_ids[0], int)
            else token_ids
        )
        lengths = np.fromiter(map(len, rows), dtype=np.uint32)
        ids = np.fromiter((token for row in rows for token in row), dtype=np.uint32)
    offsets = np.concatenate((np.zeros(1, np.uint32), lengths.cumsum(dtype=np.uint32)))
    if logits.ndim == 2:
        logits = logits[:, None, :]
    if logits.ndim != 3 or len(lengths) != logits.shape[0]:
        raise ValueError("logits must align with output token rows")
    if np.any(lengths > logits.shape[1]):
        raise ValueError("verification logits do not cover the accepted token row")

    policies = plan.sampling_params
    requested = np.fromiter(
        (
            sample and params.logprobs
            for sample, params in zip(plan.sampling_mask, policies, strict=True)
        ),
        dtype=np.bool_,
    )
    scored_lengths = lengths * requested
    score_offsets = np.concatenate(
        (np.zeros(1, np.uint32), scored_lengths.cumsum(dtype=np.uint32))
    )
    row_ids = np.repeat(np.arange(len(lengths)), scored_lengths)
    positions = np.arange(len(row_ids)) - np.repeat(score_offsets[:-1], scored_lengths)
    device = logits.device
    if len(row_ids) == len(lengths) and np.all(scored_lengths == 1):
        values = logits[:, 0].float()
    else:
        values = logits[
            torch.as_tensor(row_ids, device=device),
            torch.as_tensor(positions, device=device),
        ].float()
    selected = torch.as_tensor(
        ids[offsets[row_ids] + positions].astype(np.int64), device=device
    )
    values = values.log_softmax(dim=-1)
    chosen = values.gather(1, selected[:, None]).squeeze(1)

    # Array storage crosses into Rust without scalar boxing or a second host copy.
    # The .numpy() views retain the CPU tensors that own their buffers.
    columns = {}
    counts = np.fromiter(
        (min(params.top_logprobs, values.shape[-1]) for params in policies),
        dtype=np.uint32,
    )[row_ids]
    max_k = int(counts.max(initial=0))
    if max_k:
        top_values, top_ids = values.topk(max_k, dim=-1)
        keep = (
            torch.arange(max_k, device=device)[None, :]
            < torch.as_tensor(counts.astype(np.int64), device=device)[:, None]
        )
        columns = {
            "top_indptr": np.concatenate(
                (np.zeros(1, np.uint32), counts.cumsum(dtype=np.uint32))
            ),
            "top_token_ids": top_ids[keep].to(device="cpu", dtype=torch.uint32).numpy(),
            "top_logprobs": top_values[keep].cpu().numpy(),
        }
    return ForwardOutput(
        ids,
        offsets,
        logprobs=chosen.cpu().numpy(),
        logprob_indptr=score_offsets,
        **columns,
    )


def batch_sampling_inputs(plan, vocab_size: int) -> dict[str, object]:
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
    return {
        "temperatures": tuple(float(item.temperature) for item in params),
        "top_ps": tuple(float(item.top_p) for item in params),
        "top_ks": tuple(
            min(vocab_size, vocab_size if item.top_k is None else int(item.top_k))
            for item in params
        ),
        "seeds": seeds,
        "offsets": offsets,
        "sampling_mask": plan.sampling_mask,
    }


class TorchSampler:
    """Readable reference sampler with row-stable stateless randomness."""

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
            zip(values["temperatures"], values["sampling_mask"], strict=True)
        ):
            if not should_sample or temperature == 0.0:
                continue
            scores = logits[row].float() / temperature
            top_k = values["top_ks"][row]
            if top_k < scores.numel():
                scores, token_ids = torch.topk(scores, top_k, sorted=True)
            else:
                scores, token_ids = torch.sort(scores, descending=True)
            probabilities = torch.softmax(scores, dim=-1)
            top_p = values["top_ps"][row]
            if top_p < 1.0:
                cumulative = probabilities.cumsum(dim=-1)
                keep = cumulative - probabilities < top_p
                probabilities = probabilities * keep
                probabilities = probabilities / probabilities.sum()
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(
                self._mixed_seed(values["seeds"][row], values["offsets"][row])
            )
            selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(
                0
            )
            result[row] = token_ids[selected]

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = [
    "SAMPLING_OFFSET_STRIDE",
    "TorchSampler",
    "batch_sampling_inputs",
    "format_output",
]
