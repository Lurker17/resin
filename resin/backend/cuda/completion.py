"""Owned CUDA results and device feedback for Engine overlap scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from resin.backend.cuda.staging import TensorStager
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE


@dataclass(frozen=True, slots=True)
class DeviceFeedback:
    """Exact next decode inputs, independent of reusable graph output buffers."""

    token_ids: torch.Tensor
    positions: torch.Tensor

    def restrict(self, rows):
        # Scratch rows must remain at their reserved initial position across
        # arbitrarily many continuations; they have no logical request progress.
        self.token_ids[rows:].zero_()
        self.positions[rows:].fill_(1)
        return self

    def inputs(self):
        return {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "sampling_offsets": (self.positions + 1) * SAMPLING_OFFSET_STRIDE,
        }


def accepted_block(output, positions, draft_tokens, *, fixed_acceptance=None):
    """Apply the shared prefix-acceptance rule entirely on the device."""

    if draft_tokens:
        drafts, targets = output[:, :draft_tokens], output[:, draft_tokens:]
        if fixed_acceptance is None:
            accepted = (drafts == targets[:, :draft_tokens]).int().cumprod(1).sum(1)
        else:
            accepted = torch.full_like(positions, fixed_acceptance)
        bonus = targets.gather(1, accepted[:, None]).squeeze(1)
        columns = torch.arange(draft_tokens + 1, device=output.device)[None, :]
        tokens = torch.where(
            columns < accepted[:, None], output[:, : draft_tokens + 1], bonus[:, None]
        )
        counts = accepted + 1
    else:
        tokens = output.reshape(-1, 1)
        counts = torch.ones_like(positions)
        bonus = tokens[:, 0]
    return tokens, counts.int(), DeviceFeedback(bonus.clone(), positions + counts)


class DeviceCompletion:
    """One accepted step; CPU harvest never reruns model or acceptance work.

    Copies are queued before another replay may reuse graph buffers. The Engine
    retains the native writer lease until ``result`` has established completion.
    """

    def __init__(self, tokens, counts, feedback, *, rows, stats=None, device=None):
        self.feedback = feedback.restrict(rows)
        self._rows, self._stats = rows, stats
        self._staging = TensorStager(device or tokens.device)
        self._tokens = self._staging.output("tokens", tokens)
        self._counts = self._staging.output("counts", counts)
        self._positions = self._staging.output("positions", feedback.positions - counts)
        self._event = torch.cuda.Event()
        self._event.record(torch.cuda.current_stream(tokens.device))
        self._result = None

    @property
    def positions(self):
        self.result()
        return self._positions[: self._rows].numpy()

    def result(self):
        if self._result is None:
            self._event.synchronize()
            counts = self._counts[: self._rows].tolist()
            tokens = self._tokens[: self._rows].tolist()
            self._result = [
                row[:count] for row, count in zip(tokens, counts, strict=True)
            ]
            if self._stats is not None:
                self._stats.blocks += self._rows
                self._stats.drafts += self._rows * self._stats.draft_tokens
                self._stats.accepted += sum(counts) - self._rows
                self._stats.emitted += sum(counts)
        return self._result
