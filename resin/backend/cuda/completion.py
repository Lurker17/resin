"""Owned CUDA results and device feedback for Engine overlap scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from resin.backend.cuda.staging import TensorStager
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, token_output


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
            "token_ids": self.token_ids[:, None],
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

    def __init__(
        self,
        tokens,
        counts,
        feedback,
        *,
        rows,
        stats=None,
        device=None,
        plan=None,
        logits=None,
    ):
        self.feedback = feedback.restrict(rows)
        self._rows, self._stats = rows, stats
        self._staging = TensorStager(device or tokens.device)
        self._tokens = self._staging.output("tokens", tokens)
        self._counts = self._staging.output("counts", counts)
        self._positions = self._staging.output("positions", feedback.positions - counts)
        self._logprobs = None
        self._top_ids = self._top_logprobs = None
        if plan is not None and plan.needs_logprobs:
            self._stage_scores(tokens, plan, logits)
        self._event = torch.cuda.Event()
        self._event.record(torch.cuda.current_stream(tokens.device))
        self._result = None

    def _stage_scores(self, tokens, plan, logits):
        if logits is None:
            raise ValueError("requested logprobs require target logits")
        if logits.ndim == 2:
            logits = logits[:, None, :]
        width = tokens.shape[1]
        if logits.ndim != 3 or logits.shape[0] < self._rows or logits.shape[1] < width:
            raise ValueError("target logits must cover the accepted token grid")
        policies = plan.sampling_params
        self._requested = np.fromiter(
            (
                sample and params.logprobs
                for sample, params in zip(plan.sampling_mask, policies, strict=True)
            ),
            dtype=np.bool_,
        )
        if len(self._requested) != self._rows:
            raise ValueError("sampling policies must match active completion rows")
        self._score_rows = np.flatnonzero(self._requested)
        self._top_counts = np.fromiter(
            (
                min(policies[row].top_logprobs, logits.shape[-1])
                for row in self._score_rows
            ),
            dtype=np.uint32,
        )
        # Keep a fixed grid until harvest: accepted counts live on the device,
        # and CUDA boolean compaction would synchronize before the next launch.
        values, selected_tokens = logits[: self._rows, :width], tokens[: self._rows]
        if len(self._score_rows) != self._rows:
            row_indices = self._staging.device(
                "score_rows", self._score_rows, torch.long
            )
            values = values.index_select(0, row_indices)
            selected_tokens = selected_tokens.index_select(0, row_indices)
        values = values.float().log_softmax(dim=-1)
        selected = values.gather(-1, selected_tokens.long()[..., None]).squeeze(-1)
        self._logprobs = self._staging.output("logprobs", selected)
        top_k = int(max(self._top_counts, default=0))
        if top_k:
            top_values, top_ids = values.topk(top_k, dim=-1)
            self._top_ids = self._staging.output("top_ids", top_ids)
            self._top_logprobs = self._staging.output("top_logprobs", top_values)

    def _scored_output(self, output, counts):
        from resin import ForwardOutput

        assert self._logprobs is not None
        width = self._logprobs.shape[1]
        scored_counts = counts.astype(np.uint32) * self._requested
        kept_counts = counts[self._score_rows].astype(np.uint32)
        keep = np.arange(width)[None, :] < kept_counts[:, None]
        columns = {}
        if self._top_ids is not None:
            assert self._top_logprobs is not None
            top_counts = np.repeat(self._top_counts, kept_counts)
            top_keep = keep[..., None] & (
                np.arange(self._top_ids.shape[-1])[None, None, :]
                < self._top_counts[:, None, None]
            )
            columns = {
                "top_indptr": np.concatenate(
                    (np.zeros(1, np.uint32), top_counts.cumsum(dtype=np.uint32))
                ),
                "top_token_ids": self._top_ids.numpy()[top_keep].astype(np.uint32),
                "top_logprobs": self._top_logprobs.numpy()[top_keep],
            }
        return ForwardOutput(
            token_ids=output.token_ids,
            token_indptr=output.token_indptr,
            logprobs=self._logprobs.numpy()[keep],
            logprob_indptr=np.concatenate(
                (np.zeros(1, np.uint32), scored_counts.cumsum(dtype=np.uint32))
            ),
            **columns,
        )

    @property
    def positions(self):
        self.result()
        return self._positions[: self._rows].numpy()

    def result(self):
        if self._result is None:
            self._event.synchronize()
            counts = self._counts[: self._rows].numpy()
            output = token_output(self._tokens[: self._rows].numpy(), counts=counts)
            if self._logprobs is not None:
                output = self._scored_output(output, counts)
            self._result = output
            if self._stats is not None:
                self._stats.blocks += self._rows
                self._stats.drafts += self._rows * self._stats.draft_tokens
                emitted = int(counts.sum())
                self._stats.accepted += emitted - self._rows
                self._stats.emitted += emitted
        return self._result
