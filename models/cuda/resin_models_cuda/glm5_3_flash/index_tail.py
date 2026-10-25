"""Owned four-row raw-index tails; full history retains only compressed keys."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class IndexTail:
    keys: Tensor
    gates: Tensor
    end: Tensor

    def snapshot(self):
        # Updates replace tensors, never mutate these owned initial allocations.
        return IndexTail(self.keys, self.gates, self.end)

    def assign(self, other):
        self.keys, self.gates, self.end = other.keys, other.gates, other.end


@dataclass
class IndexUpdate:
    initial: IndexTail
    keys: Tensor
    gates: Tensor
    positions: Tensor
    offsets: Tensor


def select_tail(update: IndexUpdate, counts: Tensor | None = None) -> IndexTail:
    """Select the last four logical rows of a complete or accepted input prefix."""
    if update.keys.is_cuda:
        from resin_models_cuda.glm5_3_flash.index_tail_cuda import select_tail_cuda

        return select_tail_cuda(update, counts)
    initial = update.initial
    out = IndexTail(initial.keys.clone(), initial.gates.clone(), initial.end.clone())
    offsets = update.offsets.tolist()
    for seq, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if counts is not None:
            end = begin + min(int(counts[seq]), end - begin)
        if end <= begin or int(update.positions[end - 1]) < 0:
            continue
        first, last = int(update.positions[begin]), int(update.positions[end - 1])
        prior_end = int(initial.end[seq])
        for slot, position in enumerate(range(last - 3, last + 1)):
            if position < 0:
                out.keys[seq, slot].zero_()
                out.gates[seq, slot].zero_()
            elif first <= position <= last:
                row = begin + position - first
                out.keys[seq, slot] = update.keys[row]
                out.gates[seq, slot] = update.gates[row]
            else:
                prior_slot = position - (prior_end - 3)
                if not 0 <= prior_slot < 4:
                    raise ValueError("raw index tail does not cover the continuation")
                out.keys[seq, slot] = initial.keys[seq, prior_slot]
                out.gates[seq, slot] = initial.gates[seq, prior_slot]
        out.end[seq] = last
    return out


def record_update(cache: Any, keys: Tensor, gates: Tensor) -> IndexUpdate:
    update = IndexUpdate(
        cache.index_tail.snapshot(), keys, gates, cache.positions, cache.cu_seqlens
    )
    cache.index_update = update
    return update


def finish_update(cache: Any, update: IndexUpdate, counts: Tensor | None = None):
    cache.index_tail.assign(select_tail(update, counts))


def commit_tail(cache: Any, counts: Tensor):
    update = getattr(cache, "index_update", None)
    if update is not None:
        finish_update(cache, update, counts)
        cache.index_update = None
