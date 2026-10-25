"""Runtime-position selection of bounded raw index tails."""

from typing import Any, cast

import torch
import triton
import triton.language as tl

from resin_models_cuda.glm5_3_flash.index_tail import IndexTail


@triton.jit(do_not_specialize=["key_stride", "gate_stride"])
def _select_tail(
    OldKeys,
    OldGates,
    OldEnd,
    Keys,
    Gates,
    Positions,
    Offsets,
    Counts,
    OutKeys,
    OutGates,
    OutEnd,
    key_stride,
    gate_stride,
    WIDTH: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    slot, channel = offset // WIDTH, offset % WIDTH
    begin, end = tl.load(Offsets + seq), tl.load(Offsets + seq + 1)
    if HAS_COUNTS:
        end = begin + tl.minimum(tl.load(Counts + seq), end - begin)
    old_end = tl.load(OldEnd + seq)
    first = tl.load(Positions + begin, begin < end, other=-1)
    last = tl.load(Positions + end - 1, begin < end, other=-1)
    live = (end > begin) & (last >= 0)
    final_end = tl.where(live, last, old_end)
    position = final_end - 3 + slot
    row = begin + position - first
    current = live & (position >= first) & (row < end) & (position >= 0)
    prior = position - (old_end - 3)
    old_valid = (~current) & (position >= 0) & (prior >= 0) & (prior < 4)
    valid = offset < 4 * WIDTH
    key = tl.load(
        Keys + row.to(tl.int64) * key_stride + channel, valid & current, other=0
    )
    old_key = tl.load(
        OldKeys + seq * 4 * WIDTH + prior * WIDTH + channel, valid & old_valid, other=0
    )
    gate = tl.load(
        Gates + row.to(tl.int64) * gate_stride + channel, valid & current, other=0
    ).to(tl.float32)
    old_gate = tl.load(
        OldGates + seq * 4 * WIDTH + prior * WIDTH + channel, valid & old_valid, other=0
    )
    tl.store(OutKeys + seq * 4 * WIDTH + offset, tl.where(current, key, old_key), valid)
    tl.store(
        OutGates + seq * 4 * WIDTH + offset, tl.where(current, gate, old_gate), valid
    )
    tl.store(OutEnd + seq, final_end)


def select_tail_cuda(update, counts=None):
    initial = update.initial
    result = IndexTail(
        torch.empty_like(initial.keys),
        torch.empty_like(initial.gates),
        torch.empty_like(initial.end),
    )
    width = initial.keys.shape[-1]
    if initial.end.numel():
        cast(Any, _select_tail)[(initial.end.numel(),)](
            initial.keys,
            initial.gates,
            initial.end,
            update.keys,
            update.gates,
            update.positions,
            update.offsets,
            counts if counts is not None else initial.end,
            result.keys,
            result.gates,
            result.end,
            update.keys.stride(0),
            update.gates.stride(0),
            width,
            counts is not None,
            triton.next_power_of_2(4 * width),
        )
    return result
