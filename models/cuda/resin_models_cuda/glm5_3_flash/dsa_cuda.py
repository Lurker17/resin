"""Runtime-length sparse-table compaction for GLM's native NoPE TRT kernel."""

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["input_stride", "column_stride", "length_stride"])
def _compact(
    Indices,
    Lengths,
    Packed,
    TopkLengths,
    SequenceLengths,
    Empty,
    input_stride,
    column_stride,
    length_stride,
    WIDTH: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    value = tl.load(
        Indices + row * input_stride + col * column_stride, col < WIDTH, other=-1
    )
    valid = (col < WIDTH) & (value >= 0)
    destination = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    # TRT masks padded scores, but PV can still multiply zero by the loaded V.
    # Repeat a valid row rather than reading potentially poisoned/recycled null
    # storage: 0 * NaN would poison an otherwise valid query's entire output.
    # Empty queries retain the dummy zero row and are masked after attention.
    first_col = tl.min(tl.where(valid, col, WIDTH), 0)
    first = tl.load(
        Indices + row * input_stride + first_col * column_stride,
        count > 0,
        other=0,
    )
    # These writes are disjoint from the compact prefix written below.
    tl.store(Packed + row * PADDED + col, first, (col < PADDED) & (col >= count))
    tl.store(Packed + row * PADDED + destination, value, valid)
    length = tl.load(Lengths + row * length_stride)
    tl.store(TopkLengths + row, tl.maximum(count, 1))
    tl.store(SequenceLengths + row, tl.maximum(length, 1).to(tl.int32))
    tl.store(Empty + row, count == 0)


def sparse_metadata_cuda(indices, sequence_lengths):
    rows, width = indices.shape
    padded = triton.cdiv(width, 4) * 4
    packed = torch.empty((rows, padded), dtype=torch.int32, device=indices.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=indices.device)
    sequence = torch.empty_like(lengths)
    empty = torch.empty(rows, dtype=torch.bool, device=indices.device)
    if rows:
        cast(Any, _compact)[(rows,)](
            indices,
            sequence_lengths,
            packed,
            lengths,
            sequence,
            empty,
            indices.stride(0),
            indices.stride(1),
            sequence_lengths.stride(0),
            width,
            padded,
            triton.next_power_of_2(padded),
            num_warps=4,
        )
    return packed, lengths, sequence, empty
