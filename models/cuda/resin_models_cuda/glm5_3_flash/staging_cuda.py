"""Copy live projection rows and zero padding in one strided GPU pass."""

from typing import Any, cast

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tokens", "source_strides", "destination_strides"])
def _stage_rows(
    Source,
    Destination,
    tokens,
    source_strides,
    destination_strides,
    SHAPE: tl.constexpr,
    TOKEN_AXIS: tl.constexpr,
    RANK: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    remaining = index
    source_offset = tl.full((BLOCK,), 0, tl.int64)
    destination_offset = tl.full((BLOCK,), 0, tl.int64)
    live = index < ELEMENTS
    for axis in tl.static_range(RANK - 1, -1, -1):
        coordinate = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        source_offset += coordinate * source_strides[axis]
        destination_offset += coordinate * destination_strides[axis]
        if axis == TOKEN_AXIS:
            live = live & (coordinate < tokens)
    value = tl.load(Source + source_offset, live, other=0)
    tl.store(Destination + destination_offset, value, index < ELEMENTS)


def stage_rows(source, destination, token_axis):
    """Preserve existing allocation, pointer offsets and GEMM operand layouts."""
    elements = destination.numel()
    if elements:
        cast(Any, _stage_rows)[(triton.cdiv(elements, 1024),)](
            source,
            destination,
            source.shape[token_axis],
            source.stride(),
            destination.stride(),
            tuple(destination.shape),
            token_axis,
            destination.ndim,
            elements,
            1024,
            num_warps=4,
        )
