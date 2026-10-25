"""Fixed physical rows for pure target decode and verification projections.

The runner chooses a capacity for the executing batch bucket, independently of
the configured concurrency limit and draft width.
Prefill, draft and repair pass zero and preserve their original operations.
Only pure projections may be padded; live results are cropped before cache
updates, recurrent kernels, expert exchange or speculative acceptance.
"""

from collections.abc import Callable
from typing import TypeVar, cast

import torch
from torch import Tensor
from torch.nn import functional as F

Result = TypeVar("Result")


def fixed_rows(
    value: Tensor,
    operation: Callable[[Tensor], Result],
    *,
    projection_rows: int = 0,
    token_axis: int = 0,
    input_strides: tuple[int, ...] | None = None,
    input_storage_offset: int = 0,
) -> Result:
    """Stage one packed prefix and retain the original operator and layouts.

    All checks use metadata. Positive capacity applies even to ordinary K0
    decode. There are no row-map tensors, host transfers or indexed gathers.
    Narrow outputs retain their complete allocation, including inside CUDA
    graph pools; callers preserve their existing transpose/contiguous consumers.
    """
    if type(projection_rows) is not int or projection_rows < 0:
        raise ValueError("projection_rows must be a nonnegative integer")
    if projection_rows == 0:
        return operation(value)
    if type(token_axis) is not int or not 0 <= token_axis < value.ndim:
        raise ValueError("projection token axis is outside the operand")
    tokens = value.shape[token_axis]
    if tokens > projection_rows:
        raise ValueError("projection rows exceed the fixed capacity")
    if not tokens:
        return operation(value)
    strides = value.stride() if input_strides is None else input_strides
    if len(strides) != value.ndim or any(type(s) is not int or s <= 0 for s in strides):
        raise ValueError("projection strides must be positive integers")
    if type(input_storage_offset) is not int or input_storage_offset < 0:
        raise ValueError("projection storage offset must be a nonnegative integer")
    shape = list(value.shape)
    shape[token_axis] = projection_rows
    span = 1
    for stride, size in sorted(zip(strides, shape, strict=True)):
        if size > 1:
            if stride < span:
                raise ValueError("projection layout overlaps elements")
            span += (size - 1) * stride
    if input_storage_offset:
        size = (
            input_storage_offset
            + 1
            + sum(
                (length - 1) * stride
                for length, stride in zip(shape, strides, strict=True)
            )
        )
        operand = torch.empty(size, dtype=value.dtype, device=value.device).as_strided(
            shape, strides, input_storage_offset
        )
    else:
        operand = torch.empty_strided(
            shape, strides, dtype=value.dtype, device=value.device
        )
    if value.is_cuda:
        from resin_models_cuda.glm5_3_flash.staging_cuda import stage_rows

        stage_rows(value, operand, token_axis)
    else:
        operand.zero_()
        operand.narrow(token_axis, 0, tokens).copy_(value)
    result = operation(operand)

    def crop(item):
        if isinstance(item, Tensor):
            if item.shape[token_axis] != projection_rows:
                raise ValueError("projection operation changed the physical row count")
            return item.narrow(token_axis, 0, tokens)
        if isinstance(item, tuple):
            return tuple(crop(part) for part in item)
        raise TypeError("projection operations must return tensors or tensor tuples")

    return cast(Result, crop(result))


def linear(value: Tensor, weight: Tensor, *, projection_rows: int = 0) -> Tensor:
    return fixed_rows(
        value,
        lambda operand: F.linear(operand, weight),
        projection_rows=projection_rows,
    )


def bmm(
    value: Tensor,
    weight: Tensor,
    *,
    projection_rows: int = 0,
    input_strides: tuple[int, int, int],
    input_storage_offset: int = 0,
) -> Tensor:
    return fixed_rows(
        value,
        lambda operand: torch.bmm(operand, weight),
        projection_rows=projection_rows,
        token_axis=1,
        input_strides=input_strides,
        input_storage_offset=input_storage_offset,
    )
