"""cuTile kernels used by Resin's native DeepSeek-V4 implementation."""

from __future__ import annotations

import math
from typing import Annotated, TypeAlias

import cuda.tile as ct
import torch
from torch import Tensor
from torch.nn import functional as F

_ConstantInt: TypeAlias = ct.Constant[int]
_ConstantBool: TypeAlias = ct.Constant[bool]
_StaticLastDim: TypeAlias = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(-1,)),
]
_StaticLastTwoDims: TypeAlias = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(-2, -1)),
]
_NEGATIVE = -3.402823466e38


@ct.kernel
def _top6_sqrt_softplus_kernel(
    logits: _StaticLastDim,
    bias: _StaticLastDim,
    expert_indices: _StaticLastDim,
    expert_weights: _StaticLastDim,
    route_scale,
    num_experts: _ConstantInt,
):
    row = ct.bid(0)
    logits_tile = ct.load(logits, (row, 0), shape=(1, num_experts)).astype(ct.float32)
    bias_tile = ct.load(bias, (0,), shape=(num_experts,)).reshape((1, num_experts))
    stable_softplus = ct.maximum(logits_tile, 0.0) + ct.log(
        1.0 + ct.exp(-ct.abs(logits_tile))
    )
    softplus = ct.where(logits_tile > 20.0, logits_tile, stable_softplus)
    original = ct.sqrt(softplus)
    scores = original + bias_tile
    expert_ids = ct.arange(num_experts, dtype=ct.int32).reshape((1, num_experts))

    index0 = ct.argmax(scores, 1, keepdims=True)
    remaining = ct.where(expert_ids == index0, _NEGATIVE, scores)
    index1 = ct.argmax(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index1, _NEGATIVE, remaining)
    index2 = ct.argmax(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index2, _NEGATIVE, remaining)
    index3 = ct.argmax(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index3, _NEGATIVE, remaining)
    index4 = ct.argmax(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index4, _NEGATIVE, remaining)
    index5 = ct.argmax(remaining, 1, keepdims=True)

    indices01 = ct.cat((index0, index1), 1)
    indices23 = ct.cat((index2, index3), 1)
    indices45 = ct.cat((index4, index5), 1)

    weight0 = ct.sum(
        ct.where(expert_ids == index0, original, 0.0), axis=1, keepdims=True
    )
    weight1 = ct.sum(
        ct.where(expert_ids == index1, original, 0.0), axis=1, keepdims=True
    )
    weight2 = ct.sum(
        ct.where(expert_ids == index2, original, 0.0), axis=1, keepdims=True
    )
    weight3 = ct.sum(
        ct.where(expert_ids == index3, original, 0.0), axis=1, keepdims=True
    )
    weight4 = ct.sum(
        ct.where(expert_ids == index4, original, 0.0), axis=1, keepdims=True
    )
    weight5 = ct.sum(
        ct.where(expert_ids == index5, original, 0.0), axis=1, keepdims=True
    )
    weights01 = ct.cat((weight0, weight1), 1)
    weights23 = ct.cat((weight2, weight3), 1)
    weights45 = ct.cat((weight4, weight5), 1)
    normalization = route_scale / (
        weight0 + weight1 + weight2 + weight3 + weight4 + weight5
    )

    ct.store(expert_indices, (row, 0), indices01)
    ct.store(expert_indices, (row, 1), indices23)
    ct.store(expert_indices, (row, 2), indices45)
    ct.store(expert_weights, (row, 0), weights01 * normalization)
    ct.store(expert_weights, (row, 1), weights23 * normalization)
    ct.store(expert_weights, (row, 2), weights45 * normalization)


def top6_sqrt_softplus(
    logits: Tensor,
    bias: Tensor,
    route_scale: float,
) -> tuple[Tensor, Tensor]:
    """Apply the exact top-six routing transform to dynamic token rows."""

    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("the optimized router requires nonempty two-dimensional logits")
    rows, num_experts = logits.shape
    if num_experts & (num_experts - 1):
        raise ValueError("the optimized routing expert count must be a power of two")
    if bias.shape != (num_experts,):
        raise ValueError("DeepSeek-V4 routing bias must match the expert count")
    expert_indices = torch.empty((rows, 6), dtype=torch.int32, device=logits.device)
    expert_weights = torch.empty((rows, 6), dtype=torch.float32, device=logits.device)
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (rows,),
        _top6_sqrt_softplus_kernel,
        (logits, bias, expert_indices, expert_weights, route_scale, num_experts),
    )
    return expert_weights, expert_indices


@ct.kernel
def _fp8_quantize_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    scales: _StaticLastDim,
    width: _ConstantInt,
):
    row = ct.bid(0)
    group = ct.bid(1)
    values = ct.load(source, (row, group), shape=(1, 128)).astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(values), axis=1, keepdims=True), 1e-4)
    exponent = ct.ceil(ct.log2(maximum / 448.0))
    scale = ct.exp2(exponent)
    quantized = ct.maximum(ct.minimum(values / scale, 448.0), -448.0).astype(
        ct.float8_e4m3fn
    )
    ct.store(output, (row, group), quantized)
    # E8M0 stores the unbiased power-of-two exponent plus 127. cuda-tile 1.4
    # can consume E8M0 arrays but cannot select the required rounding mode for
    # a float-to-E8M0 conversion, so emit the exact byte representation.
    ct.store(scales, (row, group), (exponent + 127).astype(ct.uint8))


@ct.kernel
def _fp8_simulate_kernel(values: _StaticLastDim, width: _ConstantInt):
    row = ct.bid(0)
    group = ct.bid(1)
    tile = ct.load(values, (row, group), shape=(1, 64)).astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(tile), axis=1, keepdims=True), 1e-4)
    scale = ct.exp2(ct.ceil(ct.log2(maximum / 448.0)))
    simulated = ct.maximum(ct.minimum(tile / scale, 448.0), -448.0).astype(
        ct.float8_e4m3fn
    )
    ct.store(
        values,
        (row, group),
        (simulated.astype(ct.float32) * scale).astype(values.dtype),
    )


@ct.kernel
def _fp4_simulate_kernel(values: _StaticLastDim, width: _ConstantInt):
    row = ct.bid(0)
    group = ct.bid(1)
    tile = ct.load(values, (row, group), shape=(1, 32)).astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(tile), axis=1, keepdims=True), 6.0 * (2.0**-126))
    scale = ct.exp2(ct.ceil(ct.log2(maximum / 6.0)))
    simulated = ct.maximum(ct.minimum(tile / scale, 6.0), -6.0).astype(ct.float4_e2m1fn)
    ct.store(
        values,
        (row, group),
        (simulated.astype(ct.float32) * scale).astype(values.dtype),
    )


def act_quant(
    values: Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Power-of-two FP8 block quantization matching the Flash checkpoint."""

    if inplace:
        if block_size != 64:
            raise ValueError("in-place FP8 simulation requires 64-value blocks")
        destination = values
        values = values.contiguous()
        rows, width = values.numel() // values.shape[-1], values.shape[-1]
        ct.launch(
            torch.cuda.current_stream(values.device),
            (rows, width // block_size),
            _fp8_simulate_kernel,
            (values.view(rows, width), width),
        )
        destination.copy_(values)
        return destination
    if block_size != 128 or scale_fmt != "ue8m0":
        raise ValueError("DeepSeek-V4 FP8 GEMMs require 128-value UE8M0 blocks")
    if scale_dtype != torch.float8_e8m0fnu:
        raise ValueError("DeepSeek-V4 FP8 scales must use float8_e8m0fnu")
    rows, width = values.numel() // values.shape[-1], values.shape[-1]
    output = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (*values.shape[:-1], width // block_size),
        dtype=torch.uint8,
        device=values.device,
    )
    scales = scale_storage.view(scale_dtype)
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, width // block_size),
        _fp8_quantize_kernel,
        (
            values.view(rows, width),
            output.view(rows, width),
            scale_storage.view(rows, -1),
            width,
        ),
    )
    return output, scales


def fp4_act_quant(values: Tensor, block_size: int = 32, inplace: bool = False):
    """Apply the checkpoint's in-place FP4 QAT simulation."""

    if block_size != 32 or not inplace:
        raise ValueError("Resin only uses the in-place 32-value FP4 simulation")
    rows, width = values.numel() // values.shape[-1], values.shape[-1]
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, width // block_size),
        _fp4_simulate_kernel,
        (values.view(rows, width), width),
    )
    return values


@ct.kernel
def _qk_norm_rope_store_kernel(
    query: _StaticLastTwoDims,
    key_value: _StaticLastDim,
    norm_weight: _StaticLastDim,
    frequencies: _StaticLastTwoDims,
    cache: _StaticLastTwoDims,
    position,
    heads: _ConstantInt,
    dimension: _ConstantInt,
    rope_dimension: _ConstantInt,
    window: _ConstantInt,
    epsilon: float,
):
    block = ct.bid(0)
    batch = ct.bid(1)
    token = ct.load(position, (batch,), shape=(1,)).item()
    if block < heads:
        value = ct.load(
            query, (batch, 0, block, 0), shape=(1, 1, 1, dimension)
        ).reshape((dimension,))
        value = value.astype(ct.float32)
        value *= ct.rsqrt(ct.sum(value * value) / dimension + epsilon)
    else:
        value = ct.load(
            key_value, (batch, 0, 0), shape=(1, 1, dimension)
        ).reshape((dimension,))
        weight = ct.load(norm_weight, (0,), shape=(dimension,)).astype(ct.float32)
        value = value.astype(ct.float32)
        value *= ct.rsqrt(ct.sum(value * value) / dimension + epsilon) * weight

    # Match the released model's BF16 RMSNorm output before RoPE in FP32.
    value = value.astype(ct.bfloat16).astype(ct.float32)
    rope = ct.extract(
        value,
        (dimension // rope_dimension - 1,),
        shape=(rope_dimension,),
    ).reshape((rope_dimension // 2, 2))
    frequency = ct.load(
        frequencies,
        (token, 0, 0),
        shape=(1, rope_dimension // 2, 2),
    ).reshape((rope_dimension // 2, 2))
    real = ct.extract(rope, (0, 0), shape=(rope_dimension // 2, 1))
    imag = ct.extract(rope, (0, 1), shape=(rope_dimension // 2, 1))
    frequency_real = ct.extract(frequency, (0, 0), shape=(rope_dimension // 2, 1))
    frequency_imag = ct.extract(frequency, (0, 1), shape=(rope_dimension // 2, 1))
    rope = ct.cat(
        (
            real * frequency_real - imag * frequency_imag,
            real * frequency_imag + imag * frequency_real,
        ),
        axis=1,
    ).reshape((rope_dimension,))
    groups = value.reshape((dimension // rope_dimension, rope_dimension))
    group_ids = ct.arange(dimension // rope_dimension, dtype=ct.int32).reshape(
        (dimension // rope_dimension, 1)
    )
    value = ct.where(
        group_ids == dimension // rope_dimension - 1,
        ct.broadcast_to(rope.reshape((1, rope_dimension)), groups.shape),
        groups,
    ).reshape((dimension,))

    if block < heads:
        ct.store(
            query,
            (batch, 0, block, 0),
            value.reshape((1, 1, 1, dimension)).astype(query.dtype),
        )
    else:
        # QAT-simulate the non-RoPE dimensions in 64-value E4M3 groups and
        # directly store the token into the circular window cache.
        groups = value.reshape((dimension // 64, 64))
        maximum = ct.maximum(ct.max(ct.abs(groups), axis=1, keepdims=True), 1e-4)
        scale = ct.exp2(ct.ceil(ct.log2(maximum / 448.0)))
        simulated = (
            ct.maximum(ct.minimum(groups / scale, 448.0), -448.0)
            .astype(ct.float8_e4m3fn)
            .astype(ct.float32)
            * scale
        )
        group_ids = ct.arange(dimension // 64, dtype=ct.int32).reshape(
            (dimension // 64, 1)
        )
        value = ct.where(group_ids < dimension // 64 - 1, simulated, groups).reshape(
            (dimension,)
        )
        ct.store(
            cache,
            (batch, token % window, 0),
            value.reshape((1, 1, dimension)).astype(cache.dtype),
        )


def qk_norm_rope_store(
    query: Tensor,
    key_value: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    position: Tensor,
    epsilon: float,
) -> None:
    """Normalize Q/K, apply RoPE, quantize K, and update the decode cache."""

    batch_size = query.shape[0]
    if (
        query.shape[1] != 1
        or key_value.shape != (batch_size, 1, query.shape[-1])
        or position.shape != (batch_size,)
        or cache.shape[0] != batch_size
    ):
        raise ValueError("fused Q/K decode requires one token per matching batch row")
    dimension = query.shape[-1]
    rope_dimension = frequencies.shape[-2] * 2
    if dimension != 512 or rope_dimension != 64:
        raise ValueError(
            "DeepSeek-V4 fused Q/K decode requires 512D heads and 64D RoPE"
        )
    ct.launch(
        torch.cuda.current_stream(query.device),
        (query.shape[2] + 1, batch_size),
        _qk_norm_rope_store_kernel,
        (
            query,
            key_value,
            norm_weight,
            frequencies,
            cache,
            position,
            query.shape[2],
            dimension,
            rope_dimension,
            cache.shape[1],
            epsilon,
        ),
    )


@ct.kernel
def _compress_state_pool_kernel(
    values: _StaticLastDim,
    gates: _StaticLastDim,
    ape: _StaticLastDim,
    value_state: _StaticLastDim,
    score_state: _StaticLastDim,
    position,
    ratio: _ConstantInt,
    dimension: _ConstantInt,
    overlap: _ConstantBool,
):
    block = ct.bid(0)
    batch = ct.bid(1)
    token = ct.load(position, (batch,), shape=(1,)).item()
    slot = token % ratio
    state_slot = slot + ratio if overlap else slot
    current = ct.load(values, (batch, 0, block), shape=(1, 1, 128)).astype(
        ct.float32
    )
    score = ct.load(gates, (batch, 0, block), shape=(1, 1, 128)).astype(ct.float32)
    score += ct.load(ape, (slot, block), shape=(1, 128)).reshape((1, 1, 128))
    ct.store(value_state, (batch, state_slot, block), current)
    ct.store(score_state, (batch, state_slot, block), score)

    if (token + 1) % ratio == 0:
        if block < dimension // 128:
            rows = ct.arange(ratio, dtype=ct.int32).reshape((1, ratio, 1))
            if overlap:
                old_values = ct.load(
                    value_state, (batch, 0, block), shape=(1, ratio, 128)
                )
                old_scores = ct.load(
                    score_state, (batch, 0, block), shape=(1, ratio, 128)
                )
                second_block = block + dimension // 128
                new_values = ct.load(
                    value_state, (batch, 1, second_block), shape=(1, ratio, 128)
                )
                new_scores = ct.load(
                    score_state, (batch, 1, second_block), shape=(1, ratio, 128)
                )
                direct = ct.load(
                    values, (batch, 0, second_block), shape=(1, 1, 128)
                ).astype(ct.float32)
                direct_score = ct.load(
                    gates, (batch, 0, second_block), shape=(1, 1, 128)
                ).astype(ct.float32)
                direct_score += ct.load(
                    ape, (slot, second_block), shape=(1, 128)
                ).reshape((1, 1, 128))
                last = rows == slot
                new_values = ct.where(
                    last, ct.broadcast_to(direct, new_values.shape), new_values
                )
                new_scores = ct.where(
                    last, ct.broadcast_to(direct_score, new_scores.shape), new_scores
                )
                pooled_values = ct.cat((old_values, new_values), axis=1)
                pooled_scores = ct.cat((old_scores, new_scores), axis=1)
            else:
                pooled_values = ct.load(
                    value_state, (batch, 0, block), shape=(1, ratio, 128)
                )
                pooled_scores = ct.load(
                    score_state, (batch, 0, block), shape=(1, ratio, 128)
                )
                last = rows == slot
                pooled_values = ct.where(
                    last, ct.broadcast_to(current, pooled_values.shape), pooled_values
                )
                pooled_scores = ct.where(
                    last, ct.broadcast_to(score, pooled_scores.shape), pooled_scores
                )
            maximum = ct.max(pooled_scores, axis=1, keepdims=True)
            weights = ct.exp(pooled_scores - maximum)
            pooled = ct.sum(pooled_values * weights, axis=1, keepdims=True)
            pooled /= ct.sum(weights, axis=1, keepdims=True)
            ct.store(values, (batch, 0, block), pooled)

        if overlap:
            window_values = ct.load(
                value_state, (batch, 1, block), shape=(1, ratio, 128)
            )
            window_scores = ct.load(
                score_state, (batch, 1, block), shape=(1, ratio, 128)
            )
            rows = ct.arange(ratio, dtype=ct.int32).reshape((1, ratio, 1))
            last = rows == slot
            window_values = ct.where(
                last, ct.broadcast_to(current, window_values.shape), window_values
            )
            window_scores = ct.where(
                last, ct.broadcast_to(score, window_scores.shape), window_scores
            )
            ct.store(value_state, (batch, 0, block), window_values)
            ct.store(score_state, (batch, 0, block), window_scores)


@ct.kernel
def _compress_finalize_kernel(
    values: _StaticLastDim,
    norm_weight: _StaticLastDim,
    frequencies: _StaticLastTwoDims,
    cache: _StaticLastDim,
    position,
    ratio: _ConstantInt,
    dimension: _ConstantInt,
    stages: _ConstantInt,
    rotate: _ConstantBool,
    epsilon: float,
):
    batch = ct.bid(1)
    token = ct.load(position, (batch,), shape=(1,)).item()
    if (token + 1) % ratio == 0:
        value = ct.load(values, (batch, 0, 0), shape=(1, 1, dimension))
        # The released model rounds the pooled FP32 value to BF16 before RMSNorm.
        value = value.astype(ct.bfloat16).astype(ct.float32).reshape((dimension,))
        weight = ct.load(norm_weight, (0,), shape=(dimension,)).astype(ct.float32)
        value *= ct.rsqrt(ct.sum(value * value) / dimension + epsilon) * weight
        value = value.astype(ct.bfloat16).astype(ct.float32)

        rope = ct.extract(value, (dimension // 64 - 1,), shape=(64,)).reshape((32, 2))
        frequency = ct.load(
            frequencies, (token + 1 - ratio, 0, 0), shape=(1, 32, 2)
        ).reshape((32, 2))
        real = ct.extract(rope, (0, 0), shape=(32, 1))
        imag = ct.extract(rope, (0, 1), shape=(32, 1))
        frequency_real = ct.extract(frequency, (0, 0), shape=(32, 1))
        frequency_imag = ct.extract(frequency, (0, 1), shape=(32, 1))
        rope = ct.cat(
            (
                real * frequency_real - imag * frequency_imag,
                real * frequency_imag + imag * frequency_real,
            ),
            axis=1,
        ).reshape((64,))
        groups = value.reshape((dimension // 64, 64))
        group_ids = ct.arange(dimension // 64, dtype=ct.int32).reshape(
            (dimension // 64, 1)
        )
        value = ct.where(
            group_ids == dimension // 64 - 1,
            ct.broadcast_to(rope.reshape((1, 64)), groups.shape),
            groups,
        ).reshape((dimension,))

        if rotate:
            for stage in ct.static_iter(range(stages)):
                stride = 1 << stage
                grouped = value.reshape((dimension // (2 * stride), 2, stride))
                first = ct.extract(
                    grouped,
                    (0, 0, 0),
                    shape=(dimension // (2 * stride), 1, stride),
                )
                second = ct.extract(
                    grouped,
                    (0, 1, 0),
                    shape=(dimension // (2 * stride), 1, stride),
                )
                value = ct.cat((first + second, first - second), axis=1).reshape(
                    (dimension,)
                )
            value *= dimension**-0.5
            grouped = value.reshape((dimension // 32, 32))
            maximum = ct.maximum(
                ct.max(ct.abs(grouped), axis=1, keepdims=True), 6.0 * (2.0**-126)
            )
            scale = ct.exp2(ct.ceil(ct.log2(maximum / 6.0)))
            value = (
                ct.maximum(ct.minimum(grouped / scale, 6.0), -6.0)
                .astype(ct.float4_e2m1fn)
                .astype(ct.float32)
                * scale
            ).reshape((dimension,))
        else:
            grouped = value.reshape((dimension // 64, 64))
            maximum = ct.maximum(ct.max(ct.abs(grouped), axis=1, keepdims=True), 1e-4)
            scale = ct.exp2(ct.ceil(ct.log2(maximum / 448.0)))
            simulated = (
                ct.maximum(ct.minimum(grouped / scale, 448.0), -448.0)
                .astype(ct.float8_e4m3fn)
                .astype(ct.float32)
                * scale
            )
            value = ct.where(
                group_ids < dimension // 64 - 1, simulated, grouped
            ).reshape((dimension,))
        ct.store(
            cache,
            (batch, token // ratio, 0),
            value.reshape((1, 1, dimension)).astype(cache.dtype),
        )


def compress_decode(
    values: Tensor,
    gates: Tensor,
    ape: Tensor,
    value_state: Tensor,
    score_state: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    position: Tensor,
    ratio: int,
    rotate: bool,
    epsilon: float,
) -> None:
    """Update one compressor and materialize only completed cache groups."""

    dimension = cache.shape[-1]
    width = values.shape[-1]
    batch_size = values.shape[0]
    if (
        values.shape[1] != 1
        or gates.shape != values.shape
        or position.shape != (batch_size,)
        or value_state.shape[0] != batch_size
        or score_state.shape[0] != batch_size
        or cache.shape[0] != batch_size
        or width != dimension * (2 if ratio == 4 else 1)
    ):
        raise ValueError("decode compression requires one correctly sized token per row")
    stream = torch.cuda.current_stream(values.device)
    ct.launch(
        stream,
        (width // 128, batch_size),
        _compress_state_pool_kernel,
        (
            values,
            gates,
            ape,
            value_state,
            score_state,
            position,
            ratio,
            dimension,
            ratio == 4,
        ),
    )
    ct.launch(
        stream,
        (1, batch_size),
        _compress_finalize_kernel,
        (
            values,
            norm_weight,
            frequencies,
            cache,
            position,
            ratio,
            dimension,
            int(math.log2(dimension)),
            rotate,
            epsilon,
        ),
    )


@ct.kernel(occupancy=2)
def _sparse_attention_kernel(
    query: _StaticLastTwoDims,
    key_value: _StaticLastDim,
    sinks: _StaticLastDim,
    indices,
    output: _StaticLastTwoDims,
    heads: _ConstantInt,
    dimension: _ConstantInt,
    index_width: _ConstantInt,
    softmax_scale: float,
):
    sequence = ct.bid(0)
    batch = ct.bid(1)
    q = ct.load(query, (batch, sequence, 0, 0), shape=(1, 1, heads, dimension))
    q = q.reshape((heads, dimension))
    sink = ct.load(sinks, (0,), shape=(heads,)).reshape((heads, 1))
    maximum = sink
    denominator = ct.full((heads, 1), 1.0, dtype=ct.float32)
    accumulator = ct.full((heads, dimension), 0.0, dtype=ct.float32)
    dimensions = ct.arange(dimension, dtype=ct.int32).reshape((1, dimension))
    for index_block in range(index_width // 64):
        selected = ct.load(
            indices, (batch, sequence, index_block), shape=(1, 1, 64)
        ).reshape((64, 1))
        batches = ct.full((64, 1), batch, dtype=ct.int32)
        values = ct.gather(
            key_value,
            (batches, selected, dimensions),
            padding_value=0,
        )
        scores = ct.mma(
            q,
            values.permute((1, 0)),
            ct.full((heads, 64), 0.0, dtype=ct.float32),
        )
        valid = selected.permute((1, 0)) >= 0
        scores = ct.where(valid, scores * softmax_scale, _NEGATIVE)
        next_maximum = ct.maximum(maximum, ct.max(scores, axis=1, keepdims=True))
        correction = ct.exp(maximum - next_maximum)
        probabilities = ct.exp(scores - next_maximum)
        denominator = denominator * correction + ct.sum(
            probabilities, axis=1, keepdims=True
        )
        accumulator *= correction
        accumulator = ct.mma(probabilities.astype(query.dtype), values, accumulator)
        maximum = next_maximum
    ct.store(
        output,
        (batch, sequence, 0, 0),
        (accumulator / denominator)
        .reshape((1, 1, heads, dimension))
        .astype(output.dtype),
    )


def sparse_attn(
    query: Tensor,
    key_value: Tensor,
    sinks: Tensor,
    indices: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Gather sparse KV positions and apply online-softmax attention."""

    padded = math.ceil(indices.shape[-1] / 64) * 64
    if padded != indices.shape[-1]:
        indices = F.pad(indices, (0, padded - indices.shape[-1]), value=-1)
    output = torch.empty_like(query)
    ct.launch(
        torch.cuda.current_stream(query.device),
        (query.shape[1], query.shape[0], 1),
        _sparse_attention_kernel,
        (
            query,
            key_value,
            sinks,
            indices,
            output,
            query.shape[2],
            query.shape[3],
            padded,
            softmax_scale,
        ),
    )
    return output


@ct.kernel
def _hc_post_mix_kernel(
    values: _StaticLastDim,
    residual: _StaticLastDim,
    post: _StaticLastDim,
    combination: _StaticLastDim,
    output: _StaticLastDim,
):
    row = ct.bid(0)
    branch = ct.bid(1)
    block = ct.bid(2)
    value = ct.load(values, (row, block), shape=(1, 128)).astype(ct.float32)
    post_scale = ct.load(post, (row, branch), shape=(1, 1)).item()
    branches = ct.load(residual, (row, 0, block), shape=(1, 4, 128)).astype(ct.float32)
    weights = ct.load(combination, (row, 0, branch), shape=(1, 4, 1))
    mixed = value * post_scale + ct.sum(branches * weights, axis=1)
    ct.store(
        output,
        (row, branch, block),
        mixed.reshape((1, 1, 128)).astype(output.dtype),
    )


def hc_post_mix(
    values: Tensor,
    residual: Tensor,
    post: Tensor,
    combination: Tensor,
) -> Tensor:
    """Apply the four mHC output branches and residual mixing in one kernel."""

    dimension = residual.shape[-1]
    rows = residual.numel() // (4 * dimension)
    output = torch.empty_like(residual)
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, 4, dimension // 128),
        _hc_post_mix_kernel,
        (
            values.view(rows, dimension),
            residual.view(rows, 4, dimension),
            post.view(rows, 4),
            combination.view(rows, 4, 4),
            output.view(rows, 4, dimension),
        ),
    )
    return output


@ct.kernel
def _hadamard_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    dimension: _ConstantInt,
    stages: _ConstantInt,
):
    row = ct.bid(0)
    values = ct.load(source, (row, 0), shape=(1, dimension)).reshape((dimension,))
    for stage in ct.static_iter(range(stages)):
        stride = 1 << stage
        groups = dimension // (2 * stride)
        shaped = values.reshape((groups, 2, stride))
        first = ct.extract(shaped, (0, 0, 0), shape=(groups, 1, stride))
        second = ct.extract(shaped, (0, 1, 0), shape=(groups, 1, stride))
        values = ct.cat((first + second, first - second), axis=1).reshape((dimension,))
    values *= dimension**-0.5
    ct.store(output, (row, 0), values.reshape((1, dimension)).astype(output.dtype))


def hadamard_transform(values: Tensor) -> Tensor:
    """Apply a normalized Walsh-Hadamard transform to the final dimension."""

    dimension = values.shape[-1]
    if dimension not in {128, 512}:
        raise ValueError("DeepSeek-V4 Hadamard width must be 128 or 512")
    rows = values.numel() // dimension
    output = torch.empty_like(values)
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows,),
        _hadamard_kernel,
        (
            values.view(rows, dimension),
            output.view(rows, dimension),
            dimension,
            int(math.log2(dimension)),
        ),
    )
    return output


__all__ = [
    "act_quant",
    "compress_decode",
    "fp4_act_quant",
    "hadamard_transform",
    "hc_post_mix",
    "qk_norm_rope_store",
    "sparse_attn",
    "top6_sqrt_softplus",
]
