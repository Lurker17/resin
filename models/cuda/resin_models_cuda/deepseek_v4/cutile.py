"""DeepSeek-V4 cuTile kernels."""

from __future__ import annotations

import math
from typing import Annotated, TypeAlias

import cuda.tile as ct
import torch
from torch import Tensor
from torch.nn import functional as F

from resin.backend.cuda.paged_kv import PagedCacheMapping

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
_MIN_WIDE_TILE_PROGRAMS = 64
_INDEX_SCORE_TILE = 64
_CACHE_MAP_TILE = 64
_DYNAMIC_SHAPE_ALIGNMENT = 16


def _aligned_size(size: int, alignment: int = _DYNAMIC_SHAPE_ALIGNMENT) -> int:
    """Pad a dynamic dimension into one stable cuTile divisibility class."""

    return (size + alignment - 1) // (2 * alignment) * (2 * alignment) + alignment


@ct.kernel
def _paged_rows_kernel(
    parent_ids,
    child_slots,
    first_blocks,
    positions,
    output,
    block_tokens: _ConstantInt,
    divisor: _ConstantInt,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    row_scale: _ConstantInt,
    layer_offset: int,
):
    """Map logical token positions to physical cache rows in one launch."""

    tile = ct.bid(0)
    batch = ct.bid(1)
    logical_positions = ct.load(
        positions,
        (batch, tile),
        shape=(1, _CACHE_MAP_TILE),
        padding_mode=ct.PaddingMode.ZERO,
    ).astype(ct.int32)
    first = ct.load(first_blocks, (batch,), shape=(1,)).item()
    columns = logical_positions // block_tokens - first
    batches = ct.full((1, _CACHE_MAP_TILE), batch, dtype=ct.int32)
    selected_parents = ct.gather(parent_ids, (batches, columns), padding_value=0)
    selected_slots = ct.gather(child_slots, (batches, columns), padding_value=0)

    child_rows = parent_rows // packing
    offsets = (logical_positions % block_tokens) // divisor
    physical = (
        (selected_parents * parent_rows + selected_slots * child_rows) * row_scale
        + layer_offset
        + offsets
    )
    in_table = (columns >= 0) & (columns < parent_ids.shape[1])
    physical = ct.where(selected_parents > 0, physical, 0)
    physical = ct.where(in_table, physical, -1)
    ct.store(output, (batch, tile), physical.astype(ct.int32))


def paged_cache_rows(
    parent_ids: Tensor,
    child_slots: Tensor,
    first_blocks: Tensor,
    positions: Tensor,
    *,
    block_tokens: int,
    divisor: int,
    parent_rows: int,
    packing: int,
    row_scale: int,
    layer_offset: int,
) -> Tensor:
    """Map a batched logical-position tensor without PyTorch kernel chains."""

    if positions.ndim < 2 or parent_ids.ndim != 2:
        raise ValueError("paged cache mapping requires batched tensors")
    batch_size = positions.shape[0]
    if (
        child_slots.shape != parent_ids.shape
        or first_blocks.shape != (batch_size,)
        or parent_ids.shape[0] != batch_size
    ):
        raise ValueError("paged cache rows must share one batch dimension")
    logical_width = positions.numel() // batch_size
    padded_width = _aligned_size(logical_width)
    flat_positions = positions.reshape(batch_size, logical_width)
    if padded_width != logical_width:
        flat_positions = F.pad(flat_positions, (0, padded_width - logical_width))
    output = torch.empty_like(flat_positions, dtype=torch.int32)
    ct.launch(
        torch.cuda.current_stream(positions.device),
        (math.ceil(flat_positions.shape[1] / _CACHE_MAP_TILE), batch_size),
        _paged_rows_kernel,
        (
            parent_ids,
            child_slots,
            first_blocks,
            flat_positions,
            output,
            block_tokens,
            divisor,
            parent_rows,
            packing,
            row_scale,
            layer_offset,
        ),
    )
    return output[:, :logical_width].reshape(positions.shape)


@ct.kernel
def _paged_state_locations_kernel(
    parent_ids,
    child_slots,
    first_blocks,
    positions,
    output_pages,
    output_offsets,
    block_tokens: _ConstantInt,
    packing: _ConstantInt,
):
    tile = ct.bid(0)
    batch = ct.bid(1)
    logical_positions = ct.load(
        positions,
        (batch, tile),
        shape=(1, _CACHE_MAP_TILE),
        padding_mode=ct.PaddingMode.ZERO,
    ).astype(ct.int32)
    first = ct.load(first_blocks, (batch,), shape=(1,)).item()
    columns = logical_positions // block_tokens - first
    batches = ct.full((1, _CACHE_MAP_TILE), batch, dtype=ct.int32)
    selected_parents = ct.gather(parent_ids, (batches, columns), padding_value=0)
    selected_slots = ct.gather(child_slots, (batches, columns), padding_value=0)
    valid = (columns >= 0) & (columns < parent_ids.shape[1])
    physical_children = ct.where(
        selected_parents > 0, selected_parents * packing + selected_slots, 0
    )
    ct.store(
        output_pages,
        (batch, tile),
        ct.where(valid, physical_children, -1).astype(ct.int32),
    )
    ct.store(
        output_offsets,
        (batch, tile),
        ct.where(valid, logical_positions % block_tokens, 0).astype(ct.int32),
    )


def paged_state_locations(
    parent_ids: Tensor,
    child_slots: Tensor,
    first_blocks: Tensor,
    positions: Tensor,
    *,
    block_tokens: int,
    packing: int,
) -> tuple[Tensor, Tensor]:
    """Map recurrent state locations in one launch."""

    if positions.ndim < 2 or parent_ids.ndim != 2:
        raise ValueError("paged state mapping requires batched tensors")
    batch_size = positions.shape[0]
    if (
        child_slots.shape != parent_ids.shape
        or first_blocks.shape != (batch_size,)
        or parent_ids.shape[0] != batch_size
    ):
        raise ValueError("paged state rows must share one batch dimension")
    logical_width = positions.numel() // batch_size
    padded_width = _aligned_size(logical_width)
    flat_positions = positions.reshape(batch_size, logical_width)
    if padded_width != logical_width:
        flat_positions = F.pad(flat_positions, (0, padded_width - logical_width))
    output_pages = torch.empty_like(flat_positions, dtype=torch.int32)
    output_offsets = torch.empty_like(flat_positions, dtype=torch.int32)
    ct.launch(
        torch.cuda.current_stream(positions.device),
        (math.ceil(flat_positions.shape[1] / _CACHE_MAP_TILE), batch_size),
        _paged_state_locations_kernel,
        (
            parent_ids,
            child_slots,
            first_blocks,
            flat_positions,
            output_pages,
            output_offsets,
            block_tokens,
            packing,
        ),
    )
    shape = positions.shape
    return (
        output_pages[:, :logical_width].reshape(shape),
        output_offsets[:, :logical_width].reshape(shape),
    )


def _quant_groups_per_program(rows: int, group_count: int) -> int:
    """Use wider tiles only when the resulting grid retains enough parallel work."""

    if group_count % 4 == 0 and rows * (group_count // 4) >= _MIN_WIDE_TILE_PROGRAMS:
        return 4
    return 1


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
    ct.store(
        expert_weights,
        (row, 0),
        (weights01 * normalization).astype(expert_weights.dtype),
    )
    ct.store(
        expert_weights,
        (row, 1),
        (weights23 * normalization).astype(expert_weights.dtype),
    )
    ct.store(
        expert_weights,
        (row, 2),
        (weights45 * normalization).astype(expert_weights.dtype),
    )


def top6_sqrt_softplus(
    logits: Tensor,
    bias: Tensor,
    route_scale: float,
) -> tuple[Tensor, Tensor]:
    """Apply the exact top-six routing transform to dynamic token rows."""

    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(
            "the optimized router requires nonempty two-dimensional logits"
        )
    rows, num_experts = logits.shape
    if num_experts & (num_experts - 1):
        raise ValueError("the optimized routing expert count must be a power of two")
    if bias.shape != (num_experts,):
        raise ValueError("DeepSeek-V4 routing bias must match the expert count")
    expert_indices = torch.empty((rows, 6), dtype=torch.int32, device=logits.device)
    expert_weights = torch.empty((rows, 6), dtype=torch.bfloat16, device=logits.device)
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (rows,),
        _top6_sqrt_softplus_kernel,
        (logits, bias, expert_indices, expert_weights, route_scale, num_experts),
    )
    return expert_weights, expert_indices


@ct.kernel
def _hash_sqrt_softplus_kernel(
    logits: _StaticLastDim,
    token_ids,
    token_to_experts: _StaticLastDim,
    expert_indices: _StaticLastDim,
    expert_weights: _StaticLastDim,
    route_scale,
    expert_tile: _ConstantInt,
):
    """Fuse hash lookup with the exact sqrt-softplus routing weights."""

    row = ct.bid(0)
    logits_tile = ct.load(
        logits,
        (row, 0),
        shape=(1, expert_tile),
        padding_mode=ct.PaddingMode.ZERO,
    ).astype(ct.float32)
    stable_softplus = ct.maximum(logits_tile, 0.0) + ct.log(
        1.0 + ct.exp(-ct.abs(logits_tile))
    )
    original = ct.sqrt(ct.where(logits_tile > 20.0, logits_tile, stable_softplus))
    token = ct.load(token_ids, (row,), shape=(1,)).item()
    indices01 = ct.load(token_to_experts, (token, 0), shape=(1, 2)).astype(ct.int32)
    indices23 = ct.load(token_to_experts, (token, 1), shape=(1, 2)).astype(ct.int32)
    indices45 = ct.load(token_to_experts, (token, 2), shape=(1, 2)).astype(ct.int32)
    expert_ids = ct.arange(expert_tile, dtype=ct.int32).reshape((1, expert_tile))
    weights0 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices01, (0, 0), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    weights1 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices01, (0, 1), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    weights2 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices23, (0, 0), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    weights3 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices23, (0, 1), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    weights4 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices45, (0, 0), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    weights5 = ct.sum(
        ct.where(
            expert_ids == ct.extract(indices45, (0, 1), shape=(1, 1)), original, 0.0
        ),
        axis=1,
        keepdims=True,
    )
    normalization = route_scale / (
        weights0 + weights1 + weights2 + weights3 + weights4 + weights5
    )
    weight01 = ct.cat((weights0, weights1), axis=1) * normalization
    weight23 = ct.cat((weights2, weights3), axis=1) * normalization
    weight45 = ct.cat((weights4, weights5), axis=1) * normalization
    ct.store(expert_indices, (row, 0), indices01)
    ct.store(expert_indices, (row, 1), indices23)
    ct.store(expert_indices, (row, 2), indices45)
    ct.store(expert_weights, (row, 0), weight01.astype(expert_weights.dtype))
    ct.store(expert_weights, (row, 1), weight23.astype(expert_weights.dtype))
    ct.store(expert_weights, (row, 2), weight45.astype(expert_weights.dtype))


def hash_sqrt_softplus(
    logits: Tensor,
    token_ids: Tensor,
    token_to_experts: Tensor,
    route_scale: float,
) -> tuple[Tensor, Tensor]:
    """Apply hash routing without materializing PyTorch pointwise intermediates."""

    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("hash routing requires nonempty two-dimensional logits")
    rows, num_experts = logits.shape
    if token_ids.shape != (rows,) or token_to_experts.shape[-1] != 6:
        raise ValueError("hash routing inputs have incompatible shapes")
    expert_indices = torch.empty((rows, 6), dtype=torch.int32, device=logits.device)
    expert_weights = torch.empty((rows, 6), dtype=torch.bfloat16, device=logits.device)
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (rows,),
        _hash_sqrt_softplus_kernel,
        (
            logits,
            token_ids,
            token_to_experts,
            expert_indices,
            expert_weights,
            route_scale,
            1 << (num_experts - 1).bit_length(),
        ),
    )
    return expert_weights, expert_indices


@ct.kernel
def _index_score_kernel(
    query: _StaticLastTwoDims,
    cache: _StaticLastDim,
    cache_parent_ids,
    cache_child_slots,
    cache_first_blocks,
    head_weights: _StaticLastDim,
    positions,
    output,
    ratio: _ConstantInt,
    heads: _ConstantInt,
    head_dim: _ConstantInt,
    tile_width: _ConstantInt,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    row_scale: _ConstantInt,
    layer_offset: int,
    weight_scale: float,
):
    """Score one cache tile, skipping tiles beyond the runtime sequence."""

    tile = ct.bid(0)
    sequence = ct.bid(1)
    batch = ct.bid(2)
    token = ct.load(positions, (batch, sequence), shape=(1, 1)).item()
    complete = (token + 1) // ratio
    columns = tile * tile_width + ct.arange(tile_width, dtype=ct.int32)
    if tile * tile_width < complete:
        q = ct.load(
            query,
            (batch, sequence, 0, 0),
            shape=(1, 1, heads, head_dim),
        ).reshape((heads, head_dim))
        logical_positions = columns * ratio
        first = ct.load(cache_first_blocks, (batch,), shape=(1,)).item()
        page_columns = logical_positions // 128 - first
        selected_parents = ct.gather(
            cache_parent_ids,
            (
                ct.full((tile_width,), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        selected_slots = ct.gather(
            cache_child_slots,
            (
                ct.full((tile_width,), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        child_rows = parent_rows // packing
        rows = (
            (selected_parents * parent_rows + selected_slots * child_rows) * row_scale
            + layer_offset
            + (logical_positions % 128) // ratio
        ).reshape((tile_width, 1))
        dimensions = ct.arange(head_dim, dtype=ct.int32).reshape((1, head_dim))
        keys = ct.gather(cache, (rows, dimensions), padding_value=0)
        scores = ct.mma(
            q,
            keys.permute((1, 0)),
            ct.full((heads, tile_width), 0.0, dtype=ct.float32),
        )
        weights = (
            ct.load(head_weights, (batch, sequence, 0), shape=(1, 1, heads))
            .reshape((heads, 1))
            .astype(ct.float32)
        )
        reduced = ct.sum(
            ct.maximum(scores, 0.0) * weights * weight_scale,
            axis=0,
        )
        reduced = ct.where(columns < complete, reduced, float("-inf"))
    else:
        reduced = ct.full((tile_width,), float("-inf"), dtype=ct.float32)
    ct.store(
        output,
        (batch, sequence, tile),
        reduced.reshape((1, 1, tile_width)).astype(output.dtype),
    )


def index_scores(
    query: Tensor,
    cache: Tensor,
    head_weights: Tensor,
    positions: Tensor,
    ratio: int,
    weight_scale: float,
    *,
    cache_mapping: PagedCacheMapping,
    cache_width: int | None = None,
) -> Tensor:
    """Compute and collapse per-head index logits in one tensor-core launch."""

    if query.ndim != 4 or cache.ndim != 2:
        raise ValueError("index QK inputs have invalid ranks")
    batch_size, block_length, heads, head_dim = query.shape
    width = cache_width
    if width is None:
        width = (
            math.ceil(
                cache_mapping.parent_ids.shape[1] * 128 / ratio / _INDEX_SCORE_TILE
            )
            * _INDEX_SCORE_TILE
        )
    if (
        head_weights.shape != (batch_size, block_length, heads)
        or positions.shape != (batch_size, block_length)
        or cache.shape[-1] != head_dim
        or width % _INDEX_SCORE_TILE
    ):
        raise ValueError("index scoring requires matching padded inputs")
    output = torch.empty(
        (batch_size, block_length, width),
        dtype=query.dtype,
        device=query.device,
    )
    if cache_mapping.block_tokens != 128 or cache_mapping.divisor != ratio:
        raise ValueError("index score cache mapping has incompatible geometry")
    ct.launch(
        torch.cuda.current_stream(query.device),
        (width // _INDEX_SCORE_TILE, block_length, batch_size),
        _index_score_kernel,
        (
            query,
            cache,
            cache_mapping.parent_ids,
            cache_mapping.child_slots,
            cache_mapping.first_blocks,
            head_weights,
            positions,
            output,
            ratio,
            heads,
            head_dim,
            _INDEX_SCORE_TILE,
            cache_mapping.parent_rows,
            cache_mapping.packing,
            cache_mapping.row_scale,
            cache_mapping.layer_offset,
            weight_scale,
        ),
    )
    return output


@ct.kernel
def _finish_index_topk_kernel(
    indices: _StaticLastDim,
    positions,
    cache_parent_ids,
    cache_child_slots,
    cache_first_blocks,
    output: _StaticLastDim,
    ratio: _ConstantInt,
    width: _ConstantInt,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    layer_offset: int,
    offset: int,
):
    sequence = ct.bid(0)
    batch = ct.bid(1)
    token = ct.load(positions, (batch, sequence), shape=(1, 1)).item()
    # Index top-k widths are not necessarily powers of two (V4 uses 192), so
    # process them as fixed 64-wide subtiles while retaining one kernel launch.
    for tile in range(width // 64):
        values = ct.load(indices, (batch, sequence, tile), shape=(1, 1, 64))
        values = ct.where(values < (token + 1) // ratio, values + offset, -1)
        logical_positions = values * ratio
        first = ct.load(cache_first_blocks, (batch,), shape=(1,)).item()
        page_columns = logical_positions // 128 - first
        selected_parents = ct.gather(
            cache_parent_ids,
            (
                ct.full((1, 1, 64), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        selected_slots = ct.gather(
            cache_child_slots,
            (
                ct.full((1, 1, 64), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        child_rows = parent_rows // packing
        physical = (
            selected_parents * parent_rows
            + selected_slots * child_rows
            + layer_offset
            + (logical_positions % 128) // ratio
        )
        values = ct.where((values >= 0) & (selected_parents > 0), physical, -1)
        ct.store(output, (batch, sequence, tile), values.astype(ct.int32))


def finish_index_topk(
    indices: Tensor,
    positions: Tensor,
    ratio: int,
    offset: int,
    *,
    cache_mapping: PagedCacheMapping,
) -> Tensor:
    """Mask causal index results and convert them to attention indices."""

    if indices.ndim != 3 or positions.shape != indices.shape[:2]:
        raise ValueError("index top-k results must match the token positions")
    if indices.shape[-1] % 64:
        raise ValueError("index top-k width must be divisible by 64")
    output = torch.empty_like(indices, dtype=torch.int32)
    if cache_mapping.block_tokens != 128 or cache_mapping.divisor != ratio:
        raise ValueError("index top-k cache mapping has incompatible geometry")
    ct.launch(
        torch.cuda.current_stream(indices.device),
        (indices.shape[1], indices.shape[0]),
        _finish_index_topk_kernel,
        (
            indices,
            positions,
            cache_mapping.parent_ids,
            cache_mapping.child_slots,
            cache_mapping.first_blocks,
            output,
            ratio,
            indices.shape[-1],
            cache_mapping.parent_rows,
            cache_mapping.packing,
            cache_mapping.layer_offset,
            offset,
        ),
    )
    return output


@ct.kernel
def _fp8_quantize_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    scales: _StaticLastDim,
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
def _fp8_quantize_four_groups_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    scales: _StaticLastDim,
):
    row = ct.bid(0)
    tile = ct.bid(1)
    values = ct.load(source, (row, tile), shape=(1, 512)).reshape((4, 128))
    values = values.astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(values), axis=1, keepdims=True), 1e-4)
    exponent = ct.ceil(ct.log2(maximum / 448.0))
    scale = ct.exp2(exponent)
    quantized = ct.maximum(ct.minimum(values / scale, 448.0), -448.0).astype(
        ct.float8_e4m3fn
    )
    ct.store(output, (row, tile), quantized.reshape((1, 512)))
    ct.store(scales, (row, tile), (exponent + 127).astype(ct.uint8).reshape((1, 4)))


@ct.kernel
def _fp8_quantize_four_groups_packed_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    scale_storage: _StaticLastDim,
):
    """Quantize four K128 groups and emit DeepGEMM's packed MN-major scales."""

    row = ct.bid(0)
    tile = ct.bid(1)
    values = ct.load(source, (row, tile), shape=(1, 512)).reshape((4, 128))
    values = values.astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(values), axis=1, keepdims=True), 1e-4)
    exponent = ct.ceil(ct.log2(maximum / 448.0))
    scale = ct.exp2(exponent)
    quantized = ct.maximum(ct.minimum(values / scale, 448.0), -448.0).astype(
        ct.float8_e4m3fn
    )
    ct.store(output, (row, tile), quantized.reshape((1, 512)))
    # Storage is [packed_k, tma_aligned_m, 4 bytes]. Viewing it as int32 and
    # transposing produces shape [m, packed_k], stride [1, aligned_m].
    ct.store(
        scale_storage,
        (tile, row, 0),
        (exponent + 127).astype(ct.uint8).reshape((1, 1, 4)),
    )


@ct.kernel
def _fp8_quantize_moe_kernel(
    source: _StaticLastDim,
    output: _StaticLastDim,
    scale_storage: _StaticLastDim,
):
    """Quantize K128 groups and directly emit FlashInfer's K32 scale layout."""

    row = ct.bid(0)
    tile = ct.bid(1)
    values = ct.load(source, (row, tile), shape=(1, 512)).reshape((4, 128))
    values = values.astype(ct.float32)
    maximum = ct.maximum(ct.max(ct.abs(values), axis=1, keepdims=True), 1e-4)
    exponent = ct.ceil(ct.log2(maximum / 448.0))
    scale = ct.exp2(exponent)
    quantized = ct.maximum(ct.minimum(values / scale, 448.0), -448.0).astype(
        ct.float8_e4m3fn
    )
    repeated_scales = ct.broadcast_to(
        (exponent + 127).astype(ct.uint8), (4, 4)
    ).reshape((1, 16))
    ct.store(output, (row, tile), quantized.reshape((1, 512)))
    ct.store(scale_storage, (row, tile), repeated_scales)


@ct.kernel
def _fp8_simulate_kernel(values: _StaticLastDim):
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
def _fp4_simulate_kernel(values: _StaticLastDim):
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


@ct.kernel
def _fp4_simulate_four_groups_kernel(values: _StaticLastDim):
    row = ct.bid(0)
    tile = ct.bid(1)
    groups = ct.load(values, (row, tile), shape=(1, 128)).reshape((4, 32))
    groups = groups.astype(ct.float32)
    maximum = ct.maximum(
        ct.max(ct.abs(groups), axis=1, keepdims=True), 6.0 * (2.0**-126)
    )
    scale = ct.exp2(ct.ceil(ct.log2(maximum / 6.0)))
    simulated = ct.maximum(ct.minimum(groups / scale, 6.0), -6.0).astype(
        ct.float4_e2m1fn
    )
    ct.store(
        values,
        (row, tile),
        (simulated.astype(ct.float32) * scale).reshape((1, 128)).astype(values.dtype),
    )


def act_quant(
    values: Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
    *,
    packed_scales: bool = False,
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
            (values.view(rows, width),),
        )
        destination.copy_(values)
        return destination
    if block_size != 128 or scale_fmt != "ue8m0":
        raise ValueError("DeepSeek-V4 FP8 GEMMs require 128-value UE8M0 blocks")
    if scale_dtype != torch.float8_e8m0fnu:
        raise ValueError("DeepSeek-V4 FP8 scales must use float8_e8m0fnu")
    rows, width = values.numel() // values.shape[-1], values.shape[-1]
    output = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    group_count = width // block_size
    if packed_scales:
        if group_count % 4:
            raise ValueError("packed DeepGEMM scales require a multiple of four groups")
        aligned_rows = (rows + 3) // 4 * 4
        scale_storage = torch.empty(
            (group_count // 4, aligned_rows, 4),
            dtype=torch.uint8,
            device=values.device,
        )
        ct.launch(
            torch.cuda.current_stream(values.device),
            (rows, group_count // 4),
            _fp8_quantize_four_groups_packed_kernel,
            (
                values.view(rows, width),
                output.view(rows, width),
                scale_storage,
            ),
        )
        scales = scale_storage.view(torch.int32).squeeze(-1).T[:rows]
        return output, scales
    scale_storage = torch.empty(
        (*values.shape[:-1], width // block_size),
        dtype=torch.uint8,
        device=values.device,
    )
    scales = scale_storage.view(scale_dtype)
    groups_per_program = _quant_groups_per_program(rows, group_count)
    kernel = (
        _fp8_quantize_four_groups_kernel
        if groups_per_program == 4
        else _fp8_quantize_kernel
    )
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, group_count // groups_per_program),
        kernel,
        (
            values.view(rows, width),
            output.view(rows, width),
            scale_storage.view(rows, -1),
        ),
    )
    return output, scales


def moe_act_quant(
    values: Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Quantize once into the exact MXFP8 scale layout consumed by FlashInfer."""

    if block_size != 128 or scale_fmt != "ue8m0":
        raise ValueError("DeepSeek-V4 MoE requires 128-value UE8M0 blocks")
    if scale_dtype != torch.float8_e8m0fnu:
        raise ValueError("DeepSeek-V4 MoE scales must use float8_e8m0fnu")
    rows, width = values.numel() // values.shape[-1], values.shape[-1]
    if width % 512:
        raise ValueError("DeepSeek-V4 MoE hidden width must be divisible by 512")
    output = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (rows, width // 32), dtype=torch.uint8, device=values.device
    )
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, width // 512),
        _fp8_quantize_moe_kernel,
        (
            values.view(rows, width),
            output.view(rows, width),
            scale_storage,
        ),
    )
    return output, scale_storage.view(torch.float8_e4m3fn)


def fp4_act_quant(values: Tensor, block_size: int = 32, inplace: bool = False):
    """Apply the checkpoint's in-place FP4 QAT simulation."""

    if block_size != 32 or not inplace:
        raise ValueError("Resin only uses the in-place 32-value FP4 simulation")
    rows, width = values.numel() // values.shape[-1], values.shape[-1]
    group_count = width // block_size
    groups_per_program = _quant_groups_per_program(rows, group_count)
    kernel = (
        _fp4_simulate_four_groups_kernel
        if groups_per_program == 4
        else _fp4_simulate_kernel
    )
    ct.launch(
        torch.cuda.current_stream(values.device),
        (rows, group_count // groups_per_program),
        kernel,
        (values.view(rows, width),),
    )
    return values


@ct.kernel
def _swiglu_kernel(
    gate: _StaticLastDim,
    up: _StaticLastDim,
    output: _StaticLastDim,
    tile_width: _ConstantInt,
    apply_limit: _ConstantBool,
    limit: float,
):
    row = ct.bid(0)
    tile = ct.bid(1)
    gate_values = ct.load(gate, (row, tile), shape=(1, tile_width)).astype(ct.float32)
    up_values = ct.load(up, (row, tile), shape=(1, tile_width)).astype(ct.float32)
    if apply_limit:
        gate_values = ct.minimum(gate_values, limit)
        up_values = ct.maximum(ct.minimum(up_values, limit), -limit)
    activated = gate_values / (1.0 + ct.exp(-gate_values))
    ct.store(output, (row, tile), (activated * up_values).astype(output.dtype))


def swiglu(gate: Tensor, up: Tensor, limit: float = 0.0) -> Tensor:
    """Fuse shared-expert SwiGLU and its FP32 activation boundary."""

    if gate.shape != up.shape or gate.ndim < 2 or gate.shape[-1] <= 0:
        raise ValueError("fused SwiGLU inputs must have one matching feature width")
    rows, width = gate.numel() // gate.shape[-1], gate.shape[-1]
    tile_width = min(width & -width, 128)
    output = torch.empty_like(gate)
    ct.launch(
        torch.cuda.current_stream(gate.device),
        (rows, width // tile_width),
        _swiglu_kernel,
        (
            gate.view(rows, width),
            up.view(rows, width),
            output.view(rows, width),
            tile_width,
            limit > 0,
            limit,
        ),
    )
    return output


@ct.kernel
def _qk_norm_rope_kernel(
    query: _StaticLastTwoDims,
    key_value: _StaticLastDim,
    norm_weight: _StaticLastDim,
    frequencies: _StaticLastTwoDims,
    cache: _StaticLastTwoDims,
    cache_parent_ids,
    cache_child_slots,
    cache_first_blocks,
    position,
    heads: _ConstantInt,
    dimension: _ConstantInt,
    rope_dimension: _ConstantInt,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    layer_offset: int,
    epsilon: float,
):
    block = ct.bid(0)
    batch = ct.bid(1)
    sequence = ct.bid(2)
    token = ct.load(position, (batch, sequence), shape=(1, 1)).item()
    if block < heads:
        value = ct.load(
            query, (batch, sequence, block, 0), shape=(1, 1, 1, dimension)
        ).reshape((dimension,))
        value = value.astype(ct.float32)
        value *= ct.rsqrt(ct.sum(value * value) / dimension + epsilon)
    else:
        value = ct.load(
            key_value, (batch, sequence, 0), shape=(1, 1, dimension)
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
            (batch, sequence, block, 0),
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
        first = ct.load(cache_first_blocks, (batch,), shape=(1,)).item()
        column = token // 128 - first
        cache_parent = ct.gather(
            cache_parent_ids,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        cache_slot = ct.gather(
            cache_child_slots,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        child_rows = parent_rows // packing
        cache_row = (
            cache_parent * parent_rows
            + cache_slot * child_rows
            + layer_offset
            + token % 128
        )
        ct.store(
            cache,
            (cache_row, 0),
            value.reshape((1, dimension)).astype(cache.dtype),
        )


def qk_norm_rope_store(
    query: Tensor,
    key_value: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    position: Tensor,
    epsilon: float,
    *,
    cache_mapping: PagedCacheMapping,
) -> None:
    """Normalize Q/K, apply RoPE, quantize K, and update the decode cache."""

    batch_size = query.shape[0]
    if query.shape[1] != 1 or position.shape != (batch_size,):
        raise ValueError("fused Q/K decode requires one token per matching batch row")
    qk_norm_rope_block(
        query,
        key_value,
        norm_weight,
        frequencies,
        cache,
        position.view(batch_size, 1),
        epsilon,
        cache_mapping=cache_mapping,
    )


def qk_norm_rope_block(
    query: Tensor,
    key_value: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    positions: Tensor,
    epsilon: float,
    *,
    cache_mapping: PagedCacheMapping,
) -> None:
    """Normalize/rotate a verification block and store its guarded window KV."""

    batch_size, block_length = query.shape[:2]
    if (
        block_length == 0
        or key_value.shape != (batch_size, block_length, query.shape[-1])
        or positions.shape != (batch_size, block_length)
        or cache.ndim != 2
        or cache.shape[-1] != query.shape[-1]
    ):
        raise ValueError("fused Q/K verification requires matching token blocks")
    dimension = query.shape[-1]
    rope_dimension = frequencies.shape[-2] * 2
    if dimension != 512 or rope_dimension != 64:
        raise ValueError(
            "DeepSeek-V4 fused Q/K verification requires 512D heads and 64D RoPE"
        )
    if cache_mapping.block_tokens != 128 or cache_mapping.divisor != 1:
        raise ValueError("Q/K cache mapping must address 128-token window rows")
    ct.launch(
        torch.cuda.current_stream(query.device),
        (query.shape[2] + 1, batch_size, block_length),
        _qk_norm_rope_kernel,
        (
            query,
            key_value,
            norm_weight,
            frequencies,
            cache,
            cache_mapping.parent_ids,
            cache_mapping.child_slots,
            cache_mapping.first_blocks,
            positions,
            query.shape[2],
            dimension,
            rope_dimension,
            cache_mapping.parent_rows,
            cache_mapping.packing,
            cache_mapping.layer_offset,
            epsilon,
        ),
    )


def kv_norm_rope_store(
    key_value: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    positions: Tensor,
    epsilon: float,
    *,
    cache_mapping: PagedCacheMapping,
) -> None:
    """Fuse DSpark main-KV normalization, RoPE, quantization, and cache store."""

    if (
        key_value.ndim != 3
        or key_value.shape[-1] != 512
        or positions.shape != key_value.shape[:2]
        or frequencies.shape[-2:] != (32, 2)
        or norm_weight.shape != (512,)
        or cache.ndim != 2
        or cache.shape[-1] != 512
    ):
        raise ValueError("DSpark main-KV store requires matching 512D token blocks")
    if cache_mapping.block_tokens != 128 or cache_mapping.divisor != 1:
        raise ValueError("DSpark main-KV mapping must address window rows")
    # The query array is an unused view when the compile-time head count is zero.
    query_placeholder = key_value.unsqueeze(2)
    ct.launch(
        torch.cuda.current_stream(key_value.device),
        (1, key_value.shape[0], key_value.shape[1]),
        _qk_norm_rope_kernel,
        (
            query_placeholder,
            key_value,
            norm_weight,
            frequencies,
            cache,
            cache_mapping.parent_ids,
            cache_mapping.child_slots,
            cache_mapping.first_blocks,
            positions,
            0,
            512,
            64,
            cache_mapping.parent_rows,
            cache_mapping.packing,
            cache_mapping.layer_offset,
            epsilon,
        ),
    )


@ct.kernel
def _rope_last_block_kernel(
    values: _StaticLastDim,
    frequencies: _StaticLastTwoDims,
    positions,
    dimension: _ConstantInt,
    rope_dimension: _ConstantInt,
    inverse: _ConstantBool,
):
    head = ct.bid(0)
    batch = ct.bid(1)
    sequence = ct.bid(2)
    token = ct.load(positions, (batch, sequence), shape=(1, 1)).item()
    rope = ct.load(
        values,
        (batch, sequence, head, dimension // rope_dimension - 1),
        shape=(1, 1, 1, rope_dimension),
    ).reshape((rope_dimension // 2, 2))
    rope = rope.astype(ct.float32)
    frequency = ct.load(
        frequencies,
        (token, 0, 0),
        shape=(1, rope_dimension // 2, 2),
    ).reshape((rope_dimension // 2, 2))
    real = ct.extract(rope, (0, 0), shape=(rope_dimension // 2, 1))
    imag = ct.extract(rope, (0, 1), shape=(rope_dimension // 2, 1))
    frequency_real = ct.extract(frequency, (0, 0), shape=(rope_dimension // 2, 1))
    frequency_imag = ct.extract(frequency, (0, 1), shape=(rope_dimension // 2, 1))
    if inverse:
        frequency_imag = -frequency_imag
    rotated = ct.cat(
        (
            real * frequency_real - imag * frequency_imag,
            real * frequency_imag + imag * frequency_real,
        ),
        axis=1,
    ).reshape((1, 1, 1, rope_dimension))
    ct.store(
        values,
        (batch, sequence, head, dimension // rope_dimension - 1),
        rotated.astype(values.dtype),
    )


def rope_last_block_inplace(
    values: Tensor,
    frequencies: Tensor,
    positions: Tensor,
    *,
    inverse: bool = False,
) -> Tensor:
    """Rotate the final RoPE group of a batched token block in one launch."""

    if values.ndim != 4 or positions.shape != values.shape[:2]:
        raise ValueError("fused RoPE values and positions must share a token block")
    dimension = values.shape[-1]
    rope_dimension = frequencies.shape[-2] * 2
    if rope_dimension != 64 or dimension % rope_dimension:
        raise ValueError("DeepSeek-V4 fused RoPE requires a 64D final group")
    ct.launch(
        torch.cuda.current_stream(values.device),
        (values.shape[2], values.shape[0], values.shape[1]),
        _rope_last_block_kernel,
        (
            values,
            frequencies,
            positions,
            dimension,
            rope_dimension,
            inverse,
        ),
    )
    return values


@ct.kernel
def _verify_attention_metadata_kernel(
    position,
    window_parent_ids,
    window_child_slots,
    window_first_blocks,
    compressed_parent_ids,
    compressed_child_slots,
    compressed_first_blocks,
    compressed_indices,
    indices,
    sparse_lengths,
    sequence_lengths,
    window_size: _ConstantInt,
    compress_ratio: _ConstantInt,
    compressed_index_width: int,
    has_compression: _ConstantBool,
    has_explicit_compressed_indices: _ConstantBool,
    parent_rows: _ConstantInt,
    window_packing: _ConstantInt,
    compressed_packing: _ConstantInt,
    window_layer_offset: int,
    compressed_layer_offset: int,
):
    tile = ct.bid(0)
    query = ct.bid(1)
    batch = ct.bid(2)
    start = ct.load(position, (batch,), shape=(1,)).item()
    token = start + query
    # The sliding-window width includes the current token. As the causal block
    # grows, retire one pre-block history row per query instead of appending the
    # whole block to a fixed-width history (which made verify attend to as many
    # as ``window_size + block_length`` dense tokens).
    block_count = ct.minimum(query + 1, window_size)
    history_count = ct.minimum(start, ct.maximum(window_size - block_count, 0))
    block_end = history_count + block_count
    columns = tile * 64 + ct.arange(64, dtype=ct.int32)
    old_column = ct.minimum(columns, ct.maximum(history_count - 1, 0))
    old_position = start - history_count + old_column
    block_column = columns - history_count
    block_position = start + query + 1 - block_count + block_column
    window_positions = ct.where(columns < history_count, old_position, block_position)
    window_valid = (columns < history_count) | (
        (columns >= history_count) & (columns < block_end)
    )
    first_window = ct.load(window_first_blocks, (batch,), shape=(1,)).item()
    window_columns = window_positions // 128 - first_window
    window_parents = ct.gather(
        window_parent_ids,
        (
            ct.full((64,), batch, dtype=ct.int32),
            window_columns,
        ),
        padding_value=0,
    )
    window_children = ct.gather(
        window_child_slots,
        (
            ct.full((64,), batch, dtype=ct.int32),
            window_columns,
        ),
        padding_value=0,
    )
    child_rows = parent_rows // window_packing
    window_indices = (
        window_parents * parent_rows
        + window_children * child_rows
        + window_layer_offset
        + window_positions % 128
    )
    window_indices = ct.where(window_valid & (window_parents > 0), window_indices, -1)

    output = window_indices
    if has_compression:
        compressed_column = columns - block_end
        if has_explicit_compressed_indices:
            compressed = ct.gather(
                compressed_indices,
                (batch, query, compressed_column),
                mask=(columns >= block_end)
                & (compressed_column < compressed_index_width),
                padding_value=-1,
            )
        else:
            compressed_complete = ct.minimum(
                (token + 1) // compress_ratio,
                compressed_index_width,
            )
            compressed = ct.where(
                (compressed_column >= 0) & (compressed_column < compressed_complete),
                compressed_column,
                -1,
            )
        if not has_explicit_compressed_indices:
            first_compressed = ct.load(
                compressed_first_blocks, (batch,), shape=(1,)
            ).item()
            compressed_positions = compressed * compress_ratio
            compressed_columns = compressed_positions // 128 - first_compressed
            compressed_parents = ct.gather(
                compressed_parent_ids,
                (
                    ct.full((64,), batch, dtype=ct.int32),
                    compressed_columns,
                ),
                padding_value=0,
            )
            compressed_children = ct.gather(
                compressed_child_slots,
                (
                    ct.full((64,), batch, dtype=ct.int32),
                    compressed_columns,
                ),
                padding_value=0,
            )
            child_rows = parent_rows // compressed_packing
            compressed_physical = (
                compressed_parents * parent_rows
                + compressed_children * child_rows
                + compressed_layer_offset
                + (compressed_positions % 128) // compress_ratio
            )
            compressed = ct.where(
                (compressed >= 0) & (compressed_parents > 0),
                compressed_physical,
                -1,
            )
        output = ct.where(
            columns < block_end,
            output,
            compressed,
        )
    else:
        output = ct.where(columns < block_end, output, -1)
    ct.store(indices, (batch, query, tile), output.reshape((1, 1, 64)).astype(ct.int32))

    compressed_complete = 0
    if has_compression:
        compressed_complete = ct.minimum(
            (token + 1) // compress_ratio,
            compressed_index_width,
        )
    ct.scatter(
        sparse_lengths,
        (batch, query),
        (block_end + compressed_complete).astype(ct.int32),
        mask=tile == 0,
    )
    ct.scatter(
        sequence_lengths,
        (batch, query),
        (token + 1).astype(ct.int32),
        mask=tile == 0,
    )


def verify_attention_metadata(
    position: Tensor,
    block_length: int,
    window_size: int,
    window_capacity: int,
    compressed_indices: Tensor | None,
    compressed_pool_width: int,
    compress_ratio: int,
    generated_compressed_width: int = 0,
    *,
    window_mapping: PagedCacheMapping,
    compressed_mapping: PagedCacheMapping | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build all sparse-verification indices and lengths in one launch."""

    if position.ndim != 1 or block_length <= 0 or window_size <= 0:
        raise ValueError("verification metadata requires valid rows and widths")
    batch_size = position.shape[0]
    if compressed_indices is not None and generated_compressed_width:
        raise ValueError("compressed indices cannot be both supplied and generated")
    compressed_index_width = (
        generated_compressed_width
        if compressed_indices is None
        else compressed_indices.shape[-1]
    )
    if compressed_indices is not None and compressed_indices.shape[:2] != (
        batch_size,
        block_length,
    ):
        raise ValueError("compressed verification indices must match the token block")
    if (compressed_index_width > 0) != (compress_ratio > 0):
        raise ValueError("compressed index geometry requires a compression ratio")
    if (
        window_mapping.block_tokens != 128
        or window_mapping.divisor != 1
        or (
            compressed_indices is None
            and compressed_index_width > 0
            and compressed_mapping is None
        )
    ):
        raise ValueError("paged verification metadata has invalid cache mappings")
    if compressed_mapping is not None and (
        compressed_mapping.block_tokens != 128
        or compressed_mapping.divisor != compress_ratio
        or compressed_mapping.parent_rows != window_mapping.parent_rows
    ):
        raise ValueError("compressed verification mapping is incompatible")
    index_width = _aligned_size(
        window_size + compressed_index_width,
        alignment=64,
    )
    indices = torch.empty(
        (batch_size, block_length, index_width),
        dtype=torch.int32,
        device=position.device,
    )
    sparse_lengths = torch.empty(
        (batch_size, block_length), dtype=torch.int32, device=position.device
    )
    sequence_lengths = torch.empty_like(sparse_lengths)
    compressed_input = indices if compressed_indices is None else compressed_indices
    window_parent_ids = window_mapping.parent_ids
    window_child_slots = window_mapping.child_slots
    window_first_blocks = window_mapping.first_blocks
    compressed_parent_ids = (
        indices if compressed_mapping is None else compressed_mapping.parent_ids
    )
    compressed_child_slots = (
        indices if compressed_mapping is None else compressed_mapping.child_slots
    )
    compressed_first_blocks = (
        position if compressed_mapping is None else compressed_mapping.first_blocks
    )
    ct.launch(
        torch.cuda.current_stream(position.device),
        (index_width // 64, block_length, batch_size),
        _verify_attention_metadata_kernel,
        (
            position,
            window_parent_ids,
            window_child_slots,
            window_first_blocks,
            compressed_parent_ids,
            compressed_child_slots,
            compressed_first_blocks,
            compressed_input,
            indices,
            sparse_lengths,
            sequence_lengths,
            window_size,
            compress_ratio,
            compressed_index_width,
            compress_ratio > 0,
            compressed_indices is not None,
            window_mapping.parent_rows,
            window_mapping.packing,
            compressed_mapping.packing if compressed_mapping is not None else 1,
            window_mapping.layer_offset,
            compressed_mapping.layer_offset if compressed_mapping is not None else 0,
        ),
    )
    return indices, sparse_lengths, sequence_lengths


@ct.kernel
def _dspark_attention_metadata_kernel(
    position,
    window_parent_ids,
    window_child_slots,
    window_first_blocks,
    indices: _StaticLastDim,
    sparse_lengths,
    sequence_lengths,
    window_size: _ConstantInt,
    block_length: _ConstantInt,
    index_width: _ConstantInt,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    layer_offset: int,
):
    """Build compact history-plus-draft indices for FlashInfer sparse MLA."""

    query = ct.bid(0)
    batch = ct.bid(1)
    anchor = ct.load(position, (batch,), shape=(1,)).item()
    history_count = ct.minimum(anchor, window_size)
    history_start = anchor - history_count
    for tile in ct.static_iter(range(index_width // 64)):
        columns = tile * 64 + ct.arange(64, dtype=ct.int32)
        history_positions = history_start + columns
        first = ct.load(window_first_blocks, (batch,), shape=(1,)).item()
        page_columns = history_positions // 128 - first
        selected_parents = ct.gather(
            window_parent_ids,
            (
                ct.full((64,), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        selected_slots = ct.gather(
            window_child_slots,
            (
                ct.full((64,), batch, dtype=ct.int32),
                page_columns,
            ),
            padding_value=0,
        )
        child_rows = parent_rows // packing
        history = (
            selected_parents * parent_rows
            + selected_slots * child_rows
            + layer_offset
            + history_positions % 128
        )
        history = ct.where(
            (columns < history_count) & (selected_parents > 0), history, -1
        )
        # FlashInfer reserves the first ``window_size`` columns for the SWA
        # pool even when a short sequence has fewer valid history tokens.
        draft_column = columns - window_size
        draft = draft_column + batch * block_length
        output = ct.where(
            columns < window_size,
            history,
            ct.where(
                (draft_column >= 0) & (draft_column < block_length),
                draft,
                -1,
            ),
        )
        ct.store(
            indices,
            (batch, query, tile),
            output.reshape((1, 1, 64)).astype(ct.int32),
        )
    ct.store(
        sparse_lengths,
        (batch, query),
        ct.full((1, 1), window_size + block_length, dtype=ct.int32),
    )
    # FlashInfer uses the logical sequence length to identify how many leading
    # sparse indices address the dense-window pool. Draft-block indices follow
    # that prefix and address the separate block-local KV pool.
    ct.store(
        sequence_lengths,
        (batch, query),
        ct.full((1, 1), anchor, dtype=ct.int32),
    )


def dspark_attention_metadata(
    position: Tensor,
    block_length: int,
    window_size: int,
    window_capacity: int,
    *,
    window_mapping: PagedCacheMapping,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build graph-safe sparse-MLA metadata for a parallel DSpark block."""

    if (
        position.ndim != 1
        or block_length <= 0
        or window_size <= 0
        or window_capacity < window_size
    ):
        raise ValueError("DSpark attention metadata requires valid geometry")
    batch_size = position.shape[0]
    index_width = math.ceil((window_size + block_length) / 64) * 64
    indices = torch.empty(
        (batch_size, block_length, index_width),
        dtype=torch.int32,
        device=position.device,
    )
    sparse_lengths = torch.empty(
        (batch_size, block_length), dtype=torch.int32, device=position.device
    )
    sequence_lengths = torch.empty_like(sparse_lengths)
    if window_mapping.block_tokens != 128 or window_mapping.divisor != 1:
        raise ValueError("DSpark window mapping must use 128-token pages")
    ct.launch(
        torch.cuda.current_stream(position.device),
        (block_length, batch_size),
        _dspark_attention_metadata_kernel,
        (
            position,
            window_mapping.parent_ids,
            window_mapping.child_slots,
            window_mapping.first_blocks,
            indices,
            sparse_lengths,
            sequence_lengths,
            window_size,
            block_length,
            index_width,
            window_mapping.parent_rows,
            window_mapping.packing,
            window_mapping.layer_offset,
        ),
    )
    return indices, sparse_lengths, sequence_lengths


@ct.kernel
def _compress_ring_kernel(
    values: _StaticLastDim,
    gates: _StaticLastDim,
    ape: _StaticLastDim,
    value_state: _StaticLastDim,
    score_state: _StaticLastDim,
    state_parent_ids,
    state_child_slots,
    state_first_blocks,
    positions,
    ratio: _ConstantInt,
    dimension: _ConstantInt,
    block_length: _ConstantInt,
    ring_size: _ConstantInt,
    overlap: _ConstantBool,
    state_block_tokens: _ConstantInt,
    state_packing: _ConstantInt,
):
    """Append one token block to a padded ring and pool completed windows.

    Speculative decode deliberately uses a ring larger than the compressor's
    logical window.  Rejected writes therefore cannot overwrite committed
    history; the next block overwrites those speculative rows before they can
    participate in a completed window.
    """

    block = ct.bid(0)
    batch = ct.bid(1)
    rows = ct.arange(ring_size, dtype=ct.int32).reshape((1, ring_size, 1))
    if overlap:
        end = ct.load(positions, (batch, block_length - 1), shape=(1, 1)).item()
        slots = ct.arange(ring_size, dtype=ct.int32)
        chronological = end - (end - slots) % ring_size
        first = ct.load(state_first_blocks, (batch,), shape=(1,)).item()
        columns = chronological // state_block_tokens - first
        selected_parents = ct.gather(
            state_parent_ids,
            (
                ct.full((ring_size,), batch, dtype=ct.int32),
                columns,
            ),
            padding_value=0,
        )
        selected_slots = ct.gather(
            state_child_slots,
            (
                ct.full((ring_size,), batch, dtype=ct.int32),
                columns,
            ),
            padding_value=0,
        )
        valid = chronological >= 0
        pages = ct.where(
            valid & (selected_parents > 0),
            selected_parents * state_packing + selected_slots,
            -1,
        ).reshape((ring_size, 1))
        offsets = ct.where(valid, chronological % state_block_tokens, 0).reshape(
            (ring_size, 1)
        )
        dimensions = block * 32 + ct.arange(32, dtype=ct.int32).reshape((1, 32))
        first_values = (
            ct.gather(value_state, (pages, offsets, dimensions), padding_value=0)
            .reshape((1, ring_size, 32))
            .astype(ct.float32)
        )
        first_scores = (
            ct.gather(
                score_state, (pages, offsets, dimensions), padding_value=float("-inf")
            )
            .reshape((1, ring_size, 32))
            .astype(ct.float32)
        )
        second_block = block + dimension // 32
        second_dimensions = second_block * 32 + ct.arange(32, dtype=ct.int32).reshape(
            (1, 32)
        )
        second_values = (
            ct.gather(value_state, (pages, offsets, second_dimensions), padding_value=0)
            .reshape((1, ring_size, 32))
            .astype(ct.float32)
        )
        second_scores = (
            ct.gather(
                score_state,
                (pages, offsets, second_dimensions),
                padding_value=float("-inf"),
            )
            .reshape((1, ring_size, 32))
            .astype(ct.float32)
        )

    for token_index in ct.static_iter(range(block_length)):
        token = ct.load(positions, (batch, token_index), shape=(1, 1)).item()
        ape_slot = token % ratio
        ring_slot = token % ring_size
        first = ct.load(state_first_blocks, (batch,), shape=(1,)).item()
        column = token // state_block_tokens - first
        selected_parent = ct.gather(
            state_parent_ids,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        selected_slot = ct.gather(
            state_child_slots,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        state_page = selected_parent * state_packing + selected_slot
        state_offset = token % state_block_tokens
        selected = rows == ring_slot
        current = ct.load(values, (batch, token_index, block), shape=(1, 1, 32)).astype(
            ct.float32
        )
        score = ct.load(gates, (batch, token_index, block), shape=(1, 1, 32)).astype(
            ct.float32
        )
        score += ct.load(ape, (ape_slot, block), shape=(1, 32)).reshape((1, 1, 32))
        ct.store(value_state, (state_page, state_offset, block), current)
        ct.store(score_state, (state_page, state_offset, block), score)

        if overlap:
            first_values = ct.where(
                selected, ct.broadcast_to(current, first_values.shape), first_values
            )
            first_scores = ct.where(
                selected, ct.broadcast_to(score, first_scores.shape), first_scores
            )
            second_current = ct.load(
                values, (batch, token_index, second_block), shape=(1, 1, 32)
            ).astype(ct.float32)
            second_score = ct.load(
                gates, (batch, token_index, second_block), shape=(1, 1, 32)
            ).astype(ct.float32)
            second_score += ct.load(
                ape, (ape_slot, second_block), shape=(1, 32)
            ).reshape((1, 1, 32))
            second_values = ct.where(
                selected,
                ct.broadcast_to(second_current, second_values.shape),
                second_values,
            )
            second_scores = ct.where(
                selected,
                ct.broadcast_to(second_score, second_scores.shape),
                second_scores,
            )
            ct.store(
                value_state, (state_page, state_offset, second_block), second_current
            )
            ct.store(
                score_state, (state_page, state_offset, second_block), second_score
            )

        if (token + 1) % ratio == 0:
            age = (token - rows + ring_size) % ring_size
            if overlap:
                old_mask = (age >= ratio) & (age < 2 * ratio)
                new_mask = age < ratio
                pooled_values = ct.cat((first_values, second_values), axis=1)
                pooled_scores = ct.cat(
                    (
                        ct.where(old_mask, first_scores, float("-inf")),
                        ct.where(new_mask, second_scores, float("-inf")),
                    ),
                    axis=1,
                )
            else:
                pooled_positions = token + 1 - ratio + ct.arange(ratio, dtype=ct.int32)
                first = ct.load(state_first_blocks, (batch,), shape=(1,)).item()
                pooled_columns = pooled_positions // state_block_tokens - first
                selected_parents = ct.gather(
                    state_parent_ids,
                    (
                        ct.full((ratio,), batch, dtype=ct.int32),
                        pooled_columns,
                    ),
                    padding_value=0,
                )
                selected_slots = ct.gather(
                    state_child_slots,
                    (
                        ct.full((ratio,), batch, dtype=ct.int32),
                        pooled_columns,
                    ),
                    padding_value=0,
                )
                pooled_pages = (
                    selected_parents * state_packing + selected_slots
                ).reshape((ratio, 1))
                pooled_offsets = (pooled_positions % state_block_tokens).reshape(
                    (ratio, 1)
                )
                dimensions = block * 32 + ct.arange(32, dtype=ct.int32).reshape((1, 32))
                pooled_values = (
                    ct.gather(
                        value_state,
                        (pooled_pages, pooled_offsets, dimensions),
                        padding_value=0,
                    )
                    .reshape((1, ratio, 32))
                    .astype(ct.float32)
                )
                pooled_scores = (
                    ct.gather(
                        score_state,
                        (pooled_pages, pooled_offsets, dimensions),
                        padding_value=float("-inf"),
                    )
                    .reshape((1, ratio, 32))
                    .astype(ct.float32)
                )
            maximum = ct.max(pooled_scores, axis=1, keepdims=True)
            weights = ct.exp(pooled_scores - maximum)
            pooled = ct.sum(pooled_values * weights, axis=1, keepdims=True)
            pooled /= ct.sum(weights, axis=1, keepdims=True)
            ct.store(values, (batch, token_index, block), pooled)


@ct.kernel
def _clear_rejected_ring_kernel(
    value_state: _StaticLastDim,
    score_state: _StaticLastDim,
    state_parent_ids,
    state_child_slots,
    state_first_blocks,
    positions,
    processed_counts,
    block_length: _ConstantInt,
    state_block_tokens: _ConstantInt,
    state_packing: _ConstantInt,
):
    """Clear the uncommitted tail of a c128 speculative write."""

    block = ct.bid(0)
    batch = ct.bid(1)
    start = ct.load(positions, (batch,), shape=(1,)).item()
    processed = ct.load(processed_counts, (batch,), shape=(1,)).item()
    zeros = ct.full((1, 1, 128), 0.0, dtype=ct.float32)
    empty_scores = ct.full((1, 1, 128), float("-inf"), dtype=ct.float32)
    for token_index in ct.static_iter(range(block_length)):
        if token_index >= processed:
            token = start + token_index
            first = ct.load(state_first_blocks, (batch,), shape=(1,)).item()
            column = token // state_block_tokens - first
            selected_parent = ct.gather(
                state_parent_ids,
                (
                    ct.full((1,), batch, dtype=ct.int32),
                    ct.full((1,), column, dtype=ct.int32),
                ),
                padding_value=0,
            ).item()
            selected_slot = ct.gather(
                state_child_slots,
                (
                    ct.full((1,), batch, dtype=ct.int32),
                    ct.full((1,), column, dtype=ct.int32),
                ),
                padding_value=0,
            ).item()
            page = selected_parent * state_packing + selected_slot
            offset = token % state_block_tokens
            ct.store(value_state, (page, offset, block), zeros)
            ct.store(score_state, (page, offset, block), empty_scores)


@ct.kernel
def _compress_verify_finalize_kernel(
    values: _StaticLastDim,
    norm_weight: _StaticLastDim,
    frequencies: _StaticLastTwoDims,
    cache: _StaticLastDim,
    cache_parent_ids,
    cache_child_slots,
    cache_first_blocks,
    positions,
    ratio: _ConstantInt,
    dimension: _ConstantInt,
    stages: _ConstantInt,
    rotate: _ConstantBool,
    parent_rows: _ConstantInt,
    packing: _ConstantInt,
    layer_offset: int,
    epsilon: float,
):
    token_index = ct.bid(0)
    batch = ct.bid(1)
    token = ct.load(positions, (batch, token_index), shape=(1, 1)).item()
    if (token + 1) % ratio == 0:
        value = ct.load(values, (batch, token_index, 0), shape=(1, 1, dimension))
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
                ct.max(ct.abs(grouped), axis=1, keepdims=True),
                6.0 * (2.0**-126),
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
                group_ids < dimension // 64 - 1, simulated, groups
            ).reshape((dimension,))
        first = ct.load(cache_first_blocks, (batch,), shape=(1,)).item()
        column = token // 128 - first
        cache_parent = ct.gather(
            cache_parent_ids,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        cache_slot = ct.gather(
            cache_child_slots,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        child_rows = parent_rows // packing
        cache_row = (
            cache_parent * parent_rows
            + cache_slot * child_rows
            + layer_offset
            + (token % 128) // ratio
        )
        ct.store(
            cache,
            (cache_row, 0),
            value.reshape((1, dimension)).astype(cache.dtype),
        )


def compress_verify(
    values: Tensor,
    gates: Tensor,
    ape: Tensor,
    value_state: Tensor,
    score_state: Tensor,
    norm_weight: Tensor,
    frequencies: Tensor,
    cache: Tensor,
    positions: Tensor,
    ratio: int,
    rotate: bool,
    epsilon: float,
    *,
    cache_mapping: PagedCacheMapping,
    state_mapping: PagedCacheMapping,
    state_ring_size: int,
) -> None:
    """Update one fixed verification block in a padded compressor ring."""

    dimension = cache.shape[-1]
    width = values.shape[-1]
    batch_size, block_length = values.shape[:2]
    ring_size = state_ring_size
    logical_window = ratio * (2 if ratio == 4 else 1)
    if (
        block_length == 0
        or gates.shape != values.shape
        or positions.shape != (batch_size, block_length)
        or value_state.ndim != 3
        or score_state.shape != value_state.shape
        or width != dimension * (2 if ratio == 4 else 1)
        or ring_size < logical_window
        or ring_size % ratio
    ):
        raise ValueError("compression requires matching token blocks and a valid ring")
    if cache_mapping.block_tokens != 128 or cache_mapping.divisor != ratio:
        raise ValueError("compressed cache mapping has incompatible geometry")
    if state_mapping.divisor != 1 or state_mapping.row_scale != 1:
        raise ValueError("compressor state mapping has incompatible geometry")
    stream = torch.cuda.current_stream(values.device)
    ct.launch(
        stream,
        (dimension // 32, batch_size),
        _compress_ring_kernel,
        (
            values,
            gates,
            ape,
            value_state,
            score_state,
            state_mapping.parent_ids,
            state_mapping.child_slots,
            state_mapping.first_blocks,
            positions,
            ratio,
            dimension,
            block_length,
            ring_size,
            ratio == 4,
            state_mapping.block_tokens,
            state_mapping.packing,
        ),
    )
    ct.launch(
        stream,
        (block_length, batch_size),
        _compress_verify_finalize_kernel,
        (
            values,
            norm_weight,
            frequencies,
            cache,
            cache_mapping.parent_ids,
            cache_mapping.child_slots,
            cache_mapping.first_blocks,
            positions,
            ratio,
            dimension,
            int(math.log2(dimension)),
            rotate,
            cache_mapping.parent_rows,
            cache_mapping.packing,
            cache_mapping.layer_offset,
            epsilon,
        ),
    )


def clear_rejected_compress_state(
    value_state: Tensor,
    score_state: Tensor,
    positions: Tensor,
    processed_counts: Tensor,
    block_length: int,
    *,
    state_mapping: PagedCacheMapping,
    state_ring_size: int,
) -> None:
    """Discard rejected c128 rows without restoring the committed ring."""

    _, _, dimension = value_state.shape
    batch_size = positions.shape[0]
    ring_size = state_ring_size
    if (
        score_state.shape != value_state.shape
        or positions.ndim != 1
        or processed_counts.shape != (batch_size,)
        or block_length <= 0
        or ring_size < 128
        or dimension % 128
    ):
        raise ValueError("invalid speculative compressor cleanup")
    if state_mapping.divisor != 1 or state_mapping.row_scale != 1:
        raise ValueError("cleanup state mapping has incompatible geometry")
    ct.launch(
        torch.cuda.current_stream(value_state.device),
        (dimension // 128, batch_size),
        _clear_rejected_ring_kernel,
        (
            value_state,
            score_state,
            state_mapping.parent_ids,
            state_mapping.child_slots,
            state_mapping.first_blocks,
            positions,
            processed_counts,
            block_length,
            state_mapping.block_tokens,
            state_mapping.packing,
        ),
    )


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
    "clear_rejected_compress_state",
    "compress_verify",
    "dspark_attention_metadata",
    "fp4_act_quant",
    "finish_index_topk",
    "hadamard_transform",
    "hash_sqrt_softplus",
    "index_scores",
    "kv_norm_rope_store",
    "moe_act_quant",
    "paged_cache_rows",
    "paged_state_locations",
    "qk_norm_rope_store",
    "qk_norm_rope_block",
    "rope_last_block_inplace",
    "swiglu",
    "top6_sqrt_softplus",
    "verify_attention_metadata",
]
