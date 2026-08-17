"""Persistent cuTile kernels for Inkling decode fusion."""

from __future__ import annotations

from functools import cache
from typing import Annotated, TypeAlias

import cuda.tile as ct
import torch
from torch import Tensor

_StaticLastDim: TypeAlias = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(-1,)),
]
_StaticLastTwoDims: TypeAlias = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(-2, -1)),
]
_ConstantInt: TypeAlias = ct.Constant[int]

# This is a kernel scheduling choice, not model geometry. Architecture and cache
# dimensions are supplied by the caller and become compile-time parameters.
_PERSISTENT_OCCUPANCY = 2
_VALUE_TILE = 128


@cache
def _device_program_count(device_type: str, device_index: int) -> int:
    if device_type != "cuda":
        raise ValueError("persistent Inkling kernels require a CUDA tensor")
    properties = torch.cuda.get_device_properties(device_index)
    return _PERSISTENT_OCCUPANCY * properties.multi_processor_count


def _program_count(tensor: Tensor, override: int | None) -> int:
    if override is not None:
        if isinstance(override, bool) or override <= 0:
            raise ValueError("persistent program count must be positive")
        return override
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return _device_program_count(tensor.device.type, index)


@ct.kernel(occupancy=2)
def _kv_sconv_norm_store_kernel(
    key_input: _StaticLastDim,
    value_input: _StaticLastDim,
    key_state: _StaticLastDim,
    value_state: _StaticLastDim,
    key_weight: _StaticLastTwoDims,
    value_weight: _StaticLastTwoDims,
    key_norm_weight: _StaticLastDim,
    next_key_state: _StaticLastDim,
    next_value_state: _StaticLastDim,
    cache_keys: _StaticLastTwoDims,
    cache_values: _StaticLastTwoDims,
    cache_slab_ids,
    cache_slotnums,
    cache_first_blocks,
    positions,
    head_dim: _ConstantInt,
    kv_heads: _ConstantInt,
    state_length: _ConstantInt,
    page_tokens: _ConstantInt,
    slab_units: _ConstantInt,
    layer_count: _ConstantInt,
    program_count: _ConstantInt,
    layer: int,
    epsilon: float,
):
    """Fuse both KV sconv chains, K norm, state advance, and paged stores."""

    work = ct.bid(0)
    tokens = key_input.shape[1]
    work_count = key_input.shape[0] * tokens * kv_heads
    while work < work_count:
        head = work % kv_heads
        row = work // kv_heads
        sequence = row % tokens
        batch = row // tokens

        key_residual = ct.load(
            key_input,
            (batch, sequence, head),
            shape=(1, 1, head_dim),
        ).reshape((head_dim,))
        value_residual = ct.load(
            value_input,
            (batch, sequence, head),
            shape=(1, 1, head_dim),
        ).reshape((head_dim,))
        key_accumulator = ct.zeros((head_dim,), dtype=ct.float32)
        value_accumulator = ct.zeros((head_dim,), dtype=ct.float32)
        for tap in ct.static_iter(range(state_length + 1)):
            source = sequence + tap
            if source < state_length:
                key_source = ct.load(
                    key_state,
                    (batch, source, head),
                    shape=(1, 1, head_dim),
                ).reshape((head_dim,))
                value_source = ct.load(
                    value_state,
                    (batch, source, head),
                    shape=(1, 1, head_dim),
                ).reshape((head_dim,))
            else:
                key_source = ct.load(
                    key_input,
                    (batch, source - state_length, head),
                    shape=(1, 1, head_dim),
                ).reshape((head_dim,))
                value_source = ct.load(
                    value_input,
                    (batch, source - state_length, head),
                    shape=(1, 1, head_dim),
                ).reshape((head_dim,))
            key_coefficient = ct.load(
                key_weight,
                (head, 0, tap),
                shape=(head_dim, 1, 1),
            ).reshape((head_dim,))
            value_coefficient = ct.load(
                value_weight,
                (head, 0, tap),
                shape=(head_dim, 1, 1),
            ).reshape((head_dim,))
            key_accumulator += key_source.astype(ct.float32) * key_coefficient
            value_accumulator += value_source.astype(ct.float32) * value_coefficient

        # Match the checkpoint path's FP32 convolution/residual followed by a
        # BF16 boundary before K RMSNorm.
        key = (key_accumulator + key_residual.astype(ct.float32)).astype(ct.bfloat16)
        value = (value_accumulator + value_residual.astype(ct.float32)).astype(
            ct.bfloat16
        )
        key_float = key.astype(ct.float32)
        norm_weight = ct.load(key_norm_weight, (0,), shape=(head_dim,)).astype(
            ct.float32
        )
        key = (
            key_float
            * ct.rsqrt(ct.sum(key_float * key_float) / head_dim + epsilon)
            * norm_weight
        ).astype(ct.bfloat16)

        position = ct.load(positions, (batch, sequence), shape=(1, 1)).item()
        first_block = ct.load(cache_first_blocks, (batch,), shape=(1,)).item()
        column = position // page_tokens - first_block
        slab_id = ct.gather(
            cache_slab_ids,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        slotnum = ct.gather(
            cache_slotnums,
            (
                ct.full((1,), batch, dtype=ct.int32),
                ct.full((1,), column, dtype=ct.int32),
            ),
            padding_value=0,
        ).item()
        page = slab_id * slab_units + slotnum * layer_count + layer
        offset = position % page_tokens
        ct.store(
            cache_keys,
            (page, offset, head, 0),
            key.reshape((1, 1, 1, head_dim)),
        )
        ct.store(
            cache_values,
            (page, offset, head, 0),
            value.reshape((1, 1, 1, head_dim)),
        )

        state_token = sequence - (tokens - state_length)
        if state_token >= 0:
            ct.store(
                next_key_state,
                (batch, state_token, head),
                key_residual.reshape((1, 1, head_dim)),
            )
            ct.store(
                next_value_state,
                (batch, state_token, head),
                value_residual.reshape((1, 1, head_dim)),
            )
        work += program_count


def kv_sconv_norm_store(
    key_input: Tensor,
    value_input: Tensor,
    key_state: Tensor,
    value_state: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_norm_weight: Tensor,
    cache_keys: Tensor,
    cache_values: Tensor,
    cache_slab_ids: Tensor,
    cache_slotnums: Tensor,
    cache_first_blocks: Tensor,
    positions: Tensor,
    *,
    page_tokens: int,
    slab_units: int,
    layer_count: int,
    layer: int,
    epsilon: float,
    program_count: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Launch the graph-stable persistent KV/sconv fusion."""

    batch, tokens, width = key_input.shape
    state_length = key_weight.shape[-1] - 1
    head_dim = key_norm_weight.numel()
    if head_dim <= 0 or width % head_dim:
        raise ValueError("persistent Inkling KV head geometry is invalid")
    kv_heads = width // head_dim
    if (
        value_input.shape != key_input.shape
        or key_state.shape != (batch, state_length, width)
        or value_state.shape != key_state.shape
        or key_weight.shape != (width, 1, state_length + 1)
        or value_weight.shape != key_weight.shape
        or key_norm_weight.shape != (head_dim,)
        or positions.shape != (batch, tokens)
        or tokens < state_length
        or head_dim & (head_dim - 1)
    ):
        raise ValueError("persistent Inkling KV/sconv inputs have invalid geometry")
    if (
        cache_keys.shape != cache_values.shape
        or cache_keys.shape[-2:] != (kv_heads, head_dim)
        or cache_slotnums.shape != cache_slab_ids.shape
        or cache_first_blocks.shape != (batch,)
        or cache_slab_ids.shape[0] != batch
        or page_tokens <= 0
        or slab_units <= 0
        or not 0 <= layer < layer_count
    ):
        raise ValueError("persistent Inkling KV cache inputs have invalid geometry")

    next_key_state = torch.empty(
        (batch, state_length, width),
        dtype=key_state.dtype,
        device=key_state.device,
    )
    next_value_state = torch.empty_like(next_key_state)
    programs = _program_count(key_input, program_count)
    ct.launch(
        torch.cuda.current_stream(key_input.device),
        (programs,),
        _kv_sconv_norm_store_kernel,
        (
            key_input,
            value_input,
            key_state,
            value_state,
            key_weight,
            value_weight,
            key_norm_weight,
            next_key_state,
            next_value_state,
            cache_keys,
            cache_values,
            cache_slab_ids,
            cache_slotnums,
            cache_first_blocks,
            positions,
            head_dim,
            kv_heads,
            state_length,
            page_tokens,
            slab_units,
            layer_count,
            programs,
            layer,
            epsilon,
        ),
    )
    return next_key_state, next_value_state


@ct.kernel(occupancy=2)
def _sconv_residual_kernel(
    inputs: _StaticLastDim,
    state: _StaticLastDim,
    weight: _StaticLastTwoDims,
    residual: _StaticLastDim,
    output: _StaticLastDim,
    next_state: _StaticLastDim,
    state_length: _ConstantInt,
    program_count: _ConstantInt,
):
    """Fuse a width-four sconv, BF16 boundary, residual, and state advance."""

    work = ct.bid(0)
    tokens = inputs.shape[1]
    tiles = inputs.shape[2] // _VALUE_TILE
    work_count = inputs.shape[0] * tokens * tiles
    while work < work_count:
        tile = work % tiles
        row = work // tiles
        sequence = row % tokens
        batch = row // tokens

        current = ct.load(
            inputs,
            (batch, sequence, tile),
            shape=(1, 1, _VALUE_TILE),
        ).reshape((_VALUE_TILE,))
        accumulator = ct.zeros((_VALUE_TILE,), dtype=ct.float32)
        for tap in ct.static_iter(range(state_length + 1)):
            source = sequence + tap
            if source < state_length:
                source_values = ct.load(
                    state,
                    (batch, source, tile),
                    shape=(1, 1, _VALUE_TILE),
                ).reshape((_VALUE_TILE,))
            else:
                source_values = ct.load(
                    inputs,
                    (batch, source - state_length, tile),
                    shape=(1, 1, _VALUE_TILE),
                ).reshape((_VALUE_TILE,))
            coefficient = ct.load(
                weight,
                (tile, 0, tap),
                shape=(_VALUE_TILE, 1, 1),
            ).reshape((_VALUE_TILE,))
            accumulator += source_values.astype(ct.float32) * coefficient

        convolved = (accumulator + current.astype(ct.float32)).astype(ct.bfloat16)
        residual_values = ct.load(
            residual,
            (batch, sequence, tile),
            shape=(1, 1, _VALUE_TILE),
        ).reshape((_VALUE_TILE,))
        ct.store(
            output,
            (batch, sequence, tile),
            (residual_values + convolved).reshape((1, 1, _VALUE_TILE)),
        )

        state_token = sequence - (tokens - state_length)
        if state_token >= 0:
            ct.store(
                next_state,
                (batch, state_token, tile),
                current.reshape((1, 1, _VALUE_TILE)),
            )
        work += program_count


def sconv_residual(
    inputs: Tensor,
    state: Tensor,
    weight: Tensor,
    residual: Tensor,
    *,
    program_count: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Launch the graph-stable persistent sconv/residual fusion."""

    if inputs.ndim != 3:
        raise ValueError("persistent Inkling sconv input must be rank three")
    batch, tokens, width = inputs.shape
    state_length = weight.shape[-1] - 1
    if (
        residual.shape != inputs.shape
        or state.shape != (batch, state_length, width)
        or weight.shape != (width, 1, state_length + 1)
        or width % _VALUE_TILE
        or tokens < state_length
    ):
        raise ValueError("persistent Inkling sconv inputs have invalid geometry")
    output = torch.empty_like(inputs)
    next_state = torch.empty_like(state)
    programs = _program_count(inputs, program_count)
    ct.launch(
        torch.cuda.current_stream(inputs.device),
        (programs,),
        _sconv_residual_kernel,
        (
            inputs,
            state,
            weight,
            residual,
            output,
            next_state,
            state_length,
            programs,
        ),
    )
    return output, next_state


@ct.kernel(occupancy=2)
def _router_choice_kernel(
    logits: _StaticLastDim,
    bias: _StaticLastDim,
    choice: _StaticLastDim,
    routed_experts: _ConstantInt,
    program_count: _ConstantInt,
):
    """Apply the stable sigmoid and routing bias before PyTorch top-k."""

    row = ct.bid(0)
    while row < logits.shape[0]:
        routed = ct.load(logits, (row, 0), shape=(1, routed_experts)).astype(ct.float32)
        routing_bias = ct.load(bias, (0,), shape=(routed_experts,)).reshape(
            (1, routed_experts)
        )
        exponential = ct.exp(-ct.abs(routed))
        sigmoid = ct.where(
            routed >= 0.0,
            1.0 / (1.0 + exponential),
            exponential / (1.0 + exponential),
        )
        ct.store(choice, (row, 0), sigmoid + routing_bias)
        row += program_count


@ct.kernel(occupancy=2)
def _router_weights_kernel(
    logits: _StaticLastDim,
    expert_indices: _StaticLastDim,
    global_scale: _StaticLastDim,
    routed_weights: _StaticLastDim,
    shared_weights: _StaticLastDim,
    route_scale: float,
    routed_experts: _ConstantInt,
    shared_experts: _ConstantInt,
    top_k: _ConstantInt,
    selected_width: _ConstantInt,
    program_count: _ConstantInt,
):
    """Gather selected logits and normalize routed and shared expert weights."""

    row = ct.bid(0)
    while row < logits.shape[0]:
        positions = ct.arange(selected_width, dtype=ct.int32).reshape(
            (1, selected_width)
        )
        indices = ct.load(
            expert_indices,
            (row, 0),
            shape=(1, selected_width),
            padding_mode=ct.PaddingMode.ZERO,
        ).astype(ct.int32)
        indices = ct.where(
            positions < top_k,
            indices,
            routed_experts + positions - top_k,
        )
        valid = positions < top_k + shared_experts
        indices = ct.where(valid, indices, 0)
        rows = ct.full((1, selected_width), row, dtype=ct.int32)
        selected = ct.gather(logits, (rows, indices), padding_value=0).astype(
            ct.float32
        )
        selected = ct.where(valid, selected, -3.402823466e38)
        log_probabilities = ct.minimum(selected, 0.0) - ct.log(
            1.0 + ct.exp(-ct.abs(selected))
        )
        probabilities = ct.exp(
            log_probabilities - ct.max(log_probabilities, 1, keepdims=True)
        )
        probabilities /= ct.sum(probabilities, 1, keepdims=True)
        model_scale = ct.load(global_scale, (0,), shape=(1,)).item()
        probabilities = (probabilities * (route_scale * model_scale)).astype(
            ct.bfloat16
        )
        # The routed array may be narrower than the power-of-two tile. cuTile
        # ignores the partial store's out-of-bounds tail.
        ct.store(routed_weights, (row, 0), probabilities)
        shared = ct.extract(
            probabilities,
            (0, top_k // shared_experts),
            shape=(1, shared_experts),
        )
        ct.store(shared_weights, (row, 0), shared)
        row += program_count


def router_choice(
    logits: Tensor,
    bias: Tensor,
    *,
    program_count: int | None = None,
) -> Tensor:
    """Apply Inkling's graph-stable routed-expert selection transform."""

    routed_experts = bias.numel()
    shared_experts = logits.shape[1] - routed_experts
    if (
        logits.ndim != 2
        or bias.shape != (routed_experts,)
        or routed_experts <= 0
        or routed_experts & (routed_experts - 1)
        or shared_experts <= 0
    ):
        raise ValueError("persistent Inkling router inputs have invalid geometry")
    choice = torch.empty(
        (logits.shape[0], routed_experts),
        dtype=torch.float32,
        device=logits.device,
    )
    programs = _program_count(logits, program_count)
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (programs,),
        _router_choice_kernel,
        (logits, bias, choice, routed_experts, programs),
    )
    return choice


def router_weights(
    logits: Tensor,
    expert_indices: Tensor,
    global_scale: Tensor,
    route_scale: float,
    *,
    routed_experts: int,
    shared_experts: int,
    program_count: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalize the chosen routed and always-on shared expert weights."""

    rows = logits.shape[0]
    top_k = expert_indices.shape[1]
    selected_width = 1 << (top_k + shared_experts - 1).bit_length()
    if (
        logits.ndim != 2
        or logits.shape[1] != routed_experts + shared_experts
        or expert_indices.shape != (rows, top_k)
        or global_scale.shape != (1,)
        or routed_experts <= 0
        or shared_experts <= 0
        or shared_experts & (shared_experts - 1)
        or not 0 < top_k <= routed_experts
        or top_k % shared_experts
    ):
        raise ValueError("persistent Inkling router outputs have invalid geometry")
    routed = torch.empty((rows, top_k), dtype=torch.bfloat16, device=logits.device)
    shared = torch.empty(
        (rows, shared_experts), dtype=torch.bfloat16, device=logits.device
    )
    programs = _program_count(logits, program_count)
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (programs,),
        _router_weights_kernel,
        (
            logits,
            expert_indices,
            global_scale,
            routed,
            shared,
            route_scale,
            routed_experts,
            shared_experts,
            top_k,
            selected_width,
            programs,
        ),
    )
    return routed, shared


__all__ = [
    "kv_sconv_norm_store",
    "router_choice",
    "router_weights",
    "sconv_residual",
]
