from __future__ import annotations

import pytest
import torch

from resin_models_cuda.deepseek_v4.cache import DeviceMapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.fixture
def kernels():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("DeepSeek-V4 cuTile kernels require Blackwell")
    from resin_models_cuda.deepseek_v4 import cutile

    torch.manual_seed(17)
    return cutile


def _mapping(*, first_blocks=(0, 0), block_tokens=128, divisor=1, row_scale=1):
    return DeviceMapping(
        parent_ids=torch.tensor([[3, 1], [2, 4]], device="cuda", dtype=torch.int32),
        child_slots=torch.tensor([[1, 0], [0, 1]], device="cuda", dtype=torch.int32),
        first_blocks=torch.tensor(first_blocks, device="cuda", dtype=torch.int32),
        block_tokens=block_tokens,
        divisor=divisor,
        parent_rows=512,
        packing=2,
        row_scale=row_scale,
        layer_offset=64,
    )


def _rows(mapping, positions):
    columns = (positions // mapping.block_tokens - mapping.first_blocks[:, None]).long()
    parents = mapping.parent_ids.gather(1, columns)
    slots = mapping.child_slots.gather(1, columns)
    return (
        (
            parents * mapping.parent_rows
            + slots * (mapping.parent_rows // mapping.packing)
        )
        * mapping.row_scale
        + mapping.layer_offset
        + positions % mapping.block_tokens // mapping.divisor
    ).long()


def _norm_rope(values, weight, frequencies, positions):
    values = values.float()
    values *= torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    values = (values * weight).bfloat16().float()
    frequency = frequencies[positions.long()]
    if values.ndim == 4:
        frequency = frequency.unsqueeze(2)
    rotated = torch.view_as_complex(
        values[..., -64:].reshape(*values.shape[:-1], 32, 2)
    )
    values[..., -64:] = torch.view_as_real(
        rotated * torch.view_as_complex(frequency)
    ).flatten(-2)
    return values


@pytest.mark.parametrize("mode", ["decode", "verify", "kv"])
def test_fused_norm_rope_writes_only_the_addressed_packed_rows(kernels, mode):
    mapping = _mapping(first_blocks=(1, 2))
    positions = torch.tensor(
        [[254, 255, 256], [382, 383, 384]], device="cuda", dtype=torch.int32
    )
    if mode == "decode":
        positions = positions[:, :1].contiguous()
    query = torch.randn(
        2, positions.shape[1], 8, 512, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(2, positions.shape[1], 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(512, 32, device="cuda")
    frequencies = torch.stack((angles.cos(), angles.sin()), dim=-1)
    expected_query = _norm_rope(query, 1, frequencies, positions).bfloat16()
    expected_key = _norm_rope(key, weight.float(), frequencies, positions)
    groups = expected_key[..., :-64].reshape(2, positions.shape[1], 7, 64)
    scale = (
        (groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
    )
    expected_key[..., :-64] = (
        (groups / scale).to(torch.float8_e4m3fn).float() * scale
    ).flatten(-2)
    cache = torch.full((2560, 512), -7, device="cuda", dtype=torch.bfloat16)
    expected_cache = cache.clone()
    expected_cache[_rows(mapping, positions)] = expected_key.bfloat16()
    arguments = (weight, frequencies, cache, positions, 1e-6)
    if mode == "kv":
        kernels.kv_norm_rope_store(key, *arguments, cache_mapping=mapping)
    elif mode == "decode":
        kernels.qk_norm_rope_store(
            query,
            key,
            weight,
            frequencies,
            cache,
            positions[:, 0],
            1e-6,
            cache_mapping=mapping,
        )
    else:
        kernels.qk_norm_rope_block(query, key, *arguments, cache_mapping=mapping)
    if mode != "kv":
        torch.testing.assert_close(query, expected_query, atol=0.015625, rtol=0.015625)
    torch.testing.assert_close(cache, expected_cache, atol=0.015625, rtol=0.015625)


def test_index_scores_and_topk_preserve_causality_and_packed_addresses(kernels):
    mapping = _mapping(divisor=4, row_scale=4)
    positions = torch.tensor(
        [[0, 131, 255], [3, 143, 255]], device="cuda", dtype=torch.int32
    )
    query = torch.randn(2, 3, 16, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(10240, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(2, 3, 16, device="cuda", dtype=torch.bfloat16)
    logical = torch.arange(64, device="cuda").expand(2, -1) * 4
    keys = cache[_rows(mapping, logical)]
    expected = torch.einsum("bshd,bkd->bshk", query.float(), keys.float())
    expected = (expected.relu() * weights.float()[..., None] * 0.125).sum(2)
    complete = (positions + 1) // 4
    causal = torch.arange(64, device="cuda") < complete[..., None]
    expected.masked_fill_(~causal, float("-inf"))
    actual = kernels.index_scores(
        query,
        cache,
        weights,
        positions,
        4,
        0.125,
        cache_mapping=mapping,
        cache_width=128,
    )
    torch.testing.assert_close(
        actual[..., :64], expected.bfloat16(), atol=0.0625, rtol=0.01
    )
    assert actual[..., 64:].isneginf().all()
    indices = actual.topk(64, dim=-1).indices.int()
    attention_mapping = _mapping(divisor=4)
    physical = kernels.finish_index_topk(
        indices, positions, 4, 0, cache_mapping=attention_mapping
    )
    for batch in range(2):
        for token in range(3):
            valid = indices[batch, token] < complete[batch, token]
            logical_rows = _rows(attention_mapping, logical)[batch]
            assert (physical[batch, token, ~valid] == -1).all()
            torch.testing.assert_close(
                physical[batch, token, valid].long(),
                logical_rows[indices[batch, token, valid].long()],
            )


@pytest.mark.parametrize("offset", [0, 33])
def test_index_topk_keeps_negative_padding_invalid_before_offset(kernels, offset):
    mapping = _mapping(divisor=4)
    positions = torch.tensor([[0, 7], [3, 11]], device="cuda", dtype=torch.int32)
    indices = torch.full((2, 2, 64), -1, device="cuda", dtype=torch.int32)
    indices[..., :4] = torch.tensor([-1, 0, 1, 64], device="cuda")
    complete = (positions + 1) // 4
    valid = (indices >= 0) & (indices < complete[..., None])
    # Positive offsets make an unchecked -1 look like a valid logical row.
    logical_rows = torch.arange(2, device="cuda").expand(2, -1) + offset
    expected_rows = _rows(mapping, logical_rows * 4)
    expected = torch.full_like(indices, -1)
    for batch in range(2):
        for query in range(2):
            selected = valid[batch, query]
            expected[batch, query, selected] = expected_rows[
                batch, indices[batch, query, selected].long()
            ].int()

    actual = kernels.finish_index_topk(
        indices, positions, 4, offset, cache_mapping=mapping
    )

    torch.testing.assert_close(actual, expected)
    assert (actual[~valid] == -1).all()


def _sparse_mla_reference(query, window_cache, compressed_cache, rows, sinks, scale):
    """Independent FP32 softmax, with a sink logit carrying zero value."""

    outputs = []
    for batch, (window, compressed) in enumerate(rows):
        values = torch.cat(
            (
                window_cache.view(-1, 512)[window],
                compressed_cache.view(-1, 512)[compressed],
            )
        ).float()
        scores = (query[batch, 0].float()[:, None] * values[None]).sum(-1) * scale
        scores = torch.cat((scores, sinks[:, None]), dim=-1)
        probabilities = scores.softmax(dim=-1)[:, :-1]
        outputs.append((probabilities[..., None] * values[None]).sum(1))
    return torch.stack(outputs).unsqueeze(1)


@pytest.mark.parametrize("lengths", [(17,), (1, 17, 511)])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_native_sparse_mla_matches_reference_across_padding_and_graphs(
    kernels, lengths, ratio
):
    from resin_models_cuda.deepseek_v4.attention import sparse_decode_attention

    batch, heads, dimension = len(lengths), 32, 512
    parents = torch.arange(1, 1 + batch * 4, device="cuda", dtype=torch.int32).view(
        batch, 4
    )

    def mapping(divisor):
        return DeviceMapping(
            parent_ids=parents,
            child_slots=torch.zeros_like(parents),
            first_blocks=torch.zeros(batch, device="cuda", dtype=torch.int32),
            block_tokens=128,
            divisor=divisor,
            parent_rows=128,
            packing=1,
            row_scale=1,
            layer_offset=0,
        )

    # Distinct pools make a misplaced compressed row observable: sharing a
    # latent allocation can hide an incorrect SWA/compressed segment boundary.
    shape = (1 + batch * 4, 128, dimension)
    window_cache = (torch.randn(shape, device="cuda") * 0.25 - 0.5).bfloat16()
    compressed_cache = (torch.randn(shape, device="cuda") * 0.25 + 0.5).bfloat16()
    original_query = torch.randn(
        batch, 1, heads, dimension, device="cuda", dtype=torch.bfloat16
    )
    sinks = torch.linspace(-1, 1, heads, device="cuda", dtype=torch.float32)
    positions = torch.tensor(lengths, device="cuda", dtype=torch.int32) - 1
    workspace = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    rows = []
    for row, length in enumerate(lengths):
        window = torch.arange(max(0, length - 128), length, device="cuda")
        window_rows = parents[row, window // 128].long() * 128 + window % 128
        logical = torch.arange(length // ratio if ratio else 0, device="cuda")
        times = logical * (ratio or 1)
        compressed_rows = parents[row, times // 128].long() * 128 + times % 128 // (
            ratio or 1
        )
        rows.append((window_rows, compressed_rows))
    expected = _sparse_mla_reference(
        original_query, window_cache, compressed_cache, rows, sinks, dimension**-0.5
    )
    capacities = (64, 128, 512) if ratio == 4 else (4, 64, 512) if ratio else (0,)
    if max(lengths) > 256 and ratio == 4:
        capacities = (128, 192, 512)
    first_output = None
    for capacity in capacities:
        compressed = None
        if ratio == 4:
            compressed = torch.full(
                (batch, 1, capacity), -1, device="cuda", dtype=torch.int32
            )
            for row, (_, selected) in enumerate(rows):
                compressed[row, 0, : selected.numel()] = selected.int()
        indices, sparse_lengths, sequence_lengths = kernels.verify_attention_metadata(
            positions,
            1,
            128,
            128,
            compressed,
            0,
            ratio,
            capacity if ratio == 128 else 0,
            window_mapping=mapping(1),
            compressed_mapping=mapping(ratio) if ratio else None,
        )
        for row, (window, selected) in enumerate(rows):
            assert sparse_lengths[row, 0].item() == 128 + selected.numel()
            torch.testing.assert_close(indices[row, 0, : window.numel()].long(), window)
            assert (indices[row, 0, window.numel() : 128] == -1).all()
            torch.testing.assert_close(
                indices[row, 0, 128 : 128 + selected.numel()].long(), selected
            )
        query = original_query.clone()

        def run():
            return sparse_decode_attention(
                query,
                window_cache,
                compressed_cache,
                sinks,
                indices,
                sparse_lengths.flatten(),
                sequence_lengths.flatten(),
                workspace,
                dimension**-0.5,
            )

        eager = run()
        torch.testing.assert_close(
            eager.float(), expected, atol=0.015625, rtol=0.015625
        )
        if first_output is None:
            first_output = eager.clone()
        else:
            torch.testing.assert_close(
                eager, first_output, atol=0.015625, rtol=0.015625
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            query.add_(0.25)
            graph.replay()
            torch.cuda.synchronize()
            replay_reference = _sparse_mla_reference(
                query, window_cache, compressed_cache, rows, sinks, dimension**-0.5
            )
            torch.testing.assert_close(
                captured.float(), replay_reference, atol=0.015625, rtol=0.015625
            )
        finally:
            graph.reset()


def test_rejected_compressor_cleanup_preserves_accepted_and_unrelated_rows(kernels):
    mapping = _mapping(first_blocks=(15, 31), block_tokens=8)
    starts = torch.tensor([126, 254], device="cuda", dtype=torch.int32)
    processed = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    values = torch.randn(10, 8, 512, device="cuda")
    scores = torch.randn_like(values)
    expected_values, expected_scores = values.clone(), scores.clone()
    for batch in range(2):
        for offset in range(processed[batch].item(), 6):
            position = starts[batch].item() + offset
            column = position // 8 - mapping.first_blocks[batch].item()
            page = (
                mapping.parent_ids[batch, column] * 2
                + mapping.child_slots[batch, column]
            ).item()
            expected_values[page, position % 8] = 0
            expected_scores[page, position % 8] = float("-inf")
    kernels.clear_rejected_compress_state(
        values, scores, starts, processed, 6, state_mapping=mapping, state_ring_size=256
    )
    torch.testing.assert_close(values, expected_values, atol=0, rtol=0)
    torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)


@pytest.mark.parametrize("draft_tokens", [1, 2, 5])
@pytest.mark.parametrize("captured", [False, True])
def test_overlap_compression_ignores_poisoned_future_ring_slots(
    kernels, draft_tokens, captured
):
    """Compare pooled FP32 rows with a sequential oracle after rejected tails."""

    batch_size, dimension, ratio = 2, 64, 4
    block_length = draft_tokens + 1
    ring_size = 1 << (8 + draft_tokens - 1).bit_length()
    parents = torch.randperm(24, device="cuda").view(2, 12).int() + 1
    slots = torch.arange(12, device="cuda", dtype=torch.int32).expand(2, -1) % 2
    state_mapping = DeviceMapping(
        parent_ids=parents,
        child_slots=slots,
        first_blocks=torch.zeros(2, device="cuda", dtype=torch.int32),
        block_tokens=4,
        divisor=1,
        parent_rows=128,
        packing=2,
        row_scale=1,
        layer_offset=0,
    )
    cache_mapping = DeviceMapping(
        parent_ids=torch.tensor([[1], [2]], device="cuda", dtype=torch.int32),
        child_slots=torch.zeros(2, 1, device="cuda", dtype=torch.int32),
        first_blocks=torch.zeros(2, device="cuda", dtype=torch.int32),
        block_tokens=128,
        divisor=ratio,
        parent_rows=128,
        packing=1,
        row_scale=1,
        layer_offset=0,
    )
    value_state = torch.empty(50, 4, 2 * dimension, device="cuda")
    score_state = torch.empty_like(value_state)
    values = torch.empty(batch_size, block_length, 2 * dimension, device="cuda")
    gates = torch.empty_like(values)
    positions = torch.empty(batch_size, block_length, device="cuda", dtype=torch.int32)
    ape = torch.randn(ratio, 2 * dimension, device="cuda") * 0.05
    weight = torch.ones(dimension, device="cuda", dtype=torch.bfloat16)
    frequencies = torch.zeros(128, 32, 2, device="cuda")
    frequencies[..., 0] = 1
    cache = torch.full((384, dimension), -7, device="cuda", dtype=torch.bfloat16)
    history_values = torch.randn(batch_size, 64, 2 * dimension, device="cuda")
    history_gates = torch.randn_like(history_values)
    starts = (19, 23)

    def state_locations(batch, times):
        columns = times // 4
        return parents[batch, columns].long() * 2 + slots[
            batch, columns
        ].long(), times % 4

    def restore_committed_history(step):
        # A prefix can be restored into arbitrary reused memory. Neither NaN
        # values nor extreme unused scores may enter a completed live window.
        value_state.fill_(float("nan"))
        value_state[..., ::3] = 1e20
        score_state.fill_(float("nan"))
        score_state[..., ::2] = -1e20
        for batch, start in enumerate(starts):
            times = torch.arange(start + step, device="cuda")
            pages, offsets = state_locations(batch, times)
            value_state[pages, offsets] = history_values[batch, times]
            score_state[pages, offsets] = (
                history_gates[batch, times] + ape[times % ratio]
            )

    def launch():
        kernels.compress_verify(
            values,
            gates,
            ape,
            value_state,
            score_state,
            weight,
            frequencies,
            cache,
            positions,
            ratio,
            False,
            1e-6,
            cache_mapping=cache_mapping,
            state_mapping=state_mapping,
            state_ring_size=ring_size,
        )

    graph = None
    try:
        for step in range(5):
            raw = torch.randn_like(values)
            raw_gates = torch.randn_like(gates)
            times = (
                torch.tensor(starts, device="cuda")[:, None]
                + step
                + torch.arange(block_length, device="cuda")
            )
            expected = raw.clone()
            for batch in range(batch_size):
                reference_values = history_values[batch].clone()
                reference_gates = history_gates[batch].clone()
                for index in range(block_length):
                    token = int(times[batch, index])
                    reference_values[token] = raw[batch, index]
                    reference_gates[token] = raw_gates[batch, index]
                    if (token + 1) % ratio:
                        continue
                    old = torch.arange(token - 7, token - 3, device="cuda")
                    new = old + ratio
                    pooled_values = torch.cat(
                        (
                            reference_values[old, :dimension],
                            reference_values[new, dimension:],
                        )
                    )
                    pooled_scores = torch.cat(
                        (
                            reference_gates[old, :dimension]
                            + ape[old % ratio, :dimension],
                            reference_gates[new, dimension:]
                            + ape[new % ratio, dimension:],
                        )
                    )
                    expected[batch, index, :dimension] = (
                        pooled_values * pooled_scores.softmax(0)
                    ).sum(0)

            # Keep stale rejected tails for some rounds and reconstruct the
            # committed prefix into poisoned storage for others.
            if step in (0, 2, 4):
                restore_committed_history(step)
            values.copy_(raw)
            gates.copy_(raw_gates)
            positions.copy_(times)
            if captured and graph is None:
                launch()  # Compile before capture, then restore its state writes.
                restore_committed_history(step)
                values.copy_(raw)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
            if graph is None:
                launch()
            else:
                graph.replay()

            assert values.isfinite().all()
            torch.testing.assert_close(values, expected, atol=2e-6, rtol=2e-5)
            # Accept only the anchor; all other verification inputs are
            # rejected and replaced by new projections on the next replay.
            for batch, start in enumerate(starts):
                history_values[batch, start + step] = raw[batch, 0]
                history_gates[batch, start + step] = raw_gates[batch, 0]
    finally:
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize(
    "rows,dtype",
    [(rows, torch.bfloat16) for rows in (1, 8, 32, 129)]
    + [(rows, torch.float32) for rows in (1, 129)],
)
@pytest.mark.parametrize("layout", ["row", "packed", "moe"])
def test_fp8_quantization_payload_and_scale_layout(kernels, rows, dtype, layout):
    values = torch.randn(rows, 32, 128, device="cuda")
    values *= torch.logspace(-7, 4, 32, device="cuda")[None, :, None]
    values[:, 0] = 0
    values[:, 1] = 448 * 2**-10
    shape = (3, 43, 4096) if rows == 129 else (rows, 4096)
    values = values.to(dtype).reshape(shape)
    groups = values.float().reshape(rows, 32, 128)
    maximum = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    exponent = (maximum / 448).log2().ceil()
    expected = (groups / exponent.exp2()).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected_scales = (exponent.squeeze(-1) + 127).byte()
    kwargs = dict(scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    if layout == "moe":
        output, scales = kernels.moe_act_quant(values, **kwargs)
        expected_scales = expected_scales.repeat_interleave(4, dim=-1)
    else:
        output, scales = kernels.act_quant(
            values, packed_scales=layout == "packed", **kwargs
        )
    if layout == "packed":
        assert scales.dtype == torch.int32
        assert scales.shape == (rows, 8)
        assert scales.stride() == (1, (rows + 3) // 4 * 4)
    assert output.shape == values.shape
    torch.testing.assert_close(
        output.reshape(rows, 4096).view(torch.uint8),
        expected.reshape(rows, 4096).view(torch.uint8),
        atol=0,
        rtol=0,
    )
    if layout == "row":
        assert scales.dtype == torch.float32 and scales.shape == (*shape[:-1], 32)
        scales = scales.reshape(rows, 32)
        assert scales.stride(1) == (rows + 3) // 4 * 4
        assert rows == 1 or scales.stride(0) == 1
        torch.testing.assert_close(scales, exponent.squeeze(-1).exp2(), atol=0, rtol=0)
    else:
        torch.testing.assert_close(
            scales.contiguous().view(torch.uint8), expected_scales, atol=0, rtol=0
        )


@pytest.mark.parametrize("router", ["hash", "top6"])
@pytest.mark.parametrize("level", [-80, -20, -12, 0, 20])
def test_routing_normalizes_negative_logits_without_nan(kernels, router, level):
    logits = torch.linspace(level - 2, level + 2, 256, device="cuda").repeat(2, 1)
    logits[1].fill_(level)
    scores = torch.nn.functional.softplus(logits).sqrt()
    if router == "hash":
        table = torch.tensor(
            [[0, 1, 17, 53, 129, 255]], device="cuda", dtype=torch.int32
        )
        expected_ids = table.expand(2, -1)
        weights, indices = kernels.hash_sqrt_softplus(
            logits, torch.zeros(2, device="cuda", dtype=torch.int64), table, 2.5
        )
    else:
        bias = torch.arange(256, device="cuda", dtype=torch.float32) / 256
        expected_ids = (scores + bias).topk(6).indices.int()
        weights, indices = kernels.top6_sqrt_softplus(logits, bias, 2.5)
    selected = scores.gather(1, expected_ids.long())
    expected = (selected * (2.5 / selected.sum(-1, keepdim=True))).bfloat16()
    assert weights.isfinite().all()
    torch.testing.assert_close(indices, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0.008)
