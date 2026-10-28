"""BF16 arithmetic, mutable CUDA graphs, and reserved-page metadata contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from resin_models_cuda.gemma4.edge_kernels import (
    gelu_tanh_mul,
    linear_gelu,
    norm_residual,
    sparse_gather_dot,
    strided_qk_rmsnorm_rope,
    update_reserved_paged_metadata,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _capture(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def _assert_replay(graph, captured, expected):
    graph.replay()
    torch.cuda.synchronize()
    if isinstance(expected, tuple):
        for actual, wanted in zip(captured, expected, strict=True):
            if wanted is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    else:
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def _compilations(monkeypatch, kernel):
    from triton import knobs

    compiled = []
    previous = knobs.runtime.jit_post_compile_hook

    def post_compile(**event):
        if event["fn"].name == kernel:
            compiled.append(str(event["key"]))
        if previous is not None:
            return previous(**event)

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", post_compile)
    return compiled


def _discrete_storage(*shape):
    powers = torch.randint(-2, 3, shape, device="cuda").float().exp2()
    signs = torch.randint(0, 2, shape, device="cuda").float() * 2 - 1
    return (powers * signs).bfloat16()


def _norm(hidden, weight, eps):
    values = hidden.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return ((values * inverse) * weight.float()).bfloat16()


def _rope_reference(hidden, weight, cos, sin, eps):
    normalized = _norm(hidden, weight, eps)
    left, right = normalized.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    # Separate BF16 products and addition expose accidental FMA/round changes.
    return normalized * cos[:, None, :] + rotated * sin[:, None, :]


def _qk_inputs(tokens, width, with_key):
    storage = _discrete_storage(tokens, 11, width + 16)
    frequencies = torch.randn(tokens, width // 2, device="cuda")
    if width == 512:
        frequencies[:, width // 8 :] = 0  # Global attention rotates a partial head.
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return (
        storage[:, :8, 3 : 3 + width],
        storage[:, 8:9, 3 : 3 + width] if with_key else None,
        torch.randn(width, dtype=torch.bfloat16, device="cuda"),
        torch.randn(width, dtype=torch.bfloat16, device="cuda") if with_key else None,
        frequencies.cos().bfloat16(),
        frequencies.sin().bfloat16(),
    )


@pytest.mark.parametrize("width", [256, 512])
@pytest.mark.parametrize("with_key", [False, True])
@torch.inference_mode()
def test_strided_norm_rope_rounding_replay_and_request_values(
    width, with_key, monkeypatch
):
    torch.manual_seed(71)
    inputs = _qk_inputs(3, width, with_key)

    def operation():
        return strided_qk_rmsnorm_rope(*inputs, eps=1e-6)

    graph, captured = _capture(operation)
    compiled = _compilations(monkeypatch, "_strided_qk_rmsnorm_rope")
    for _ in range(3):
        for value in inputs:
            if value is not None:
                value.copy_(torch.randn_like(value))
        _assert_replay(graph, captured, operation())
    for tokens, eps in [(1, 3e-6), (65, 1e-5), (513, 1e-7)]:
        changed = _qk_inputs(tokens, width, with_key)
        query, key, qw, kw, cos, sin = changed
        actual_q, actual_k = strided_qk_rmsnorm_rope(*changed, eps=eps)
        torch.testing.assert_close(
            actual_q, _rope_reference(query, qw, cos, sin, eps), atol=0, rtol=0
        )
        if key is None:
            assert actual_k is None
        else:
            torch.testing.assert_close(
                actual_k, _rope_reference(key, kw, cos, sin, eps), atol=0, rtol=0
            )
    assert compiled == [], f"ordinary request values caused compilation: {compiled}"


def _residual_inputs(tokens, scaled):
    storage = _discrete_storage(tokens, 1600)
    return (
        storage[:, 3:1539],
        torch.randn(1536, dtype=torch.bfloat16, device="cuda"),
        torch.randn_like(storage)[:, 7:1543],
        torch.tensor([1.125], dtype=torch.bfloat16, device="cuda") if scaled else None,
    )


def _residual_reference(hidden, weight, residual, scale, eps):
    added = _norm(hidden, weight, eps) + residual
    return added if scale is None else added * scale


@pytest.mark.parametrize("scaled", [False, True])
@torch.inference_mode()
def test_post_norm_residual_bf16_rounding_replay_and_request_values(
    scaled, monkeypatch
):
    torch.manual_seed(83)
    hidden, weight, residual, scale = _residual_inputs(3, scaled)

    def operation():
        return norm_residual(hidden, weight, residual, scale=scale, eps=1e-6)

    graph, captured = _capture(operation)
    compiled = _compilations(monkeypatch, "_norm_residual")
    for iteration in range(3):
        h, w, r, _ = _residual_inputs(3, scaled)
        hidden.copy_(h)
        weight.copy_(w)
        residual.copy_(r)
        if scale is not None:
            scale.fill_(0.75 + iteration / 8)
        expected = _residual_reference(hidden, weight, residual, scale, 1e-6)
        _assert_replay(graph, captured, expected)
    for tokens, eps in [(1, 3e-6), (65, 1e-5), (513, 1e-7)]:
        h, w, r, s = _residual_inputs(tokens, scaled)
        torch.testing.assert_close(
            norm_residual(h, w, r, scale=s, eps=eps),
            _residual_reference(h, w, r, s, eps),
            atol=0,
            rtol=0,
        )
    assert compiled == [], f"ordinary request values caused compilation: {compiled}"


@pytest.mark.parametrize("width", [127, 256])
@torch.inference_mode()
def test_sparse_head_fp64_oracle_and_changed_input_graph(width):
    torch.manual_seed(29)
    hidden = torch.randn(3, width, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1024, width, dtype=hidden.dtype, device="cuda")
    indices = torch.randint(1024, (3, 74), device="cuda")[:, ::2]

    def operation():
        return sparse_gather_dot(hidden, weights, indices)

    graph, captured = _capture(operation)
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        weights.copy_(torch.randn_like(weights))
        indices.copy_(torch.randint(1024, indices.shape, device="cuda"))
        expected = (
            (hidden.double()[:, None, :] * weights[indices].double()).sum(-1).bfloat16()
        )
        actual = operation()
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        _assert_replay(graph, captured, actual)


@torch.inference_mode()
def test_cublaslt_gelu_accumulator_oracle_and_changed_input_graph():
    torch.manual_seed(41)
    # The active producer-prefill regime; small shapes use the model's baseline.
    hidden = torch.randn(8192, 1536, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(6144, 1536, dtype=hidden.dtype, device="cuda") * 0.02
    zero_bias = torch.zeros(6144, dtype=hidden.dtype, device="cuda")

    def operation():
        return linear_gelu(hidden, weight, zero_bias=zero_bias)

    graph, captured = _capture(operation)
    selected_rows = [0, 127, 4096, 8191]
    for _ in range(2):
        hidden.copy_(torch.randn_like(hidden))
        weight.mul_(-0.5)
        actual = operation()
        # Lt applies GELU before the final BF16 cast, with no BF16 GEMM round.
        projected = F.linear(hidden[selected_rows].double(), weight.double())
        expected = F.gelu(projected, approximate="tanh").bfloat16()
        torch.testing.assert_close(
            actual[selected_rows], expected, atol=0.015625, rtol=0.01
        )
        _assert_replay(graph, captured, actual)


@pytest.mark.parametrize("width", [256, 6144])
@torch.inference_mode()
def test_gelu_mul_retains_bf16_intermediate_round_and_replays(width):
    torch.manual_seed(43)
    gate = torch.randn(65, width, dtype=torch.bfloat16, device="cuda") * 3
    up = torch.randn_like(gate)

    def operation():
        return gelu_tanh_mul(gate, up)

    graph, captured = _capture(operation)
    for _ in range(3):
        gate.copy_(torch.randn_like(gate) * 3)
        up.copy_(torch.randn_like(up))
        actual = operation()
        expected = F.gelu(gate, approximate="tanh") * up
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        _assert_replay(graph, captured, actual)


@torch.inference_mode()
def test_reserved_csr_append_lookup_replays_positions_and_recycled_pages():
    from resin.backend.sampling import SAMPLING_OFFSET_STRIDE

    rows, width, window = 32, 4, 512
    initial = np.resize(
        [58, 63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705], rows
    ).astype(np.int64)
    initial[-1] = 1  # Padded scratch row owns only physical parent zero.
    inputs = {
        "positions": torch.as_tensor(initial, device="cuda"),
        "sampling_offsets": torch.empty(rows, dtype=torch.int64, device="cuda"),
    }
    tables, firsts = {}, {}
    for kind, page_width in (("local", 137), ("global", 160)):
        first = (
            np.maximum(0, (initial + 7) // 64 + 1 - page_width)
            if kind == "local"
            else np.zeros(rows, dtype=np.int64)
        )
        table = np.zeros((rows, page_width), dtype=np.int32)
        for row in range(rows - 1):
            count = (initial[row] + 7) // 64 - first[row] + 1
            table[row, :count] = (np.arange(count) * 17 + row * 271 + 31) % 10007 + 1
        tables[kind], firsts[kind] = table, first
        inputs[f"envelope_{kind}_page_table"] = torch.as_tensor(table, device="cuda")
        inputs[f"envelope_{kind}_first_pages"] = torch.as_tensor(first, device="cuda")
        for prefix in ("assistant_", "verify_"):
            inputs[f"{prefix}{kind}_kv_indptr"] = torch.empty(
                rows + 1, dtype=torch.int32, device="cuda"
            )
            inputs[f"{prefix}{kind}_page_indices"] = torch.zeros(
                rows * page_width, dtype=torch.int32, device="cuda"
            )
            inputs[f"{prefix}{kind}_last_page_lengths"] = torch.empty(
                rows, dtype=torch.int32, device="cuda"
            )
        for field in ("append_pages", "append_offsets"):
            inputs[f"verify_{kind}_{field}"] = torch.empty(
                rows * width, dtype=torch.int64, device="cuda"
            )

    def operation():
        return update_reserved_paged_metadata(
            inputs, query_width=width, sliding_window=window
        )

    graph, _ = _capture(operation)
    for step in (0, 1, 4, 2):
        positions = initial + (np.arange(rows) + step) % 5
        positions[-1] = 1
        inputs["positions"].copy_(torch.as_tensor(positions, device="cuda"))
        graph.replay()
        np.testing.assert_array_equal(
            inputs["sampling_offsets"].cpu().numpy(),
            (positions + 1) * SAMPLING_OFFSET_STRIDE,
        )
        for kind in ("local", "global"):
            for prefix, end in (
                ("assistant_", positions),
                ("verify_", positions + width),
            ):
                first = (
                    np.maximum(0, positions - window) // 64
                    if prefix == "assistant_" and kind == "local"
                    else firsts[kind]
                )
                counts = (end + 63) // 64 - first
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_kv_indptr"].cpu().numpy(),
                    np.concatenate(([0], np.cumsum(counts))),
                )
                expected = np.concatenate(
                    [
                        tables[kind][
                            row,
                            int(first[row] - firsts[kind][row]) : int(
                                first[row] - firsts[kind][row] + counts[row]
                            ),
                        ]
                        for row in range(rows)
                    ]
                )
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_page_indices"][: len(expected)]
                    .cpu()
                    .numpy(),
                    expected,
                )
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_last_page_lengths"].cpu().numpy(),
                    (end - 1) % 64 + 1,
                )
            absolute = positions[:, None] + np.arange(width)[None]
            expected = np.take_along_axis(
                tables[kind], absolute // 64 - firsts[kind][:, None], axis=1
            )
            np.testing.assert_array_equal(
                inputs[f"verify_{kind}_append_pages"]
                .cpu()
                .numpy()
                .reshape(rows, width),
                expected,
            )
            np.testing.assert_array_equal(
                inputs[f"verify_{kind}_append_offsets"]
                .cpu()
                .numpy()
                .reshape(rows, width),
                absolute % 64,
            )
    assert inputs["verify_local_last_page_lengths"][-1].item() == 5


@pytest.fixture
def _reference_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _attention_oracle(query, keys, values, lengths, indices, *, width, window):
    outputs, logsumexp = [], []
    for row, length in enumerate(lengths):
        key = keys[indices[row].long()].flatten(0, 1)[:length, 0].float()
        value = values[indices[row].long()].flatten(0, 1)[:length, 0].float()
        q = query[row * width : (row + 1) * width].float()
        scores = torch.einsum("qhd,kd->qhk", q, key)
        positions = torch.arange(length, device=query.device)
        query_positions = length - width + torch.arange(width, device=query.device)
        valid = positions[None, :] <= query_positions[:, None]
        if window >= 0:
            valid &= positions[None, :] >= query_positions[:, None] - window
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        outputs.append(torch.einsum("qhk,kd->qhd", scores.softmax(-1), value))
        # FA2 returns the base-2 accumulator, not natural-log logsumexp.
        logsumexp.append(scores.logsumexp(-1) / math.log(2))
    return torch.cat(outputs), torch.cat(logsumexp)


def _metadata(lengths, *, page_width=132, variant=0):
    counts = [(length + 63) // 64 for length in lengths]
    indptr = torch.zeros(len(counts) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    # Distinct row banks and recycled physical order expose stale page IDs.
    rows = [
        (row * page_width + (torch.arange(count) + 17 * variant) % page_width)
        for row, count in enumerate(counts)
    ]
    indices = torch.cat(rows).to(device="cuda", dtype=torch.int32)
    last = torch.tensor(
        [(length - 1) % 64 + 1 for length in lengths], dtype=torch.int32
    )
    return indptr, indices, last, rows


def _plan(wrapper, metadata, *, heads, dimension, width, window, uniform=False):
    indptr, indices, last, _ = metadata
    kwargs = (
        {"q_len_per_req": width, "is_causal": width > 1}
        if uniform
        else {"causal": True}
    )
    args = (indptr, indices, last)
    if not uniform:
        args = (torch.arange(0, len(indptr) * width, width, dtype=torch.int32), *args)
    wrapper.plan(
        *args,
        heads,
        1,
        dimension,
        64,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=1.0,
        window_left=window,
        disable_split_kv=True,
        **kwargs,
    )


@pytest.mark.parametrize("dimension,window", [(256, 511), (512, -1)])
@pytest.mark.parametrize("batch", [1, 8])
@torch.inference_mode()
def test_uniform_q4_verify_matches_causal_oracle_and_changed_input_graph(
    dimension, window, batch, _reference_precision
):
    from resin_models_cuda.gemma4.edge_kernels import UniformCausalPagedVerifyWrapper

    torch.manual_seed(119)
    capacity = batch * 132
    keys = (
        torch.randn(capacity, 64, 1, dimension, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    values = torch.randn_like(keys)
    query = (
        torch.randn(batch * 4, 8, dimension, device="cuda", dtype=torch.bfloat16) * 0.2
    )
    wrapper = UniformCausalPagedVerifyWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        query_width=4,
        split_kv=True,
        use_cuda_graph=True,
        backend="fa2",
        paged_kv_indptr_buf=torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buf=torch.empty(capacity, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len_buf=torch.empty(batch, dtype=torch.int32, device="cuda"),
    )
    _plan(
        wrapper,
        _metadata([65] * batch),
        heads=8,
        dimension=dimension,
        width=4,
        window=window,
    )
    graph, captured = _capture(lambda: wrapper.run(query, (keys, values)))
    for iteration, length in enumerate(
        [65, 66, 127, 128, 129, 511, 512, 513, 8192, 8193]
    ):
        lengths = [length + row % 3 for row in range(batch)]
        metadata = _metadata(lengths, variant=iteration % 2)
        _plan(wrapper, metadata, heads=8, dimension=dimension, width=4, window=window)
        query.add_(0.001)
        expected, expected_lse = _attention_oracle(
            query, keys, values, lengths, metadata[3], width=4, window=window
        )
        eager, lse = wrapper.run(query, (keys, values), return_lse=True)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.float(), expected, rtol=0.015, atol=0.003)
        torch.testing.assert_close(lse.float(), expected_lse, rtol=2e-4, atol=3e-4)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def _schedule(wrapper, batch, width, heads):
    info = tuple(wrapper._plan_info)
    assert len(info) == 15 and info[13] and not info[14], info
    padded, _, total_rows, tile, request, qo, kv, _, out, chunk, _, _, valid, _, _ = (
        info
    )
    real_ctas = batch * math.ceil(width * heads / tile)
    ranges = [
        (request, real_ctas * 4),
        (qo, real_ctas * 4),
        (kv, real_ctas * 4),
        (out, (batch + 1) * 4),
        (chunk, 4),
        (valid, padded),
        (total_rows, 4),
    ]
    # Ignore uninitialized padded request/tile entries; block_valid masks them.
    integer = wrapper._int_workspace_buffer
    return info, tuple(
        bytes(integer[start : start + count].cpu().tolist()) for start, count in ranges
    )


@pytest.mark.parametrize(
    "width,heads,dimension,window",
    [
        (1, 4, 256, 511),
        (1, 4, 512, -1),
        (4, 8, 256, 511),
        (4, 8, 512, -1),
    ],
)
@pytest.mark.parametrize("batch", [32, 256])
@torch.inference_mode()
def test_no_split_schedule_is_invariant_and_graph_reads_changed_csr(
    batch, width, heads, dimension, window, _reference_precision
):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    torch.manual_seed(781 + dimension + width)
    keys = (
        torch.randn(batch * 132, 64, 1, dimension, device="cuda", dtype=torch.bfloat16)
        * 0.25
    )
    values = torch.randn_like(keys) * 0.25
    query = (
        torch.randn(
            batch * width, heads, dimension, device="cuda", dtype=torch.bfloat16
        )
        * 0.25
    )
    indptr_buffer = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
    indices_buffer = torch.empty(batch * 132, dtype=torch.int32, device="cuda")
    last_buffer = torch.empty(batch, dtype=torch.int32, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buffer,
        paged_kv_indices_buffer=indices_buffer,
        paged_kv_last_page_len_buffer=last_buffer,
    )
    choices = (1, 8, 9, 127, 128, 129, 130)
    cases = []
    for variant in range(4):
        lengths = []
        for row in range(batch):
            count = choices[(row + variant) % len(choices)]
            last = (width, 63, 64, 1)[(row + variant) % 4]
            lengths.append(
                (count - 1) * 64 + max(last, width)
                if count == 1
                else (count - 1) * 64 + last
            )
        cases.append((lengths, _metadata(lengths, variant=variant)))
    schedule = None
    for _, metadata in cases:
        _plan(
            wrapper,
            metadata,
            heads=heads,
            dimension=dimension,
            width=width,
            window=window,
            uniform=True,
        )
        current = _schedule(wrapper, batch, width, heads)
        if schedule is None:
            schedule = current
        assert current == schedule, "fixed no-split schedule changed with runtime CSR"
    _plan(
        wrapper,
        cases[0][1],
        heads=heads,
        dimension=dimension,
        width=width,
        window=window,
        uniform=True,
    )
    graph, captured = _capture(lambda: wrapper.run(query, (keys, values)))
    frozen_plan = wrapper._plan_info
    for lengths, metadata in cases:
        indptr, indices, last, page_rows = metadata
        # Deliberately no plan() call after capture. Only mutable CSR and Q change.
        indptr_buffer.copy_(indptr)
        indices_buffer[: len(indices)].copy_(indices)
        last_buffer.copy_(last)
        query.copy_(torch.randn_like(query) * 0.25)
        graph.replay()
        expected, _ = _attention_oracle(
            query, keys, values, lengths, page_rows, width=width, window=window
        )
        torch.cuda.synchronize()
        assert wrapper._plan_info is frozen_plan
        torch.testing.assert_close(captured.float(), expected, rtol=0.02, atol=0.002)
