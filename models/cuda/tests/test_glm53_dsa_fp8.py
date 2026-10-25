"""Independent raw-E4M3 oracle and graph contracts for NoPE512 TRT attention."""

import gc

import pytest
import torch

from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.dsa import (
    DsaWorkspace,
    fp8_sparse_attention,
    sparse_metadata,
)

_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
GPU = pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")


def _fixture(rows=5, topk=2056, *, pages=40, dtype=torch.float8_e4m3fn):
    generator = torch.Generator().manual_seed(23012)
    # The pooled-index tail is physically interleaved:72 latent-width rows/page.
    # Poison every gap and the null page, so accidental dense reads are visible.
    storage = torch.full((pages, 72, 512), float("nan"), dtype=torch.float32).to(dtype)
    cache = storage.as_strided((pages, 64, 512), (72 * 512, 512, 1))
    values = torch.randn(pages - 1, 64, 512, generator=generator) * 0.4
    cache[1:] = values.to(dtype)
    q = (torch.randn(rows, 64, 512, generator=generator) * 0.4).bfloat16()
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    counts = [0, 1, 5, 63, min(2051, (pages - 1) * 64)]
    physical = torch.cat([torch.arange(p * 72, p * 72 + 64) for p in range(1, pages)])
    for row in range(rows):
        count = min(counts[row % len(counts)], topk)
        if count:
            # Spread holes through the row and retain a valid causal tail at end.
            slots = torch.linspace(0, topk - 1, count).long()
            indices[row, slots] = physical[
                torch.randperm(len(physical), generator=generator)[:count]
            ].int()
    return q, indices, cache


def _reference(query, indices, cache, *, quantize_probability=False):
    # Separate FP64 equation, on independently rounded E4M3 operands. Do not
    # invoke production metadata or its CPU attention path for the GPU oracle.
    rows = ((cache.shape[0] - 1) * cache.stride(0) + 64 * 512) // 512
    flat = cache.as_strided((rows, 512), (512, 1)).float().double()
    q = query.to(torch.float8_e4m3fn).float().double()
    result = torch.zeros_like(query)
    for row in range(len(query)):
        ids = indices[row][indices[row] >= 0].long()
        if len(ids):
            kv = flat[ids]
            score = q[row] @ kv.T / 16
            exponent = (score - score.max(-1, keepdim=True).values).exp()
            denominator = exponent.sum(-1, keepdim=True)
            if quantize_probability:
                # Pinned FlashInfer0.7.1 FP8 FMHA uses E4M3 P for the second
                # tensor-core product, with an unrounded softmax denominator.
                # Source: helpers_softmax._compute_fp8_p_regs_and_local_sums,
                # helpers_common._fp8_log2_quant_scale = log2(448):
                # https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/flashinfer/attention/prims_ts/kernels/fmha_decode/fmha_decode_resources/helpers_softmax.py
                # Independently evaluate the equation in FP64, rounding only P.
                # Multi-tile online/split softmax can choose different maxima;
                # the existing GPU tolerance covers that schedule difference.
                exponent = (448 * exponent).to(
                    torch.float8_e4m3fn
                ).float().double() / 448
            probability = exponent / denominator
            result[row] = probability @ kv
    return result


def test_metadata_stably_compacts_holes_and_keeps_duplicate_selection():
    ids = torch.tensor([[7, -1, 3, -1, 7], [-1, -1, -1, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([131071, 0], dtype=torch.int64)
    packed, active, actual_lengths, empty = sparse_metadata(ids, lengths)
    assert packed.tolist() == [[7, 3, 7, 7, 7, 7, 7, 7], [0] * 8]
    assert active.tolist() == [3, 1]
    assert actual_lengths.tolist() == [131071, 1]
    assert empty.tolist() == [False, True]
    assert ids.tolist() == [[7, -1, 3, -1, 7], [-1] * 5]


def test_fp8_reference_uses_quantized_queries_and_storage_coordinates():
    q, ids, cache = _fixture(topk=12, pages=3)
    actual = kernels.sparse_attention(
        q,
        ids,
        cache,
        1 / 16,
        sequence_lengths=torch.tensor([0, 1, 127, 65536, 131071]),
        max_seq_len=131072,
    )
    expected = _reference(q, ids, cache)
    # FP32 versus FP64 softmax can cross one BF16 rounding boundary.
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.008)
    assert torch.isfinite(actual).all()
    assert actual[0].count_nonzero() == 0


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@GPU
def test_cuda_metadata_strides_changed_contexts_and_no_jit(length_dtype):
    from resin_models_cuda.glm5_3_flash.dsa_cuda import _compact

    storage = torch.full((5, 2 * 2056 + 13), -1, dtype=torch.int32, device="cuda")
    ids = storage[:, : 2 * 2056 : 2]
    length_storage = torch.zeros(10, dtype=length_dtype, device="cuda")
    lengths = length_storage[::2]
    for count in (1, 8, 64):
        ids[:, :count] = torch.arange(count, device="cuda") * 72
        lengths.fill_(count)
        sparse_metadata(ids, lengths)
    device = torch.cuda.current_device()
    variants = set(_compact.device_caches[device][0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sparse_metadata(ids, lengths)
    for count, context in ((0, 0), (17, 8193), (511, 65537), (2053, 131071)):
        ids.fill_(-1)
        if count:
            ids[:, :count] = torch.arange(count, device="cuda").flip(0) * 72
        lengths.fill_(context)
        graph.replay()
        expected = sparse_metadata(ids.cpu(), lengths.cpu())
        for actual, oracle in zip(output, expected, strict=True):
            torch.testing.assert_close(actual.cpu(), oracle, atol=0, rtol=0)
        # Also invoke the normal eager adapter at new runtime context values.
        sparse_metadata(ids, lengths)
    assert set(_compact.device_caches[device][0]) == variants


@GPU
@pytest.mark.parametrize("topk", [8, 2056])
def test_trt_fp8_matches_independent_quantized_oracle_and_graph_changes(
    topk, monkeypatch
):
    q, ids, cache = _fixture(topk=topk)
    device_storage = torch.full(
        (cache.shape[0], 72, 512), float("nan"), device="cuda"
    ).to(cache.dtype)
    device_cache = device_storage.as_strided(cache.shape, cache.stride())
    device_cache.copy_(cache)
    # Explicitly poison the reserved dummy token too: idle output must be zero.
    device_cache[0].copy_(cache[0])
    device_q, device_ids = q.cuda(), ids.cuda()
    lengths = torch.tensor([0, 1, 257, 8192, 8192], dtype=torch.int64, device="cuda")
    workspace = DsaWorkspace("cuda")

    def run():
        return fp8_sparse_attention(
            device_q,
            device_ids,
            device_cache,
            1 / 16,
            workspace=workspace,
            sequence_lengths=lengths,
            max_seq_len=131072,
        )

    for _ in range(3):
        run()
    from flashinfer.mla import _core

    from resin_models_cuda.glm5_3_flash.dsa_cuda import _compact

    # The exact pinned TRT wrapper compiles its C++/CUDA module through this
    # cached factory. Warm one legal regime, then fail if unseen runtime lengths
    # cause another factory invocation. This covers the consumer, not only our
    # metadata Triton kernel. No context grid is precompiled.
    compiled = _core.get_trtllm_gen_fmha_module()
    misses = _core.get_trtllm_gen_fmha_module.cache_info().misses
    variants = set(_compact.device_caches[torch.cuda.current_device()][0])

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("runtime context requested a new TRT compiled module")

    monkeypatch.setattr(_core, "gen_trtllm_gen_fmha_module", unexpected_compile)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for seed, context in ((31, 4097), (32, 65537), (33, 131071)):
        q.copy_(
            torch.randn(
                q.shape, generator=torch.Generator().manual_seed(seed)
            ).bfloat16()
            * 0.3
        )
        device_q.copy_(q)
        # Changed rows alter active lengths and values while the graph retains
        # its original allocations and fixed launch geometry.
        ids[2] = ids[2].roll(1)
        device_ids.copy_(ids)
        lengths[2:] = context
        # Change actual FP8 KV contents as well as query values/indices, without
        # changing the view, page strides, workspace or captured pointers.
        changed = (
            torch.randn(64, 512, generator=torch.Generator().manual_seed(seed + 10))
            * 0.4
        )
        cache[1] = changed.to(cache.dtype)
        device_cache[1].copy_(cache[1])
        gc.collect()
        graph.replay()
        expected = _reference(q, ids, cache, quantize_probability=True)
        torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0.035)
        assert torch.isfinite(actual).all()
        assert actual[0].count_nonzero() == 0
        torch.testing.assert_close(actual, run(), atol=0, rtol=0)
    assert _core.get_trtllm_gen_fmha_module() is compiled
    assert _core.get_trtllm_gen_fmha_module.cache_info().misses == misses
    assert set(_compact.device_caches[torch.cuda.current_device()][0]) == variants


@GPU
def test_trt_fp8_sparse_storage_row_addresses_exceed_two_gibibytes():
    # This reproduces the cache's high page IDs without touching the entire
    # allocation. TMA indices are512-byte row coordinates, not int32 byteoffsets.
    pages = 60003
    cache = torch.empty_strided(
        (pages, 64, 512), (72 * 512, 512, 1), dtype=torch.float8_e4m3fn, device="cuda"
    )
    assert 60000 * cache.stride(0) > 2**31
    q = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.full((1, 8), -1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([131071], dtype=torch.int32, device="cuda")
    workspace = DsaWorkspace("cuda")
    values = torch.randn(2, 512, dtype=torch.bfloat16) * 0.3
    for page in (60000, 60001):
        cache[page, 3:5] = values.to(device="cuda", dtype=cache.dtype)
    ids[0, :2] = torch.tensor([60000 * 72 + 3, 60000 * 72 + 4], device="cuda")

    def run():
        return fp8_sparse_attention(
            q,
            ids,
            cache,
            1 / 16,
            workspace=workspace,
            sequence_lengths=lengths,
            max_seq_len=131072,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for page in (60000, 60001):
        ids[0, :2] = torch.tensor([page * 72 + 3, page * 72 + 4], device="cuda")
        graph.replay()
        quant_q = q.cpu().to(torch.float8_e4m3fn).float().double()
        quant_k = values.to(cache.dtype).float().double()
        score = quant_q @ quant_k.T / 16
        exponent = (score - score.max(-1, keepdim=True).values).exp()
        rounded = (448 * exponent).to(cache.dtype).float().double() / 448
        expected = (rounded / exponent.sum(-1, keepdim=True) @ quant_k).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0.035)


@GPU
def test_trt_fp8_probability_operand_matches_independent_basis_oracle():
    # Two basis-vector K/V rows reveal the probability operand directly. This
    # distinguishes the required448×E4M3 rounding from a plainFP64 softmax oracle
    # without changing the original random-fixture tolerances.
    q = torch.zeros(1, 64, 512, dtype=torch.bfloat16)
    q[0, :, 0] = torch.linspace(-24, 24, 64).bfloat16()
    storage = torch.full((2, 72, 512), float("nan")).to(torch.float8_e4m3fn)
    cache = storage.as_strided((2, 64, 512), (72 * 512, 512, 1))
    values = torch.zeros(2, 512)
    values[0, 0], values[1, 1] = 1, 1
    cache[1, :2] = values.to(cache.dtype)
    indices = torch.tensor([[72, 73, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    ideal = _reference(q, indices, cache)
    expected = _reference(q, indices, cache, quantize_probability=True)
    assert (
        (ideal.float() - expected.float()).abs() > 0.004 + 0.035 * ideal.float().abs()
    ).any()
    device_storage = storage.cuda()
    actual = fp8_sparse_attention(
        q.cuda(),
        indices.cuda(),
        device_storage.as_strided(cache.shape, cache.stride()),
        1 / 16,
        workspace=DsaWorkspace("cuda"),
        sequence_lengths=torch.tensor([2], dtype=torch.int32, device="cuda"),
        max_seq_len=131072,
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
