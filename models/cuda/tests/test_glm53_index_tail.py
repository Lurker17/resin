"""Bounded index history versus the independent full-history pool reference."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry
from resin_models_cuda.glm5_3_flash.index_tail import IndexTail, commit_tail
from resin_models_cuda.glm5_3_flash.kernels import store_sparse

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cache(device="cpu", *, bounded=True, origin=0, batch=1):
    pages, width, dim = 80, 8, 4
    cache = SimpleNamespace(
        page_table=torch.arange(
            1, 1 + batch * 8, dtype=torch.int32, device=device
        ).reshape(batch, 8),
        page_tokens=64,
        storage_origin=origin,
        latent=torch.zeros(pages, 64, width, dtype=torch.bfloat16, device=device),
        pooled_keys=torch.zeros(pages, 16, dim, dtype=torch.bfloat16, device=device),
        index_keys=torch.zeros(pages, 64, dim, dtype=torch.bfloat16, device=device),
        index_gates=torch.zeros(pages, 64, dim, dtype=torch.float32, device=device),
    )
    if bounded:
        cache.index_tail = IndexTail(
            torch.zeros(batch, 4, dim, dtype=torch.bfloat16, device=device),
            torch.zeros(batch, 4, dim, dtype=torch.float32, device=device),
            torch.full((batch,), -1, dtype=torch.int64, device=device),
        )
    return cache


def _set(cache, starts, counts):
    device = cache.latent.device
    cache.positions = torch.tensor(
        [
            p
            for start, count in zip(starts, counts)
            for p in range(start, start + count)
        ],
        dtype=torch.long,
        device=device,
    )
    cache.sequence_indices = torch.repeat_interleave(
        torch.arange(len(starts), device=device), torch.tensor(counts, device=device)
    )
    cache.cu_seqlens = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device
    )


def _inputs(device="cpu", rows=300):
    gen = torch.Generator().manual_seed(913)
    return (
        torch.randn(rows, 8, generator=gen).bfloat16().to(device),
        torch.randn(rows, 4, generator=gen).bfloat16().to(device),
        torch.randn(rows, 4, generator=gen).float().to(device),
        torch.randn(4, 4, generator=gen).bfloat16().to(device),
    )


def _equal(actual, expected):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("chunks", [[1, 2, 61, 1, 63, 3, 65], [129, 67], [4] * 49])
def test_chunked_tail_matches_full_history(origin, chunks):
    bounded, full = _cache(origin=origin), _cache(bounded=False, origin=origin)
    latent, keys, gates, ape = _inputs()
    begin = 0
    for count in chunks:
        for cache in (bounded, full):
            _set(cache, [begin], [count])
            store_sparse(
                latent[begin : begin + count],
                keys[begin : begin + count],
                gates[begin : begin + count],
                ape,
                cache,
            )
        _equal(bounded.latent, full.latent)
        _equal(bounded.pooled_keys, full.pooled_keys)
        end = begin + count
        assert bounded.index_tail.end.tolist() == [end - 1]
        if end >= 4:
            _equal(bounded.index_tail.keys[0], keys[end - 4 : end])
            _equal(bounded.index_tail.gates[0], gates[end - 4 : end])
        begin = end


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("start", [3, 4, 63, 64, 65])
@pytest.mark.parametrize("accepted", [1, 2, 4, 7, 8])
def test_speculative_accepted_tail_then_continuation(origin, start, accepted):
    bounded, full = _cache(origin=origin), _cache(bounded=False, origin=origin)
    latent, keys, gates, ape = _inputs()
    for cache in (bounded, full):
        _set(cache, [0], [start])
        store_sparse(latent[:start], keys[:start], gates[:start], ape, cache)
        _set(cache, [start], [8])
        store_sparse(
            latent[start : start + 8],
            keys[start : start + 8],
            gates[start : start + 8],
            ape,
            cache,
        )
    commit_tail(bounded, torch.tensor([accepted]))
    end = start + accepted
    assert bounded.index_tail.end.tolist() == [end - 1]
    _equal(bounded.index_tail.keys[0], keys[end - 4 : end])
    # Future rejected projections are replaced before they become visible.
    for cache in (bounded, full):
        _set(cache, [end], [9])
        store_sparse(latent[200:209], keys[200:209], gates[200:209], ape, cache)
    _equal(bounded.latent, full.latent)
    _equal(bounded.pooled_keys, full.pooled_keys)


@pytest.mark.parametrize("start", [4, 64, 128])
def test_mtp_rewrite_completed_pool_then_repair(start):
    bounded, full = _cache(origin=1), _cache(bounded=False, origin=1)
    latent, keys, gates, ape = _inputs()
    for cache in (bounded, full):
        _set(cache, [0], [start])
        store_sparse(latent[:start], keys[:start], gates[:start], ape, cache)
        # The first next draft rewrites p-1, possibly completing the old pool.
        _set(cache, [start - 1], [1])
        store_sparse(latent[200:201], keys[200:201], gates[200:201], ape, cache)
    _equal(bounded.pooled_keys, full.pooled_keys)
    repair_initial = bounded.index_tail.snapshot()
    for step in range(2):
        _set(bounded, [start + step], [1])
        store_sparse(
            latent[210 + step : 211 + step],
            keys[210 + step : 211 + step],
            gates[210 + step : 211 + step],
            ape,
            bounded,
        )
    bounded.index_tail.assign(repair_initial)
    for cache in (bounded, full):
        _set(cache, [start], [4])
        store_sparse(latent[220:224], keys[220:224], gates[220:224], ape, cache)
    commit_tail(bounded, torch.tensor([1]))
    for cache in (bounded, full):
        _set(cache, [start], [4])
        store_sparse(latent[230:234], keys[230:234], gates[230:234], ape, cache)
    _equal(bounded.pooled_keys, full.pooled_keys)


def _geometry():
    return CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention"), 16, 8, 4, 2, 8, 3, 64, True
    )


def _group(parents, slots=None):
    parents = torch.tensor(parents, dtype=torch.int32)
    return DevicePageGroup(
        parents,
        torch.zeros_like(parents)
        if slots is None
        else torch.tensor(slots, dtype=torch.int32),
        torch.zeros(parents.shape[0], dtype=torch.int32),
    )


def test_tail_prefix_snapshot_and_fork_are_engine_owned():
    g = _geometry()
    arena = CacheArena("cpu", g.layout(6), g, max_batch_size=1)
    cache = arena._make_graph_cache((_group([[1, 2]]), _group([[3, 4]])))
    latent, keys, gates, ape = _inputs()
    batch = cache.batch(
        torch.arange(64),
        torch.zeros(64, dtype=torch.long),
        torch.tensor([0, 64]),
        torch.tensor([0]),
    )
    target, mtp = batch.layer(1), batch.layer(2)
    store_sparse(latent[:64], keys[:64], gates[:64], ape, target)
    # Prompt priming is one logical token behind the target at publication.
    mtp.positions = torch.arange(63)
    mtp.sequence_indices = torch.zeros(63, dtype=torch.long)
    mtp.cu_seqlens = torch.tensor([0, 63])
    store_sparse(latent[:63], keys[:63], gates[:63], ape, mtp)
    batch.store(torch.zeros(1, 16, dtype=torch.bfloat16))
    prefix = arena.storage[3].clone()
    resumed = cache.batch(
        torch.tensor([64]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([64])
    )
    assert resumed.layer(1).index_tail.end.tolist() == [63]
    assert resumed.layer(2).index_tail.end.tolist() == [62]
    fork = cache.batch(
        torch.tensor([64]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([64])
    )
    old = fork.layer(2).index_tail.keys.clone()
    resumed.layer(2).positions = torch.tensor([63])
    store_sparse(latent[63:64], keys[63:64], gates[63:64], ape, resumed.layer(2))
    resumed.store(torch.zeros(1, 16, dtype=torch.bfloat16))
    _equal(arena.storage[3], prefix)
    _equal(fork.layer(2).index_tail.keys, old)
    assert resumed.layer(2).index_tail.end.tolist() == [63]
    # New sequence reuse clears even initialized stale parent contents.
    fresh = cache.batch(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0])
    )
    assert fresh.layer(2).index_tail.end.tolist() == [-1]
    assert not fresh.layer(2).index_tail.keys.count_nonzero()


def test_full_pages_hold_no_raw_history_and_tail_size_is_context_independent():
    g = _geometry()
    arena = CacheArena("cpu", g.layout(3), g, max_batch_size=1)
    assert not hasattr(arena, "index_keys") and not hasattr(arena, "index_gates")
    assert arena.index_tail_keys.shape[-3:] == (2, 4, 4)
    assert g.unit_bytes == 64 * 8 * 2 + 16 * 4 * 2
    assert replace(g, state_tokens=8192).state_bytes == g.state_bytes


@CUDA
@pytest.mark.parametrize("origin", [0, 1])
def test_cuda_tail_graph_changed_inputs_pages_counts_and_idle(origin):
    actual = _cache("cuda", origin=origin, batch=2)
    reference = _cache("cuda", bounded=False, origin=origin, batch=2)
    latent, keys, gates, ape = _inputs("cuda")
    # Seed both sequences on either side of a physical page boundary.
    for cache in (actual, reference):
        _set(cache, [0, 0], [63, 64])
        store_sparse(latent[:127], keys[:127], gates[:127], ape, cache)
    initial = actual.index_tail.snapshot()
    for cache in (actual, reference):
        _set(cache, [63, 64], [8, 8])
    source = [x[150:166].clone() for x in (latent, keys, gates)]
    counts = torch.tensor([1, 8], device="cuda")

    def run():
        actual.index_tail.assign(initial)
        store_sparse(*source, ape, actual)
        commit_tail(actual, counts)
        return actual.index_tail.snapshot()

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for value in (1, 4, 8):
        counts.copy_(torch.tensor([value, 8], device="cuda"))
        source[1].copy_(keys[170:186])
        source[2].copy_(gates[170:186])
        graph.replay()
        for seq, begin in enumerate((0, 8)):
            accepted = value if seq == 0 else 8
            expected = torch.cat(
                (initial.keys[seq], source[1][begin : begin + accepted])
            )[-4:]
            _equal(result.keys[seq], expected)
        store_sparse(*source, ape, reference)
        _equal(actual.pooled_keys, reference.pooled_keys)
    # Idle rows preserve the immutable starting tail, no empty-source read.
    _set(actual, [63, 64], [0, 0])
    actual.index_tail.assign(initial)
    store_sparse(latent[:0], keys[:0], gates[:0], ape, actual)
    _equal(actual.index_tail.keys, initial.keys)


@CUDA
def test_tail_runtime_lengths_counts_strides_and_batches_do_not_recompile():
    from resin_models_cuda.glm5_3_flash import index_tail_cuda, kernels_cuda

    def variants():
        device = torch.cuda.current_device()
        return {
            name: set(function.device_caches[device][0])
            for name, function in (
                ("store", kernels_cuda._store),
                ("compress_tail", kernels_cuda._compress_tail),
                ("select_tail", index_tail_cuda._select_tail),
            )
        }

    for iteration, (batch, count, accepted, stride) in enumerate(
        [(2, 8, 3, 8), (1, 7, 1, 12), (3, 13, 6, 16), (2, 64, 8, 8)]
    ):
        cache = _cache("cuda", batch=batch)
        _set(cache, [0] * batch, [count] * batch)
        latent, _, _, ape = _inputs("cuda", rows=batch * count)
        # The same feature width appears at different packed projection strides.
        source = torch.randn(batch * count, stride, device="cuda")
        keys = source.bfloat16()[:, :4]
        gates = source[:, 4:8]
        store_sparse(latent, keys, gates, ape, cache)
        commit_tail(cache, torch.full((batch,), accepted, device="cuda"))
        torch.cuda.synchronize()
        if iteration == 0:
            baseline = variants()
        else:
            assert variants() == baseline


@pytest.mark.parametrize("origin", [0, 1])
def test_long_prefill_uses_current_chunk_scratch_not_four_slot_ring(origin):
    bounded = _cache(origin=origin)
    full = _cache(bounded=False, origin=origin)
    latent, keys, gates, ape = _inputs(rows=260)
    for cache in (bounded, full):
        _set(cache, [0], [3])
        store_sparse(latent[:3], keys[:3], gates[:3], ape, cache)
        _set(cache, [3], [257])
        store_sparse(latent[3:], keys[3:], gates[3:], ape, cache)
    _equal(bounded.pooled_keys, full.pooled_keys)
    _equal(bounded.index_tail.keys[0], keys[-4:])


@CUDA
@pytest.mark.parametrize("origin", [0, 1])
def test_fp8_store_exact_bytes_and_72_row_envelope(origin):
    # Consumer-independent conversion oracle, including the MTP page crossing.
    cache = _cache("cuda", origin=origin)
    storage = torch.zeros((80, 72, 512), dtype=torch.uint8, device="cuda")
    cache.latent = storage.view(torch.float8_e4m3fn).as_strided(
        (80, 64, 512), (72 * 512, 512, 1)
    )
    _set(cache, [0], [70])
    latent = torch.randn(70, 512, dtype=torch.bfloat16, device="cuda") * 2
    _, keys, gates, ape = _inputs("cuda", rows=70)
    store_sparse(latent, keys, gates, ape, cache)
    expected = torch.zeros_like(storage)
    quantized = latent.to(torch.float8_e4m3fn).view(torch.uint8)
    for row in range(70):
        physical = row + origin
        page = 1 + physical // 64
        expected[page, physical % 64].copy_(quantized[row])
    _equal(storage, expected)
    # A changed-input graph retains exactly the same physical byte mapping.
    initial = cache.index_tail.snapshot()

    def run():
        cache.index_tail.assign(initial)
        store_sparse(latent, keys, gates, ape, cache)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    latent.mul_(0.5)
    graph.replay()
    quantized = latent.to(torch.float8_e4m3fn).view(torch.uint8)
    for row in range(70):
        physical = row + origin
        expected[1 + physical // 64, physical % 64].copy_(quantized[row])
    _equal(storage, expected)
