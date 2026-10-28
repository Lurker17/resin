"""Reusable vision tensors own storage and have bounded retention."""

import gc
import weakref

import pytest
import torch

from resin.backend.cuda.vision_cache import VisionFeatureCache


def test_cache_owns_compact_detached_storage_without_retaining_source():
    cache = VisionFeatureCache()
    source = torch.arange(100.0, requires_grad=True)
    view = source[1:17:2]
    reference = weakref.ref(source)
    expected = view.detach().clone()
    owned, absent = cache.put("image", (view, None))
    assert absent is None
    assert owned is not None and not owned.requires_grad
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()
    with torch.no_grad():
        source.zero_()
    torch.testing.assert_close(owned, expected)
    del view, source
    gc.collect()
    assert reference() is None
    cached = cache.get("image")
    assert cached is not None and cached[0] is owned
    assert cache.size_bytes == expected.numel() * expected.element_size()


@pytest.mark.parametrize("limits", [{"max_entries": 2}, {"max_bytes": 16}])
def test_lru_eviction_keeps_active_references_valid(limits):
    cache = VisionFeatureCache(**limits)
    (first,) = cache.put("a", (torch.ones(2),))
    cache.put("b", (torch.full((2,), 2.0),))
    cached = cache.get("a")  # Touch a: b is now least recently used.
    assert cached is not None and cached[0] is first
    cache.put("c", (torch.full((2,), 3.0),))
    assert cache.get("b") is None
    assert len(cache) == 2 and cache.size_bytes == 16
    cache.put("d", (torch.full((2,), 4.0),))
    assert cache.get("a") is None
    torch.testing.assert_close(first, torch.ones(2))
    cache.clear()
    assert len(cache) == 0 and cache.size_bytes == 0
    torch.testing.assert_close(first, torch.ones(2))


def test_large_entries_and_disabled_caches_do_not_retain_tensors():
    for cache in (VisionFeatureCache(max_bytes=4), VisionFeatureCache(max_entries=0)):
        workspace = torch.ones(2)
        (result,) = cache.put("large", (workspace,))
        workspace.zero_()
        torch.testing.assert_close(result, torch.ones(2))
        assert cache.get("large") is None
        assert len(cache) == 0 and cache.size_bytes == 0


def test_replacing_a_key_accounts_for_old_storage():
    cache = VisionFeatureCache(max_bytes=16)
    cache.put("a", (torch.ones(4),))
    cache.put("a", (torch.ones(1),))
    assert len(cache) == 1 and cache.size_bytes == 4
    cache.put("a", (torch.ones(5),))
    assert len(cache) == 0 and cache.size_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ownership")
def test_cache_waits_for_producer_stream_and_survives_workspace_overwrite():
    cache = VisionFeatureCache()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        workspace = torch.empty(1_000_000, device="cuda")
        workspace.fill_(7)
        cache.put("image", (workspace[::2],))
        workspace.zero_()
    cached = cache.get("image")
    assert cached is not None
    (owned,) = cached
    assert owned is not None
    cache.clear()
    torch.testing.assert_close(owned, torch.full_like(owned, 7))
