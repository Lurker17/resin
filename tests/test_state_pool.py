from __future__ import annotations

import pytest

from resin import StatePool


class TestStatePool:
    def test_reference_counts_and_bulk_allocation(self) -> None:
        pool = StatePool(3)
        first, second = pool.allocate(2)
        assert (pool.free_count, pool.active_count) == (1, 2)
        pool.retain([first, first])
        assert pool.counts(first) == (3, 0)
        pool.release([first, first, second])
        assert pool.counts(first) == (1, 0)
        assert not pool.contains(second)

    def test_completion_pin_defers_reuse(self) -> None:
        pool = StatePool(1)
        first = pool.allocate()[0]
        pool.pin([first])
        pool.release([first])
        assert (pool.free_count, pool.retiring_count) == (0, 1)
        with pytest.raises(MemoryError):
            pool.allocate()

        pool.unpin([first])
        second = pool.allocate()[0]
        assert first.slot == second.slot
        assert first.generation != second.generation
        assert not pool.contains(first)
        with pytest.raises(ValueError, match="stale"):
            pool.counts(first)

    def test_pool_identity_and_failed_bulk_operation_are_safe(self) -> None:
        first_pool = StatePool(1)
        second_pool = StatePool(1)
        handle = first_pool.allocate()[0]
        with pytest.raises(ValueError, match="different pool"):
            second_pool.retain([handle])
        with pytest.raises(ValueError, match="exceeds"):
            first_pool.release([handle, handle])
        assert first_pool.counts(handle) == (1, 0)
