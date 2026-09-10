from __future__ import annotations

import math

import pytest

from resin.backend.capacity import plan_cache_capacity


def test_capacity_plan_accounts_for_protected_memory_overhead_and_null_parent() -> None:
    plan = plan_cache_capacity(
        1_000,
        800,
        0.9,
        100,
        reserved_parent_count=1,
        overhead_bytes=50,
    )

    assert plan.protected_free_bytes == math.ceil(1_000 * 0.1)
    assert plan.cache_budget_bytes == 650
    assert plan.parent_capacity == 5
    assert plan.allocation_bytes == 600
    assert plan.overhead_bytes == 50


def test_capacity_plan_rejects_a_budget_without_one_usable_parent() -> None:
    with pytest.raises(MemoryError, match="reserved and usable"):
        plan_cache_capacity(1_000, 200, 0.9, 100)
