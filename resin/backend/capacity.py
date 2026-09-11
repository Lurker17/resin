"""Framework-neutral device-memory budgeting for cache parent pools."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheCapacityPlan:
    """Post-load memory budget for one physical cache-parent allocation."""

    total_memory_bytes: int
    free_memory_bytes: int
    memory_utilization: float
    protected_free_bytes: int
    overhead_bytes: int
    cache_budget_bytes: int
    bytes_per_parent: int
    reserved_parent_count: int
    parent_capacity: int
    allocation_bytes: int


def plan_cache_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    memory_utilization: float,
    bytes_per_parent: int,
    *,
    reserved_parent_count: int = 1,
    overhead_bytes: int = 0,
) -> CacheCapacityPlan:
    """Protect unused VRAM, subtract overhead, and size usable parent IDs."""

    if total_memory_bytes <= 0 or not 0 < free_memory_bytes <= total_memory_bytes:
        raise ValueError("device memory totals must be positive and internally valid")
    if not 0.0 < memory_utilization < 1.0:
        raise ValueError("memory_utilization must be between zero and one")
    if bytes_per_parent <= 0:
        raise ValueError("bytes_per_parent must be positive")
    if reserved_parent_count <= 0:
        raise ValueError("reserved_parent_count must be positive")
    if overhead_bytes < 0:
        raise ValueError("overhead_bytes cannot be negative")

    protected_free_bytes = math.ceil(total_memory_bytes * (1.0 - memory_utilization))
    cache_budget_bytes = free_memory_bytes - protected_free_bytes - overhead_bytes
    parent_capacity = cache_budget_bytes // bytes_per_parent - reserved_parent_count
    if parent_capacity <= 0:
        raise MemoryError("cache budget cannot hold its reserved and usable parents")
    allocation_bytes = (parent_capacity + reserved_parent_count) * bytes_per_parent
    return CacheCapacityPlan(
        total_memory_bytes,
        free_memory_bytes,
        memory_utilization,
        protected_free_bytes,
        overhead_bytes,
        cache_budget_bytes,
        bytes_per_parent,
        reserved_parent_count,
        parent_capacity,
        allocation_bytes,
    )
