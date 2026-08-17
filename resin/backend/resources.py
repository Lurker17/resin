"""Framework-neutral physical resource bindings."""

from __future__ import annotations


class CachePoolBinding:
    """Permanently bind one physical allocation to one native cache pool."""

    def __init__(self, layout, *, resource: str = "cache") -> None:
        self.layout = layout
        self.resource = resource
        self.pool_id: int | None = None

    def validate(self, pool_id: int, layout) -> int:
        if isinstance(pool_id, bool) or not isinstance(pool_id, int) or pool_id <= 0:
            raise ValueError("cache pool id must be a positive integer")
        if layout is not self.layout:
            raise ValueError(
                "cache pool binding must use this allocation's frozen layout object"
            )
        if self.pool_id is not None and self.pool_id != pool_id:
            raise ValueError(
                f"this {self.resource} allocation is already bound to a different "
                "cache pool"
            )
        return pool_id

    def commit(self, pool_id: int) -> int:
        self.pool_id = pool_id
        return pool_id

    def bind(self, pool_id: int, layout) -> int:
        return self.commit(self.validate(pool_id, layout))

    def bind_resources(self, binding) -> None:
        """Consume the native Engine binding exposed by the runner contract."""

        if binding.cache_pool_id is None or binding.cache_layout is None:
            raise ValueError(f"{self.resource} requires managed cache resources")
        self.bind(binding.cache_pool_id, binding.cache_layout)
