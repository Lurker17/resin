"""Backend-neutral GPT-OSS cache geometry and attention batching."""

from __future__ import annotations

from resin import BatchPhase
from resin.backend.paged_attention import (
    PagedAttentionBatch as AttentionBatch,
)
from resin.backend.paged_attention import (
    PagedAttentionGroupSpec,
    build_paged_attention_batch,
)

PAGE_SIZE = 128
SLIDING_WINDOW = 128
PREFILL_CHUNK_SIZE = 8192
# Paged prefill retains the whole query plus the preceding attention window.
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1
WINDOW_TABLE_WIDTH = 1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE

ATTENTION_GROUPS = (
    PagedAttentionGroupSpec("full", 0),
    PagedAttentionGroupSpec("sliding", 1, WINDOW_TABLE_WIDTH),
)


def build_attention_batches(
    plan, cache_layout
) -> tuple[tuple[AttentionBatch, ...], AttentionBatch | None]:
    """Return one prefill batch or one decode batch from a homogeneous plan."""

    batch = build_paged_attention_batch(
        plan,
        cache_layout,
        ATTENTION_GROUPS,
        max_query_tokens=PREFILL_CHUNK_SIZE,
    )
    if batch.phase == BatchPhase.PREFILL:
        return (batch,), None
    return (), batch


__all__ = [
    "ATTENTION_GROUPS",
    "AttentionBatch",
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "SLIDING_WINDOW",
    "WINDOW_RETENTION_TOKENS",
    "WINDOW_TABLE_WIDTH",
    "build_attention_batches",
]
