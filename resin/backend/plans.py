"""Small framework-neutral plan adapters for readable model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resin._resin import BatchPlan


@dataclass(frozen=True, slots=True)
class PlanRow:
    request_id: int
    token_ids: list[int]
    positions: list[int]
    cache: Any


def plan_positions(plan: BatchPlan) -> list[int]:
    """Return the absolute position of every scheduled token."""

    positions: list[int] = []
    for index, context_length in enumerate(plan.context_lengths):
        start, end = plan.token_indptr[index : index + 2]
        positions.extend(range(context_length, context_length + end - start))
    return positions


def plan_rows(
    plan: BatchPlan, caches: dict[int, Any], vocab_size: int
) -> list[PlanRow]:
    """Validate and split one native plan into request-local rows."""

    rows = []
    for index, sequence_id in enumerate(plan.sequence_ids):
        start, end = plan.token_indptr[index : index + 2]
        cache = caches.get(sequence_id)
        offset = 0 if cache is None else cache.length
        positions = list(
            range(
                plan.context_lengths[index], plan.context_lengths[index] + end - start
            )
        )
        expected = list(range(offset, offset + end - start))
        if positions != expected:
            raise ValueError(
                f"non-contiguous positions for request {sequence_id}: "
                f"expected {expected}, got {positions}"
            )
        row_tokens = plan.token_ids[start:end]
        if any(token < 0 or token >= vocab_size for token in row_tokens):
            raise ValueError(f"token id is out of range for request {sequence_id}")
        rows.append(PlanRow(sequence_id, row_tokens, positions, cache))
    return rows
