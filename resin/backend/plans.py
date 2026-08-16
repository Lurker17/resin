"""Small framework-neutral plan adapters for readable model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from resin._resin import BatchPlan


@dataclass(frozen=True, slots=True)
class PlanRow:
    request_id: int
    token_ids: npt.NDArray[np.uint32]
    positions: npt.NDArray[np.uint32]
    cache: Any


def plan_rows(
    plan: BatchPlan, caches: dict[int, Any], vocab_size: int
) -> list[PlanRow]:
    """Validate and split one native plan into request-local rows."""

    rows = []
    for index, sequence_id in enumerate(plan.sequence_ids):
        start, end = plan.token_indptr[index : index + 2]
        start = int(start)
        end = int(end)
        sequence_id = int(sequence_id)
        cache = caches.get(sequence_id)
        offset = 0 if cache is None else cache.length
        positions = plan.positions[start:end]
        expected = np.arange(offset, offset + end - start, dtype=np.uint32)
        if not np.array_equal(positions, expected):
            raise ValueError(
                f"non-contiguous positions for request {sequence_id}: "
                f"expected {expected.tolist()}, got {positions.tolist()}"
            )
        row_tokens = plan.token_ids[start:end]
        if np.any(row_tokens >= vocab_size):
            raise ValueError(f"token id is out of range for request {sequence_id}")
        rows.append(PlanRow(sequence_id, row_tokens, positions, cache))
    return rows
