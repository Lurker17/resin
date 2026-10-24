"""Explicit native outputs for test runners, without importing Torch."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from resin import ForwardOutput


def forward_output(
    token_rows: Sequence[int] | Sequence[Sequence[int]],
) -> ForwardOutput:
    rows = (
        [[token] for token in cast(Sequence[int], token_rows)]
        if len(token_rows) and np.isscalar(token_rows[0])
        else cast(Sequence[Sequence[int]], token_rows)
    )
    lengths = np.fromiter((len(row) for row in rows), dtype=np.uint32)
    tokens = [token for row in rows for token in row]
    if any(
        isinstance(token, (bool, np.bool_))
        or not isinstance(token, (int, np.integer))
        or not 0 <= token <= 0xFFFFFFFF
        for token in tokens
    ):
        raise ValueError("test output token IDs must be uint32 integers")
    return ForwardOutput(
        token_ids=np.asarray(tokens, dtype=np.uint32),
        token_indptr=np.concatenate(
            (np.zeros(1, dtype=np.uint32), lengths.cumsum(dtype=np.uint32))
        ),
    )
