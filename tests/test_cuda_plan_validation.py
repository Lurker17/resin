"""Model-specific limits remain checked after native plan validation."""

import pytest

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.runner import validate_hybrid_plan


def _validate(plan):
    return validate_hybrid_plan(
        plan, label="test", max_batch_size=2, max_seq_len=8, decode_lookahead=2
    )


def test_native_prefill_and_decode_keep_their_distinct_capacity_limits():
    prefill = BatchPlan([1, 2], [6, 4], [1, 2, 3], [0, 2, 3])
    decode = BatchPlan([1, 2], [5, 3], [1, 2], [0, 1, 2], phase=BatchPhase.DECODE)
    assert _validate(prefill) == ((0, 2), (2, 3))
    assert _validate(decode) == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    "plan,error",
    [
        (BatchPlan([1, 2, 3], [0, 0, 0], [1, 2, 3], [0, 1, 2, 3]), "batch size"),
        (BatchPlan([1], [7], [1, 2], [0, 2]), "max_seq_len"),
        (BatchPlan([1], [6], [1], [0, 1], phase=BatchPhase.DECODE), "max_seq_len"),
        (BatchPlan([1], [3], [1, 2], [0, 2], phase=BatchPhase.DECODE), "one anchor"),
        (BatchPlan([1], [0], [1], [0, 1], phase=BatchPhase.DECODE), "cached prefix"),
    ],
)
def test_native_valid_plans_still_obey_model_capacity_and_decode_rules(plan, error):
    with pytest.raises(ValueError, match=error):
        _validate(plan)
