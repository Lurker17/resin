"""Model-specific limits remain checked after native plan validation."""

import pytest

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.runner import validate_hybrid_plan


def _validate(plan):
    return validate_hybrid_plan(
        plan, label="test", max_batch_size=2, max_seq_len=8, decode_lookahead=2
    )


def test_native_prefill_and_decode_keep_their_distinct_capacity_limits():
    prefill = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[6, 4],
        token_ids=[1, 2, 3],
        token_indptr=[0, 2, 3],
    )
    decode = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[5, 3],
        token_ids=[1, 2],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    assert _validate(prefill) == ((0, 2), (2, 3))
    assert _validate(decode) == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    "plan,error",
    [
        (
            BatchPlan(
                sequence_ids=[1, 2, 3],
                context_lengths=[0, 0, 0],
                token_ids=[1, 2, 3],
                token_indptr=[0, 1, 2, 3],
            ),
            "batch size",
        ),
        (
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[7],
                token_ids=[1, 2],
                token_indptr=[0, 2],
            ),
            "max_seq_len",
        ),
        (
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[6],
                token_ids=[1],
                token_indptr=[0, 1],
                phase=BatchPhase.DECODE,
            ),
            "max_seq_len",
        ),
        (
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[3],
                token_ids=[1, 2],
                token_indptr=[0, 2],
                phase=BatchPhase.DECODE,
            ),
            "one anchor",
        ),
        (
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[0],
                token_ids=[1],
                token_indptr=[0, 1],
                phase=BatchPhase.DECODE,
            ),
            "cached prefix",
        ),
    ],
)
def test_native_valid_plans_still_obey_model_capacity_and_decode_rules(plan, error):
    with pytest.raises(ValueError, match=error):
        _validate(plan)
