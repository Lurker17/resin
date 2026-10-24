from __future__ import annotations

import numpy as np
import pytest
import torch

from resin.backend.cuda.completion import accepted_block
from resin.backend.speculative import accept_speculative_tokens


@pytest.mark.parametrize("draft_tokens", [0, 1, 3, 4, 8])
@pytest.mark.parametrize("fixed", [False, True])
def test_device_acceptance_matches_shared_reference(draft_tokens, fixed):
    generator = torch.Generator().manual_seed(15)
    drafts = torch.randint(0, 5, (128, draft_tokens), generator=generator)
    targets = torch.randint(0, 5, (128, draft_tokens + 1), generator=generator)
    for row in range(128):
        prefix = row % (draft_tokens + 1)
        targets[row, :prefix] = drafts[row, :prefix]
    positions = torch.arange(128, dtype=torch.int64) * 7
    fixed_depth = draft_tokens // 2 if fixed else None
    output = torch.cat((drafts, targets), dim=1) if draft_tokens else targets
    tokens, counts, feedback = accepted_block(
        output, positions, draft_tokens, fixed_acceptance=fixed_depth
    )
    expected = accept_speculative_tokens(
        drafts.tolist(), targets.tolist(), benchmark_accepted_drafts=fixed_depth
    )
    actual = [
        row[:count] for row, count in zip(tokens.tolist(), counts.tolist(), strict=True)
    ]
    assert actual == expected.token_rows
    assert counts.tolist() == expected.processed_counts
    assert feedback.token_ids.tolist() == [row[-1] for row in actual]
    assert feedback.positions.tolist() == (positions + counts).tolist()
    assert (
        feedback.inputs()["sampling_offsets"].tolist()
        == ((positions + counts + 1) * 32).tolist()
    )
    # Next replay is allowed to overwrite its source output immediately.
    output.fill_(-1)
    assert feedback.token_ids.tolist() == [row[-1] for row in actual]


def test_padded_feedback_never_advances_outside_reserved_scratch_positions():
    positions = torch.ones(8, dtype=torch.int64)
    output = torch.tensor([[3, 4, 5, 3, 4, 5, 6]] * 8, dtype=torch.int64)
    for _ in range(256):
        _, _, feedback = accepted_block(output, positions, 3)
        feedback.restrict(5)
        positions = feedback.positions
        assert positions[5:].tolist() == [1, 1, 1]
        assert feedback.token_ids[5:].tolist() == [0, 0, 0]
    assert positions[:5].tolist() == [1025] * 5


def test_completion_owns_copied_results_and_counts_stats_once(monkeypatch):
    from resin.backend.cuda import completion as runtime
    from resin.backend.speculative import SpeculativeStats

    class Stager:
        def __init__(self, device):
            pass

        def output(self, name, source):
            return source.clone()

    class Event:
        syncs = 0

        def record(self, stream):
            pass

        def synchronize(self):
            self.syncs += 1

    monkeypatch.setattr(runtime, "TensorStager", Stager)
    monkeypatch.setattr(runtime.torch.cuda, "Event", Event)
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda device: None)
    output = torch.tensor([[3, 4, 5, 3, 9, 7, 6]] * 4, dtype=torch.int64)
    tokens, counts, feedback = accepted_block(
        output, torch.ones(4, dtype=torch.int64), 3
    )
    stats = SpeculativeStats(3)
    pending = runtime.DeviceCompletion(tokens, counts, feedback, rows=3, stats=stats)
    tokens.fill_(-1)
    counts.fill_(-1)

    def forbid_list(tensor):
        raise AssertionError("completion must not box tokens into Python lists")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", forbid_list)
        result = pending.result()
        assert pending.result() is result
    np.testing.assert_array_equal(result.token_ids, [3, 9] * 3)
    np.testing.assert_array_equal(result.token_indptr, [0, 2, 4, 6])
    assert pending.positions.tolist() == [1, 1, 1]
    assert pending.feedback.positions.tolist() == [3, 3, 3, 1]
    assert isinstance(pending._event, Event)
    assert pending._event.syncs == 1
    assert stats.snapshot() == {
        "blocks": 3,
        "drafts": 9,
        "accepted": 3,
        "emitted": 6,
        "acceptance_rate": 1 / 3,
    }


def test_feedback_fits_the_captured_decode_token_column_and_metadata_buffers():
    from resin.backend.cuda.staging import TensorStager

    output = torch.tensor([[3, 4, 5, 3, 9, 7, 6]] * 12, dtype=torch.int64)
    _, _, feedback = accepted_block(output, torch.ones(12, dtype=torch.int64), 3)
    feedback.restrict(8)
    # These are the actual shapes/dtypes allocated by the shared CUDA runner.
    targets = {
        "token_ids": torch.empty((12, 1), dtype=torch.int64),
        "positions": torch.empty(12, dtype=torch.int64),
        "sampling_offsets": torch.empty(12, dtype=torch.int64),
    }
    TensorStager("cpu").copy_into(targets, feedback.inputs())
    assert targets["token_ids"][:, 0].tolist() == [9] * 8 + [0] * 4
    assert targets["positions"].tolist() == [3] * 8 + [1] * 4
    assert targets["sampling_offsets"].tolist() == [128] * 8 + [64] * 4


def test_staging_error_identifies_the_device_field_shape_and_dtype():
    from resin.backend.cuda.staging import StagingError, TensorStager

    targets = {"positions": torch.zeros(4, dtype=torch.int64)}
    with pytest.raises(
        StagingError,
        match=r"positions.*shape \(4,\).*torch.int32.*expected shape \(4,\).*torch.int64",
    ):
        TensorStager("cpu").copy_into(
            targets, {"positions": torch.ones(4, dtype=torch.int32)}
        )
    assert targets["positions"].tolist() == [0] * 4
