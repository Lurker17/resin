"""Image identity and cached features survive follower plan serialization."""

import pickle

import pytest
import torch
from test_deepseek_v41_vision import media_fixture, plan_for, runner_fixture

from resin import BatchPhase, BatchPlan
from resin.backend.parallel import batch_plan_from_payload, batch_plan_payload
from resin.models.deepseek_v41.media import DeepSeekV41Image, DeepSeekV41Media


def transported(plan):
    return batch_plan_from_payload(pickle.loads(pickle.dumps(batch_plan_payload(plan))))


def test_serialized_prefill_chunks_and_decode_reuse_image_features():
    media = media_fixture()
    leader, _ = runner_fixture()
    follower, calls = runner_fixture()
    previous_media = media
    for start, end in ((0, 5), (5, 7), (7, len(media.token_ids))):
        original = plan_for(media, start, end)
        restored = transported(original)
        assert restored.media[0] is not previous_media
        assert restored.media[0].digest == media.digest
        previous_media = restored.media[0]
        follower._validate_plan(restored)
        input_ids = torch.tensor([media.token_ids[start:end]])
        torch.testing.assert_close(
            follower._media_embeddings(restored, 0, input_ids),
            leader._media_embeddings(original, 0, input_ids),
        )
    assert len(calls) == 1
    cached = follower._media_features[42]
    for position in range(len(media.token_ids), len(media.token_ids) + 2):
        decode = transported(
            BatchPlan(
                [42],
                [position],
                [21],
                [0, 1],
                phase=BatchPhase.DECODE,
                media=[media],
            )
        )
        assert decode.media[0] is not cached[0]
        assert follower._validate_plan(decode) == ((0, 1),)
        assert follower._media_embeddings(decode, 0, torch.tensor([[21]])) is None
        assert follower._media_features[42] is cached
    assert len(calls) == 1
    follower.release([42])
    assert not follower._media_features


@pytest.mark.parametrize("change", ["pixels", "prompt"])
@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_serialized_plans_reject_changed_media_or_prompt(change, phase):
    media = media_fixture()
    follower, _ = runner_fixture()
    initial = transported(plan_for(media, 0, 5))
    follower._validate_plan(initial)
    follower._media_embeddings(initial, 0, torch.tensor([media.token_ids[:5]]))
    if change == "pixels":
        original = media.images[0]
        image = DeepSeekV41Image(
            original.start,
            original.patches + 1,
            original.n_vit_h,
            original.n_vit_w,
            original.types,
        )
        changed = DeepSeekV41Media(media.token_ids, (image,))
        assert changed.token_ids == media.token_ids
        assert changed.digest != media.digest
    else:
        # The digest covers image inputs. Changing text outside the next chunk
        # must still invalidate the live sequence, even with identical images.
        changed = DeepSeekV41Media(media.token_ids[:-1] + (11,), media.images)
        assert changed.digest == media.digest
    plan = (
        plan_for(changed, 5, 7)
        if phase == BatchPhase.PREFILL
        else BatchPlan(
            [42],
            [len(media.token_ids)],
            [21],
            [0, 1],
            phase=phase,
            media=[changed],
        )
    )
    with pytest.raises(ValueError, match="changed its media or prompt"):
        follower._validate_plan(transported(plan))
