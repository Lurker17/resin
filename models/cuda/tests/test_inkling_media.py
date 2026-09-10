from __future__ import annotations

import pytest
import torch

from resin.models.inkling.media import InklingMedia


def test_media_owns_pixels_and_salts_only_image_placeholders() -> None:
    pixels = torch.arange(2 * 2 * 40 * 40 * 3, dtype=torch.float32).reshape(
        2, 2, 40, 40, 3
    )
    media = InklingMedia(
        token_ids=(7, 200054, 8, 200054, 9),
        image_token_id=200054,
        pixel_values=pixels,
    )
    pixels.zero_()

    assert media.image_positions == (1, 3)
    assert media.token_spans == ()
    assert media.pixel_values is not None
    assert torch.count_nonzero(media.pixel_values)
    assert len(media.cache_key) == len(media.token_ids)
    assert media.cache_key[0] == 7
    assert media.cache_key[2] == 8
    assert media.cache_key[4] == 9
    assert media.cache_key[1] != 200054
    assert media.cache_key[3] != 200054


def test_media_requires_one_patch_per_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder count"):
        InklingMedia(
            token_ids=(200054, 200054),
            image_token_id=200054,
            pixel_values=torch.zeros(1, 2, 40, 40, 3),
        )
