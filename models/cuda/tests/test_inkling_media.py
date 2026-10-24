from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from resin.models.inkling.media import InklingMedia


def test_media_owns_pixels_and_salts_only_image_placeholders() -> None:
    pixels = np.arange(2 * 2 * 40 * 40 * 3, dtype=np.float32).reshape(2, 2, 40, 40, 3)
    media = InklingMedia(
        token_ids=(7, 200054, 8, 200054, 9),
        image_token_id=200054,
        pixel_values=pixels,
    )
    pixels.fill(0)

    assert media.image_positions == (1, 3)
    assert media.token_spans == ()
    assert media.pixel_values is not None
    assert np.count_nonzero(media.pixel_values)
    assert not np.shares_memory(media.pixel_values, pixels)
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
            pixel_values=np.zeros((1, 2, 40, 40, 3), dtype=np.float32),
        )


def test_runner_converts_native_pixels_once_and_reuses_features() -> None:
    from resin_models_cuda.inkling.cuda import InklingCudaRunner

    seen = []

    def vision(pixels):
        seen.append(pixels)
        return torch.ones((1, 3), dtype=pixels.dtype)

    runner: Any = object.__new__(InklingCudaRunner)
    runner.device = torch.device("cpu")
    runner.config = SimpleNamespace(hidden_size=3)
    runner._model = SimpleNamespace(vision=vision)
    runner._media_features = {}
    media = InklingMedia(
        token_ids=(7, 200054, 9),
        image_token_id=200054,
        pixel_values=np.full((1, 2, 40, 40, 3), 0.5, dtype=np.float32),
    )

    features = runner._features_for(1, media)
    assert runner._features_for(1, media) is features
    assert len(seen) == 1
    assert seen[0].dtype == torch.bfloat16
    assert tuple(seen[0].shape) == (1, 2, 40, 40, 3)
    assert np.all(media.pixel_values == 0.5)
