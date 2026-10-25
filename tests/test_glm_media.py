"""GLM's native image processor, payload ownership, and transport contract."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_glm_recipe import backend as backend

import resin._resin as native
from resin.recipe import (
    Glm53FlashMedia,
    Image,
    Message,
    Role,
    Text,
    Video,
    get_native_recipe,
)

FIXTURE = Path(__file__).parent / "recipe_fuzz/templates/glm53_images_hf.npz"


def render(backend, images, **config):
    recipe = get_native_recipe("glm5_3_flash", backend, media_config=config)
    return recipe.render(
        [Message(Role.USER, [Text("Inspect"), *[Image(source=i) for i in images]])]
    )


def test_native_processor_matches_pinned_hf_geometry_pixels_and_padding(backend):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata"]))
        assert (
            metadata["transformers_revision"]
            == "50e35cc0ca18ec19b80653690de553e6decf74f5"
        )
        for case in metadata["cases"]:
            name = case["name"]
            media = render(backend, [fixture[name + "_input"]], **case["config"]).media
            assert isinstance(media, Glm53FlashMedia)
            np.testing.assert_array_equal(media.image_grid_thw, fixture[name + "_grid"])
            np.testing.assert_allclose(
                media.pixel_values,
                fixture[name + "_pixels"],
                atol=1e-5,
                rtol=1e-5,
                err_msg=name,
            )
            assert len(media.image_positions) == np.prod(media.image_grid_thw) // 4


def test_image_order_cache_identity_and_distributed_transport(backend):
    first = np.zeros((28, 56, 3), dtype=np.float32)
    second = np.ones((56, 28, 3), dtype=np.float32)
    prompt = render(backend, [first, second], min_image_tokens=1)
    media = prompt.media
    assert isinstance(media, Glm53FlashMedia)
    assert media.image_grid_thw.tolist() == [[1, 2, 4], [1, 4, 2]]
    np.testing.assert_array_equal(
        media.pixel_values[:8],
        render(backend, [first], min_image_tokens=1).media.pixel_values,
    )
    np.testing.assert_array_equal(
        media.pixel_values[8:],
        render(backend, [second], min_image_tokens=1).media.pixel_values,
    )
    assert len(media.image_positions) == 4
    assert media.token_ids == tuple(prompt.token_ids)
    assert all(
        media.token_ids[p] == media.image_token_id for p in media.image_positions
    )
    assert media.image_positions[2] - media.image_positions[1] == 3
    assert not media.token_spans  # Chunked prefill can cross image boundaries.
    stored = media.pixel_values.copy()
    first.fill(0.5)
    second.fill(0.25)
    np.testing.assert_array_equal(stored, media.pixel_values)
    changed = render(backend, [first, second], min_image_tokens=1).media
    assert changed.token_ids == media.token_ids
    assert changed.digest != media.digest and changed.cache_key != media.cache_key
    for i, token in enumerate(media.token_ids):
        if i not in media.image_positions:
            assert media.cache_key[i] == token
    restored = native._media_from_buffers(*native._media_to_buffers(media))
    assert isinstance(restored, Glm53FlashMedia)
    assert restored.digest == media.digest and restored.cache_key == media.cache_key
    assert restored.image_positions == media.image_positions
    np.testing.assert_array_equal(restored.pixel_values, media.pixel_values)
    np.testing.assert_array_equal(restored.image_grid_thw, media.image_grid_thw)


def test_payload_rejects_mismatched_slots_grids_and_video(backend):
    media = render(
        backend, [np.zeros((28, 28, 3), np.float32)], min_image_tokens=1
    ).media
    values: dict[str, Any] = dict(
        token_ids=media.token_ids,
        image_positions=media.image_positions,
        pixel_values=media.pixel_values,
        image_grid_thw=media.image_grid_thw,
        spatial_merge_size=2,
        image_token_id=media.image_token_id,
    )
    for overrides in (
        {"image_positions": [0]},
        {"image_grid_thw": np.array([[2, 2, 2]], np.int64)},
        {"image_grid_thw": np.array([[1, 3, 2]], np.int64)},
        {"pixel_values": media.pixel_values[:-1]},
        {"spatial_merge_size": 0},
    ):
        with pytest.raises(ValueError, match="GLM"):
            arguments: dict[str, Any] = dict(values)
            arguments.update(overrides)
            Glm53FlashMedia(**arguments)
    recipe = get_native_recipe("glm5_3_flash", backend)
    with pytest.raises(ValueError, match="video"):
        recipe.render(
            [Message(Role.USER, [Video(source=np.zeros((2, 28, 28, 3), np.float32))])]
        )


@pytest.mark.parametrize(
    "config",
    [
        {"patch_size": 0},
        {"max_image_tokens": 0},
        {"min_image_tokens": 20, "max_image_tokens": 10},
        {"resample": 1},
        {"image_std": [1, 0, 1]},
        {"do_resize": False},
    ],
)
def test_invalid_image_geometry_is_rejected_before_packing(backend, config):
    with pytest.raises(ValueError, match="GLM"):
        render(backend, [np.zeros((7, 11, 3), np.float32)], **config)
