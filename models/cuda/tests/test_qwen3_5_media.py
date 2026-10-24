from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from recipe_fuzz.qwen import QwenAdapter

import resin
from resin import MediaInput
from resin.models.qwen3_5.media import Qwen35Media
from resin.recipe import Qwen35Recipe


def _media() -> Qwen35Media:
    return Qwen35Media(
        token_ids=(2, 8, 8, 8, 8, 3),
        image_positions=(1, 2, 3, 4),
        pixel_values=np.ones((16, 12), dtype=np.float32),
        image_grid_thw=np.array([[1, 4, 4]], dtype=np.int64),
        spatial_merge_size=2,
        image_token_id=8,
    )


def test_qwen_media_owns_pixels_cache_identity_and_mrope_positions() -> None:
    media = _media()

    assert isinstance(media, MediaInput)
    assert media.token_spans == ()
    assert media.position_ids.tolist() == [
        [0, 1, 1, 1, 1, 3],
        [0, 1, 1, 2, 2, 3],
        [0, 1, 2, 1, 2, 3],
    ]
    assert media.rope_delta == -2
    assert len(media.digest) == 64
    assert media.cache_key[0] == media.token_ids[0]
    assert media.cache_key[1] != media.token_ids[1]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("pixel_values", torch.ones(16, 12)),
        ("pixel_values", np.ones((16, 12), dtype=np.float64)),
        ("image_grid_thw", torch.tensor([[1, 4, 4]])),
        ("image_grid_thw", np.array([[1, 4, 4]], dtype=np.int32)),
    ],
)
def test_qwen_media_requires_exact_numpy_storage(field: str, invalid: Any) -> None:
    arrays: dict[str, Any] = {
        "pixel_values": np.ones((16, 12), dtype=np.float32),
        "image_grid_thw": np.array([[1, 4, 4]], dtype=np.int64),
    }
    arrays[field] = invalid
    with pytest.raises(TypeError):
        Qwen35Media(
            (2, 8, 8, 8, 8, 3),
            (1, 2, 3, 4),
            **arrays,
            spatial_merge_size=2,
            image_token_id=8,
        )


def test_qwen_media_copies_strided_numpy_arrays_in_logical_order() -> None:
    pixels = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)[:, ::2]
    grid = np.array([[1, 99, 4, 99, 4, 99]], dtype=np.int64)[:, ::2]
    expected_pixels, expected_grid = pixels.copy(), grid.copy()
    assert not pixels.flags.c_contiguous and not grid.flags.c_contiguous
    media = Qwen35Media((2, 8, 8, 8, 8, 3), (1, 2, 3, 4), pixels, grid, 2, 8)
    pixels.fill(0)
    grid.fill(0)

    np.testing.assert_array_equal(media.pixel_values, expected_pixels)
    np.testing.assert_array_equal(media.image_grid_thw, expected_grid)
    assert not np.shares_memory(media.pixel_values, pixels)
    assert not np.shares_memory(media.image_grid_thw, grid)


def test_qwen_media_keeps_adjacent_images_distinct_for_mrope() -> None:
    media = Qwen35Media(
        token_ids=(2, 8, 8, 8, 8, 8, 8, 3),
        image_positions=(1, 2, 3, 4, 5, 6),
        pixel_values=np.ones((24, 12), dtype=np.float32),
        image_grid_thw=np.array([[1, 4, 4], [1, 2, 4]], dtype=np.int64),
        spatial_merge_size=2,
        image_token_id=8,
    )
    assert media.position_ids.tolist() == [
        [0, 1, 1, 1, 1, 3, 3, 5],
        [0, 1, 1, 2, 2, 3, 3, 5],
        [0, 1, 2, 1, 2, 3, 4, 5],
    ]
    assert media.rope_delta == -2


def _data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_qwen_recipe_normalizes_images_and_attaches_processor_payload() -> None:
    tokenizer = QwenAdapter().codec.tokenizer
    image_id = tokenizer.token_to_id("<|image_pad|>")

    model_recipe = Qwen35Recipe(
        tokenizer,
        media_config={
            "patch_size": 2,
            "temporal_patch_size": 1,
            "merge_size": 1,
            "min_pixels": 16,
            "max_pixels": 16,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        },
    )
    output = model_recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image(_data_url()), resin.recipe.Text("describe")],
            )
        ]
    )
    assert output.stop_token_ids == model_recipe.get_stop_token_ids()
    assert isinstance(output.media, Qwen35Media)
    assert output.token_ids.count(image_id) == 4
    assert output.media.pixel_values.shape == (4, 12)
    assert output.media.pixel_values.dtype == np.float32
    assert output.media.image_grid_thw.tolist() == [[1, 2, 2]]
    np.testing.assert_allclose(
        output.media.pixel_values, [[1.0] * 4 + [-1.0] * 8] * 4, atol=1e-6
    )


def test_native_qwen_rotary_positions_cover_prompt_and_decode_continuation():
    from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

    runner = object.__new__(Qwen3_5CudaRunner)
    runner.device = torch.device("cpu")
    media = _media()
    positions = torch.arange(8)[None]
    result = runner._prefill_rotary_positions(media, positions)
    expected = np.concatenate((media.position_ids, [[4, 5]] * 3), axis=1)
    np.testing.assert_array_equal(result[:, 0].numpy(), expected)


def test_qwen_recipe_rejects_video_instead_of_dropping_its_embeddings() -> None:
    model_recipe = Qwen35Recipe(QwenAdapter().codec.tokenizer)
    with pytest.raises(ValueError, match="does not support video"):
        model_recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [resin.recipe.Video(np.zeros((1, 2, 2, 3), dtype=np.float32))],
                )
            ]
        )
