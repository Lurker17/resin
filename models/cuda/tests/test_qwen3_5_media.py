from __future__ import annotations

import base64
import io

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
        pixel_values=torch.ones(16, 12),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
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


def _data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_qwen_recipe_normalizes_images_and_attaches_processor_payload() -> None:
    tokenizer = QwenAdapter().codec.tokenizer
    image_id = tokenizer.token_to_id("<|image_pad|>")

    class ImageProcessor:
        merge_size = 2
        seen = None

        def __call__(self, **kwargs):
            self.seen = kwargs
            return {
                "pixel_values": torch.ones(16, 12),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }

    class Processor:
        image_processor = ImageProcessor()
        image_token = "<|image_pad|>"

        def __init__(self):
            self.tokenizer = tokenizer

        def replace_image_token(self, output, image_idx):
            return self.image_token * 4

    processor = Processor()
    model_recipe = Qwen35Recipe(processor)
    output = model_recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image(_data_url()), resin.recipe.Text("describe")],
            ),
        ]
    )
    assert output.stop_token_ids == model_recipe.get_stop_token_ids()
    assert isinstance(output.media, Qwen35Media)
    assert output.token_ids.count(image_id) == 4
    assert processor.image_processor.seen is not None
    assert processor.image_processor.seen["images"][0].dtype == np.float32
    assert processor.image_processor.seen["return_tensors"] == "pt"
    assert processor.image_processor.seen["do_rescale"] is False


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
