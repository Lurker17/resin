from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image
from recipe_fuzz.gemma import GemmaAdapter

import resin
from resin._recipe_media import normalized_rgb_from_data_url
from resin.models.gemma4.media import Gemma4Media
from resin.recipe import Gemma4Recipe


def _data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 1), (255, 128, 0)).save(output, format="PNG")
    payload = base64.b64encode(output.getvalue()).decode()
    return f"data:image/png;base64,{payload}"


def test_data_url_adapter_ends_at_owned_normalized_rgb() -> None:
    image = normalized_rgb_from_data_url(_data_url())

    assert image.dtype == np.float32
    assert image.shape == (1, 2, 3)
    assert image.flags.c_contiguous
    assert image.min() == 0.0
    assert image.max() == 1.0
    with pytest.raises(ValueError, match="remote fetching"):
        normalized_rgb_from_data_url("https://example.com/image.png")


@pytest.mark.parametrize("source_kind", ["data_url", "bytes", "rgb"])
def test_recipe_attaches_media_and_preserves_token_controls(source_kind) -> None:
    tokenizer = GemmaAdapter().codec.tokenizer
    image_id = tokenizer.token_to_id("<|image|>")

    class ImageProcessor:
        seen = None

        def __call__(self, **kwargs):
            self.seen = kwargs
            return {
                "pixel_values": torch.ones(1, 9, 12),
                "image_position_ids": torch.zeros(1, 9, 2, dtype=torch.int64),
            }

    class Processor:
        image_token = "<|image|>"
        image_processor = ImageProcessor()

        def __init__(self):
            self.tokenizer = tokenizer

        def replace_image_token(self, output, image_idx):
            return self.image_token

        def create_mm_token_type_ids(self, ids):
            return [[int(token == image_id) for token in row] for row in ids]

    processor = Processor()
    model_recipe = Gemma4Recipe(processor, eos_token_ids=(1, 106))
    data_url = _data_url()
    image = {
        "data_url": lambda: resin.recipe.Image(data_url),
        "bytes": lambda: resin.recipe.Image.from_bytes(
            base64.b64decode(data_url.split(",", 1)[1]), media_type="image/png"
        ),
        "rgb": lambda: resin.recipe.Image.from_rgb(
            normalized_rgb_from_data_url(data_url)
        ),
    }[source_kind]()
    rendered = model_recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [image, resin.recipe.Text("describe")],
            ),
        ],
        options=resin.recipe.RenderOptions(enable_thinking=True),
    )
    assert isinstance(rendered.media, Gemma4Media)
    assert rendered.token_ids.count(image_id) == 1
    assert rendered.stop_token_ids == [1, 106]
    assert processor.image_processor.seen is not None
    assert processor.image_processor.seen["images"][0].dtype == np.float32
    assert processor.image_processor.seen["do_rescale"] is False
    assert model_recipe.create_parser(rendered) is not model_recipe.create_parser(
        rendered
    )
