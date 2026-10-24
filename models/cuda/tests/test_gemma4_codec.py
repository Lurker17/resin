from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image
from recipe_fuzz.gemma import GemmaAdapter

import resin
from resin.models.gemma4.media import Gemma4Media
from resin.recipe import Gemma4Recipe


def _data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 1), (255, 128, 0)).save(output, format="PNG")
    payload = base64.b64encode(output.getvalue()).decode()
    return f"data:image/png;base64,{payload}"


@pytest.mark.parametrize("source_kind", ["data_url", "bytes", "rgb"])
def test_recipe_attaches_media_and_preserves_token_controls(source_kind) -> None:
    tokenizer = GemmaAdapter().codec.tokenizer
    image_id = tokenizer.token_to_id("<|image|>")

    model_recipe = Gemma4Recipe(
        tokenizer,
        eos_token_ids=(1, 106),
        media_config={
            "image_processor": {
                "patch_size": 1,
                "pooling_kernel_size": 1,
                "max_soft_tokens": 70,
                "do_resize": False,
            }
        },
    )
    data_url = _data_url()
    image = {
        "data_url": lambda: resin.recipe.Image(data_url),
        "bytes": lambda: resin.recipe.Image.from_bytes(
            base64.b64decode(data_url.split(",", 1)[1]), media_type="image/png"
        ),
        "rgb": lambda: resin.recipe.Image.from_rgb(
            np.array([[[1.0, 128 / 255, 0.0]] * 2], dtype=np.float32)
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
    assert rendered.token_ids.count(image_id) == 2
    assert rendered.stop_token_ids == [1, 106]
    pixels = rendered.media.pixel_values
    assert pixels is not None
    assert pixels.dtype == np.float32
    assert pixels.shape == (1, 70, 3)
    np.testing.assert_array_equal(
        pixels[0, :2], np.array([[1.0, 128 / 255, 0.0]] * 2, dtype=np.float32)
    )
    assert not np.any(pixels[0, 2:])
    assert model_recipe.create_parser(rendered) is not model_recipe.create_parser(
        rendered
    )
