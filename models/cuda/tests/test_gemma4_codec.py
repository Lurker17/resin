from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

import resin.models.gemma4.codec as codec_module
from resin.models.gemma4.codec import Gemma4Codec
from resin.models.gemma4.media import Gemma4Media
from resin.server.media import normalized_rgb_from_data_url


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


def test_codec_attaches_media_and_keeps_codec_work_out_of_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [12]

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(map(str, token_ids))

    class ImageProcessor:
        seen = None

        def __call__(self, **kwargs):
            self.seen = kwargs
            return {
                "pixel_values": torch.ones(1, 9, 12),
                "image_position_ids": torch.zeros(1, 9, 2, dtype=torch.int64),
            }

    class Processor:
        tokenizer = Tokenizer()
        image_token = "<|image|>"
        image_processor = ImageProcessor()

        def replace_image_token(self, output, image_idx):
            return self.image_token

        def create_mm_token_type_ids(self, ids):
            return [[int(token == 8) for token in row] for row in ids]

    parser = object()

    class RecordingRecipe:
        def __init__(self, tokenizer, *, model):
            assert model == "gemma4"

        def render_media_ids(self, request, replacements):
            assert replacements == {"<|image|>": ["<|image|>"]}
            assert request["extra"] == {"enable_thinking": True}
            return [2, 8]

        def create_parser(self, request, prompt_ids):
            return parser

    monkeypatch.setattr(codec_module, "Recipe", RecordingRecipe)
    processor = Processor()
    codec = Gemma4Codec(processor, eos_token_ids=(1, 106))
    rendered = codec.render(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url()}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            "stop": ["stop"],
            "extra": {"enable_thinking": True},
        }
    )

    assert isinstance(rendered["media"], Gemma4Media)
    assert rendered["token_ids"] == [2, 8]
    assert rendered["stop_token_ids"] == [1, 106]
    assert processor.image_processor.seen is not None
    assert processor.image_processor.seen["images"][0].dtype == np.float32
    assert processor.image_processor.seen["do_rescale"] is False
    assert codec.create_parser({}, []) is parser
