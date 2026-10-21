from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

import resin.models.qwen3_5.codec as codec_module
from resin import MediaInput
from resin.models.qwen3_5.codec import Qwen35Codec
from resin.models.qwen3_5.media import Qwen35Media


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


def test_qwen_codec_normalizes_images_and_attaches_processor_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<|image_pad|>": 8}[token]

        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [12]

    class RecordingRecipe:
        request = None

        def __init__(self, tokenizer, *, model):
            assert model == "qwen3_5"

        @classmethod
        def render_media_ids(cls, request, replacements):
            cls.request = request
            assert replacements == {"<|image_pad|>": ["<|image_pad|>" * 4]}
            return [2, 8, 8, 8, 8, 3]

        @staticmethod
        def get_stop_token_ids():
            return [1]

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
        tokenizer = Tokenizer()
        image_processor = ImageProcessor()
        image_token = "<|image_pad|>"

        def replace_image_token(self, output, image_idx):
            return self.image_token * 4

    monkeypatch.setattr(codec_module, "Recipe", RecordingRecipe)
    processor = Processor()
    codec = Qwen35Codec(processor)
    output = codec.render(
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
            "stop": ["done"],
        }
    )

    assert output["stop_token_ids"] == [1, 12]
    assert isinstance(output["media"], Qwen35Media)
    assert RecordingRecipe.request is not None
    assert RecordingRecipe.request["messages"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "describe"},
    ]
    assert processor.image_processor.seen is not None
    assert processor.image_processor.seen["images"][0].dtype == np.float32
    assert processor.image_processor.seen["return_tensors"] == "pt"
    assert processor.image_processor.seen["do_rescale"] is False


def test_qwen_codec_rejects_video_instead_of_dropping_its_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = Qwen35Codec.__new__(Qwen35Codec)
    monkeypatch.setattr(codec, "processor", None, raising=False)
    monkeypatch.setattr(codec, "recipe", None, raising=False)
    monkeypatch.setattr(
        codec_module,
        "prepare_recipe_inputs",
        lambda *_: {"pixel_values_videos": torch.zeros(1, 1, 1)},
    )
    with pytest.raises(ValueError, match="image input only"):
        codec.render({"messages": []})
