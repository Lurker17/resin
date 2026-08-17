from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from renderers.base import MultiModalData, PlaceholderRange, RenderedTokens

from resin import MediaInput
from resin_models_cuda.qwen3_5.codec import Qwen35Codec
from resin_models_cuda.qwen3_5.media import Qwen35Media


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


def test_qwen_codec_normalizes_images_and_attaches_renderer_sidecar() -> None:
    class Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<|image_pad|>": 8, "<think>": 9}[token]

        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [12]

    class Renderer:
        seen = None

        @classmethod
        def render(cls, messages, **kwargs):
            cls.seen = (messages, kwargs)
            return RenderedTokens(
                token_ids=[2, 8, 8, 8, 8, 3],
                multi_modal_data=MultiModalData(
                    mm_placeholders={"image": [PlaceholderRange(1, 4)]},
                    mm_items={
                        "image": [
                            {
                                "pixel_values": np.ones((16, 12), np.float32),
                                "image_grid_thw": np.array([[1, 4, 4]], np.int64),
                            }
                        ]
                    },
                ),
            )

        @staticmethod
        def get_stop_token_ids():
            return [1]

    processor = SimpleNamespace(
        tokenizer=Tokenizer(), image_processor=SimpleNamespace(merge_size=2)
    )
    codec = Qwen35Codec(Renderer(), processor)
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
    assert isinstance(Renderer.seen[0][0]["content"][0]["image"], Image.Image)
    assert Renderer.seen[1] == {"tools": None, "add_generation_prompt": True}
