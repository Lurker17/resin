from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from resin_models_cuda.gemma4.codec import Gemma4Codec, normalized_rgb_from_data_url
from resin_models_cuda.gemma4.media import Gemma4Media


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


def test_codec_attaches_media_and_keeps_codec_work_out_of_engine() -> None:
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [12]

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(map(str, token_ids))

    class Processor:
        tokenizer = Tokenizer()
        seen = None

        @classmethod
        def apply_chat_template(cls, messages, **kwargs):
            cls.seen = (messages, kwargs)
            return {
                "input_ids": torch.tensor([[2, 8]]),
                "mm_token_type_ids": torch.tensor([[0, 1]]),
                "pixel_values": torch.ones(1, 9, 12),
                "image_position_ids": torch.zeros(1, 9, 2, dtype=torch.int64),
            }

    codec = Gemma4Codec(Processor(), eos_token_ids=(1, 106))
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
        }
    )

    assert isinstance(rendered["media"], Gemma4Media)
    assert rendered["token_ids"] == [2, 8]
    assert rendered["stop_token_ids"] == [1, 106, 12]
    image = Processor.seen[0][0]["content"][0]["image"]
    assert image.dtype == np.float32
    assert Processor.seen[1]["processor_kwargs"] == {"do_rescale": False}

    parser = codec.create_parser({}, [])
    assert parser.feed([3]) == [{"type": "text_delta", "delta": "3"}]
    assert parser.feed([4]) == [{"type": "text_delta", "delta": "4"}]

