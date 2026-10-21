from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from resin import Engine, MediaInput, MediaTokenSpan
from resin.models.gemma4.media import Gemma4Media
from resin.server.media import decode_image


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_encoded_image_boundary_is_bytes_plus_explicit_mime() -> None:
    image = decode_image(_png_bytes(), "image/png")

    assert image.mode == "RGB"
    assert image.size == (4, 3)
    with pytest.raises(ValueError, match="media type"):
        decode_image(_png_bytes(), "application/octet-stream")


def test_media_owns_processor_tensors_and_absolute_placeholder_mapping() -> None:
    pixels = torch.ones(1, 9, 12)
    output = {
        "input_ids": torch.tensor([[2, 8, 8, 3, 9, 9]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 2, 2]]),
        "pixel_values": pixels,
        "image_position_ids": torch.zeros(1, 9, 2, dtype=torch.int64),
        "pixel_values_videos": torch.ones(1, 1, 9, 12),
        "video_position_ids": torch.zeros(1, 1, 9, 2, dtype=torch.int64),
    }

    media = Gemma4Media.from_processor_output(output)
    pixels.zero_()

    assert isinstance(media, MediaInput)
    assert media.token_ids == (2, 8, 8, 3, 9, 9)
    assert media.image_positions == (1, 2)
    assert media.video_positions == (4, 5)
    assert media.block_ids == (-1, 0, 0, -1, 1, 1)
    assert media.token_spans == (
        MediaTokenSpan(position=1, length=2),
        MediaTokenSpan(position=4, length=2),
    )
    assert media.pixel_values is not None
    assert media.pixel_values.sum().item() == 108
    media.validate_placeholder_tokens(image_token_id=8, video_token_id=9)
    assert len(media.digest) == 64
    assert len(media.cache_key) == len(media.token_ids)
    assert isinstance(hash(media.cache_key), int)
    assert media.cache_key[0] == media.token_ids[0]
    assert media.cache_key[1] != media.token_ids[1]


def test_canonical_normalized_arrays_are_copied_before_processing() -> None:
    class Processor:
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

    image = np.ones((3, 4, 3), dtype=np.float32)
    media = Gemma4Media.from_images(Processor(), [image], prompt="describe")
    image.fill(0)

    assert Processor.seen is not None
    passed = Processor.seen[0][0]["content"][0]["image"]
    assert passed.sum() == 36
    assert Processor.seen[1]["processor_kwargs"] == {"do_rescale": False}
    assert media.image_positions == (1,)


def test_submit_media_uses_its_tokenized_prompt_without_engine_media_logic() -> None:
    class Runner:
        plans = []

        @classmethod
        def forward(cls, plan):
            cls.plans.append(plan)
            return [7]

    media = Gemma4Media(
        token_ids=(2, 8),
        mm_token_type_ids=(0, 1),
        pixel_values=torch.ones(1, 9, 12),
        image_position_ids=torch.zeros(1, 9, 2, dtype=torch.int64),
    )
    engine = Engine(Runner())

    engine.submit(media=media, max_new_tokens=1)
    engine.step()

    assert Runner.plans[0].token_ids.tolist() == [2, 8]
    assert Runner.plans[0].media == (media,)
