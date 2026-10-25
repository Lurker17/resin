from __future__ import annotations

import io

import numpy as np
import pytest
from output_helpers import forward_output
from PIL import Image
from recipe_fuzz.gemma import GemmaAdapter

import resin
from resin import Engine, MediaInput, MediaTokenSpan, SamplingParams
from resin.recipe import Gemma4Media, get_native_recipe


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_encoded_image_boundary_is_bytes_plus_explicit_mime() -> None:
    recipe = get_native_recipe(
        "gemma4",
        GemmaAdapter().recipe.tokenizer,
        media_config={
            "image_processor": {
                "patch_size": 1,
                "pooling_kernel_size": 1,
                "max_soft_tokens": 70,
                "do_resize": False,
            }
        },
    )
    prompt = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image.from_bytes(_png_bytes(), media_type="image/png")],
            )
        ]
    )
    assert isinstance(prompt.media, Gemma4Media)
    pixels = prompt.media.pixel_values
    assert pixels is not None
    assert pixels.shape == (1, 70, 3)
    np.testing.assert_allclose(
        pixels[0, :12], np.tile(np.array([10, 20, 30]) / 255, (12, 1))
    )
    with pytest.raises(ValueError, match="media type"):
        recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [
                        resin.recipe.Image.from_bytes(
                            _png_bytes(), media_type="application/octet-stream"
                        )
                    ],
                )
            ]
        )


def test_media_owns_processor_tensors_and_absolute_placeholder_mapping() -> None:
    pixels = np.ones((1, 9, 12), dtype=np.float32)
    media = Gemma4Media(
        token_ids=(2, 8, 8, 3, 9, 9),
        mm_token_type_ids=(0, 1, 1, 0, 2, 2),
        pixel_values=pixels,
        image_position_ids=np.zeros((1, 9, 2), dtype=np.int64),
        pixel_values_videos=np.ones((1, 1, 9, 12), dtype=np.float32),
        video_position_ids=np.zeros((1, 1, 9, 2), dtype=np.int64),
    )
    pixels.fill(0)

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


@pytest.mark.parametrize("is_video", [False, True])
def test_native_patches_match_reference_without_resize_and_own_input(is_video) -> None:
    from transformers.models.gemma4.image_processing_gemma4 import (
        Gemma4ImageProcessor,
        Gemma4ImageProcessorKwargs,
    )
    from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

    config: Gemma4ImageProcessorKwargs = {
        "patch_size": 2,
        "pooling_kernel_size": 1,
        "max_soft_tokens": 70,
        "do_resize": False,
        "do_normalize": True,
        "image_mean": [0.5, 0.25, 0.0],
        "image_std": [0.5, 0.25, 1.0],
    }
    image = (np.arange(24).reshape(2, 4, 3) / 24).astype(np.float32)
    if is_video:
        image = np.stack([image, image / 2], dtype=np.float32)
        expected = Gemma4VideoProcessor(**config, do_sample_frames=False)(
            videos=[image],
            do_rescale=False,
            return_tensors="np",
            input_data_format="channels_last",
        )
        block = resin.recipe.Video(source=image)
        field = "video_processor"
    else:
        expected = Gemma4ImageProcessor(**config)(
            images=[image],
            do_rescale=False,
            return_tensors="np",
            input_data_format="channels_last",
        )
        block = resin.recipe.Image.from_rgb(image)
        field = "image_processor"
    native_config = {**config, "do_sample_frames": False} if is_video else config
    recipe = get_native_recipe(
        "gemma4", GemmaAdapter().recipe.tokenizer, media_config={field: native_config}
    )
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.USER, [block, resin.recipe.Text("describe")]
        )
    ]
    prompt = recipe.render(messages)
    marker = "<|video|>" if is_video else "<|image|>"
    frame = "<|image>" + marker * 2 + "<image|>"
    expansion = " ".join(["00:00 " + frame] * 2) if is_video else frame
    expected_text = recipe.debug_render_text(messages).replace(marker, expansion)
    assert prompt.token_ids.tolist() == recipe.tokenizer.encode(
        expected_text, add_special_tokens=False
    )
    image.fill(0)
    assert isinstance(prompt.media, Gemma4Media)
    if is_video:
        np.testing.assert_array_equal(
            prompt.media.pixel_values_videos, expected["pixel_values_videos"]
        )
        np.testing.assert_array_equal(
            prompt.media.video_position_ids, expected["video_position_ids"]
        )
        assert len(prompt.media.video_positions) == 4
    else:
        np.testing.assert_array_equal(
            prompt.media.pixel_values, expected["pixel_values"]
        )
        np.testing.assert_array_equal(
            prompt.media.image_position_ids, expected["image_position_ids"]
        )
        assert len(prompt.media.image_positions) == 2


def test_submit_media_uses_its_tokenized_prompt_without_engine_media_logic() -> None:
    class Runner:
        plans = []

        @classmethod
        def forward(cls, plan):
            cls.plans.append(plan)
            return forward_output([7])

    media = Gemma4Media(
        token_ids=(2, 8),
        mm_token_type_ids=(0, 1),
        pixel_values=np.ones((1, 9, 12), dtype=np.float32),
        image_position_ids=np.zeros((1, 9, 2), dtype=np.int64),
    )
    engine = Engine(Runner())

    engine.submit(media=media, sampling_params=SamplingParams(max_output_tokens=1))
    engine.step()

    assert Runner.plans[0].token_ids.tolist() == [2, 8]
    assert Runner.plans[0].media == (media,)
