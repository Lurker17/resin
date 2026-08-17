from __future__ import annotations

import numpy as np

from resin.server.media import canonical_rgb_image, normalize_image_messages


def test_image_helpers_return_owned_processor_inputs_without_mutating_messages() -> (
    None
):
    image = np.ones((2, 3, 3), dtype=np.float32)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image": image},
                {"type": "input_text", "text": "describe"},
            ],
        }
    ]

    normalized = normalize_image_messages(messages)
    normalized_image = normalized[0]["content"][0]["image"]

    assert messages[0]["content"][0]["type"] == "input_image"
    assert normalized[0]["content"][0]["type"] == "image"
    assert normalized[0]["content"][1] == {"type": "text", "text": "describe"}
    assert normalized_image is not image
    assert normalized_image.flags.c_contiguous

    image.fill(0)
    assert normalized_image.sum() == 18


def test_canonical_rgb_image_rejects_implicit_dtype_conversion() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    try:
        canonical_rgb_image(image)
    except TypeError as error:
        assert "dtype float32" in str(error)
    else:
        raise AssertionError("uint8 input must be rejected")
