from __future__ import annotations

import numpy as np

from resin._recipe_media import canonical_rgb_image, normalize_image


def test_image_helpers_return_owned_processor_inputs():
    image = np.ones((2, 3, 3), dtype=np.float32)
    normalized = normalize_image(image)
    assert normalized is not image
    assert normalized.flags.c_contiguous
    image.fill(0)
    assert normalized.sum() == 18


def test_canonical_rgb_image_rejects_implicit_dtype_conversion() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    try:
        canonical_rgb_image(image)
    except TypeError as error:
        assert "dtype float32" in str(error)
    else:
        raise AssertionError("uint8 input must be rejected")
