from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from resin_models_cuda.kimi_k25.media import KimiK25Media


def _rendered(*, first_hash: str = "first") -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=[10, 99, 11, 99, 12],
        multi_modal_data=SimpleNamespace(
            mm_placeholders={
                "image": [
                    SimpleNamespace(offset=1, length=1),
                    SimpleNamespace(offset=3, length=1),
                ]
            },
            mm_items={
                "image": [
                    {
                        "pixel_values": np.zeros((24, 3, 14, 14), np.float32),
                        "grid_thws": np.array([[1, 4, 6]], np.int64),
                    },
                    {
                        "pixel_values": np.zeros((4, 3, 14, 14), np.float32),
                        "grid_thws": np.array([[1, 2, 2]], np.int64),
                    },
                ]
            },
            mm_hashes={"image": [first_hash, "second"]},
        ),
    )


def test_kimi_media_expands_patch_placeholders_for_the_scheduler() -> None:
    media = KimiK25Media.from_rendered(_rendered())

    assert media.token_ids == (10, *([99] * 6), 11, 99, 12)
    assert [(span.position, span.length) for span in media.image_spans] == [
        (1, 6),
        (8, 1),
    ]
    assert media.pixel_values.shape == (28, 3, 14, 14)
    assert media.grid_thws.tolist() == [[1, 4, 6], [1, 2, 2]]
    assert media.token_spans == ()


def test_kimi_media_hash_salts_expanded_prefix_identity() -> None:
    first = KimiK25Media.from_rendered(_rendered(first_hash="one"))
    second = KimiK25Media.from_rendered(_rendered(first_hash="two"))

    assert first.cache_key[0] == second.cache_key[0] == 10
    assert first.cache_key[1:7] != second.cache_key[1:7]
    assert first.cache_key[7:] == second.cache_key[7:]


def test_kimi_media_identity_survives_tensor_parallel_serialization() -> None:
    media = KimiK25Media.from_rendered(_rendered())
    restored = pickle.loads(pickle.dumps(media))

    assert restored is not media
    assert restored.cache_key == media.cache_key
    assert restored.image_hashes == media.image_hashes
