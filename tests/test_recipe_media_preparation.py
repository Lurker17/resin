"""Pixel strategy contracts independent of native chat formatting."""

from __future__ import annotations

import base64
import hashlib
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from resin._recipe_media import (
    _DeepSeekPreparer,
    _ProcessorPreparer,
    digest_tensors,
    make_preparer,
    normalize_image,
)
from resin.models.deepseek_v41.media import DeepSeekV41Media
from resin.models.gemma4.media import Gemma4Media
from resin.models.inkling.media import InklingMedia
from resin.models.qwen3_5.media import Qwen35Media
from resin.protocols import MediaTokenSpan


def test_image_sources_share_one_owned_rgb_boundary() -> None:
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    raw = encoded.getvalue()
    expected = pixels.astype(np.float32) / np.float32(255)
    sources = (
        expected,
        (raw, "image/png"),
        "data:image/png;base64," + base64.b64encode(raw).decode(),
    )
    for source in sources:
        image = normalize_image(source)
        assert np.array_equal(image, expected)
        assert image.flags.c_contiguous
        assert not np.shares_memory(image, expected)


def test_unsupported_media_rejected_before_processor_runs() -> None:
    processor = SimpleNamespace(tokenizer=object())
    for model in ("inkling", "qwen3_5"):
        preparer = make_preparer(model, processor)
        with pytest.raises(ValueError, match="does not support video"):
            preparer.prepare([("video", np.zeros((2, 2, 2, 3), dtype=np.float32))])
    with pytest.raises(ValueError, match="requires a pixel processor"):
        make_preparer("inkling", object()).prepare(
            [("image", np.zeros((2, 3, 3), dtype=np.float32))]
        )


def test_gemma_preparation_preserves_interleaved_geometry_and_atomic_spans() -> None:
    seen: dict[str, Any] = {}
    pixels = torch.ones((2, 1, 6))
    video_pixels = torch.ones((1, 2, 1, 6))

    def images(**kwargs):
        seen["images"] = kwargs["images"]
        return {"pixel_values": pixels, "image_position_ids": torch.zeros((2, 1, 2))}

    def videos(**kwargs):
        seen["videos"] = kwargs["videos"]
        return {
            "pixel_values_videos": video_pixels,
            "video_position_ids": torch.zeros((1, 2, 1, 2)),
            "video_metadata": ["temporary"],
        }

    processor = SimpleNamespace(
        tokenizer=object(),
        image_processor=images,
        video_processor=videos,
        image_token="<image>",
        video_token="<video>",
        replace_image_token=lambda prepared, image_idx: f"image{image_idx}",
        replace_video_token=lambda prepared, video_idx: f"video{video_idx}",
        create_mm_token_type_ids=lambda rows: [
            [1 if token == 8 else 2 if token == 9 else 0 for token in rows[0]]
        ],
    )
    image = np.ones((2, 3, 3), dtype=np.float32)
    frames = np.ones((2, 2, 3, 3), dtype=np.float32)
    preparer = make_preparer("gemma4", processor)
    assert isinstance(preparer, _ProcessorPreparer)
    replacements, state = preparer.prepare(
        [("image", image), ("video", frames), ("image", image * 0.5)]
    )
    assert replacements == {"<image>": ["image0", "image1"], "<video>": ["video0"]}
    assert "video_metadata" not in state
    assert np.array_equal(seen["images"][1], image * 0.5)
    assert not np.shares_memory(seen["images"][0], image)
    assert not np.shares_memory(seen["videos"][0], frames)
    media = preparer.finalize(state, [0, 8, 9, 9, 8, 0])
    assert isinstance(media, Gemma4Media)
    assert media.image_positions == (1, 4)
    assert media.video_positions == (2, 3)
    assert media.token_spans == (MediaTokenSpan(1, 4),)
    assert media.cache_key[0] == media.cache_key[-1] == 0
    pixels.zero_()
    video_pixels.zero_()
    assert media.pixel_values is not None and media.pixel_values.sum() == 12
    assert (
        media.pixel_values_videos is not None and media.pixel_values_videos.sum() == 12
    )


def test_qwen_and_inkling_bind_ids_without_temporary_token_tensors() -> None:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 8)
    qwen = make_preparer(
        "qwen3_5",
        SimpleNamespace(
            tokenizer=tokenizer, image_processor=SimpleNamespace(merge_size=2)
        ),
    )
    assert isinstance(qwen, _ProcessorPreparer)
    media = qwen.finalize(
        {
            "pixel_values": torch.ones(4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        [1, 8, 2],
    )
    assert isinstance(media, Qwen35Media)
    assert media.image_positions == (1,)
    assert media.position_ids.tolist() == [[0, 1, 2]] * 3
    assert media.token_spans == ()
    inkling = make_preparer(
        "inkling", SimpleNamespace(tokenizer=tokenizer, image_token_id=8)
    )
    assert isinstance(inkling, _ProcessorPreparer)
    media = inkling.finalize({"pixel_values": torch.ones(2, 1, 3, 2, 3)}, [1, 8, 8, 2])
    assert isinstance(media, InklingMedia)
    assert media.image_positions == (1, 2)
    assert media.token_spans == ()


def test_deepseek_preparation_preserves_geometry_positions_and_cache_identity() -> None:
    tokenizer = SimpleNamespace(token_to_id=lambda token: 8)
    preparer = make_preparer(
        "deepseek_v41", tokenizer, vision_config={"min_pixels": 0}, image_token_id=8
    )
    assert isinstance(preparer, _DeepSeekPreparer)
    source = np.ones((14, 14, 3), dtype=np.float32)
    replacements, state = preparer.prepare([("image", source), ("image", source)])
    assert len(next(iter(replacements.values()))) == 2
    media = preparer.finalize(state, [1, 8, 8, 8, 8, 2, 8, 8, 8, 8, 3])
    assert isinstance(media, DeepSeekV41Media)
    assert [image.start for image in media.images] == [1, 6]
    assert media.token_spans == ()
    assert media.cache_key[0] == 1 and media.cache_key[5] == 2
    repeated = preparer.finalize(state, list(media.token_ids))
    assert isinstance(repeated, DeepSeekV41Media)
    assert media.digest == repeated.digest
    with pytest.raises(ValueError, match="contiguous"):
        preparer.finalize(state, [1, 8, 8, 8, 2, 8, 8, 8, 8, 8, 3])


def test_buffer_hash_preserves_existing_digest_encoding() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16).T
    expected = hashlib.sha256(b"namespace")
    for part in (b"metadata", b"pixels", b"torch.bfloat16", b"(2, 2)"):
        expected.update(part)
    expected.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    assert (
        digest_tensors(b"namespace", [("pixels", tensor)], metadata=[b"metadata"])
        == expected.hexdigest()
    )


def test_gemma_text_only_skips_media_embeddings_and_custom_attention_mask() -> None:
    import resin
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

    runner = object.__new__(Gemma4CudaRunner)
    native: Any = SimpleNamespace(sequence_ids=[7, 8], media=(None, None))
    batch: Any = SimpleNamespace(phase=resin.BatchPhase.PREFILL)
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = hidden.clone()
    features = runner._merge_media_embeddings(native, batch, hidden)
    assert features == {}
    assert torch.equal(hidden, expected)
    assert runner._local_custom_mask(batch, features) is None
