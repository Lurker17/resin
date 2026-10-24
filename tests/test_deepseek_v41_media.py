"""Image codec equivalence, cache identity and scheduler-neutral ownership."""

import base64
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from deepseek_recipe import ImageInfo, MultiModalData
from PIL import Image
from recipe_fuzz.deepseek import DeepSeekV41Adapter
from tokenizers import Tokenizer

import resin
from resin import Engine, MediaInput, RunnerConfig, SchedulerConfig
from resin._recipe_media import digest_tensors
from resin.models.deepseek_v41 import (
    DeepSeekV41Image,
    DeepSeekV41Media,
)
from resin.models.deepseek_v41.media import (
    IMAGE_TOKEN,
    DeepSeekV41ImageProcessor,
    image_token_types,
)
from resin.recipe import DeepSeekV41Recipe

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_v41_images.json").read_text()
)


def _pattern(width, height):
    y, x, channel = np.indices((height, width, 3), dtype=np.int32)
    return ((17 * x + 29 * y + 71 * channel) % 256).astype(np.uint8)


def _data_url(rgb):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


@pytest.fixture(scope="module")
def codec():
    tokenizer = Tokenizer.from_str(DeepSeekV41Adapter().codec.tokenizer.to_str())
    tokenizer.add_special_tokens([IMAGE_TOKEN])
    return DeepSeekV41Recipe(tokenizer)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_pixels_and_patch_order_match_published_processor(case):
    processor = DeepSeekV41ImageProcessor(max_wh_ratio=case["max_wh_ratio"])
    rgb = _pattern(case["width"], case["height"])
    image = processor.prepare(rgb.astype(np.float32) / 255)
    assert (image.n_vit_h, image.n_vit_w) == tuple(case["grid"])
    assert image.types == image_token_types(*case["llm_grid"])
    assert len(image.types) <= 1024
    assert image.patches.dtype == torch.bfloat16
    assert image.patches.is_contiguous()
    assert (
        hashlib.sha256(image.patches.view(torch.uint8).numpy().tobytes()).hexdigest()
        == case["patch_sha256"]
    )


def test_geometry_matches_official_recipe_for_varied_aspect_ratios():
    randomizer = random.Random(1729)
    processor = DeepSeekV41ImageProcessor()
    for _ in range(300):
        width, height = (randomizer.randint(1, 65536) for _ in range(2))
        h, w = processor.grid(width, height)
        expected = (
            MultiModalData([ImageInfo(b"", width, height)]).image_token_adjustment() + 1
        )
        assert len(image_token_types((h + 2) // 3, (w + 2) // 3)) == expected


def test_media_codec_expands_only_explicit_images_and_owns_inputs(codec):
    rgb = _pattern(17, 31)
    second = np.zeros((2, 4, 3), dtype=np.float32)
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.USER,
            [
                resin.recipe.Text("literal " + IMAGE_TOKEN),
                resin.recipe.Image(_data_url(rgb)),
                resin.recipe.Text("compare"),
                resin.recipe.Image(second),
            ],
        )
    ]
    rendered = codec.render(
        messages,
        options=resin.recipe.RenderOptions(
            thinking_mode="thinking", reasoning_effort=37
        ),
    )
    media = rendered.media
    assert isinstance(media, MediaInput)
    assert isinstance(media, DeepSeekV41Media)
    assert len(media.images) == 2
    assert media.token_ids == tuple(rendered.token_ids)
    assert rendered.stop_token_ids == codec.get_stop_token_ids()
    assert media.token_spans == ()
    positions = {
        position
        for image in media.images
        for position in range(image.start, image.start + len(image.types))
    }
    assert rendered.token_ids.count(media.image_token_id) == len(positions)
    assert all(
        (token != key) == (index in positions)
        for index, (token, key) in enumerate(
            zip(media.token_ids, media.cache_key, strict=True)
        )
    )
    assert IMAGE_TOKEN in codec.tokenizer.decode(
        rendered.token_ids[: media.images[0].start], skip_special_tokens=False
    )
    assert codec.render_ids(messages) == codec.render(messages).token_ids
    second.fill(1)
    assert media.images[1].patches.max().item() < 0


def test_image_identity_changes_only_its_media_cache_identity(codec):
    def render(color):
        return codec.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [
                        resin.recipe.Image(np.full((2, 4, 3), color, dtype=np.float32)),
                        resin.recipe.Text("describe"),
                    ],
                )
            ]
        ).media

    black, white, black_again = render(0), render(1), render(0)
    assert black.token_ids == white.token_ids
    assert black.cache_key != white.cache_key
    assert black.digest == black_again.digest
    assert black.cache_key == black_again.cache_key


def test_prepared_media_rejects_bad_layout_and_owns_patch_storage():
    patches = torch.ones((1, 3, 14, 14), dtype=torch.bfloat16)
    image = DeepSeekV41Image(1, patches, 1, 1, (0, 1, 2, 3))
    patches.zero_()
    assert image.patches.min().item() == 1
    media = DeepSeekV41Media((7, 129264, 129264, 129264, 129264), (image,))
    with pytest.raises(ValueError, match="different placeholder"):
        DeepSeekV41Media((7, 129264, 129264, 129264, 8), (image,))
    with pytest.raises(ValueError, match="outside an image span"):
        DeepSeekV41Media((*media.token_ids, 129264), (image,))
    with pytest.raises(ValueError, match="3x3 aligner"):
        DeepSeekV41Image(1, image.patches, 1, 1, (0, 1, 3))
    with pytest.raises(ValueError, match="ordered, disjoint"):
        DeepSeekV41Media(media.token_ids, (image, image))


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32, torch.int64]
)
def test_shared_digest_supports_torch_dtypes_and_scalar_shapes(dtype):
    tensor = torch.tensor([[0, 1], [2, 3]]).to(dtype)
    assert digest_tensors(b"test", [("x", tensor)]) == digest_tensors(
        b"test", [("x", tensor.clone())]
    )
    assert digest_tensors(b"test", [("x", tensor)]) != digest_tensors(
        b"test", [("x", tensor.T)]
    )
    assert digest_tensors(b"test", [("x", tensor[0, 0])]) != digest_tensors(
        b"test", [("x", tensor[0, :1])]
    )


def test_engine_slices_causal_image_spans_across_prefill_chunks(codec):
    class Runner:
        runner_config = RunnerConfig(continuation_prefill=True)
        contexts = []

        @classmethod
        def forward(cls, plan):
            cls.contexts.append(plan)
            return [7]

    rendered = codec.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [
                    resin.recipe.Image(np.zeros((2, 4, 3), dtype=np.float32)),
                    resin.recipe.Text("describe"),
                ],
            )
        ]
    )
    engine = Engine(Runner(), scheduler_config=SchedulerConfig(prefill_chunk_size=32))
    engine.submit(media=rendered.media, max_new_tokens=1)
    while engine.has_work():
        engine.step()
    assert len(Runner.contexts) > 1
    assert all(plan.media == (rendered.media,) for plan in Runner.contexts)
    assert sum(plan.num_tokens for plan in Runner.contexts) == len(rendered.token_ids)
