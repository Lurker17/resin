from __future__ import annotations

import torch
from test_qwen3_5_config import _qwen38

from resin import CacheGroupFamily
from resin_models_cuda.qwen3_5.cache import CacheArena, CacheGeometry
from resin_models_cuda.qwen3_5.config import Qwen35Config


def test_qwen38_cache_geometry_accounts_for_fp32_recurrent_state() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    geometry = CacheGeometry.from_config(config, include_mtp=True)

    assert geometry.recurrent_bytes == 144 * 1024**2
    assert geometry.state_bytes == 153_954_304
    assert geometry.slab_bytes == 152 * 1024**2
    assert geometry.slab_units == 304
    assert geometry.target_blocks_per_slab == 19
    assert geometry.state_blocks_per_slab == 1
    assert geometry.mtp_blocks_per_slab == 304
    layout = geometry.layout(10)
    assert layout.groups[1].family is CacheGroupFamily.STATE
    assert len(layout.groups) == 3


def test_qwen35_cache_omits_mtp_group_when_speculation_is_disabled() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    geometry = CacheGeometry.from_config(config, include_mtp=False)
    assert len(geometry.layout(10).groups) == 2


def test_qwen38_fp8_kv_halves_each_attention_unit() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    bf16 = CacheGeometry.from_config(config, include_mtp=True)
    fp8 = CacheGeometry.from_config(config, include_mtp=True, kv_element_bytes=1)

    assert fp8.unit_bytes * 2 == bf16.unit_bytes
    assert fp8.target_blocks_per_slab == 37
    assert fp8.slab_bytes == 148 * 1024**2


def test_mixed_dtype_state_views_share_the_byte_arena() -> None:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    config = Qwen35Config.from_hugging_face(values)
    geometry = CacheGeometry.from_config(config, include_mtp=True)
    arena = CacheArena(torch, torch.device("cpu"), geometry.layout(2), geometry)

    assert arena.storage.dtype is torch.uint8
    assert arena.keys.dtype is torch.bfloat16
    assert arena.recurrent_states.dtype is torch.float32
    assert arena.convolution_states.dtype is torch.bfloat16
    assert arena.target_hidden_states.dtype is torch.bfloat16
    arena.recurrent_states[1, 0, 0, 0, 0, 0] = 7.0
    arena.convolution_states[1, 0, 0, 0, 0] = 3.0
    arena.target_hidden_states[1, 0, 0] = 5.0
    assert arena.recurrent_states[1, 0, 0, 0, 0, 0].item() == 7.0
    assert arena.convolution_states[1, 0, 0, 0, 0].item() == 3.0
    assert arena.target_hidden_states[1, 0, 0].item() == 5.0
