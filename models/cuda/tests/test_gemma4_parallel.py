from __future__ import annotations

import copy
from dataclasses import replace
from typing import cast

import pytest
import torch
from test_gemma4_assistant import _raw_config
from test_gemma4_model import _small_config

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.models.gemma4.assistant import (
    Gemma4AssistantAttention,
    Gemma4AssistantConfig,
)
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin_models_cuda.dense_parallel import TensorParallelSum
from resin_models_cuda.gemma4.cuda import _Gemma4KvArena, plan_gemma4_kv_capacity
from resin_models_cuda.gemma4.model import (
    Gemma4Attention,
    RotaryEmbedding,
)
from resin_models_cuda.gemma4.parallel import attention_geometry, shard_attention


class _Parallel:
    def __init__(self, rank, size):
        self.layout = TensorParallelLayout(rank, size)
        self.rank, self.size = rank, size
        self.enabled = size > 1
        self.calls = []
        self.other = None

    def sum(self, tensor):
        self.calls.append(tensor.clone())
        return tensor if self.other is None else tensor + self.other


def _config():
    return replace(
        _small_config(),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=8,
    )


def _attention(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1)[None],
        key.transpose(0, 1)[None],
        value.transpose(0, 1)[None],
        is_causal=True,
        scale=1.0,
        enable_gqa=True,
    )[0].transpose(0, 1)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("assistant", [False, True])
@torch.inference_mode()
def test_attention_tp_shards_kv_heads_and_reduces_output_before_norm(layer, assistant):
    torch.manual_seed(91)
    config = _config()
    target = Gemma4Attention(config, layer).float()
    if assistant:
        draft_config = replace(
            Gemma4AssistantConfig.from_dict(_raw_config()),
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            global_head_dim=config.global_head_dim,
            layer_types=config.layer_types,
        )
        full = Gemma4AssistantAttention(draft_config, layer).float()
    else:
        full = target
    shards, contexts = [], []
    for rank in range(2):
        parallel = _Parallel(rank, 2)
        local = copy.deepcopy(full)
        shard_attention(local, cast(TensorParallelContext, parallel))
        assert local.num_heads == 2
        assert isinstance(local.o_proj, TensorParallelSum)
        if isinstance(local, Gemma4Attention):
            assert local.num_kv_heads == (2 if local.is_local else 1)
            local.pack_projections()
        shards.append(local)
        contexts.append(parallel)
    target.pack_projections()
    hidden = torch.randn(7, config.hidden_size)
    rotary = RotaryEmbedding(config)
    rotary.materialize(torch.device("cpu"))
    cos, sin = rotary(torch.arange(7), config.layer_types[layer], hidden.dtype)
    target_query, key, value = target.project_qkv(hidden, cos, sin)
    query = (
        full.project_q(hidden, cos, sin)
        if isinstance(full, Gemma4AssistantAttention)
        else target_query
    )
    attended = _attention(query, key, value)
    expected = full.o_proj(attended.flatten(1))
    local_outputs = []
    for rank, shard in enumerate(shards):
        if assistant:
            local_query = shard.project_q(hidden, cos, sin)
            local_key, local_value = (
                key.chunk(2, dim=1)[rank],
                value.chunk(2, dim=1)[rank],
            )
        else:
            local_query, local_key, local_value = shard.project_qkv(hidden, cos, sin)
            torch.testing.assert_close(local_key, key.chunk(2, dim=1)[rank])
            torch.testing.assert_close(local_value, value.chunk(2, dim=1)[rank])
        torch.testing.assert_close(local_query, query.chunk(2, dim=1)[rank])
        local_attended = _attention(local_query, local_key, local_value).flatten(1)
        local_outputs.append((local_attended, shard.o_proj.local(local_attended)))
    for rank, (local_attended, _partial) in enumerate(local_outputs):
        contexts[rank].other = local_outputs[1 - rank][1]
        actual = shards[rank].o_proj(local_attended)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert len(contexts[rank].calls) == 1


@pytest.mark.parametrize("size", [2, 4])
def test_attention_geometry_shrinks_cache_without_changing_slab_packing(size):
    config = replace(_config(), num_global_key_value_heads=4)
    local = attention_geometry(config, TensorParallelLayout(0, size))
    assert local.num_attention_heads == config.num_attention_heads // size
    assert local.num_key_value_heads == config.num_key_value_heads // size
    assert local.num_global_key_value_heads == config.num_global_key_value_heads // size
    # Use a layer ratio whose local/global head widths share integral slabs.
    config = replace(
        config,
        num_hidden_layers=3,
        layer_types=(SLIDING_ATTENTION,) * 2 + ("full_attention",),
    )
    local = attention_geometry(config, TensorParallelLayout(0, size))
    assert local.history_blocks_per_parent == config.history_blocks_per_parent
    original = _Gemma4KvArena("cpu", config, 2, dtype=torch.bfloat16)
    sharded = _Gemma4KvArena("cpu", local, 2, dtype=torch.bfloat16)
    assert sharded.storage.numel() * size == original.storage.numel()
    assert sharded.local_keys.shape[-2] == local.num_key_value_heads
    assert sharded.global_keys.shape[-2] == local.num_global_key_value_heads
    capacity = plan_gemma4_kv_capacity(
        96 << 30, 60 << 30, 0.8, tensor_parallel_size=size
    )
    baseline = plan_gemma4_kv_capacity(96 << 30, 60 << 30, 0.8)
    assert capacity.bytes_per_parent * size == baseline.bytes_per_parent


def test_attention_tp_rejects_unshardable_global_kv_even_when_queries_divide():
    with pytest.raises(ValueError, match="global KV heads"):
        attention_geometry(_config(), TensorParallelLayout(0, 4))
    assistant = Gemma4AssistantConfig.from_dict(_raw_config())
    with pytest.raises(ValueError, match="global KV heads"):
        attention_geometry(assistant, TensorParallelLayout(0, 8))


def test_attention_sharding_must_precede_qkv_packing():
    attention = Gemma4Attention(_config(), 0)
    attention.pack_projections()
    with pytest.raises(ValueError, match="before packing"):
        shard_attention(attention, cast(TensorParallelContext, _Parallel(0, 2)))
