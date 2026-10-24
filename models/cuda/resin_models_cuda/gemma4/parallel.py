"""Tensor-parallel preparation for Gemma target and shared-KV assistant layers."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from torch import nn

from resin.backend.cuda.parallel import (
    TensorParallelContext,
    TensorParallelLayout,
    TensorShard,
)
from resin.models.gemma4.assistant import (
    Gemma4AssistantAttention,
    Gemma4AssistantConfig,
)
from resin.models.gemma4.config import Gemma4Nvfp4Config
from resin_models_cuda.gemma4.model import Gemma4Attention

_Config = TypeVar("_Config", Gemma4Nvfp4Config, Gemma4AssistantConfig)


def attention_geometry(config: _Config, parallel: TensorParallelLayout) -> _Config:
    """Shard both KV geometries; spare ranks replicate their complete head shard."""

    if not parallel.enabled:
        return config
    return replace(
        config,
        num_attention_heads=parallel.even_shard(
            config.num_attention_heads, name="Gemma query heads"
        ).size,
        num_key_value_heads=parallel.even_shard(
            config.num_key_value_heads, name="Gemma local KV heads"
        ).size,
        num_global_key_value_heads=parallel.even_shard(
            config.num_global_key_value_heads, name="Gemma global KV heads"
        ).size,
    )


def shard_attention(
    module: Gemma4Attention | Gemma4AssistantAttention,
    parallel: TensorParallelContext,
) -> nn.Module:
    """Partition loaded Q/K/V heads and sum the output before residual/RMSNorm."""

    from resin_models_cuda.dense_parallel import TensorParallelSum, shard_linear

    if not parallel.enabled:
        return module
    query = parallel.layout.even_shard(module.num_heads, name="Gemma query heads")
    query_width = TensorShard(
        query.start * module.head_dim, query.stop * module.head_dim
    )
    projections = [("q_proj", query_width)]
    if isinstance(module, Gemma4Attention):
        if module.qkv_proj is not None:
            raise ValueError("shard Gemma attention before packing QKV projections")
        kv = parallel.layout.even_shard(module.num_kv_heads, name="Gemma KV heads")
        kv_width = TensorShard(kv.start * module.head_dim, kv.stop * module.head_dim)
        projections.append(("k_proj", kv_width))
        if module.v_proj is not None:
            projections.append(("v_proj", kv_width))
        module.num_kv_heads = kv.size
    for name, shard in projections:
        setattr(
            module,
            name,
            shard_linear(getattr(module, name), shard, output=True, rank=parallel.rank),
        )
    setattr(
        module,
        "o_proj",
        TensorParallelSum(
            shard_linear(module.o_proj, query_width, output=False, rank=parallel.rank),
            parallel,
        ),
    )
    module.num_heads = query.size
    return module
