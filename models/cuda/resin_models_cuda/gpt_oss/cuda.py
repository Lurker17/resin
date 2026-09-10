"""Thin GPT-OSS frontend for the reusable paged CUDA runner."""

from __future__ import annotations

import torch
from flashinfer.gemm import tinygemm_bf16
from torch import Tensor

from resin.backend.capacity import CacheCapacityPlan
from resin.backend.transformer import MLP, TopKRouter
from resin.models.gpt_oss.config import GptOssConfig
from resin.models.gpt_oss.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    WINDOW_TABLE_WIDTH,
    build_attention_batches,
)
from resin_models_cuda.gpt_oss.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.gpt_oss.cutile import top4_softmax
from resin_models_cuda.paged import PagedCudaRunner, plan_paged_kv_capacity

_TINYGEMM_MAX_TOKENS = 128


def _router_logits(router: TopKRouter, hidden: Tensor) -> Tensor:
    """Use Hopper TinyGEMM for skinny decode routers and GEMM otherwise."""

    if hidden.shape[0] <= _TINYGEMM_MAX_TOKENS:
        output = torch.empty(
            (hidden.shape[0], router.weight.shape[0]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        tinygemm_bf16(
            hidden,
            router.weight,
            output,
            router.bias,
            use_pdl=False,
        )
        return output
    return torch.nn.functional.linear(hidden, router.weight, router.bias)


def _route(router: TopKRouter, logits: Tensor) -> tuple[Tensor, Tensor]:
    if router.top_k == 4 and logits.shape[-1] == 128:
        return top4_softmax(logits)
    values, indices = torch.topk(logits, router.top_k, dim=-1, sorted=True)
    return indices, torch.softmax(values, dim=-1)


def plan_gpt_oss_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    *,
    attention_layers_per_group: int = 18,
    tensor_parallel_size: int = 1,
) -> CacheCapacityPlan:
    return plan_paged_kv_capacity(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        attention_layers_per_group=attention_layers_per_group,
        tensor_parallel_size=tensor_parallel_size,
        page_size=PAGE_SIZE,
        num_kv_heads=8,
        head_dim=64,
    )


class GptOssCudaRunner(PagedCudaRunner):
    """GPT-OSS BF16 attention on FA4 over one native cache pool."""

    model_label = "GPT-OSS"
    default_model = DEFAULT_MODEL
    page_size = PAGE_SIZE
    sliding_window = SLIDING_WINDOW
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    window_retention_tokens = WINDOW_RETENTION_TOKENS
    window_table_width = WINDOW_TABLE_WIDTH
    default_vocab_size = 201_088
    config_from_dict = staticmethod(GptOssConfig.from_dict)
    model_resolver = staticmethod(resolve_model_path)
    model_loader = staticmethod(load_model)
    attention_batch_builder = staticmethod(build_attention_batches)

    @staticmethod
    def _feed_forward(mlp: MLP, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits = _router_logits(mlp.router, flat)
        indices, weights = _route(mlp.router, logits)
        return mlp.experts(flat, indices, weights, logits).reshape(shape)


__all__ = [
    "GptOssCudaRunner",
    "plan_gpt_oss_kv_capacity",
    "_route",
    "_router_logits",
]
