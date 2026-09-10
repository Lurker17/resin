"""Thin GPT-OSS frontend for the reusable paged sink-MoE CUDA runner."""

from __future__ import annotations

from pathlib import Path

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.quantization import QuantizationConfig
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
from resin_models_cuda.paged_moe import (
    PagedMoECudaRunner,
    PagedMoEKvCapacityPlan,
    _route,
    _router_logits,
    plan_paged_moe_kv_capacity,
)

GptOssKvCapacityPlan = PagedMoEKvCapacityPlan


def plan_gpt_oss_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    *,
    attention_layers_per_group: int = 18,
    tensor_parallel_size: int = 1,
) -> GptOssKvCapacityPlan:
    return plan_paged_moe_kv_capacity(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        attention_layers_per_group=attention_layers_per_group,
        tensor_parallel_size=tensor_parallel_size,
        page_size=PAGE_SIZE,
        num_kv_heads=8,
        head_dim=64,
    )


class GptOssCudaRunner(PagedMoECudaRunner):
    """GPT-OSS BF16 attention on FA4 over one native cache pool."""

    model_label = "GPT-OSS"
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

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_parent_capacity: int | None = None,
        revision: str | None = None,
        device: str | None = None,
        dtype=None,
        quantization: QuantizationConfig | None = None,
        kernel_attention=None,
        kernel_moe=None,
        cuda_graph_decode_shapes: object = None,
        tensor_parallel: TensorParallelContext | None = None,
        fast_allreduce: bool = True,
        gpu_memory_utilization: float = 0.95,
    ) -> None:
        super().__init__(
            model,
            cache_parent_capacity=cache_parent_capacity,
            revision=revision,
            device=device,
            dtype=dtype,
            quantization=quantization,
            kernel_attention=kernel_attention,
            kernel_moe=kernel_moe,
            cuda_graph_decode_shapes=cuda_graph_decode_shapes,
            tensor_parallel=tensor_parallel,
            fast_allreduce=fast_allreduce,
            gpu_memory_utilization=gpu_memory_utilization,
        )


__all__ = [
    "GptOssCudaRunner",
    "GptOssKvCapacityPlan",
    "plan_gpt_oss_kv_capacity",
    "_route",
    "_router_logits",
]
