"""Cross-model paged FA4 execution for sink-attention decoders.

Rust owns one flat cache pool and supplies ordered History and Window plans.
Concrete models supply checkpoint, cache, and feed-forward hooks as class data;
attention and optional MoE kernels remain explicit constructor arguments.

The runner has no request-ID cache map and no Python page allocator.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from flashinfer.comm import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.mnnvl import TorchDistBackend
from quack import rmsnorm
from quack.rotary import apply_rotary
from torch import Tensor

from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    RunnerConfig,
)
from resin.backend.capacity import CacheCapacityPlan, plan_cache_capacity
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphStep,
    GraphInputError,
    GraphShape,
    graph_shapes,
)
from resin.backend.cuda.paged_kv import (
    DevicePagedAttentionBatch,
    PagedKvArena,
    empty_decode_paged_attention_batch,
    pad_paged_attention_table,
    paged_attention_graph_inputs,
    paged_attention_host_inputs,
    stage_paged_attention_batch,
)
from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.cuda.runner import CudaRunner, capture_sampling_inputs
from resin.backend.cuda.staging import TensorStager
from resin.backend.paged_attention import PagedAttentionBatch
from resin.backend.quantization import QuantizationConfig

if TYPE_CHECKING:
    from resin import BatchPlan


_BF16_BYTES = 2


def plan_paged_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    *,
    attention_layers_per_group: int,
    tensor_parallel_size: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> CacheCapacityPlan:
    """Size the fungible KV pool from post-load free device memory.

    ``gpu_memory_utilization`` is the total model-plus-cache fraction. Model,
    communication, and persistent kernel allocations have already been made,
    so protecting ``(1 - utilization) * total`` from the current free memory
    gives the exact remaining KV budget. History and sliding-window groups use
    one shared physical parent pool and therefore need no capacity split.
    """

    for name, value in (
        ("attention_layers_per_group", attention_layers_per_group),
        ("tensor_parallel_size", tensor_parallel_size),
        ("page_size", page_size),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_kv_heads % tensor_parallel_size:
        raise ValueError("tensor_parallel_size must divide the model's KV heads")

    bytes_per_parent = (
        2
        * attention_layers_per_group
        * page_size
        * (num_kv_heads // tensor_parallel_size)
        * head_dim
        * _BF16_BYTES
    )
    return plan_cache_capacity(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        bytes_per_parent,
    )


class PagedCudaRunner(CudaRunner):
    """FA4 runner configured by a thin concrete model subclass."""

    model_label = "paged"
    default_model = None
    page_size = 0
    sliding_window = 0
    prefill_chunk_size = 0
    window_retention_tokens = 0
    window_table_width = 0
    default_vocab_size = 0
    config_from_dict = None
    model_resolver = None
    model_loader = None
    attention_batch_builder = None

    def __init__(
        self,
        model: str | Path | None = None,
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
        required_hooks = (
            self.config_from_dict,
            self.model_resolver,
            self.model_loader,
            self.attention_batch_builder,
        )
        if any(hook is None for hook in required_hooks):
            raise TypeError("paged runner subclass is missing a model hook")
        if any(
            value <= 0
            for value in (
                self.page_size,
                self.sliding_window,
                self.prefill_chunk_size,
                self.window_retention_tokens,
                self.window_table_width,
            )
        ):
            raise TypeError("paged runner subclass has invalid cache geometry")
        if kernel_attention is None:
            from flash_attn.cute import flash_attn_varlen_func

            kernel_attention = flash_attn_varlen_func
        if not callable(kernel_attention):
            raise TypeError("kernel_attention must be callable")

        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.tensor_parallel_rank = self.tensor_parallel.rank
        self.tensor_parallel_size = self.tensor_parallel.size
        super().__init__(
            device=device,
            label=self.model_label,
            kv_cache_dtype="bfloat16",
        )

        self._fa4 = kernel_attention
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError(
                f"the {self.model_label} FA4 runner requires an SM90 GPU"
            )
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)

        resolved_dtype = torch.bfloat16 if dtype is None else dtype
        if isinstance(resolved_dtype, str):
            resolved_dtype = getattr(torch, resolved_dtype.removeprefix("torch."), None)
        if resolved_dtype is not torch.bfloat16:
            raise ValueError(
                f"the {self.model_label} FA4 runner requires bfloat16 activations"
            )

        selected_model = self.default_model if model is None else model
        if selected_model is None:
            raise TypeError("model is required when the runner has no default_model")
        self.model_path = self.model_resolver(
            selected_model,
            revision,
            quantization=quantization,
            kernel_moe=kernel_moe,
        )
        self._model, self.config = self.model_loader(
            self.model_path,
            device=self.device,
            dtype=torch.bfloat16,
            quantization=quantization,
            kernel_moe=kernel_moe,
            tensor_parallel=self.tensor_parallel.layout,
        )
        args = self._model.args
        checkpoint_args = self.config_from_dict(self.config)
        expected_attention = (
            checkpoint_args.num_attention_heads // self.tensor_parallel_size,
            checkpoint_args.num_key_value_heads // self.tensor_parallel_size,
            checkpoint_args.head_dim,
            self.sliding_window,
        )
        actual_attention = (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.sliding_window,
        )
        if actual_attention != expected_attention:
            raise ValueError(
                f"the {self.model_label} FA4 geometry is "
                f"{expected_attention}, got {actual_attention}"
            )

        self.num_query_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.eos_token_id = self.config.get("eos_token_id")

        self._allreduce_workspace: AllReduceFusionWorkspace | None = None
        self.allreduce_backend = "local"
        if self.tensor_parallel.enabled and fast_allreduce:
            self._allreduce_workspace = create_allreduce_fusion_workspace(
                backend="auto",
                world_size=self.tensor_parallel_size,
                rank=self.tensor_parallel_rank,
                max_token_num=self.prefill_chunk_size,
                hidden_dim=self.hidden_size,
                dtype=torch.bfloat16,
                comm_backend=TorchDistBackend(group=self.tensor_parallel.process_group),
                group=self.tensor_parallel.process_group,
            )
            self.allreduce_backend = self._allreduce_workspace.backend
        elif self.tensor_parallel.enabled:
            self.allreduce_backend = "nccl"

        for layer in self._model.model.layers:
            layer.self_attn.pack_qkv_projection()
            # MNNVL's fused RMSNorm reads gamma in the communication tensor's
            # dtype. The checkpoint values are BF16 even though RMSNorm
            # parameters are initially materialized as FP32.
            layer.input_layernorm.to(dtype=torch.bfloat16)
            layer.post_attention_layernorm.to(dtype=torch.bfloat16)
        self._model.model.norm.to(dtype=torch.bfloat16)

        self._full_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "full_attention"
        )
        self._sliding_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "sliding_attention"
        )
        self._full_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._full_layers)
        }
        self._sliding_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._sliding_layers)
        }
        if len(self._full_layers) != len(self._sliding_layers):
            raise ValueError(
                f"{self.model_label} History and Window groups must have equal geometry"
            )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_paged_kv_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            attention_layers_per_group=len(self._full_layers),
            tensor_parallel_size=self.tensor_parallel_size,
            page_size=self.page_size,
            num_kv_heads=checkpoint_args.num_key_value_heads,
            head_dim=checkpoint_args.head_dim,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        if self.tensor_parallel.enabled:
            capacity_tensor = torch.tensor(
                local_capacity, dtype=torch.int64, device=self.device
            )
            dist.all_reduce(
                capacity_tensor,
                op=dist.ReduceOp.MIN,
                group=self.tensor_parallel.process_group,
            )
            local_capacity = int(capacity_tensor.item())
        self.cache_parent_capacity = local_capacity
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + 1) * capacity_plan.bytes_per_parent,
        )
        self.cache_layout = CacheLayout(
            self.cache_parent_capacity,
            [
                CacheGroupLayout(self.page_size),
                CacheGroupLayout(
                    self.page_size,
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.window_retention_tokens,
                ),
            ],
        )
        # Sparse native plans remain chronological; the runner only derives
        # dense FA4 tables and page-aligned kernel rounds from each frozen plan.
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        if any(width > self.cache_parent_capacity for _, width in decode_graph_shapes):
            raise ValueError("CUDA graph table width exceeds cache capacity")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    lambda shape: self._graph_capture_inputs(shape, self.vocab_size),
                )
            }
        )
        # FA4's SM90 TMA path can load a complete physical page even when
        # ``seqused_k`` masks its unused tail.  Keep unwritten rows finite so a
        # masked PV operation can never encounter uninitialized NaN/Inf data.
        self._kv = PagedKvArena(
            self.device,
            self.cache_layout,
            layer_count=len(self._full_layers),
            page_size=self.page_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            resource=f"{self.model_label} KV",
        )
        self._configure_cache_binding(self._kv.binding)

        for layer in self._model.model.layers:
            if layer.self_attn.sinks.dtype is not torch.bfloat16:
                raise TypeError("FA4 learnable sink tensors must be bfloat16")

    @classmethod
    def _graph_capture_inputs(
        cls, shape: GraphShape, vocab_size: int | None = None
    ) -> dict[str, tuple]:
        """Return valid synthetic decode inputs backed only by null page zero."""

        batch_size, full_width = shape
        vocab_size = cls.default_vocab_size if vocab_size is None else vocab_size
        rows = tuple(range(batch_size))
        zeros = (0,) * batch_size
        return {
            "token_ids": zeros,
            "positions": zeros,
            "token_indices": rows,
            "sequence_indices": rows,
            "full_page_table": (0,) * (batch_size * full_width),
            "full_seqused_k": (1,) * batch_size,
            "sliding_page_table": (0,) * (batch_size * cls.window_table_width),
            "sliding_seqused_k": (1,) * batch_size,
            "full_append_slots": zeros,
            "sliding_append_slots": zeros,
            **capture_sampling_inputs(batch_size, vocab_size, stochastic=True),
        }

    def _after_cuda_graph_capture(self) -> None:
        # Capture writes synthetic K/V to parent zero, which the native
        # allocator permanently reserves. Restore its null-page contract
        # before a request can reach the runner.
        self._kv.storage[:, :, 0].zero_()
        torch.cuda.synchronize(self.device)

    def _tensorize(
        self, batch: PagedAttentionBatch, staging: TensorStager, prefix: str
    ) -> DevicePagedAttentionBatch:
        return stage_paged_attention_batch(batch, staging, prefix)

    def _apply_yarn_rotary(self, hidden, cos, sin):
        """Apply the model's already-materialized YaRN cos/sin tensors."""

        # The runner already gathers the YaRN rows for every packed token. A
        # synthetic batch dimension lets Quack consume those rows directly,
        # independent of the requests they came from.
        apply_rotary(
            hidden.unsqueeze(0),
            cos.squeeze(0),
            sin.squeeze(0),
            interleaved=False,
            inplace=True,
        )
        return hidden

    def _attention(
        self,
        layer_index: int,
        attention,
        hidden,
        position_embeddings,
        batch: DevicePagedAttentionBatch,
    ):
        token_count = hidden.shape[1]
        flat = hidden.reshape(token_count, self.hidden_size)
        query, key, value = attention.qkv_proj(flat).split(
            (
                self.num_query_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        query = query.view(token_count, self.num_query_heads, self.head_dim)
        key = key.view(token_count, self.num_kv_heads, self.head_dim)
        value = value.view(token_count, self.num_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        query = self._apply_yarn_rotary(query, cos, sin)
        key = self._apply_yarn_rotary(key, cos, sin)
        layer_type = attention.layer_type
        if query.dtype is not torch.bfloat16:
            raise TypeError("FA4 query tensor must be bfloat16")
        if key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
            raise TypeError("FA4 K/V tensors must be bfloat16")
        if attention.sinks.dtype is not torch.bfloat16:
            raise TypeError("FA4 learnable sink tensor must be bfloat16")

        sliding = layer_type == "sliding_attention"
        if sliding:
            slot = self._sliding_layer_slots[layer_index]
            group = batch.group("sliding")
        else:
            slot = self._full_layer_slots[layer_index]
            group = batch.group("full")
        key_cache = self._kv.keys[slot]
        value_cache = self._kv.values[slot]

        key_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, key
        )
        value_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, value
        )
        output = self._fa4(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=group.max_length,
            min_seqlen_k=group.min_length,
            seqused_q=None,
            seqused_k=group.lengths,
            page_table=group.page_table,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(self.sliding_window - 1, 0) if sliding else (None, None),
            softcap=0.0,
            num_splits=1,
            pack_gqa=True,
            deterministic=False,
            return_lse=False,
            learnable_sink=attention.sinks,
        )
        if isinstance(output, tuple):
            output = output[0]
        if output.dtype is not torch.bfloat16:
            raise TypeError("FA4 output tensor must be bfloat16")
        return attention.o_proj(output.reshape(1, token_count, -1))

    def _allreduce_residual_rmsnorm(
        self,
        partial: Tensor,
        residual: Tensor,
        weight: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        workspace = self._allreduce_workspace
        if workspace is None:
            reduced = self.tensor_parallel.sum(partial)
            normalized, combined = rmsnorm(
                reduced,
                weight,
                residual=residual,
                eps=eps,
                prenorm=True,
            )
            return normalized, combined

        shape = partial.shape
        flat_partial = partial.reshape(-1, shape[-1])
        flat_residual = residual.reshape(-1, shape[-1])
        combined = torch.empty_like(flat_partial)
        normalized = torch.empty_like(flat_partial)
        allreduce_fusion(
            input=flat_partial,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
            residual_in=flat_residual,
            residual_out=combined,
            norm_out=normalized,
            rms_gamma=weight,
            rms_eps=eps,
        )
        return normalized.view(shape), combined.view(shape)

    @staticmethod
    def _feed_forward(feed_forward, hidden: Tensor) -> Tensor:
        """Run the model-owned dense or routed feed-forward implementation."""

        return feed_forward(hidden)

    def _forward_batch(self, token_ids, positions, batch: DevicePagedAttentionBatch):
        selected_tokens = token_ids.index_select(0, batch.token_indices)
        selected_positions = positions.index_select(0, batch.token_indices)
        hidden = self._model.model.embed_tokens(selected_tokens).unsqueeze(0)
        position_embeddings = self._model.model.rotary_emb(
            selected_positions.unsqueeze(0), hidden.dtype
        )
        layers = self._model.model.layers
        normalized = rmsnorm(
            hidden,
            layers[0].input_layernorm.weight,
            eps=layers[0].input_layernorm.eps,
        )
        for layer_index, layer in enumerate(layers):
            attention_out = self._attention(
                layer_index,
                layer.self_attn,
                normalized,
                position_embeddings,
                batch,
            )
            normalized_mlp_input, hidden = self._allreduce_residual_rmsnorm(
                attention_out,
                hidden,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            feed_forward_output = self._feed_forward(
                layer.mlp, normalized_mlp_input
            )
            next_norm = (
                layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(layers)
                else self._model.model.norm
            )
            normalized, hidden = self._allreduce_residual_rmsnorm(
                feed_forward_output,
                hidden,
                next_norm.weight,
                next_norm.eps,
            )
        final_hidden = normalized.squeeze(0)
        return final_hidden.index_select(0, batch.last_token_indices)

    @staticmethod
    def _pad_page_table(batch: PagedAttentionBatch, width: int):
        """Pad each row by repeating one page already owned by that row."""

        try:
            return pad_paged_attention_table(batch.group("full"), width)
        except ValueError as error:
            raise GraphInputError(str(error)) from error

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, full_width = shape
        def empty(dtype, *shape):
            return torch.empty(shape, dtype=dtype, device=self.device)

        batch = empty_decode_paged_attention_batch(
            self.device,
            batch_size,
            {
                "full": (full_width, self.page_size),
                "sliding": (self.window_table_width, self.page_size),
            },
        )
        token_ids = empty(torch.long, batch_size)
        positions = empty(torch.long, batch_size)
        final_hidden = empty(torch.bfloat16, batch_size, self.hidden_size)
        output_ids = empty(torch.long, batch_size)
        sampling = (
            self._sampler.empty(batch_size, self.device)
            if hasattr(self._sampler, "empty")
            else None
        )

        def launch() -> None:
            hidden = self._forward_batch(token_ids, positions, batch)
            final_hidden.index_copy_(0, batch.sequence_indices, hidden)
            logits = self._model.lm_head(final_hidden)
            if sampling is None:
                self._sampler(logits, out=output_ids)
            else:
                self._sampler.sample_tensors(logits, sampling, out=output_ids)

        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            **paged_attention_graph_inputs(batch),
        }
        if sampling is not None:
            inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
        )

    def _graph_forward(
        self, plan: BatchPlan, host: PagedAttentionBatch
    ) -> list[int] | None:
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        rows = len(plan.sequence_ids)
        full = host.group("full")
        sliding = host.group("sliding")
        if (
            host.phase != BatchPhase.DECODE
            or host.query_lengths != (1,) * rows
            or sliding.width != self.window_table_width
        ):
            raise GraphInputError("plan is not a fixed-shape decode batch")
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] == rows and candidate[1] >= full.width
            ),
        )
        if shape is None:
            return None
        inputs = {
            "token_ids": plan.token_ids,
            "positions": plan.positions,
            **paged_attention_host_inputs(
                host,
                {"full": shape[1], "sliding": self.window_table_width},
            ),
        }
        if hasattr(self._sampler, "host_inputs"):
            inputs.update(self._sampler.host_inputs(plan, self.vocab_size))
        return self._run_cuda_graph("decode", shape, inputs)

    def forward(self, plan: BatchPlan) -> list[int]:
        self._check_cuda_graphs()
        host_positions = plan.positions

        prefill_batches, decode_batch = self.attention_batch_builder(
            plan, self.cache_layout
        )
        host_batches = prefill_batches + (
            () if decode_batch is None else (decode_batch,)
        )
        covered = {index for batch in host_batches for index in batch.sequence_indices}
        if covered != set(range(len(plan.sequence_ids))):
            raise ValueError(
                "native attention batches do not cover every plan sequence"
            )

        if not prefill_batches and decode_batch is not None:
            graph_output = self._graph_forward(plan, decode_batch)
            if graph_output is not None:
                return graph_output

        staging = TensorStager(self.device)
        token_ids = staging.device("token_ids", plan.token_ids, torch.long)
        positions = staging.device("positions", host_positions, torch.long)
        final_hidden = torch.empty(
            len(plan.sequence_ids),
            self.hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        with self._cuda_failure:
            with torch.inference_mode():
                for index, host_batch in enumerate(prefill_batches):
                    batch = self._tensorize(host_batch, staging, f"prefill.{index}")
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                if decode_batch is not None:
                    batch = self._tensorize(decode_batch, staging, "decode")
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                logits = self._model.lm_head(final_hidden)

            return self._sampler(logits, plan).tolist()


__all__ = [
    "PagedCudaRunner",
    "plan_paged_kv_capacity",
]
