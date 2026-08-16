"""Blackwell CUDA forward pass for NVIDIA's Gemma 4 31B NVFP4 checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    ResourceBinding,
    RunnerConfig,
)
from resin.backend.cuda.graphs import (
    CudaGraphCache,
    CudaGraphStats,
    CudaGraphStep,
    GraphInputError,
    GraphShape,
    graph_shapes,
)
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.loading import default_device
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE
from resin.backend.speculative import accept_speculative_tokens
from resin.protocols import SyncRunner
from resin_models_cuda.gemma4.assistant_checkpoint import (
    DEFAULT_ASSISTANT,
    load_assistant,
    resolve_assistant_path,
)
from resin_models_cuda.gemma4.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.gemma4.config import SLIDING_ATTENTION
from resin_models_cuda.gemma4.media import Gemma4Media
from resin_models_cuda.gemma4.model import Gemma4ForConditionalGeneration
from resin_models_cuda.gemma4.nvfp4 import prepare_nvfp4_modules
from resin_models_cuda.gemma4.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    Gemma4BatchPlan,
    PagedAttentionPlan,
    build_gemma4_batch_plan,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)

if TYPE_CHECKING:
    from resin import BatchPlan

_WORKSPACE_BYTES = 128 * 1024 * 1024
_LOCAL_GRAPH_PAGE_CAPACITY = (
    1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE
)


def _select_nvfp4_backend(requested: str, cudnn_version: int | None) -> str:
    if requested not in {"auto", "cudnn", "cute-dsl"}:
        raise ValueError("nvfp4_backend must be 'auto', 'cudnn', or 'cute-dsl'")
    if requested == "auto":
        return "cudnn" if (cudnn_version or 0) >= 92301 else "cute-dsl"
    return requested


@dataclass(frozen=True, slots=True)
class Gemma4KvCapacityPlan:
    total_memory_bytes: int
    free_memory_bytes: int
    static_memory_fraction: float
    protected_free_bytes: int
    workspace_bytes: int
    cache_budget_bytes: int
    bytes_per_parent: int
    parent_capacity: int
    allocation_bytes: int


def plan_gemma4_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    static_memory_fraction: float,
    *,
    workspace_bytes: int = 2 * _WORKSPACE_BYTES,
) -> Gemma4KvCapacityPlan:
    if total_memory_bytes <= 0 or not 0 < free_memory_bytes <= total_memory_bytes:
        raise ValueError("CUDA memory totals must be positive and internally valid")
    if not 0.0 < static_memory_fraction < 1.0:
        raise ValueError("static_memory_fraction must be between zero and one")
    if workspace_bytes < 0:
        raise ValueError("workspace_bytes cannot be negative")
    # One FP8 parent holds all 50 local layers for one 64-token page.
    bytes_per_parent = 2 * 50 * PAGE_SIZE * 16 * 256
    protected_free_bytes = math.ceil(
        total_memory_bytes * (1.0 - static_memory_fraction)
    )
    cache_budget_bytes = free_memory_bytes - protected_free_bytes - workspace_bytes
    # Parent zero is a permanent graph-only null page. Native Engine page IDs
    # remain positive and map directly into the remaining physical slab.
    parent_capacity = cache_budget_bytes // bytes_per_parent - 1
    if parent_capacity <= 0:
        raise ValueError("GPU memory budget leaves no room for Gemma 4 KV cache")
    return Gemma4KvCapacityPlan(
        total_memory_bytes=total_memory_bytes,
        free_memory_bytes=free_memory_bytes,
        static_memory_fraction=static_memory_fraction,
        protected_free_bytes=protected_free_bytes,
        workspace_bytes=workspace_bytes,
        cache_budget_bytes=cache_budget_bytes,
        bytes_per_parent=bytes_per_parent,
        parent_capacity=parent_capacity,
        allocation_bytes=(parent_capacity + 1) * bytes_per_parent,
    )


class _Gemma4KvArena:
    def __init__(self, torch_module: Any, device, config, parent_capacity: int) -> None:
        self.dtype = torch_module.float8_e4m3fn
        self.parent_capacity = parent_capacity
        physical_parent_capacity = parent_capacity + 1
        local_shape = (
            physical_parent_capacity,
            2,
            config.local_layer_count,
            PAGE_SIZE,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.storage = torch_module.empty(local_shape, dtype=self.dtype, device=device)
        local = self.storage.permute(1, 2, 0, 3, 4, 5)
        self.local_keys, self.local_values = local[0], local[1]

        packing = config.history_blocks_per_parent
        global_shape = (
            physical_parent_capacity * packing,
            2,
            config.global_layer_count,
            PAGE_SIZE,
            config.num_global_key_value_heads,
            config.global_head_dim,
        )
        global_view = self.storage.view(global_shape).permute(1, 2, 0, 3, 4, 5)
        self.global_keys, self.global_values = global_view[0], global_view[1]


@dataclass(frozen=True, slots=True)
class _DevicePagedPlan:
    qo_indptr: Tensor
    kv_indptr: Tensor
    page_indices: Tensor
    last_page_lengths: Tensor
    append_pages: Tensor
    append_offsets: Tensor


@dataclass(frozen=True, slots=True)
class _MediaFeatures:
    media: Gemma4Media
    image: Tensor | None
    video: Tensor | None
    positions: dict[int, tuple[str, int]]


class Gemma4CudaRunner(SyncRunner):
    """Dynamic paged forward pass with FP8 KV and W4A16 NVFP4 MLPs."""

    page_size = PAGE_SIZE
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    sliding_window = SLIDING_WINDOW

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_parent_capacity: int | None = None,
        revision: str | None = None,
        assistant_model: str | Path | None = None,
        assistant_revision: str | None = None,
        speculative_tokens: int = 0,
        device: str | None = None,
        nvfp4_backend: str = "auto",
        cuda_graph_decode_shapes: object | None = None,
        static_memory_fraction: float = 0.95,
    ) -> None:
        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens < 0
            or speculative_tokens > 8
        ):
            raise ValueError("speculative_tokens must be an integer between 0 and 8")
        self.device = torch.device(device or default_device(torch, require_cuda=True))
        if self.device.type != "cuda":
            raise RuntimeError("Gemma4CudaRunner requires CUDA")
        capability = torch.cuda.get_device_capability(self.device)
        if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "Gemma 4 NVFP4 W4A16 requires Blackwell-class FP4 hardware, "
                f"got sm{capability[0]}{capability[1]}"
            )
        nvfp4_backend = _select_nvfp4_backend(
            nvfp4_backend, torch.backends.cudnn.version()
        )

        self.model_path = resolve_model_path(model, revision)
        self._model, self.config, self.raw_config = load_model(
            self.model_path, device=self.device
        )
        self._model.prepare_attention()
        prepare_nvfp4_modules(self._model, nvfp4_backend)
        self.nvfp4_backend = nvfp4_backend
        self.speculative_tokens = speculative_tokens
        self._assistant = None
        self.assistant_path: Path | None = None
        if speculative_tokens:
            selected_assistant = assistant_model or DEFAULT_ASSISTANT
            self.assistant_path = resolve_assistant_path(
                selected_assistant, assistant_revision
            )
            self._assistant, assistant_config = load_assistant(
                self.assistant_path, device=self.device
            )
            if assistant_config.backbone_hidden_size != self.config.hidden_size:
                raise ValueError(
                    "assistant backbone width does not match Gemma 4 target"
                )
            if assistant_config.vocab_size != self.config.vocab_size:
                raise ValueError("assistant vocabulary does not match Gemma 4 target")
        self._speculative_blocks = 0
        self._speculative_drafts = 0
        self._speculative_accepted = 0
        self._speculative_emitted = 0
        self._graph_attention: dict[GraphShape, tuple[Any, Any]] = {}
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_gemma4_kv_capacity(
            total_memory,
            free_memory,
            static_memory_fraction,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        self.cache_parent_capacity = local_capacity
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + 1) * capacity_plan.bytes_per_parent,
        )

        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)
        maximum_history_pages = (
            self.config.max_position_embeddings + PAGE_SIZE - 1
        ) // PAGE_SIZE
        if any(batch > 8 for batch, _ in decode_graph_shapes):
            raise ValueError("Gemma 4 CUDA graph batch capacity exceeds 8")
        if any(width > maximum_history_pages for _, width in decode_graph_shapes):
            raise ValueError("Gemma 4 CUDA graph page width exceeds context length")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=lambda: torch.cuda.synchronize(self.device),
        )

        self.cache_layout = CacheLayout(
            self.cache_parent_capacity,
            [
                CacheGroupLayout(
                    PAGE_SIZE,
                    blocks_per_parent=self.config.history_blocks_per_parent,
                ),
                CacheGroupLayout(
                    PAGE_SIZE,
                    retention=CacheRetention.WINDOW,
                    window_tokens=WINDOW_RETENTION_TOKENS,
                ),
            ],
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=speculative_tokens,
            cache_layout=self.cache_layout,
        )
        self._binding = CachePoolBinding(self.cache_layout, resource="Gemma 4 KV")
        self._cache = _Gemma4KvArena(
            torch, self.device, self.config, self.cache_parent_capacity
        )
        self._sampler = FlashInferSampler(torch)
        self._media_features: dict[int, _MediaFeatures] = {}
        self._target_hidden: dict[int, Tensor] = {}
        self._build_attention_wrappers()

    @property
    def model(self) -> Gemma4ForConditionalGeneration:
        return self._model

    @property
    def model_memory_bytes(self) -> int:
        tensors = list(self._model.parameters()) + list(self._model.buffers())
        if self._assistant is not None:
            tensors += list(self._assistant.parameters()) + list(
                self._assistant.buffers()
            )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        acceptance_rate = (
            self._speculative_accepted / self._speculative_drafts
            if self._speculative_drafts
            else 0.0
        )
        return {
            "draft_tokens": self.speculative_tokens,
            "blocks": self._speculative_blocks,
            "drafts": self._speculative_drafts,
            "accepted": self._speculative_accepted,
            "emitted": self._speculative_emitted,
            "acceptance_rate": acceptance_rate,
        }

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    def _build_attention_wrappers(self) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        local_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        global_workspace = torch.zeros_like(local_workspace)
        self._workspaces = (local_workspace, global_workspace)
        self._local_prefill = BatchPrefillWithPagedKVCacheWrapper(
            local_workspace, "NHD", backend="auto"
        )
        self._global_prefill = BatchPrefillWithPagedKVCacheWrapper(
            global_workspace, "NHD", backend="auto"
        )
        self._local_decode = BatchDecodeWithPagedKVCacheWrapper(
            local_workspace, "NHD", backend="auto"
        )
        self._global_decode = BatchDecodeWithPagedKVCacheWrapper(
            global_workspace, "NHD", backend="auto"
        )

    def bind_resources(self, binding: ResourceBinding) -> None:
        if binding.cache_pool_id is None or binding.cache_layout is None:
            raise ValueError("Gemma 4 requires an Engine-owned cache pool")
        self._binding.bind(binding.cache_pool_id, binding.cache_layout)

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._media_features.pop(request_id, None)
            self._target_hidden.pop(request_id, None)

    def _device_plan(self, plan: PagedAttentionPlan) -> _DevicePagedPlan:
        def tensor(values, dtype):
            return torch.as_tensor(values, dtype=dtype, device=self.device)

        append_slots = tensor(plan.append_slots, torch.int64)
        return _DevicePagedPlan(
            qo_indptr=tensor(plan.qo_indptr, torch.int32),
            kv_indptr=tensor(plan.kv_indptr, torch.int32),
            page_indices=tensor(plan.page_indices, torch.int32),
            last_page_lengths=tensor(plan.last_page_lengths, torch.int32),
            append_pages=torch.div(append_slots, PAGE_SIZE, rounding_mode="floor"),
            append_offsets=append_slots.remainder(PAGE_SIZE),
        )

    def _plan_attention(
        self,
        batch: Gemma4BatchPlan,
        local: _DevicePagedPlan,
        global_: _DevicePagedPlan,
        custom_mask: Tensor | None,
    ) -> tuple[Any, Any]:
        if batch.phase == BatchPhase.PREFILL:
            self._local_prefill.plan(
                local.qo_indptr,
                local.kv_indptr,
                local.page_indices,
                local.last_page_lengths,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                PAGE_SIZE,
                custom_mask=custom_mask,
                causal=custom_mask is None,
                sm_scale=1.0,
                window_left=SLIDING_WINDOW - 1,
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
            )
            self._global_prefill.plan(
                global_.qo_indptr,
                global_.kv_indptr,
                global_.page_indices,
                global_.last_page_lengths,
                self.config.num_attention_heads,
                self.config.num_global_key_value_heads,
                self.config.global_head_dim,
                PAGE_SIZE,
                causal=True,
                sm_scale=1.0,
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
            )
            return self._local_prefill, self._global_prefill

        common = {
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.bfloat16,
            "kv_data_type": self._cache.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": 1.0,
        }
        self._local_decode.plan(
            local.kv_indptr,
            local.page_indices,
            local.last_page_lengths,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            PAGE_SIZE,
            window_left=SLIDING_WINDOW - 1,
            **common,
        )
        self._global_decode.plan(
            global_.kv_indptr,
            global_.page_indices,
            global_.last_page_lengths,
            self.config.num_attention_heads,
            self.config.num_global_key_value_heads,
            self.config.global_head_dim,
            PAGE_SIZE,
            **common,
        )
        return self._local_decode, self._global_decode

    def _features_for(self, sequence_id: int, media: Gemma4Media) -> _MediaFeatures:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing.media is not media:
                raise ValueError("a live Gemma 4 sequence changed its media object")
            return existing
        media.validate_placeholder_tokens(
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
        )
        with torch.inference_mode():
            image = (
                self._model.get_image_features(
                    media.pixel_values.to(self.device, dtype=torch.bfloat16),
                    media.image_position_ids.to(self.device),
                )
                if media.pixel_values is not None
                else None
            )
            video = (
                self._model.get_video_features(
                    media.pixel_values_videos.to(self.device, dtype=torch.bfloat16),
                    media.video_position_ids.to(self.device),
                )
                if media.pixel_values_videos is not None
                else None
            )
        if image is not None and len(image) != len(media.image_positions):
            raise ValueError("vision features do not match image placeholder count")
        if video is not None and len(video) != len(media.video_positions):
            raise ValueError("vision features do not match video placeholder count")
        positions = {
            position: ("image", index)
            for index, position in enumerate(media.image_positions)
        }
        positions.update(
            {
                position: ("video", index)
                for index, position in enumerate(media.video_positions)
            }
        )
        features = _MediaFeatures(media, image, video, positions)
        self._media_features[sequence_id] = features
        return features

    @staticmethod
    def _validate_unsplit_vision_blocks(
        media: Gemma4Media, query_start: int, query_end: int
    ) -> None:
        blocks = media.block_ids
        for position in range(query_start, min(query_end, len(blocks))):
            block = blocks[position]
            if block < 0:
                continue
            start = position
            while start and blocks[start - 1] == block:
                start -= 1
            end = position + 1
            while end < len(blocks) and blocks[end] == block:
                end += 1
            if start < query_start or end > query_end:
                raise ValueError(
                    "a Gemma 4 vision block crosses a prefill chunk boundary; "
                    "reduce earlier prompt length or increase the runner chunk size"
                )

    def _merge_media_embeddings(
        self,
        native: BatchPlan,
        batch: Gemma4BatchPlan,
        hidden: Tensor,
    ) -> dict[int, _MediaFeatures]:
        features_by_row = {}
        sequence_ids = np.asarray(native.sequence_ids, dtype=np.uint64)
        for row, media in enumerate(native.media):
            if media is None:
                continue
            if not isinstance(media, Gemma4Media):
                raise TypeError("Gemma4CudaRunner media must be Gemma4Media")
            start, end = map(int, batch.token_indptr[row : row + 2])
            positions = batch.positions[start:end]
            for flat_index, position in zip(range(start, end), positions, strict=True):
                absolute = int(position)
                if absolute < len(media.token_ids) and (
                    int(batch.token_ids[flat_index]) != media.token_ids[absolute]
                ):
                    raise ValueError(
                        "scheduled token does not match Gemma4Media prompt"
                    )
            vision_positions = set(media.image_positions + media.video_positions)
            if not any(int(position) in vision_positions for position in positions):
                continue
            self._validate_unsplit_vision_blocks(
                media, int(positions[0]), int(positions[-1]) + 1
            )
            features = self._features_for(int(sequence_ids[row]), media)
            features_by_row[row] = features
            for flat_index, position in zip(range(start, end), positions, strict=True):
                absolute = int(position)
                location = features.positions.get(absolute)
                if location is None:
                    continue
                kind, feature_index = location
                source = features.image if kind == "image" else features.video
                hidden[flat_index].copy_(source[feature_index])
        return features_by_row

    def _local_custom_mask(
        self,
        batch: Gemma4BatchPlan,
        features_by_row: dict[int, _MediaFeatures],
    ) -> Tensor | None:
        if batch.phase != BatchPhase.PREFILL or not features_by_row:
            return None
        masks = []
        any_bidirectional = False
        for row in range(len(batch.token_indptr) - 1):
            start, end = map(int, batch.token_indptr[row : row + 2])
            query = batch.positions[start:end, None]
            first = batch.local.first_token_positions[row]
            keys = np.arange(first, first + batch.local.kv_lengths[row])[None, :]
            allowed = (keys <= query) & (keys >= query - SLIDING_WINDOW + 1)
            features = features_by_row.get(row)
            if features is not None:
                block_ids = np.asarray(features.media.block_ids, dtype=np.int64)
                query_blocks = np.full(query.shape, -1, dtype=np.int64)
                key_blocks = np.full(keys.shape, -1, dtype=np.int64)
                valid_q = query[:, 0] < len(block_ids)
                valid_k = keys[0] < len(block_ids)
                query_blocks[valid_q, 0] = block_ids[query[valid_q, 0]]
                key_blocks[0, valid_k] = block_ids[keys[0, valid_k]]
                same = (query_blocks >= 0) & (query_blocks == key_blocks)
                if np.any(same & ~allowed):
                    any_bidirectional = True
                allowed |= same & (keys >= query - SLIDING_WINDOW + 1)
            masks.append(allowed.reshape(-1))
        if not any_bidirectional:
            return None
        return torch.as_tensor(
            np.concatenate(masks), dtype=torch.bool, device=self.device
        )

    def _run_transformer(
        self,
        hidden: Tensor,
        positions: Tensor,
        local: _DevicePagedPlan,
        global_: _DevicePagedPlan,
        local_wrapper: Any,
        global_wrapper: Any,
    ) -> Tensor:
        rotary = self._model.language_model.rotary_emb
        embeddings = {
            kind: rotary(positions, kind, hidden.dtype)
            for kind in set(self.config.layer_types)
        }
        local_layer = 0
        global_layer = 0
        for layer in self._model.language_model.layers:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            cos, sin = embeddings[layer.self_attn.layer_type]
            query, key, value = layer.self_attn.project_qkv(normalized, cos, sin)
            if layer.self_attn.layer_type == SLIDING_ATTENTION:
                keys = self._cache.local_keys[local_layer]
                values = self._cache.local_values[local_layer]
                append_pages = local.append_pages
                append_offsets = local.append_offsets
                wrapper = local_wrapper
                local_layer += 1
            else:
                keys = self._cache.global_keys[global_layer]
                values = self._cache.global_values[global_layer]
                append_pages = global_.append_pages
                append_offsets = global_.append_offsets
                wrapper = global_wrapper
                global_layer += 1
            keys[append_pages, append_offsets] = key.to(self._cache.dtype)
            values[append_pages, append_offsets] = value.to(self._cache.dtype)
            attended = wrapper.run(query, (keys, values))
            hidden = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
            hidden = residual + layer.post_attention_layernorm(hidden)

            residual = hidden
            hidden = layer.pre_feedforward_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + layer.post_feedforward_layernorm(hidden)
            hidden *= layer.layer_scalar

        return self._model.language_model.norm(hidden)

    @staticmethod
    def _graph_capture_inputs(
        shape: GraphShape, vocab_size: int = 262_208
    ) -> dict[str, tuple]:
        """Synthetic decode rows backed only by graph-reserved parent zero."""

        batch_size, history_width = shape
        rows = tuple(range(batch_size))
        zeros = (0,) * batch_size
        inputs = {
            "token_ids": zeros,
            "positions": zeros,
            "local_kv_indptr": tuple(range(batch_size + 1)),
            "local_page_indices": (0,) * (batch_size * _LOCAL_GRAPH_PAGE_CAPACITY),
            "local_last_page_lengths": (1,) * batch_size,
            "local_append_pages": zeros,
            "local_append_offsets": rows,
            "global_kv_indptr": tuple(range(batch_size + 1)),
            "global_page_indices": (0,) * (batch_size * history_width),
            "global_last_page_lengths": (1,) * batch_size,
            "global_append_pages": zeros,
            "global_append_offsets": rows,
            "sampling_temperatures": (1.0,) * batch_size,
            "sampling_top_ps": (1.0,) * batch_size,
            "sampling_top_ks": (vocab_size,) * batch_size,
            "sampling_seeds": zeros,
            "sampling_offsets": zeros,
            "sampling_stochastic_mask": (False,) * batch_size,
        }
        return inputs

    def capture_cuda_graphs(self) -> CudaGraphStats:
        """Capture configured decode batch buckets before serving requests."""

        captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape not in captured:
                self._graphs.capture(
                    shape, self._graph_capture_inputs(shape, self.config.vocab_size)
                )
        if self._graphs.shapes:
            self._cache.storage[0].zero_()
            torch.cuda.synchronize(self.device)
        return self._graphs.stats

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        batch_size, history_width = shape

        def empty(dtype, *dimensions):
            return torch.empty(dimensions, dtype=dtype, device=self.device)

        def attention(width: int, *, local: bool):
            workspace = torch.zeros(
                _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
            )
            kv_indptr = empty(torch.int32, batch_size + 1)
            page_indices = empty(torch.int32, batch_size * width)
            last_page_lengths = empty(torch.int32, batch_size)
            append_pages = empty(torch.int64, batch_size)
            append_offsets = empty(torch.int64, batch_size)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=page_indices,
                paged_kv_last_page_len_buffer=last_page_lengths,
                backend="auto",
            )
            initial_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            initial_indices = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            initial_last = torch.ones(batch_size, dtype=torch.int32, device=self.device)
            wrapper.plan(
                initial_indptr,
                initial_indices,
                initial_last,
                self.config.num_attention_heads,
                (
                    self.config.num_key_value_heads
                    if local
                    else self.config.num_global_key_value_heads
                ),
                self.config.head_dim if local else self.config.global_head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
                sm_scale=1.0,
                window_left=SLIDING_WINDOW - 1 if local else -1,
                disable_split_kv=True,
                non_blocking=False,
            )
            plan = _DevicePagedPlan(
                qo_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indptr=kv_indptr,
                page_indices=page_indices,
                last_page_lengths=last_page_lengths,
                append_pages=append_pages,
                append_offsets=append_offsets,
            )
            return wrapper, plan

        local_wrapper, local = attention(_LOCAL_GRAPH_PAGE_CAPACITY, local=True)
        global_wrapper, global_ = attention(history_width, local=False)
        self._graph_attention[shape] = (local_wrapper, global_wrapper)
        token_ids = empty(torch.int64, batch_size)
        positions = empty(torch.int64, batch_size)
        output_ids = empty(torch.int64, batch_size)
        sampling = self._sampler.empty(batch_size, self.device)

        def launch() -> None:
            hidden = self._model.embed_tokens(token_ids)
            hidden = self._run_transformer(
                hidden,
                positions,
                local,
                global_,
                local_wrapper,
                global_wrapper,
            )
            logits = self._model.logits(hidden)
            self._sampler.sample_tensors(logits, sampling, out=output_ids)

        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "local_kv_indptr": local.kv_indptr,
            "local_page_indices": local.page_indices,
            "local_last_page_lengths": local.last_page_lengths,
            "local_append_pages": local.append_pages,
            "local_append_offsets": local.append_offsets,
            "global_kv_indptr": global_.kv_indptr,
            "global_page_indices": global_.page_indices,
            "global_last_page_lengths": global_.last_page_lengths,
            "global_append_pages": global_.append_pages,
            "global_append_offsets": global_.append_offsets,
        }
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(inputs=inputs, output=output_ids, launch=launch)

    def _prepare_graph_attention(
        self, shape: GraphShape, inputs: dict[str, object]
    ) -> None:
        local_wrapper, global_wrapper = self._graph_attention[shape]

        def plan(wrapper, name: str, *, local: bool) -> None:
            host_indptr = inputs[f"{name}_kv_indptr"]
            page_count = int(np.asarray(host_indptr)[-1])
            indptr = torch.as_tensor(host_indptr, dtype=torch.int32, device=self.device)
            indices = torch.as_tensor(
                inputs[f"{name}_page_indices"],
                dtype=torch.int32,
                device=self.device,
            )
            last = torch.as_tensor(
                inputs[f"{name}_last_page_lengths"],
                dtype=torch.int32,
                device=self.device,
            )
            wrapper.plan(
                indptr,
                indices[:page_count],
                last,
                self.config.num_attention_heads,
                (
                    self.config.num_key_value_heads
                    if local
                    else self.config.num_global_key_value_heads
                ),
                self.config.head_dim if local else self.config.global_head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
                sm_scale=1.0,
                window_left=SLIDING_WINDOW - 1 if local else -1,
                disable_split_kv=True,
                non_blocking=False,
            )

        plan(local_wrapper, "local", local=True)
        plan(global_wrapper, "global", local=False)

    @staticmethod
    def _graph_page_inputs(
        name: str,
        plan: PagedAttentionPlan,
        *,
        active_rows: int,
        graph_rows: int,
        width: int,
    ) -> dict[str, object]:
        row_widths = np.diff(plan.kv_indptr)
        if len(row_widths) != active_rows or np.any(row_widths > width):
            raise GraphInputError(f"{name} page table exceeds CUDA graph shape")
        indices = np.zeros(graph_rows * width, dtype=np.int32)
        indptr = np.zeros(graph_rows + 1, dtype=np.int32)
        cursor = 0
        for row in range(graph_rows):
            if row < active_rows:
                start, end = map(int, plan.kv_indptr[row : row + 2])
                count = end - start
                indices[cursor : cursor + count] = plan.page_indices[start:end]
            else:
                count = 1
            cursor += count
            indptr[row + 1] = cursor
        last_page_lengths = np.ones(graph_rows, dtype=np.int32)
        last_page_lengths[:active_rows] = plan.last_page_lengths
        append_pages = np.zeros(graph_rows, dtype=np.int64)
        append_offsets = np.arange(graph_rows, dtype=np.int64)
        append_pages[:active_rows] = plan.append_slots // PAGE_SIZE
        append_offsets[:active_rows] = plan.append_slots % PAGE_SIZE
        return {
            f"{name}_kv_indptr": indptr,
            f"{name}_page_indices": indices,
            f"{name}_last_page_lengths": last_page_lengths,
            f"{name}_append_pages": append_pages,
            f"{name}_append_offsets": append_offsets,
        }

    def _graph_forward(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> list[int] | None:
        if not self._graphs.shapes:
            return None
        active_rows = native.num_sequences
        history_width = int(np.diff(batch.global_.kv_indptr).max())
        local_width = int(np.diff(batch.local.kv_indptr).max())
        if local_width > _LOCAL_GRAPH_PAGE_CAPACITY:
            raise GraphInputError("local page table exceeds CUDA graph shape")
        shape = self._graphs.match(
            lambda candidate: (
                candidate[0] >= active_rows and candidate[1] >= history_width
            )
        )
        if shape is None:
            self._graphs.fallback("shape")
            return None
        graph_rows, graph_history_width = shape
        padding_rows = graph_rows - active_rows
        inputs: dict[str, object] = {
            "token_ids": np.pad(batch.token_ids, (0, padding_rows)),
            "positions": np.pad(batch.positions, (0, padding_rows)),
        }
        inputs.update(
            self._graph_page_inputs(
                "local",
                batch.local,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=_LOCAL_GRAPH_PAGE_CAPACITY,
            )
        )
        inputs.update(
            self._graph_page_inputs(
                "global",
                batch.global_,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=graph_history_width,
            )
        )
        sampling = self._sampler.host_inputs(native, self.config.vocab_size)
        padding_defaults = {
            "sampling_temperatures": 1.0,
            "sampling_top_ps": 1.0,
            "sampling_top_ks": self.config.vocab_size,
            "sampling_seeds": 0,
            "sampling_offsets": 0,
            "sampling_stochastic_mask": False,
        }
        for name, default in padding_defaults.items():
            sampling[name] = (*sampling[name], *((default,) * padding_rows))
        inputs.update(sampling)
        self._prepare_graph_attention(shape, inputs)
        return self._graphs.run(shape, inputs)[:active_rows]

    def _eager_transformer(self, native: BatchPlan, batch: Gemma4BatchPlan) -> Tensor:
        token_ids = torch.as_tensor(
            batch.token_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.positions, dtype=torch.int64, device=self.device
        )
        hidden = self._model.embed_tokens(token_ids)
        features = self._merge_media_embeddings(native, batch, hidden)
        local_mask = self._local_custom_mask(batch, features)
        local = self._device_plan(batch.local)
        global_ = self._device_plan(batch.global_)
        local_wrapper, global_wrapper = self._plan_attention(
            batch, local, global_, local_mask
        )
        return self._run_transformer(
            hidden,
            positions,
            local,
            global_,
            local_wrapper,
            global_wrapper,
        )

    def _store_last_hidden(
        self, native: BatchPlan, batch: Gemma4BatchPlan, hidden: Tensor
    ) -> None:
        for sequence_id, index in zip(
            native.sequence_ids, batch.last_token_indices, strict=True
        ):
            self._target_hidden[int(sequence_id)] = hidden[int(index)].detach().clone()

    def _sample_target(self, logits: Tensor, plan: BatchPlan, step: int) -> Tensor:
        return self._sampler(
            logits,
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _single_target_decode(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> list[int]:
        rows = native.num_sequences
        target = expand_gemma4_decode_batch(batch, np.empty((rows, 0), dtype=np.int64))
        hidden = self._eager_transformer(native, target)
        self._store_last_hidden(native, target, hidden)
        logits = self._model.logits(hidden[target.last_token_indices])
        return self._sample_target(logits, native, 0).tolist()

    def _speculative_decode(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> list[list[int]] | list[int]:
        assistant = self._assistant
        if assistant is None:
            raise RuntimeError("speculative decode requires the Gemma 4 assistant")
        sequence_ids = [int(value) for value in native.sequence_ids]
        if any(sequence_id not in self._target_hidden for sequence_id in sequence_ids):
            # A complete prefix-cache hit has target K/V but no sequence-local
            # final hidden state. Materialize it with one ordinary target step;
            # all subsequent steps can draft normally.
            return self._single_target_decode(native, batch)

        context = gemma4_assistant_context(batch)
        local = self._device_plan(context.local)
        global_ = self._device_plan(context.global_)
        local_wrapper, global_wrapper = self._plan_attention(
            context, local, global_, custom_mask=None
        )
        anchors = torch.as_tensor(
            batch.token_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.positions, dtype=torch.int64, device=self.device
        )
        target_hidden = torch.stack(
            [self._target_hidden[sequence_id] for sequence_id in sequence_ids]
        )
        last_ids = anchors
        last_hidden = target_hidden
        draft_columns = []
        shared_local = (self._cache.local_keys[-1], self._cache.local_values[-1])
        shared_global = (self._cache.global_keys[-1], self._cache.global_values[-1])
        for _ in range(self.speculative_tokens):
            token_embedding = self._model.embed_tokens(last_ids)
            inputs_embeds = torch.cat((token_embedding, last_hidden), dim=-1)
            draft_logits, last_hidden = assistant.forward_step(
                inputs_embeds,
                positions,
                local_wrapper=local_wrapper,
                global_wrapper=global_wrapper,
                local_kv=shared_local,
                global_kv=shared_global,
            )
            last_ids = draft_logits.argmax(dim=-1)
            draft_columns.append(last_ids)
        drafts = torch.stack(draft_columns, dim=1)

        verify = expand_gemma4_decode_batch(batch, drafts.cpu().numpy())
        verify_hidden = self._eager_transformer(native, verify)
        width = self.speculative_tokens + 1
        verify_hidden_rows = verify_hidden.view(
            native.num_sequences, width, self.config.hidden_size
        )
        verify_logits = self._model.logits(verify_hidden).view(
            native.num_sequences, width, self.config.vocab_size
        )
        target_columns = [
            self._sample_target(verify_logits[:, index].contiguous(), native, index)
            for index in range(width)
        ]
        acceptance = accept_speculative_tokens(
            drafts.tolist(),
            torch.stack(target_columns, dim=1).tolist(),
        )
        for row, (sequence_id, accepted) in enumerate(
            zip(sequence_ids, acceptance.accepted_counts, strict=True)
        ):
            self._target_hidden[sequence_id] = (
                verify_hidden_rows[row, accepted].detach().clone()
            )
        self._speculative_blocks += native.num_sequences
        self._speculative_drafts += native.num_sequences * self.speculative_tokens
        self._speculative_accepted += sum(acceptance.accepted_counts)
        self._speculative_emitted += sum(map(len, acceptance.token_rows))
        return acceptance.token_rows

    @torch.inference_mode()
    def forward(self, plan: BatchPlan) -> list[int] | list[list[int]]:
        self._graphs.check()
        batch = build_gemma4_batch_plan(
            plan,
            history_blocks_per_parent=self.config.history_blocks_per_parent,
            parent_capacity=self.cache_parent_capacity,
            decode_lookahead=self.speculative_tokens,
        )
        if batch.phase == BatchPhase.DECODE:
            if self.speculative_tokens:
                if self._graphs.shapes:
                    self._graphs.fallback("speculative")
                return self._speculative_decode(plan, batch)
            graph_output = self._graph_forward(plan, batch)
            if graph_output is not None:
                return graph_output
        elif self._graphs.shapes:
            self._graphs.fallback("prefill")
        hidden = self._eager_transformer(plan, batch)
        self._store_last_hidden(plan, batch, hidden)
        last = torch.as_tensor(
            batch.last_token_indices, dtype=torch.int64, device=self.device
        )
        logits = self._model.logits(hidden[last])
        return self._sample_target(logits, plan, 0).tolist()


__all__ = [
    "Gemma4CudaRunner",
    "Gemma4KvCapacityPlan",
    "plan_gemma4_kv_capacity",
]
