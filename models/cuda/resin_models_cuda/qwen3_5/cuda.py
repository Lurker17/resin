"""Single-GPU Qwen3.5 hybrid runner with optional bundled-MTP decoding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from resin import BatchPhase, ResourceBinding, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.failures import CudaFailureGuard
from resin.backend.cuda.graphs import (
    CudaGraphCache,
    CudaGraphStats,
    CudaGraphStep,
    GraphShape,
    graph_shapes,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
    FlashInferSampler,
    SamplingTensors,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.loading import default_device
from resin.backend.plans import PagedCachePlan
from resin.backend.speculative import SpeculativeStats
from resin.backend.torch import use_default_dtype
from resin.protocols import Runner
from resin_models_cuda.nvfp4 import prepare_nvfp4_modules
from resin_models_cuda.qwen3_5.attention import (
    GraphAttentionSpec,
    GraphPagedAttentionRuntime,
    PagedAttentionRuntime,
)
from resin_models_cuda.qwen3_5.cache import (
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
    DeviceCache,
)
from resin_models_cuda.qwen3_5.checkpoint import (
    DEFAULT_MODEL,
    is_modelopt_nvfp4,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.qwen3_5.media import Qwen35Media

if TYPE_CHECKING:
    from resin import BatchPlan

_CACHE_WORKSPACE_BYTES = 512 * 1024**2
_GRAPH_WORKSPACE_BYTES = 128 * 1024**2


class Qwen3_5CudaRunner(Runner):
    """Dense Qwen3.5/Qwen3.8 text runner over paged FA4 and DeltaNet state."""

    page_size = 128
    prefill_chunk_size = PREFILL_CHUNK_SIZE

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 32,
        speculative_tokens: int = 0,
        device: str | None = None,
        gpu_memory_utilization: float = 0.90,
        cache_slab_capacity: int | None = None,
        kv_cache_dtype: str = "fp8",
        nvfp4_backend: str = "auto",
        cuda_graph_decode_shapes: object | None = None,
    ) -> None:
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens < 0
        ):
            raise ValueError("speculative_tokens must be a non-negative integer")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_slab_capacity is not None and (
            isinstance(cache_slab_capacity, bool)
            or not isinstance(cache_slab_capacity, int)
            or cache_slab_capacity <= 0
        ):
            raise ValueError("cache_slab_capacity must be a positive integer")
        if kv_cache_dtype not in {"fp8", "bfloat16"}:
            raise ValueError("kv_cache_dtype must be 'fp8' or 'bfloat16'")
        if nvfp4_backend not in {"auto", "cudnn", "cute-dsl"}:
            raise ValueError("nvfp4_backend must be 'auto', 'cudnn', or 'cute-dsl'")

        self.device = torch.device(device or default_device(torch, require_cuda=True))
        if self.device.type != "cuda":
            raise RuntimeError("Qwen3_5CudaRunner requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 9:
            raise RuntimeError("Qwen3_5CudaRunner requires Hopper or newer CUDA")
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        except ImportError as error:
            raise ImportError(
                "Qwen3_5CudaRunner requires flash-linear-attention and FlashInfer"
            ) from error
        del chunk_gated_delta_rule, BatchDecodeWithPagedKVCacheWrapper

        self._cuda_failure = CudaFailureGuard(
            lambda: torch.cuda.synchronize(self.device),
            label="Qwen3.5 eager execution",
        )
        self.model_path = resolve_model_path(model, revision)
        self.speculative_tokens = speculative_tokens
        self._model, self.config, self.raw_config = load_model(
            self.model_path,
            device=self.device,
            include_mtp=speculative_tokens > 0,
            include_vision=True,
        )
        self.image_token_id = self.raw_config.get("image_token_id")
        vision_config = self.raw_config.get("vision_config") or {}
        self.spatial_merge_size = int(vision_config.get("spatial_merge_size", 1))
        self._media_features: dict[int, tuple[Qwen35Media, torch.Tensor]] = {}
        self.weight_quantization = (
            "modelopt-fp8-nvfp4" if is_modelopt_nvfp4(self.raw_config) else "bfloat16"
        )
        self.nvfp4_backend: str | None = None
        if self.weight_quantization != "bfloat16":
            capability = torch.cuda.get_device_capability(self.device)
            if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
                raise RuntimeError(
                    "Qwen3.5 NVFP4 weights require Blackwell-class FP4 hardware, "
                    f"got sm{capability[0]}{capability[1]}"
                )
            if nvfp4_backend == "auto":
                nvfp4_backend = (
                    "cudnn"
                    if (torch.backends.cudnn.version() or 0) >= 92301
                    else "cute-dsl"
                )
            prepare_nvfp4_modules(self._model, nvfp4_backend)
            self.nvfp4_backend = nvfp4_backend
        selected_max_seq_len = (
            self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        )
        if (
            isinstance(selected_max_seq_len, bool)
            or not isinstance(selected_max_seq_len, int)
            or selected_max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        if selected_max_seq_len > self.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.max_seq_len = selected_max_seq_len
        self.max_batch_size = max_batch_size
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self._sampler = FlashInferSampler(torch)
        self._speculation = SpeculativeStats(speculative_tokens)
        self._graph_device_caches: dict[GraphShape, DeviceCache] = {}
        self._graph_attention: dict[GraphShape, GraphPagedAttentionRuntime] = {}
        self.kv_cache_dtype = kv_cache_dtype
        cache_dtype = torch.float8_e4m3fn if kv_cache_dtype == "fp8" else torch.bfloat16

        self.cache_geometry = CacheGeometry.from_config(
            self.config,
            include_mtp=speculative_tokens > 0,
            kv_element_bytes=cache_dtype.itemsize,
        )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry.slab_bytes,
            overhead_bytes=_CACHE_WORKSPACE_BYTES,
        )
        slab_capacity = capacity.parent_capacity
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        self.cache_capacity_plan = replace(
            capacity,
            parent_capacity=slab_capacity,
            allocation_bytes=(slab_capacity + capacity.reserved_parent_count)
            * capacity.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(slab_capacity)
        self._cache = CacheArena(
            torch,
            self.device,
            self.cache_layout,
            self.cache_geometry,
            kv_dtype=cache_dtype,
        )
        self._attention = PagedAttentionRuntime(
            self.device,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            workspace_bytes=_CACHE_WORKSPACE_BYTES,
        )
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] > self.max_seq_len for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity exceeds max_seq_len")
        if any(shape[1] <= speculative_tokens + 1 for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity is too small")
        self._graph_workspace = torch.empty(
            _GRAPH_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=self._cuda_failure,
        )
        lookahead = speculative_tokens + 1 if speculative_tokens else 0
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=lookahead,
            cache_layout=self.cache_layout,
        )

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    def bind_resources(self, binding: ResourceBinding) -> None:
        self._cache.binding.bind_resources(binding)

    @staticmethod
    def _media_at(plan: BatchPlan, row: int) -> Qwen35Media | None:
        media = plan.media[row]
        if media is None:
            return None
        if not isinstance(media, Qwen35Media):
            raise TypeError("Qwen3_5CudaRunner media must be Qwen35Media")
        return media

    def _rope_deltas(self, plan: BatchPlan) -> torch.Tensor:
        return torch.tensor(
            [
                media.rope_delta if (media := self._media_at(plan, row)) else 0
                for row in range(plan.num_sequences)
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _decode_rotary_positions(
        self, plan: BatchPlan, positions: torch.Tensor
    ) -> torch.Tensor:
        adjusted = positions + self._rope_deltas(plan)
        return adjusted[None, :, None].expand(3, -1, -1)

    def _prefill_rotary_positions(
        self,
        media: Qwen35Media | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if media is None:
            return positions[None].expand(3, -1, -1)
        columns = []
        for position in positions[0].tolist():
            absolute = int(position)
            if absolute < len(media.token_ids):
                columns.append(media.position_ids[:, absolute])
            else:
                columns.append(
                    torch.full((3,), absolute + media.rope_delta, dtype=torch.int64)
                )
        return torch.stack(columns, dim=1).to(self.device)[:, None]

    def _features_for(self, sequence_id: int, media: Qwen35Media) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing[0] is not media:
                raise ValueError("a live Qwen sequence changed its media object")
            return existing[1]
        visual = self._model.visual
        if visual is None:
            raise RuntimeError("Qwen checkpoint did not load a visual tower")
        with torch.inference_mode():
            output = visual(
                media.pixel_values.to(self.device, dtype=torch.bfloat16),
                media.image_grid_thw.to(self.device),
            )
            features = output.pooler_output.detach()
        if features.shape != (len(media.image_positions), self.config.hidden_size):
            raise ValueError("Qwen vision features do not match image placeholders")
        self._media_features[sequence_id] = (media, features)
        return features

    def _merge_media_embeddings(
        self,
        plan: BatchPlan,
        row: int,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Qwen35Media | None:
        media = self._media_at(plan, row)
        if media is None:
            return None
        position_list = [int(position) for position in positions[0].tolist()]
        for token_id, position in zip(
            token_ids[0].tolist(), position_list, strict=True
        ):
            if (
                position < len(media.token_ids)
                and int(token_id) != media.token_ids[position]
            ):
                raise ValueError("scheduled token does not match Qwen35Media prompt")
        feature_indices = {
            position: index for index, position in enumerate(media.image_positions)
        }
        if not any(position in feature_indices for position in position_list):
            return media
        features = self._features_for(int(plan.sequence_ids[row]), media)
        for query_index, position in enumerate(position_list):
            feature_index = feature_indices.get(position)
            if feature_index is not None:
                embeddings[0, query_index].copy_(features[feature_index])
        return media

    @staticmethod
    def _capture_inputs(shape: GraphShape, vocab_size: int) -> dict[str, object]:
        batch_size, _ = shape
        return {
            "token_ids": (0,) * batch_size,
            "positions": (1,) * batch_size,
            "rope_deltas": (0,) * batch_size,
            "sampling_temperatures": (1.0,) * batch_size,
            "sampling_top_ps": (1.0,) * batch_size,
            "sampling_top_ks": (vocab_size,) * batch_size,
            "sampling_seeds": tuple(range(batch_size)),
            "sampling_offsets": (0,) * batch_size,
            "sampling_stochastic_mask": (False,) * batch_size,
        }

    def capture_cuda_graphs(self) -> CudaGraphStats:
        """Capture every configured Qwen decode bucket before serving."""

        captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape in captured:
                continue
            inputs = self._capture_inputs(shape, self.vocab_size)
            inputs.update(self._cache.graph_shape_capture_inputs(*shape))
            for index, call in enumerate(self._graph_attention_for(shape).calls):
                inputs[f"attention.{index}.logical_pages"] = (0,) * len(
                    call.prepared.logical_pages
                )
            self._graphs.capture(shape, inputs)
        if self._graphs.shapes:
            self._cache.reset_null()
            torch.cuda.synchronize(self.device)
        return self._graphs.stats

    def _graph_sample(
        self,
        logits: torch.Tensor,
        sampling: SamplingTensors,
        step: int,
    ) -> torch.Tensor:
        tensors = SamplingTensors(
            temperatures=sampling.temperatures,
            top_ps=sampling.top_ps,
            top_ks=sampling.top_ks,
            seeds=sampling.seeds,
            offsets=sampling.offsets + step * SAMPLING_OFFSET_STRIDE,
            stochastic_mask=sampling.stochastic_mask,
        )
        return self._sampler.sample_tensors(logits.float(), tensors)

    def _graph_specs(self) -> tuple[GraphAttentionSpec, ...]:
        target = self.config.full_layer_count
        if not self.speculative_tokens:
            return (GraphAttentionSpec(0, target, 1, 1),)
        width = self.speculative_tokens + 1
        return (
            GraphAttentionSpec(0, target, 1, 1),
            *(
                GraphAttentionSpec(2, 1, 1, step + 1)
                for step in range(self.speculative_tokens)
            ),
            *(GraphAttentionSpec(0, target, 1, step + 2) for step in range(width)),
            GraphAttentionSpec(2, 1, width, width + 1),
        )

    def _graph_attention_for(self, shape: GraphShape) -> GraphPagedAttentionRuntime:
        runtime = self._graph_attention.get(shape)
        if runtime is None:
            cache = self._graph_device_caches.get(shape)
            if cache is None:
                cache = self._cache.graph_cache(*shape)
                self._graph_device_caches[shape] = cache
            runtime = GraphPagedAttentionRuntime(
                self.device,
                cache,
                self._graph_specs(),
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                workspace=self._graph_workspace,
            )
            self._graph_attention[shape] = runtime
        return runtime

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        attention = self._graph_attention_for(shape)
        cache = self._graph_device_caches[shape]
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        rope_deltas = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        output_width = 2 * self.speculative_tokens + 2 if self.speculative_tokens else 1
        output_ids = torch.empty(
            (batch_size, output_width), dtype=torch.long, device=self.device
        )

        def ordinary_launch() -> None:
            attention.reset()
            recurrent, convolution, _ = self._initial_state(cache, positions)
            hidden, recurrent, convolution = self._model.language_model(
                token_ids[:, None],
                positions[:, None],
                cache=cache,
                recurrent_states=recurrent,
                convolution_states=convolution,
                sequence_lengths=positions + 1,
                max_sequence_length=graph_seq_len,
                attention_runtime=attention,
                rotary_positions=(positions + rope_deltas)[None, :, None].expand(
                    3, -1, -1
                ),
            )
            self._store_state(cache, positions, recurrent, convolution, hidden[:, -1])
            output_ids[:, 0].copy_(
                self._graph_sample(self._model.logits(hidden[:, -1]), sampling, 0)
            )

        def speculative_launch() -> None:
            attention.reset()
            recurrent, convolution, _ = self._initial_state(cache, positions)
            anchor_hidden, recurrent, convolution = self._model.language_model(
                token_ids[:, None],
                positions[:, None],
                cache=cache,
                recurrent_states=recurrent,
                convolution_states=convolution,
                sequence_lengths=positions + 1,
                max_sequence_length=graph_seq_len,
                attention_runtime=attention,
                rotary_positions=(positions + rope_deltas)[None, :, None].expand(
                    3, -1, -1
                ),
            )
            first = self._graph_sample(
                self._model.logits(anchor_hidden[:, -1]), sampling, 0
            )

            draft_columns = []
            draft_input = first
            draft_hidden = anchor_hidden[:, -1:]
            for step in range(self.speculative_tokens):
                draft_position = positions + step
                draft_hidden = self._mtp_forward(
                    cache,
                    draft_input[:, None],
                    draft_hidden,
                    draft_position[:, None],
                    attention_runtime=attention,
                    max_sequence_length=graph_seq_len,
                    rope_deltas=rope_deltas,
                )
                draft_input = self._graph_sample(
                    self._model.logits(draft_hidden[:, -1]), sampling, step + 1
                )
                draft_columns.append(draft_input)
            drafts = torch.stack(draft_columns, dim=1)
            verify_ids = torch.cat((first[:, None], drafts), dim=1)

            active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            selected_recurrent = recurrent
            selected_convolution = convolution
            selected_hidden = anchor_hidden[:, -1]
            verify_hidden = []
            target_columns = []
            work_recurrent = recurrent
            work_convolution = convolution
            for step in range(self.speculative_tokens + 1):
                verify_position = positions + step + 1
                hidden, work_recurrent, work_convolution = self._model.language_model(
                    verify_ids[:, step : step + 1],
                    verify_position[:, None],
                    cache=cache,
                    recurrent_states=work_recurrent,
                    convolution_states=work_convolution,
                    sequence_lengths=verify_position + 1,
                    max_sequence_length=graph_seq_len,
                    attention_runtime=attention,
                    rotary_positions=(verify_position + rope_deltas)[
                        None, :, None
                    ].expand(3, -1, -1),
                )
                current_hidden = hidden[:, -1]
                selected_recurrent = torch.where(
                    active[:, None, None, None, None],
                    work_recurrent,
                    selected_recurrent,
                )
                selected_convolution = torch.where(
                    active[:, None, None, None],
                    work_convolution,
                    selected_convolution,
                )
                selected_hidden = torch.where(
                    active[:, None], current_hidden, selected_hidden
                )
                verify_hidden.append(current_hidden)
                target = self._graph_sample(
                    self._model.logits(current_hidden), sampling, step + 1
                )
                target_columns.append(target)
                if step < self.speculative_tokens:
                    active = active & (drafts[:, step] == target)
            targets = torch.stack(target_columns, dim=1)
            matches = drafts == targets[:, : self.speculative_tokens]
            accepted = torch.cumprod(matches.to(torch.int64), dim=1).sum(1)
            processed = accepted + 1

            repair_ids = torch.cat((drafts, targets[:, -1:]), dim=1)
            correction = targets.gather(1, accepted[:, None])
            repair_ids = repair_ids.scatter(1, accepted[:, None], correction)
            repair_positions = positions[:, None] + torch.arange(
                1,
                self.speculative_tokens + 2,
                dtype=torch.long,
                device=self.device,
            )
            self._mtp_forward(
                cache,
                repair_ids,
                torch.stack(verify_hidden, dim=1),
                repair_positions,
                attention_runtime=attention,
                max_sequence_length=graph_seq_len,
                rope_deltas=rope_deltas,
            )
            self._store_state(
                cache,
                positions + processed,
                selected_recurrent,
                selected_convolution,
                selected_hidden,
            )
            output_ids[:, 0].copy_(first)
            output_ids[:, 1 : self.speculative_tokens + 1].copy_(drafts)
            output_ids[:, self.speculative_tokens + 1 :].copy_(targets)

        launch = speculative_launch if self.speculative_tokens else ordinary_launch
        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "rope_deltas": rope_deltas,
        }
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(attention.graph_inputs())
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            prepare=attention.reset,
            validate_eager_output=False,
        )

    def _graph_forward(
        self, plan: BatchPlan, cache_plan: PagedCachePlan
    ) -> list[int] | list[list[int]] | None:
        if not self._graphs.shapes:
            return None
        batch_size = plan.num_sequences
        required_length = int(plan.positions.max()) + self.speculative_tokens + 2
        if not self.speculative_tokens:
            required_length = int(plan.positions.max()) + 1
        shape = self._graphs.match(
            lambda candidate: (
                candidate[0] >= batch_size and candidate[1] >= required_length
            )
        )
        if shape is None:
            self._graphs.fallback("shape")
            return None
        padding = shape[0] - batch_size
        inputs: dict[str, object] = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
            "rope_deltas": np.pad(self._rope_deltas(plan).cpu().numpy(), (0, padding)),
        }
        inputs.update(
            self._cache.graph_plan_inputs(self._graph_device_caches[shape], cache_plan)
        )
        inputs.update(
            self._graph_attention[shape].plan(cache_plan, np.asarray(plan.positions))
        )
        sampling_inputs = self._sampler.host_inputs(plan, self.vocab_size)
        defaults = {
            "sampling_temperatures": 1.0,
            "sampling_top_ps": 1.0,
            "sampling_top_ks": self.vocab_size,
            "sampling_seeds": 0,
            "sampling_offsets": 0,
            "sampling_stochastic_mask": False,
        }
        for name, default in defaults.items():
            sampling_inputs[name] = (
                *sampling_inputs[name],
                *((default,) * padding),
            )
        inputs.update(sampling_inputs)
        output = self._graphs.run(shape, inputs)[:batch_size]
        if not self.speculative_tokens:
            return [int(row[0]) for row in output]
        drafts = [row[1 : self.speculative_tokens + 1] for row in output]
        targets = [row[self.speculative_tokens + 1 :] for row in output]
        accepted = self._speculation.accept(drafts, targets)
        return [
            [int(row[0]), *tokens]
            for row, tokens in zip(output, accepted.token_rows, strict=True)
        ]

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        rows = plan.num_sequences
        if rows == 0 or rows > self.max_batch_size:
            raise ValueError("Qwen3.5 batch size is outside runner capacity")
        indptr = np.asarray(plan.token_indptr)
        if (
            len(indptr) != rows + 1
            or indptr[0] != 0
            or indptr[-1] != len(plan.token_ids)
            or np.any(indptr[1:] <= indptr[:-1])
        ):
            raise ValueError("Qwen3.5 sequence metadata is malformed")
        if len(plan.context_lengths) != rows or len(plan.sampling_mask) != rows:
            raise ValueError("Qwen3.5 row metadata is malformed")
        slices = tuple((int(start), int(end)) for start, end in zip(indptr, indptr[1:]))
        for row, (start, end) in enumerate(slices):
            context = int(plan.context_lengths[row])
            expected = np.arange(context, context + end - start)
            if not np.array_equal(plan.positions[start:end], expected):
                raise ValueError("Qwen3.5 positions must continue their context")
            extra = (
                self.speculative_tokens + 1
                if plan.phase == BatchPhase.DECODE and self.speculative_tokens
                else 0
            )
            if context + end - start + extra > self.max_seq_len:
                raise ValueError("Qwen3.5 sequence exceeds max_seq_len")
        if plan.phase == BatchPhase.DECODE:
            if any(end - start != 1 for start, end in slices):
                raise ValueError("Qwen3.5 decode rows must contain one anchor token")
            if any(int(context) == 0 for context in plan.context_lengths):
                raise ValueError("Qwen3.5 decode requires a cached prefix")
        elif plan.phase != BatchPhase.PREFILL:
            raise ValueError("Qwen3.5 plan phase must be prefill or decode")
        return slices

    def _sample(self, logits: torch.Tensor, plan: BatchPlan, step: int = 0):
        return self._sampler(
            logits.float(),
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _initial_state(
        self, cache: DeviceCache, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = (contexts - 1).clamp_min(0)[:, None]
        recurrent, convolution, target_hidden = cache.state_at(positions)
        recurrent = recurrent[:, 0].clone()
        convolution = convolution[:, 0].clone()
        target_hidden = target_hidden[:, 0].clone()
        new = contexts == 0
        recurrent.masked_fill_(new[:, None, None, None, None], 0)
        convolution.masked_fill_(new[:, None, None, None], 0)
        target_hidden.masked_fill_(new[:, None], 0)
        return recurrent, convolution, target_hidden

    @staticmethod
    def _store_state(
        cache: DeviceCache,
        positions: torch.Tensor,
        recurrent: torch.Tensor,
        convolution: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> None:
        cache.store_state(
            positions[:, None],
            recurrent[:, None],
            convolution[:, None],
            target_hidden[:, None],
        )

    def _mtp_forward(
        self,
        cache: DeviceCache,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention_runtime: object | None = None,
        max_sequence_length: int | None = None,
        rope_deltas: torch.Tensor | None = None,
        rotary_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mtp = self._model.mtp
        if mtp is None:
            raise RuntimeError("Qwen3.5 MTP was not loaded")
        embeddings = self._model.language_model.embed_tokens(input_ids)
        selected_rotary_positions = positions
        if rotary_positions is not None:
            selected_rotary_positions = rotary_positions
        elif rope_deltas is not None:
            selected_rotary_positions = (positions + rope_deltas[:, None])[None].expand(
                3, -1, -1
            )
        rotary = self._model.language_model.rotary(
            selected_rotary_positions, embeddings.dtype
        )
        sequence_lengths = positions[:, -1] + 1
        if max_sequence_length is None:
            max_sequence_length = int(sequence_lengths.max().item())
        return mtp(
            embeddings,
            previous_hidden,
            positions,
            rotary=rotary,
            cache=cache,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            attention_runtime=attention_runtime or self._attention,
        )

    def _prefill_mtp(
        self,
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        context: int,
        sampled_token: torch.Tensor | None,
        media: Qwen35Media | None,
    ) -> None:
        if self._model.mtp is None:
            return
        if context:
            self._mtp_forward(
                cache,
                token_ids[:, :1],
                previous_hidden[:, None],
                positions[:, :1] - 1,
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, :1] - 1
                ),
            )
        if token_ids.shape[1] > 1:
            self._mtp_forward(
                cache,
                token_ids[:, 1:],
                target_hidden[:, :-1],
                positions[:, :-1],
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, :-1]
                ),
            )
        if sampled_token is not None:
            self._mtp_forward(
                cache,
                sampled_token[:, None],
                target_hidden[:, -1:],
                positions[:, -1:],
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, -1:]
                ),
            )

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[int]:
        final_logits = []
        outputs = []
        states = []
        for row, (start, end) in enumerate(rows):
            row_cache = cache.row(row)
            row_ids = token_ids[start:end][None]
            row_positions = positions[start:end][None]
            context = int(plan.context_lengths[row])
            contexts = torch.tensor([context], dtype=torch.long, device=self.device)
            recurrent, convolution, previous_hidden = self._initial_state(
                row_cache, contexts
            )
            embeddings = self._model.language_model.embed_tokens(row_ids)
            media = self._merge_media_embeddings(
                plan, row, row_ids, row_positions, embeddings
            )
            hidden, recurrent, convolution = self._model.language_model(
                row_ids,
                row_positions,
                cache=row_cache,
                recurrent_states=recurrent,
                convolution_states=convolution,
                sequence_lengths=row_positions[:, -1] + 1,
                max_sequence_length=context + end - start,
                attention_runtime=self._attention,
                input_embeddings=embeddings,
                rotary_positions=self._prefill_rotary_positions(media, row_positions),
            )
            final_logits.append(self._model.logits(hidden[:, -1]))
            states.append(
                (
                    row_cache,
                    row_ids,
                    row_positions,
                    hidden,
                    recurrent,
                    convolution,
                    previous_hidden,
                    context,
                )
            )
        sampled = self._sample(torch.cat(final_logits), plan)
        for row, state in enumerate(states):
            (
                row_cache,
                row_ids,
                row_positions,
                hidden,
                recurrent,
                convolution,
                previous_hidden,
                context,
            ) = state
            sampled_token = sampled[row : row + 1] if plan.sampling_mask[row] else None
            media = self._media_at(plan, row)
            self._prefill_mtp(
                row_cache,
                row_ids,
                row_positions,
                hidden,
                previous_hidden,
                context,
                sampled_token,
                media,
            )
            self._store_state(
                row_cache,
                row_positions[:, -1],
                recurrent,
                convolution,
                hidden[:, -1],
            )
            outputs.append(int(sampled[row]))
        return outputs

    def _ordinary_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[int]:
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        recurrent, convolution, _ = self._initial_state(cache, contexts)
        hidden, recurrent, convolution = self._model.language_model(
            anchors[:, None],
            positions[:, None],
            cache=cache,
            recurrent_states=recurrent,
            convolution_states=convolution,
            sequence_lengths=positions + 1,
            max_sequence_length=int(positions.max().item()) + 1,
            attention_runtime=self._attention,
            rotary_positions=self._decode_rotary_positions(plan, positions),
        )
        self._store_state(cache, positions, recurrent, convolution, hidden[:, -1])
        return self._sample(self._model.logits(hidden[:, -1]), plan).tolist()

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[list[int]]:
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        rope_deltas = self._rope_deltas(plan)
        initial_recurrent, initial_convolution, _ = self._initial_state(cache, contexts)
        anchor_hidden, anchor_recurrent, anchor_convolution = (
            self._model.language_model(
                anchors[:, None],
                positions[:, None],
                cache=cache,
                recurrent_states=initial_recurrent,
                convolution_states=initial_convolution,
                sequence_lengths=positions + 1,
                max_sequence_length=int(positions.max().item()) + 1,
                attention_runtime=self._attention,
                rotary_positions=(positions + rope_deltas)[None, :, None].expand(
                    3, -1, -1
                ),
            )
        )
        first = self._sample(self._model.logits(anchor_hidden[:, -1]), plan, 0)

        drafts = []
        draft_input = first
        draft_hidden = anchor_hidden[:, -1:]
        for step in range(self.speculative_tokens):
            draft_position = positions + step
            draft_hidden = self._mtp_forward(
                cache,
                draft_input[:, None],
                draft_hidden,
                draft_position[:, None],
                rope_deltas=rope_deltas,
            )
            draft_input = self._sample(
                self._model.logits(draft_hidden[:, -1]), plan, step + 1
            )
            drafts.append(draft_input)
        draft_ids = torch.stack(drafts, dim=1)

        verify_ids = torch.cat((first[:, None], draft_ids), dim=1)
        verify_positions = positions[:, None] + torch.arange(
            1, self.speculative_tokens + 2, device=self.device
        )
        verify_hidden, verify_recurrent, verify_convolution = (
            self._model.language_model(
                verify_ids,
                verify_positions,
                cache=cache,
                recurrent_states=anchor_recurrent,
                convolution_states=anchor_convolution,
                sequence_lengths=verify_positions[:, -1] + 1,
                max_sequence_length=int(verify_positions.max().item()) + 1,
                attention_runtime=self._attention,
                rotary_positions=(verify_positions + rope_deltas[:, None])[None].expand(
                    3, -1, -1
                ),
            )
        )
        verify_logits = self._model.logits(verify_hidden)
        targets = torch.stack(
            [
                self._sample(verify_logits[:, step].contiguous(), plan, step + 1)
                for step in range(self.speculative_tokens + 1)
            ],
            dim=1,
        )
        acceptance = self._speculation.accept(draft_ids.tolist(), targets.tolist())

        rows = []
        for row, accepted in enumerate(acceptance.accepted_counts):
            processed = accepted + 1
            row_cache = cache.row(row)
            if accepted == self.speculative_tokens:
                committed_hidden = verify_hidden[row : row + 1]
                recurrent = verify_recurrent[row : row + 1]
                convolution = verify_convolution[row : row + 1]
            else:
                committed_hidden, recurrent, convolution = self._model.language_model(
                    verify_ids[row : row + 1, :processed],
                    verify_positions[row : row + 1, :processed],
                    cache=row_cache,
                    recurrent_states=anchor_recurrent[row : row + 1],
                    convolution_states=anchor_convolution[row : row + 1],
                    sequence_lengths=verify_positions[row : row + 1, processed - 1] + 1,
                    max_sequence_length=int(verify_positions[row, processed - 1].item())
                    + 1,
                    attention_runtime=self._attention,
                    rotary_positions=(
                        verify_positions[row : row + 1, :processed]
                        + rope_deltas[row : row + 1, None]
                    )[None].expand(3, -1, -1),
                )
            correction = targets[row, accepted : accepted + 1]
            repair_ids = torch.cat((draft_ids[row, :accepted], correction))[None]
            repair_positions = verify_positions[row : row + 1, :processed]
            self._mtp_forward(
                row_cache,
                repair_ids,
                committed_hidden[:, :processed],
                repair_positions,
                rope_deltas=rope_deltas[row : row + 1],
            )
            commit_position = positions[row : row + 1] + processed
            self._store_state(
                row_cache,
                commit_position,
                recurrent,
                convolution,
                committed_hidden[:, processed - 1],
            )
            rows.append([int(first[row]), *acceptance.token_rows[row]])
        return rows

    def forward(self, plan: BatchPlan) -> list[int] | list[list[int]]:
        rows = self._validate_plan(plan)
        lookahead = (
            self.speculative_tokens + 1
            if plan.phase == BatchPhase.DECODE and self.speculative_tokens
            else 0
        )
        cache_plan = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=lookahead
        )
        with self._cuda_failure:
            if plan.phase == BatchPhase.DECODE:
                graph_output = self._graph_forward(plan, cache_plan)
                if graph_output is not None:
                    return graph_output
            staging = TensorStager(self.device)
            cache = self._cache.tensorize(cache_plan, staging)
            token_ids = staging.device("token_ids", plan.token_ids, torch.long)
            positions = staging.device("positions", plan.positions, torch.long)
            with (
                torch.inference_mode(),
                use_default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if plan.phase == BatchPhase.PREFILL:
                    return self._prefill(plan, rows, cache, token_ids, positions)
                if self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                return self._ordinary_decode(plan, cache, token_ids, positions)

    def release(self, sequence_ids: list[int]) -> None:
        """Release request-local image embeddings after Engine retirement."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)


__all__ = ["Qwen3_5CudaRunner"]
