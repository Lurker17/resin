"""CUDA runner for the Kimi K2.5 family with paged EAGLE3 decoding."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
import torch.distributed as dist

from resin import BatchPhase, ResourceBinding, RunnerConfig
from resin.backend.cuda.graphs import (
    CudaGraphCache,
    CudaGraphStats,
    CudaGraphStep,
    GraphShape,
    graph_shapes,
)
from resin.backend.cuda.parallel import FlashInferAllReduce, TensorParallelContext
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
    FlashInferSampler,
    SamplingTensors,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.speculative import accept_speculative_tokens
from resin.protocols import SyncRunner
from resin_models_cuda.kimi_k25.cache import (
    GRAPH_SCRATCH_PARENTS,
    CacheArena,
    CacheGeometry,
    DeviceCache,
    build_cache_plan,
    plan_cache_capacity,
)
from resin_models_cuda.kimi_k25.checkpoint import (
    load_draft_config,
    load_target_config,
    resolve_draft_path,
    resolve_target_path,
)
from resin_models_cuda.kimi_k25.config import EagleVersion
from resin_models_cuda.kimi_k25.media import KimiK25Media
from resin_models_cuda.kimi_k25.model import KimiForCausalLM
from resin_models_cuda.kimi_k25.vision import KimiVisionEncoder
from resin_models_cuda.kimi_k25.weights import (
    load_draft_weights,
    load_target_weights,
    load_vision_weights,
)

if TYPE_CHECKING:
    from resin import BatchPlan


@contextmanager
def _default_dtype(dtype: Any) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class KimiK25CudaRunner(SyncRunner):
    """Dynamically batched Kimi target plus LightSeek EAGLE3/3.1 draft."""

    def __init__(
        self,
        target_model: str | Path,
        draft_model: str | Path,
        *,
        max_seq_len: int,
        max_batch_size: int = 1,
        tensor_parallel: TensorParallelContext,
        device: str | torch.device | None = None,
        cuda_graph_decode_shapes: object | None = None,
        fast_allreduce: bool = True,
        speculative_tokens: int = 3,
        prefill_chunk_size: int = 4096,
        eagle_version: EagleVersion | str | None = None,
        gpu_memory_utilization: float = 0.90,
        cache_parent_capacity: int | None = None,
    ) -> None:
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("max_batch_size", max_batch_size),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or not 0 <= speculative_tokens <= 3
        ):
            raise ValueError("speculative_tokens must be between zero and three")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        if not torch.cuda.is_available():
            raise RuntimeError("KimiK25CudaRunner requires CUDA")

        self.tensor_parallel = tensor_parallel
        self.tensor_parallel_rank = tensor_parallel.rank
        self.tensor_parallel_size = tensor_parallel.size
        self.device = torch.device(device or f"cuda:{tensor_parallel.rank}")
        if self.device.type != "cuda":
            raise RuntimeError("KimiK25CudaRunner requires a CUDA device")
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("the Kimi NVFP4 runner requires Blackwell")
        if self.tensor_parallel_size != 4:
            raise ValueError("the Kimi NVFP4 runner currently requires TP=4")
        if not dist.is_initialized():
            raise RuntimeError("Kimi TP requires an initialized process group")
        if (
            dist.get_world_size() != self.tensor_parallel_size
            or dist.get_rank() != self.tensor_parallel_rank
        ):
            raise ValueError("the default process group does not match TP layout")

        self.target_path = resolve_target_path(target_model)
        self.draft_path = resolve_draft_path(draft_model)
        self.config = load_target_config(self.target_path)
        self.draft_config = load_draft_config(self.draft_path)
        self.config.validate_tensor_parallel(self.tensor_parallel_size)
        self.draft_config.validate_target(self.config)
        if eagle_version is not None:
            expected_eagle = EagleVersion(eagle_version)
            if self.draft_config.eagle_version is not expected_eagle:
                raise ValueError(
                    f"expected {expected_eagle.value}, got "
                    f"{self.draft_config.eagle_version.value} draft weights"
                )
        if max_seq_len > self.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the Kimi checkpoint context length")
        if max_batch_size * max_seq_len > 2**32 - 1:
            raise ValueError("max_batch_size * max_seq_len exceeds scheduler metadata")

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.speculative_tokens = speculative_tokens
        self.prefill_chunk_size = min(prefill_chunk_size, max_seq_len)
        self.physical_seq_len = max_seq_len + speculative_tokens
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self._sampler = FlashInferSampler(torch)
        self._speculative_blocks = 0
        self._speculative_drafts = 0
        self._speculative_accepted = 0
        self._speculative_emitted = 0
        self._graph_device_caches: dict[GraphShape, DeviceCache] = {}
        self._media_features: dict[
            int, tuple[tuple[int, ...], tuple[torch.Tensor, ...]]
        ] = {}

        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=1)
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("CUDA graph batch capacity exceeds max_batch_size")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=lambda: torch.cuda.synchronize(self.device),
        )

        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=self.device)
        max_allreduce_tokens = max(
            self.prefill_chunk_size,
            max_batch_size * (speculative_tokens + 1),
        )
        self._allreduce = FlashInferAllReduce(
            tensor_parallel,
            max_token_num=max_allreduce_tokens,
            hidden_dim=self.config.hidden_size,
            dtype=torch.bfloat16,
            enabled=fast_allreduce,
        )
        self.allreduce_backend = self._allreduce.backend
        with _default_dtype(torch.bfloat16), torch.device(self.device):
            self._model = KimiForCausalLM(
                self.config,
                self.draft_config,
                tensor_parallel,
                self._allreduce.sum,
                workspace,
                self.physical_seq_len,
            )
        self._model.eval()
        if tensor_parallel.rank == 0:
            print("[kimi] loading target weights", flush=True)
        self.weight_load = {
            "target": load_target_weights(
                self.target_path, self._model, tp=tensor_parallel
            )
        }
        if tensor_parallel.rank == 0:
            print("[kimi] loading draft weights", flush=True)
        self.weight_load["draft"] = load_draft_weights(
            self.draft_path, self._model, tp=tensor_parallel
        )
        if tensor_parallel.rank == 0:
            print("[kimi] preparing NVFP4 weights", flush=True)
        self._model.prepare_nvfp4()
        if tensor_parallel.rank == 0:
            print("[kimi] loading vision tower and projector", flush=True)
        with _default_dtype(torch.bfloat16), torch.device(self.device):
            self._vision = KimiVisionEncoder(self.target_path)
        self._vision.eval()
        self.weight_load["vision"] = load_vision_weights(self.target_path, self._vision)

        self.cache_geometry = CacheGeometry.build(
            self.config.num_hidden_layers,
            latent_width=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            hidden_size=self.config.hidden_size,
        )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry,
            reserved_parent_count=1 + GRAPH_SCRATCH_PARENTS,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        capacity_tensor = torch.tensor(
            local_capacity, dtype=torch.int64, device=self.device
        )
        dist.all_reduce(
            capacity_tensor,
            op=dist.ReduceOp.MIN,
            group=self.tensor_parallel.process_group,
        )
        local_capacity = int(capacity_tensor.item())
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + capacity_plan.reserved_parent_count)
            * capacity_plan.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(local_capacity)
        self._cache = CacheArena(
            torch, self.device, self.cache_layout, self.cache_geometry
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=max_batch_size * self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=speculative_tokens,
            cache_layout=self.cache_layout,
        )

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        acceptance_rate = (
            self._speculative_accepted / self._speculative_drafts
            if self._speculative_drafts
            else 0.0
        )
        return {
            "blocks": self._speculative_blocks,
            "drafts": self._speculative_drafts,
            "accepted": self._speculative_accepted,
            "emitted": self._speculative_emitted,
            "acceptance_rate": acceptance_rate,
        }

    def bind_resources(self, binding: ResourceBinding) -> None:
        """Bind the single fungible Kimi cache arena to its Engine pool."""

        if binding.cache_pool_id is None or binding.cache_layout is None:
            raise ValueError("Kimi requires managed cache resources")
        self._cache.bind(binding.cache_pool_id, binding.cache_layout)

    def _cache_plan(self, plan: BatchPlan):
        return build_cache_plan(
            plan,
            self.cache_layout,
            decode_lookahead=self.speculative_tokens,
        )

    def _reset_graph_scratch(self) -> None:
        self._cache.reset_scratch()

    @staticmethod
    def _capture_inputs(shape: GraphShape, vocab_size: int) -> dict[str, tuple]:
        (batch_size,) = shape
        return {
            "token_ids": (0,) * batch_size,
            "positions": (1,) * batch_size,
            "sampling_temperatures": (1.0,) * batch_size,
            "sampling_top_ps": (1.0,) * batch_size,
            "sampling_top_ks": (vocab_size,) * batch_size,
            "sampling_seeds": tuple(range(batch_size)),
            "sampling_offsets": (0,) * batch_size,
            "sampling_stochastic_mask": (False,) * batch_size,
        }

    def capture_cuda_graphs(self) -> CudaGraphStats:
        """Capture every configured decode bucket before serving requests."""

        captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape in captured:
                continue
            inputs = self._capture_inputs(shape, self.vocab_size)
            inputs.update(
                self._cache.graph_shape_capture_inputs(shape[0], self.physical_seq_len)
            )
            self._graphs.capture(shape, inputs)
        if self._graphs.shapes:
            self._reset_graph_scratch()
            torch.cuda.synchronize(self.device)
        return self._graphs.stats

    def _broadcast_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        dist.broadcast(token_ids, src=0, group=self.tensor_parallel.process_group)
        return token_ids

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
        return self._broadcast_tokens(self._sampler.sample_tensors(logits, tensors))

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        (batch_size,) = shape
        graph_seq_len = self.physical_seq_len
        cache = self._cache.graph_cache(batch_size, graph_seq_len)
        self._graph_device_caches[shape] = cache
        token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        output_width = (
            1 if not self.speculative_tokens else 2 * self.speculative_tokens + 1
        )
        output_ids = torch.empty(
            (batch_size, output_width), dtype=torch.long, device=self.device
        )

        def launch() -> None:
            with _default_dtype(torch.bfloat16), torch.device(self.device):
                if not self.speculative_tokens:
                    query_positions = positions[:, None]
                    logits, _ = self._model.target_forward(
                        token_ids,
                        query_positions,
                        cache,
                        positions + 1,
                        graph_seq_len,
                        full_logits=False,
                    )
                    output_ids[:, 0].copy_(self._graph_sample(logits, sampling, 0))
                    return

                draft_tokens = []
                draft_token = token_ids
                draft_hidden = cache.aux_at(positions - 1)[:, None]
                for step in range(self.speculative_tokens):
                    draft_position = positions[:, None] + step - 1
                    draft_logits, draft_hidden = self._model.draft(
                        draft_token,
                        draft_hidden,
                        draft_position,
                        cache,
                        positions + step,
                        graph_seq_len,
                    )
                    if draft_logits is None:
                        raise RuntimeError("Kimi draft did not return logits")
                    draft_token = self._broadcast_tokens(
                        draft_logits[:, -1].argmax(dim=-1)
                    )[:, None]
                    draft_tokens.append(draft_token[:, 0])

                drafts = torch.stack(draft_tokens, dim=1)
                verify_ids = torch.cat((token_ids, drafts), dim=1)
                verify_positions = positions[:, None] + torch.arange(
                    self.speculative_tokens + 1, device=self.device
                )
                verify_logits, target_hidden = self._model.target_forward(
                    verify_ids,
                    verify_positions,
                    cache,
                    positions + self.speculative_tokens + 1,
                    graph_seq_len,
                    full_logits=True,
                )
                target_tokens = torch.stack(
                    [
                        self._graph_sample(
                            verify_logits[:, step].contiguous(), sampling, step
                        )
                        for step in range(self.speculative_tokens + 1)
                    ],
                    dim=1,
                )
                self._model.draft(
                    target_tokens,
                    target_hidden,
                    verify_positions,
                    cache,
                    positions + self.speculative_tokens + 1,
                    graph_seq_len,
                    return_logits=False,
                )
                output_ids[:, : self.speculative_tokens].copy_(drafts)
                output_ids[:, self.speculative_tokens :].copy_(target_tokens)

        inputs = {"token_ids": token_ids, "positions": positions}
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            prepare=self._reset_graph_scratch,
        )

    def _accept_speculative(
        self, draft_rows: list[list[int]], target_rows: list[list[int]]
    ) -> list[list[int]]:
        acceptance = accept_speculative_tokens(draft_rows, target_rows)
        batch_size = len(draft_rows)
        self._speculative_blocks += batch_size
        self._speculative_drafts += batch_size * self.speculative_tokens
        self._speculative_accepted += sum(acceptance.accepted_counts)
        self._speculative_emitted += sum(map(len, acceptance.token_rows))
        return acceptance.token_rows

    def _graph_forward(self, plan: BatchPlan, cache_plan):
        if not self._graphs.shapes:
            return None
        batch_size = plan.num_sequences
        shape = self._graphs.match(lambda candidate: candidate[0] >= batch_size)
        if shape is None:
            self._graphs.fallback("shape")
            return None
        padding = shape[0] - batch_size
        inputs: dict[str, object] = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
        }
        inputs.update(
            self._cache.graph_plan_inputs(self._graph_device_caches[shape], cache_plan)
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
            return [row[0] for row in output]
        width = self.speculative_tokens
        return self._accept_speculative(
            [row[:width] for row in output],
            [row[width:] for row in output],
        )

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        rows = plan.num_sequences
        if rows == 0 or rows > self.max_batch_size:
            raise ValueError("Kimi batch size is outside runner capacity")
        indptr = plan.token_indptr
        if (
            len(indptr) != rows + 1
            or indptr[0] != 0
            or indptr[-1] != len(plan.token_ids)
            or np.any(indptr[1:] <= indptr[:-1])
        ):
            raise ValueError("Kimi sequence metadata is malformed")
        if len(plan.context_lengths) != rows or len(plan.sampling_mask) != rows:
            raise ValueError("Kimi requires row-aligned context and sampling metadata")
        if len(set(plan.sequence_ids)) != rows:
            raise ValueError("Kimi sequence IDs must be unique within a batch")
        slices = tuple((int(start), int(end)) for start, end in zip(indptr, indptr[1:]))
        positions = plan.positions
        for row, (start, end) in enumerate(slices):
            context = int(plan.context_lengths[row])
            expected = np.arange(context, context + end - start, dtype=positions.dtype)
            if not np.array_equal(positions[start:end], expected):
                raise ValueError("Kimi positions must continue each sequence context")
            if context + end - start > self.max_seq_len:
                raise ValueError("Kimi sequence exceeds max_seq_len")
        if plan.phase == BatchPhase.DECODE:
            if any(end - start != 1 for start, end in slices):
                raise ValueError("Kimi decode rows must contain one token")
            if any(context == 0 for context in plan.context_lengths):
                raise ValueError("Kimi speculative decode requires a cached prefix")
        elif plan.phase != BatchPhase.PREFILL:
            raise ValueError("Kimi plan phase must be prefill or decode")
        return slices

    def _sample_target(
        self, logits: torch.Tensor, plan: BatchPlan, step: int = 0
    ) -> torch.Tensor:
        return self._broadcast_tokens(
            self._sampler(
                logits,
                plan,
                offset_delta=step * SAMPLING_OFFSET_STRIDE,
            )
        )

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[int]:
        logits = []
        projected_rows = []
        for row, (start, end) in enumerate(rows):
            row_cache = cache.row(row, start, end)
            row_ids = token_ids[start:end][None]
            row_positions = positions[start:end][None]
            input_embeddings = self._model.target.embed_tokens(row_ids)
            self._merge_media_embeddings(
                plan,
                row,
                row_ids,
                row_positions,
                input_embeddings,
            )
            row_logits, projected = self._model.target_forward(
                row_ids,
                row_positions,
                row_cache,
                row_positions[:, -1] + 1,
                self.physical_seq_len,
                full_logits=False,
                prefill_length=int(plan.context_lengths[row]) + end - start,
                input_embeddings=input_embeddings,
            )
            logits.append(row_logits)
            projected_rows.append(projected)
        anchors = self._sample_target(torch.cat(logits), plan)

        if self.speculative_tokens:
            for row, ((start, end), projected) in enumerate(
                zip(rows, projected_rows, strict=True)
            ):
                row_cache = cache.row(row, start, end)
                context = int(plan.context_lengths[row])
                should_sample = bool(plan.sampling_mask[row])
                row_ids = token_ids[start:end][None]
                if context:
                    previous = row_cache.aux_at(
                        torch.tensor([context - 1], device=self.device)
                    )[:, None]
                    draft_ids = row_ids
                    draft_hidden = torch.cat((previous, projected), dim=1)
                    draft_start = context - 1
                else:
                    draft_ids = row_ids[:, 1:]
                    draft_hidden = projected
                    draft_start = 0
                if should_sample:
                    draft_ids = torch.cat(
                        (draft_ids, anchors[row : row + 1, None]), dim=1
                    )
                draft_hidden = draft_hidden[:, : draft_ids.shape[1]]
                if draft_ids.shape[1] == 0:
                    continue
                draft_positions = torch.arange(
                    draft_start,
                    draft_start + draft_ids.shape[1],
                    device=self.device,
                )[None]
                self._model.draft(
                    draft_ids,
                    draft_hidden,
                    draft_positions,
                    row_cache,
                    draft_positions[:, -1] + 1,
                    self.physical_seq_len,
                    return_logits=False,
                    prefill_length=draft_start + draft_ids.shape[1],
                )
        return anchors.tolist()

    def _features_for(
        self, sequence_id: int, media: KimiK25Media
    ) -> tuple[torch.Tensor, ...]:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing[0] != media.cache_key:
                raise ValueError("a live Kimi sequence changed its media content")
            return existing[1]
        with torch.inference_mode():
            features = self._vision(
                media.pixel_values.to(self.device, dtype=torch.bfloat16),
                media.grid_thws.to(self.device),
            )
        expected = tuple(span.length for span in media.image_spans)
        actual = tuple(len(tensor) for tensor in features)
        if actual != expected or any(
            tensor.shape[1:] != (self.config.hidden_size,) for tensor in features
        ):
            raise ValueError(
                f"Kimi vision features {actual} do not match placeholders {expected}"
            )
        detached = tuple(tensor.detach() for tensor in features)
        self._media_features[sequence_id] = (media.cache_key, detached)
        return detached

    def _merge_media_embeddings(
        self,
        plan: BatchPlan,
        row: int,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> None:
        media = plan.media[row]
        if media is None:
            return
        if not isinstance(media, KimiK25Media):
            raise TypeError("KimiK25CudaRunner media must be KimiK25Media")
        absolute_positions = positions[0].tolist()
        for token_id, position in zip(token_ids[0].tolist(), absolute_positions):
            if (
                position < len(media.token_ids)
                and token_id != media.token_ids[position]
            ):
                raise ValueError("scheduled token does not match KimiK25Media prompt")
        overlaps = [
            (index, span)
            for index, span in enumerate(media.image_spans)
            if span.position < absolute_positions[-1] + 1
            and span.position + span.length > absolute_positions[0]
        ]
        if not overlaps:
            return
        features = self._features_for(int(plan.sequence_ids[row]), media)
        query_start = absolute_positions[0]
        query_end = absolute_positions[-1] + 1
        for index, span in overlaps:
            start = max(query_start, span.position)
            end = min(query_end, span.position + span.length)
            hidden[0, start - query_start : end - query_start].copy_(
                features[index][start - span.position : end - span.position]
            )

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[list[int]]:
        anchors = token_ids.view(plan.num_sequences, 1)
        drafts = []
        draft_token = anchors
        draft_hidden = cache.aux_at(positions - 1)[:, None]
        for step in range(self.speculative_tokens):
            draft_position = positions[:, None] + step - 1
            draft_logits, draft_hidden = self._model.draft(
                draft_token,
                draft_hidden,
                draft_position,
                cache,
                positions + step,
                self.physical_seq_len,
            )
            if draft_logits is None:
                raise RuntimeError("Kimi draft did not return logits")
            draft_token = self._broadcast_tokens(draft_logits[:, -1].argmax(dim=-1))[
                :, None
            ]
            drafts.append(draft_token[:, 0])
        draft_ids = torch.stack(drafts, dim=1)
        verify_ids = torch.cat((anchors, draft_ids), dim=1)
        verify_positions = positions[:, None] + torch.arange(
            self.speculative_tokens + 1, device=self.device
        )
        verify_logits, target_hidden = self._model.target_forward(
            verify_ids,
            verify_positions,
            cache,
            positions + self.speculative_tokens + 1,
            self.physical_seq_len,
            full_logits=True,
        )
        target_ids = torch.stack(
            [
                self._sample_target(verify_logits[:, step].contiguous(), plan, step)
                for step in range(self.speculative_tokens + 1)
            ],
            dim=1,
        )
        self._model.draft(
            target_ids,
            target_hidden,
            verify_positions,
            cache,
            positions + self.speculative_tokens + 1,
            self.physical_seq_len,
            return_logits=False,
        )
        return self._accept_speculative(draft_ids.tolist(), target_ids.tolist())

    def forward(self, plan: BatchPlan) -> list[int] | list[list[int]]:
        """Run one Engine-planned prefill or decode step."""

        self._graphs.check()
        rows = self._validate_plan(plan)
        cache_plan = self._cache_plan(plan)
        self._cache.prepare_step(cache_plan, plan.context_lengths)
        if plan.phase == BatchPhase.DECODE:
            graph_output = self._graph_forward(plan, cache_plan)
            if graph_output is not None:
                return graph_output
        elif self._graphs.shapes:
            self._graphs.fallback("prefill")
        try:
            staging = TensorStager(self.device)
            cache = self._cache.tensorize(cache_plan, staging)
            token_ids = staging.device("token_ids", plan.token_ids, torch.long)
            positions = staging.device("positions", plan.positions, torch.long)
            with (
                torch.inference_mode(),
                _default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if plan.phase == BatchPhase.PREFILL:
                    return self._prefill(plan, rows, cache, token_ids, positions)
                if self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                query_positions = positions[:, None]
                logits, _ = self._model.target_forward(
                    token_ids.view(plan.num_sequences, 1),
                    query_positions,
                    cache,
                    positions + 1,
                    self.physical_seq_len,
                    full_logits=False,
                )
                return self._sample_target(logits, plan).tolist()
        except BaseException:
            torch.cuda.synchronize(self.device)
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Pages are retired by the Engine after this callback returns."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)


__all__ = ["KimiK25CudaRunner"]
