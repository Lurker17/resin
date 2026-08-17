"""Single-Blackwell Inkling-Small target with gamma-five DSpark decoding."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

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
from resin.backend.plans import PagedCachePlan
from resin.backend.speculative import SpeculativeStats
from resin.protocols import Runner
from resin_models_cuda.inkling.cache import (
    PREFILL_CHUNK_SIZE,
    STATE_WIDTH_PER_LAYER,
    CacheArena,
    CacheGeometry,
    DeviceCache,
)
from resin_models_cuda.inkling.checkpoint import (
    load_draft_config,
    load_target_config,
    resolve_draft_path,
    resolve_target_path,
)
from resin_models_cuda.inkling.dspark import DSparkDraft
from resin_models_cuda.inkling.media import InklingMedia
from resin_models_cuda.inkling.model import (
    InklingForCausalLM,
    select_convolution_state,
)
from resin_models_cuda.inkling.weights import load_draft_weights, load_target_weights

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


class InklingCudaRunner(Runner):
    """Inkling-Small NVFP4 runner with image input and DSpark speculation."""

    def __init__(
        self,
        target_model: str | Path,
        draft_model: str | Path,
        *,
        max_seq_len: int,
        max_batch_size: int = 128,
        device: str | torch.device = "cuda:0",
        speculative_tokens: int = 5,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        gpu_memory_utilization: float = 0.90,
        cache_slab_capacity: int | None = None,
        cuda_graph_decode_shapes: object | None = None,
    ) -> None:
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("max_batch_size", max_batch_size),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if speculative_tokens not in (0, 5):
            raise ValueError("Inkling DSpark supports exactly gamma=5 or disabled")
        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise ValueError("Inkling recurrent checkpoints require 4096-token chunks")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_slab_capacity is not None and (
            isinstance(cache_slab_capacity, bool)
            or not isinstance(cache_slab_capacity, int)
            or cache_slab_capacity <= 0
        ):
            raise ValueError("cache_slab_capacity must be a positive integer")
        if not torch.cuda.is_available():
            raise RuntimeError("InklingCudaRunner requires CUDA")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("InklingCudaRunner requires a CUDA device")
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("Inkling-Small NVFP4 requires a Blackwell GPU")
        self._cuda_failure = CudaFailureGuard(
            lambda: torch.cuda.synchronize(self.device),
            label="Inkling eager execution",
        )

        self.target_path = resolve_target_path(target_model)
        self.draft_path = resolve_draft_path(draft_model)
        self.config = load_target_config(self.target_path)
        self.draft_config = load_draft_config(self.draft_path)
        self.draft_config.validate_target(self.config)
        if max_seq_len > min(
            self.config.max_position_embeddings,
            self.draft_config.max_position_embeddings,
        ):
            raise ValueError("max_seq_len exceeds the target or draft context length")

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.speculative_tokens = speculative_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.physical_seq_len = max_seq_len + speculative_tokens
        self.vocab_size = self.config.unpadded_vocab_size
        self.eos_token_id = self.config.eos_token_id
        self._sampler = FlashInferSampler(torch)
        self._speculation = SpeculativeStats(speculative_tokens)
        self._graph_device_caches: dict[GraphShape, DeviceCache] = {}
        self._media_features: dict[int, tuple[InklingMedia, torch.Tensor]] = {}

        with _default_dtype(torch.bfloat16), torch.device(self.device):
            self._model = InklingForCausalLM(
                self.config, self.draft_config.target_layer_ids
            )
            self._draft = DSparkDraft(
                self.draft_config,
                self.config,
                self._model.target,
                self.physical_seq_len,
            )
        self._model.eval()
        self._draft.eval()
        print("[inkling] loading target weights", flush=True)
        self.weight_load = {
            "target": load_target_weights(self.target_path, self._model)
        }
        print("[inkling] loading DSpark weights", flush=True)
        self.weight_load["draft"] = load_draft_weights(self.draft_path, self._draft)
        print("[inkling] preparing routed MoE weights", flush=True)
        self._model.prepare()

        self.cache_geometry = CacheGeometry()
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry.bytes_per_slab,
        )
        slab_capacity = capacity.parent_capacity
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        self.cache_capacity_plan = replace(
            capacity,
            parent_capacity=slab_capacity,
            allocation_bytes=(
                slab_capacity + capacity.reserved_parent_count
            )
            * capacity.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(slab_capacity)
        self._cache = CacheArena(torch, self.device, self.cache_layout)
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] > self.physical_seq_len for shape in decode_graph_shapes):
            raise ValueError("CUDA graph sequence capacity exceeds max_seq_len")
        if any(shape[1] <= speculative_tokens + 1 for shape in decode_graph_shapes):
            raise ValueError("Inkling CUDA graph sequence capacity is too small")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=self._cuda_failure,
        )
        self._graph_state_inputs = (
            torch.empty(
                (
                    max_batch_size,
                    speculative_tokens + 1,
                    self.config.num_hidden_layers,
                    STATE_WIDTH_PER_LAYER,
                ),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if speculative_tokens
            else None
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.speculative_tokens,
            cache_layout=self.cache_layout,
        )

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot()

    def bind_resources(self, binding: ResourceBinding) -> None:
        self._cache.binding.bind_resources(binding)

    @staticmethod
    def _capture_inputs(shape: GraphShape, vocab_size: int) -> dict[str, object]:
        batch_size, _ = shape
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
        """Capture every configured, fixed-grid decode bucket."""

        captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape in captured:
                continue
            inputs = self._capture_inputs(shape, self.vocab_size)
            inputs.update(self._cache.graph_shape_capture_inputs(*shape))
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

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        cache = self._cache.graph_cache(batch_size, graph_seq_len)
        self._graph_device_caches[shape] = cache
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
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
                initial_state = self._initial_state(cache, positions)
                if not self.speculative_tokens:
                    logits, _, final_state, _ = self._model.target(
                        token_ids[:, None],
                        positions[:, None],
                        cache,
                        initial_state,
                        positions + 1,
                        graph_seq_len,
                        full_logits=False,
                    )
                    self._store_state(cache, positions, final_state)
                    output_ids[:, 0].copy_(
                        self._graph_sample(logits[:, -1], sampling, 0)
                    )
                    return

                state_inputs = self._graph_state_inputs
                if state_inputs is None:
                    raise RuntimeError("Inkling speculative state workspace is missing")
                state_inputs = state_inputs[:batch_size]
                drafts, _ = self._draft.propose(
                    token_ids,
                    positions,
                    cache,
                    graph_seq_len,
                    lambda logits, step: self._graph_sample(logits, sampling, step),
                )
                verify_ids = torch.cat((token_ids[:, None], drafts), dim=1)
                offsets = torch.arange(self.speculative_tokens + 1, device=self.device)
                verify_positions = positions[:, None] + offsets
                logits, taps, final_state, _ = self._model.target(
                    verify_ids,
                    verify_positions,
                    cache,
                    initial_state,
                    positions + self.speculative_tokens + 1,
                    graph_seq_len,
                    full_logits=True,
                    state_inputs_out=state_inputs,
                )
                targets = torch.stack(
                    [
                        self._graph_sample(logits[:, step].contiguous(), sampling, step)
                        for step in range(self.speculative_tokens + 1)
                    ],
                    dim=1,
                )
                matches = drafts == targets[:, : self.speculative_tokens]
                accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(1)
                processed = accepted.to(torch.long) + 1
                select_convolution_state(
                    initial_state,
                    state_inputs,
                    processed,
                    out=final_state,
                )
                self._store_state(cache, positions + processed - 1, final_state)
                self._draft.inject_target(taps, verify_positions, cache)
                output_ids[:, : self.speculative_tokens].copy_(drafts)
                output_ids[:, self.speculative_tokens :].copy_(targets)

        inputs = {"token_ids": token_ids, "positions": positions}
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            prepare=self._cache.reset_null,
        )

    def _graph_forward(self, plan: BatchPlan, cache_plan: Any):
        if not self._graphs.shapes:
            return None
        batch_size = plan.num_sequences
        required_length = int(plan.positions.max()) + self.speculative_tokens + 1
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
        return self._speculation.accept(
            [row[:width] for row in output],
            [row[width:] for row in output],
        ).token_rows

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        rows = plan.num_sequences
        if rows == 0 or rows > self.max_batch_size:
            raise ValueError("Inkling batch size is outside runner capacity")
        indptr = np.asarray(plan.token_indptr)
        if (
            len(indptr) != rows + 1
            or indptr[0] != 0
            or indptr[-1] != len(plan.token_ids)
            or np.any(indptr[1:] <= indptr[:-1])
        ):
            raise ValueError("Inkling sequence metadata is malformed")
        if len(plan.context_lengths) != rows or len(plan.sampling_mask) != rows:
            raise ValueError("Inkling row metadata is malformed")
        slices = tuple((int(start), int(end)) for start, end in zip(indptr, indptr[1:]))
        for row, (start, end) in enumerate(slices):
            context = int(plan.context_lengths[row])
            expected = np.arange(context, context + end - start)
            if not np.array_equal(plan.positions[start:end], expected):
                raise ValueError("Inkling positions must continue their context")
            if context + end - start > self.max_seq_len:
                raise ValueError("Inkling sequence exceeds max_seq_len")
        if plan.phase == BatchPhase.DECODE:
            if any(end - start != 1 for start, end in slices):
                raise ValueError("Inkling decode rows must contain one anchor token")
            if any(int(context) == 0 for context in plan.context_lengths):
                raise ValueError("Inkling decode requires a cached prefix")
        elif plan.phase != BatchPhase.PREFILL:
            raise ValueError("Inkling plan phase must be prefill or decode")
        return slices

    def _sample(self, logits: torch.Tensor, plan: BatchPlan, step: int = 0):
        return self._sampler(
            logits.float(),
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _initial_state(
        self, cache: DeviceCache, contexts: torch.Tensor
    ) -> torch.Tensor:
        positions = (contexts - 1).clamp_min(0)[:, None]
        state = cache.state_at(positions)[:, 0].clone()
        return torch.where(
            (contexts == 0)[:, None, None, None],
            torch.zeros((), dtype=state.dtype, device=state.device),
            state,
        )

    @staticmethod
    def _store_state(
        cache: DeviceCache, positions: torch.Tensor, state: torch.Tensor
    ) -> None:
        cache.store_state(positions[:, None], state[:, None])

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[int]:
        logits = []
        for row, (start, end) in enumerate(rows):
            row_cache = cache.row(row)
            row_ids = token_ids[start:end][None]
            row_positions = positions[start:end][None]
            context = int(plan.context_lengths[row])
            contexts = torch.tensor([context], dtype=torch.long, device=self.device)
            state = self._initial_state(row_cache, contexts)
            input_embeddings = self._model.target.embed_tokens(row_ids)
            self._merge_media_embeddings(
                plan,
                row,
                row_ids,
                row_positions,
                input_embeddings,
            )
            row_logits, taps, final_state, _ = self._model.target(
                row_ids,
                row_positions,
                row_cache,
                state,
                row_positions[:, -1] + 1,
                context + end - start,
                full_logits=False,
                prefill=True,
                input_embeddings=input_embeddings,
            )
            self._store_state(row_cache, row_positions[:, -1], final_state)
            if self.speculative_tokens:
                self._draft.inject_target(taps, row_positions, row_cache)
            logits.append(row_logits[:, -1])
        return self._sample(torch.cat(logits), plan).tolist()

    def _features_for(
        self, sequence_id: int, media: InklingMedia
    ) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing[0] is not media:
                raise ValueError("a live Inkling sequence changed its media object")
            return existing[1]
        pixels = media.pixel_values
        if pixels is None:
            raise ValueError("Inkling image placeholders require pixel_values")
        features = self._model.vision(
            pixels.to(self.device, dtype=torch.bfloat16)
        ).detach()
        if features.shape != (len(media.image_positions), self.config.hidden_size):
            raise ValueError("Inkling vision features do not match placeholders")
        self._media_features[sequence_id] = (media, features)
        return features

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
        if not isinstance(media, InklingMedia):
            raise TypeError("InklingCudaRunner media must be InklingMedia")
        flat_ids = token_ids[0]
        flat_positions = positions[0]
        position_list = flat_positions.tolist()
        for token_id, position in zip(flat_ids.tolist(), position_list):
            if position < len(media.token_ids) and token_id != media.token_ids[position]:
                raise ValueError("scheduled token does not match InklingMedia prompt")
        feature_indices = {
            position: index for index, position in enumerate(media.image_positions)
        }
        if not any(position in feature_indices for position in position_list):
            return
        features = self._features_for(int(plan.sequence_ids[row]), media)
        for query_index, position in enumerate(position_list):
            feature_index = feature_indices.get(position)
            if feature_index is not None:
                hidden[0, query_index].copy_(features[feature_index])

    def _ordinary_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[int]:
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        state = self._initial_state(cache, contexts)
        logits, _, final_state, _ = self._model.target(
            token_ids[:, None],
            positions[:, None],
            cache,
            state,
            positions + 1,
            int(positions.max().item()) + 1,
            full_logits=False,
        )
        self._store_state(cache, positions, final_state)
        return self._sample(logits[:, -1], plan).tolist()

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[list[int]]:
        drafts, _ = self._draft.propose(
            anchors,
            positions,
            cache,
            int(positions.max().item()) + self.speculative_tokens,
            lambda logits, step: self._sample(logits, plan, step),
        )
        verify_ids = torch.cat((anchors[:, None], drafts), dim=1)
        verify_positions = positions[:, None] + torch.arange(
            self.speculative_tokens + 1, device=self.device
        )
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        initial_state = self._initial_state(cache, contexts)
        state_inputs = torch.empty(
            (
                anchors.shape[0],
                self.speculative_tokens + 1,
                self.config.num_hidden_layers,
                STATE_WIDTH_PER_LAYER,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        logits, taps, final_state, _ = self._model.target(
            verify_ids,
            verify_positions,
            cache,
            initial_state,
            positions + self.speculative_tokens + 1,
            int(positions.max().item()) + self.speculative_tokens + 1,
            full_logits=True,
            state_inputs_out=state_inputs,
        )
        targets = torch.stack(
            [
                self._sample(logits[:, step].contiguous(), plan, step)
                for step in range(self.speculative_tokens + 1)
            ],
            dim=1,
        )
        acceptance = self._speculation.accept(drafts.tolist(), targets.tolist())
        processed = torch.tensor(
            [count + 1 for count in acceptance.accepted_counts],
            dtype=torch.long,
            device=self.device,
        )
        select_convolution_state(
            initial_state,
            state_inputs,
            processed,
            out=final_state,
        )
        self._store_state(cache, positions + processed - 1, final_state)
        self._draft.inject_target(taps, verify_positions, cache)
        return acceptance.token_rows

    def forward(self, plan: BatchPlan) -> list[int] | list[list[int]]:
        rows = self._validate_plan(plan)
        cache_plan = PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=(
                self.speculative_tokens if plan.phase == BatchPhase.DECODE else 0
            ),
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
                _default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if plan.phase == BatchPhase.PREFILL:
                    return self._prefill(plan, rows, cache, token_ids, positions)
                if self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                return self._ordinary_decode(plan, cache, token_ids, positions)

    def release(self, sequence_ids: list[int]) -> None:
        """The Engine retires each sequence's slabs after this callback."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)


__all__ = ["InklingCudaRunner"]
