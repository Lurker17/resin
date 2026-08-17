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
from resin.backend.cuda.sampling import SAMPLING_OFFSET_STRIDE, FlashInferSampler
from resin.backend.cuda.staging import TensorStager
from resin.backend.loading import default_device
from resin.backend.plans import PagedCachePlan
from resin.backend.speculative import SpeculativeStats
from resin.backend.torch import use_default_dtype
from resin.protocols import Runner
from resin_models_cuda.qwen3_5.cache import (
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
    DeviceCache,
)
from resin_models_cuda.qwen3_5.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)

if TYPE_CHECKING:
    from resin import BatchPlan

_CACHE_WORKSPACE_BYTES = 512 * 1024**2


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
            from flash_attn.cute import flash_attn_varlen_func
        except ImportError as error:
            raise ImportError(
                "Qwen3_5CudaRunner requires flash-linear-attention and "
                "FlashAttention-4"
            ) from error
        del chunk_gated_delta_rule, flash_attn_varlen_func

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
        )
        selected_max_seq_len = (
            self.config.max_position_embeddings
            if max_seq_len is None
            else max_seq_len
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

        self.cache_geometry = CacheGeometry.from_config(
            self.config, include_mtp=speculative_tokens > 0
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
            torch, self.device, self.cache_layout, self.cache_geometry
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
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    def bind_resources(self, binding: ResourceBinding) -> None:
        self._cache.binding.bind_resources(binding)

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
        slices = tuple(
            (int(start), int(end)) for start, end in zip(indptr, indptr[1:])
        )
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
    ) -> torch.Tensor:
        mtp = self._model.mtp
        if mtp is None:
            raise RuntimeError("Qwen3.5 MTP was not loaded")
        embeddings = self._model.language_model.embed_tokens(input_ids)
        rotary = self._model.language_model.rotary(positions, embeddings.dtype)
        sequence_lengths = positions[:, -1] + 1
        return mtp(
            embeddings,
            previous_hidden,
            positions,
            rotary=rotary,
            cache=cache,
            sequence_lengths=sequence_lengths,
            max_sequence_length=int(sequence_lengths.max().item()),
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
    ) -> None:
        if self._model.mtp is None:
            return
        if context:
            self._mtp_forward(
                cache,
                token_ids[:, :1],
                previous_hidden[:, None],
                positions[:, :1] - 1,
            )
        if token_ids.shape[1] > 1:
            self._mtp_forward(
                cache,
                token_ids[:, 1:],
                target_hidden[:, :-1],
                positions[:, :-1],
            )
        if sampled_token is not None:
            self._mtp_forward(
                cache,
                sampled_token[:, None],
                target_hidden[:, -1:],
                positions[:, -1:],
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
            hidden, recurrent, convolution = self._model.language_model(
                row_ids,
                row_positions,
                cache=row_cache,
                recurrent_states=recurrent,
                convolution_states=convolution,
                sequence_lengths=row_positions[:, -1] + 1,
                max_sequence_length=context + end - start,
            )
            final_logits.append(self._model.logits(hidden[:, -1]))
            states.append(
                (row_cache, row_ids, row_positions, hidden, recurrent, convolution, previous_hidden, context)
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
            self._prefill_mtp(
                row_cache,
                row_ids,
                row_positions,
                hidden,
                previous_hidden,
                context,
                sampled_token,
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
        )
        self._store_state(
            cache, positions, recurrent, convolution, hidden[:, -1]
        )
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
        initial_recurrent, initial_convolution, _ = self._initial_state(
            cache, contexts
        )
        anchor_hidden, anchor_recurrent, anchor_convolution = (
            self._model.language_model(
                anchors[:, None],
                positions[:, None],
                cache=cache,
                recurrent_states=initial_recurrent,
                convolution_states=initial_convolution,
                sequence_lengths=positions + 1,
                max_sequence_length=int(positions.max().item()) + 1,
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
                committed_hidden, recurrent, convolution = (
                    self._model.language_model(
                        verify_ids[row : row + 1, :processed],
                        verify_positions[row : row + 1, :processed],
                        cache=row_cache,
                        recurrent_states=anchor_recurrent[row : row + 1],
                        convolution_states=anchor_convolution[row : row + 1],
                        sequence_lengths=verify_positions[
                            row : row + 1, processed - 1
                        ]
                        + 1,
                        max_sequence_length=int(
                            verify_positions[row, processed - 1].item()
                        )
                        + 1,
                    )
                )
            correction = targets[row, accepted : accepted + 1]
            repair_ids = torch.cat((draft_ids[row, :accepted], correction))[None]
            repair_positions = verify_positions[row : row + 1, :processed]
            self._mtp_forward(
                row_cache,
                repair_ids,
                committed_hidden[:, :processed],
                repair_positions,
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
                    return self._speculative_decode(
                        plan, cache, token_ids, positions
                    )
                return self._ordinary_decode(plan, cache, token_ids, positions)


__all__ = ["Qwen3_5CudaRunner"]
