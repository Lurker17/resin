"""Native DeepSeek-V4-Flash-0731 execution on Blackwell with TP sharding.

Resin owns the model hierarchy, kernels, request lifetime, sampling, and
scheduling. Sequence state occupies compact runner-owned rows; a later
DP-attention/prefix-cache implementation can replace that state contract
without treating compressed and recurrent state as fungible pages.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from resin import BatchPhase
from resin.backend.checks import positions as check_positions
from resin.backend.checks import token_ids as check_token_ids
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
from resin.backend.speculative import accept_speculative_tokens
from resin.protocols import SyncRunner
from resin_models_cuda.deepseek_v4.checkpoint import (
    load_config,
    resolve_model_path,
)
from resin_models_cuda.deepseek_v4.weights import load_deepseek_v4_weights

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


class DeepSeekV4FlashCudaRunner(SyncRunner):
    """Dynamically batched runner over Resin's DeepSeek-V4 implementation."""

    supports_continuation_prefill = False

    def __init__(
        self,
        model: str | Path,
        *,
        max_seq_len: int,
        max_batch_size: int = 1,
        tensor_parallel: TensorParallelContext,
        device: str | torch.device | None = None,
        cuda_graph_decode_shapes: object = None,
        fast_allreduce: bool = True,
        speculative_tokens: int = 0,
        profile_graph_phases: bool = False,
    ) -> None:
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size * max_seq_len > 2**32 - 1:
            raise ValueError("max_batch_size * max_seq_len exceeds scheduler metadata")
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens < 0
        ):
            raise ValueError("speculative_tokens must be a non-negative integer")
        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeekV4FlashCudaRunner requires CUDA")

        self.tensor_parallel = tensor_parallel
        self.tensor_parallel_rank = tensor_parallel.rank
        self.tensor_parallel_size = tensor_parallel.size
        self.device = torch.device(device or f"cuda:{tensor_parallel.rank}")
        if self.device.type != "cuda":
            raise RuntimeError("DeepSeekV4FlashCudaRunner requires a CUDA device")
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("the DeepSeek-V4 Flash runner requires Blackwell")
        if self.tensor_parallel_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "DeepSeek-V4 TP requires an initialized process group"
                )
            if (
                dist.get_world_size() != self.tensor_parallel_size
                or dist.get_rank() != self.tensor_parallel_rank
            ):
                raise ValueError("the default process group does not match TP layout")

        self.model_path = resolve_model_path(model)
        self.config = load_config(self.model_path)
        self.config.validate_tensor_parallel(self.tensor_parallel_size)
        if speculative_tokens > self.config.dspark_block_size:
            raise ValueError(
                "speculative_tokens exceeds this checkpoint's DSpark block size"
            )
        if speculative_tokens and not self.config.dspark_target_layer_ids:
            raise ValueError("speculative_tokens requires a DSpark checkpoint")
        if max_seq_len > 1_048_576:
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.speculative_tokens = speculative_tokens
        self.profile_graph_phases = profile_graph_phases
        self._dspark_phase_events: dict[GraphShape, tuple[torch.cuda.Event, ...]] = {}
        self._last_dspark_phase_ms: dict[str, float] | None = None
        self.max_batch_tokens = max_batch_size * max_seq_len
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = 1
        self._sampler = FlashInferSampler(torch)
        self._active_sequence_ids: list[int] = []
        self._speculative_blocks = 0
        self._speculative_drafts = 0
        self._speculative_accepted = 0
        self._speculative_emitted = 0
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=2)
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("CUDA graph batch capacity exceeds max_batch_size")
        physical_seq_len = max_seq_len + speculative_tokens
        if speculative_tokens:
            physical_seq_len += 1
        if any(shape[1] > physical_seq_len for shape in decode_graph_shapes):
            raise ValueError("CUDA graph sequence capacity exceeds max_seq_len")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=lambda: torch.cuda.synchronize(self.device),
        )
        # Import only after CUDA validation: importing this module initializes
        # cuTile and is intentionally not part of Resin's CPU package path.
        from resin_models_cuda.deepseek_v4 import model as native_model

        inference_config_path = self.model_path / "inference/config.json"
        try:
            inference_values = json.loads(inference_config_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError("DeepSeek-V4 inference/config.json is invalid") from error
        if not isinstance(inference_values, dict):
            raise ValueError("DeepSeek-V4 inference config must contain an object")

        inference_values.update(
            {
                "max_batch_size": max_batch_size,
                "max_seq_len": physical_seq_len,
                "dspark_block_size": speculative_tokens,
            }
        )
        if speculative_tokens:
            if int(inference_values.get("n_mtp_layers", 0)) <= 0:
                raise ValueError("DSpark checkpoint has no speculative stages")
        else:
            inference_values["n_mtp_layers"] = 0
        model_fields = native_model.ModelArgs.__dataclass_fields__
        args = native_model.ModelArgs(
            **{
                name: value
                for name, value in inference_values.items()
                if name in model_fields
            }
        )
        with _default_dtype(torch.bfloat16), torch.device(self.device):
            self._model = native_model.Transformer(args)
        self._allreduce = FlashInferAllReduce(
            self.tensor_parallel,
            max_token_num=max_batch_size * max(1, speculative_tokens + 1),
            hidden_dim=self.config.hidden_size,
            dtype=torch.bfloat16,
            enabled=fast_allreduce,
        )
        index_widths = sorted(
            {
                min(
                    ((args.max_seq_len + 255) // 256) * 64,
                    ((shape[1] + 255) // 256) * 64,
                )
                for shape in decode_graph_shapes
            }
        )
        self._index_allreduces = {
            width: FlashInferAllReduce(
                self.tensor_parallel,
                max_token_num=max_batch_size * max(1, speculative_tokens + 1),
                hidden_dim=(width + 7) // 8 * 8,
                dtype=torch.bfloat16,
                enabled=fast_allreduce,
            )
            for width in index_widths
        }

        def reduce_index_scores(tensor: torch.Tensor) -> torch.Tensor:
            reducer = self._index_allreduces.get(tensor.shape[-1])
            if reducer is None:
                return self.tensor_parallel.sum(tensor)
            padding = reducer.hidden_dim - tensor.shape[-1]
            if padding == 0:
                return reducer.sum(tensor)
            padded = F.pad(tensor, (0, padding))
            return reducer.sum(padded)[..., : tensor.shape[-1]].contiguous()

        self.allreduce_backend = self._allreduce.backend
        self._model.set_allreduce(
            self._allreduce.sum,
            indexer_reducer=reduce_index_scores,
        )
        self.weight_load = load_deepseek_v4_weights(
            self.model_path,
            self._model,
            rank=self.tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
            device=self.device,
        )
        for module in self._model.modules():
            if isinstance(module, native_model.MoE):
                module.prepare_fused_weights()

    @property
    def cuda_graph_stats(self) -> CudaGraphStats:
        return self._graphs.stats

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        acceptance = (
            self._speculative_accepted / self._speculative_drafts
            if self._speculative_drafts
            else 0.0
        )
        return {
            "blocks": self._speculative_blocks,
            "drafts": self._speculative_drafts,
            "accepted": self._speculative_accepted,
            "emitted": self._speculative_emitted,
            "acceptance_rate": acceptance,
        }

    @property
    def last_dspark_phase_ms(self) -> dict[str, float] | None:
        """Return device timings from the most recent profiled graph replay."""

        if self._last_dspark_phase_ms is None:
            return None
        return dict(self._last_dspark_phase_ms)

    @staticmethod
    def _graph_capture_inputs(
        shape: GraphShape, vocab_size: int = 129_280
    ) -> dict[str, tuple]:
        """Return deterministic synthetic inputs for one decode graph."""

        batch_size, sequence_capacity = shape
        if sequence_capacity < 2:
            raise ValueError("DeepSeek-V4 decode graphs require capacity >= 2")
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

    def _sequence_state_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the model buffers whose first dimension is a sequence row."""

        suffixes = ("kv_cache", "kv_state", "score_state")
        for name, buffer in self._model.named_buffers():
            if name.endswith(suffixes) and buffer.shape[0] == self.max_batch_size:
                yield name, buffer

    def _reset_model_state(self) -> None:
        """Restore static recurrent/cache buffers before a new sequence."""

        with torch.no_grad():
            for name, buffer in self._sequence_state_buffers():
                if name.endswith("score_state"):
                    buffer.fill_(float("-inf"))
                else:
                    buffer.zero_()

    def _compact_model_state(self, source_rows: tuple[int, ...]) -> None:
        """Move surviving sequence rows to a dense prefix in stable order."""

        with torch.no_grad():
            for name, buffer in self._sequence_state_buffers():
                for destination, source in enumerate(source_rows):
                    if destination != source:
                        buffer[destination].copy_(buffer[source])
                inactive = buffer[len(source_rows) :]
                if name.endswith("score_state"):
                    inactive.fill_(float("-inf"))
                else:
                    inactive.zero_()

    def _copy_model_state_row(self, source: int, destination: int) -> None:
        """Copy one complete sequence row after a ragged single-row prefill."""

        if source == destination:
            return
        with torch.no_grad():
            for _, buffer in self._sequence_state_buffers():
                buffer[destination].copy_(buffer[source])

    def capture_cuda_graphs(self) -> CudaGraphStats:
        """Capture configured decode capacities before accepting requests."""

        captured = set(self._graphs.stats["captured_shapes"])
        for shape in self._graphs.shapes:
            if shape not in captured:
                self._graphs.capture(
                    shape, self._graph_capture_inputs(shape, self.vocab_size)
                )
        if self._graphs.shapes:
            self._reset_model_state()
            torch.cuda.synchronize(self.device)
        return self._graphs.stats

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        if self.speculative_tokens:
            return self._new_dspark_graph_step(shape)
        batch_size, graph_seq_len = shape
        token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        output_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)

        def launch() -> None:
            with _default_dtype(torch.bfloat16), torch.device(self.device):
                logits = self._model.forward_decode(
                    token_ids,
                    positions,
                    graph_seq_len,
                )
                self._sampler.sample_tensors(logits, sampling, out=output_ids)
                if self.tensor_parallel.enabled:
                    dist.broadcast(
                        output_ids,
                        src=0,
                        group=self.tensor_parallel.process_group,
                    )

        inputs = {"token_ids": token_ids, "positions": positions}
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(inputs=inputs, output=output_ids, launch=launch)

    def _new_dspark_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        block_size = self.speculative_tokens
        token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        output_ids = torch.empty(
            (batch_size, 2 * block_size + 1),
            dtype=torch.long,
            device=self.device,
        )
        sampling = self._sampler.empty(batch_size, self.device)
        phase_events: tuple[torch.cuda.Event, ...] | None = None
        if self.profile_graph_phases:
            phase_events = tuple(
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)
            )
            self._dspark_phase_events[shape] = phase_events

        def sample(logits: torch.Tensor, step: int) -> torch.Tensor:
            step_sampling = SamplingTensors(
                temperatures=sampling.temperatures,
                top_ps=sampling.top_ps,
                top_ks=sampling.top_ks,
                seeds=sampling.seeds,
                offsets=sampling.offsets + step * SAMPLING_OFFSET_STRIDE,
                stochastic_mask=sampling.stochastic_mask,
            )
            return self._broadcast_tokens(
                self._sampler.sample_tensors(logits, step_sampling)
            )

        def launch() -> None:
            with _default_dtype(torch.bfloat16), torch.device(self.device):
                if phase_events is not None:
                    phase_events[0].record()
                drafts = self._broadcast_tokens(self._model.draft(token_ids, positions))
                if phase_events is not None:
                    phase_events[1].record()
                verify_ids = torch.cat((token_ids, drafts), dim=1)
                verify_logits, verify_hidden = self._model.forward_verify(
                    verify_ids, positions, graph_seq_len
                )
                if phase_events is not None:
                    phase_events[2].record()
                if verify_hidden is None:
                    raise RuntimeError(
                        "target did not produce DSpark verification states"
                    )
                self._model.commit_dspark_main_block(verify_hidden, positions)
                output_ids[:, :block_size].copy_(drafts)
                for index in range(block_size + 1):
                    output_ids[:, block_size + index].copy_(
                        sample(verify_logits[:, index], index)
                    )
                if phase_events is not None:
                    phase_events[3].record()

        inputs = {"token_ids": token_ids, "positions": positions}
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            prepare=self._reset_model_state,
        )

    def _graph_forward(self, plan: BatchPlan) -> list[int] | list[list[int]] | None:
        if not self._graphs.shapes:
            return None
        batch_size = plan.num_sequences
        sequence_length = max(plan.positions) + 1
        shape = self._graphs.match(
            lambda candidate: (
                candidate[0] == batch_size and candidate[1] >= sequence_length
            )
        )
        if shape is None:
            self._graphs.fallback("shape")
            return None
        inputs: dict[str, object] = {
            "token_ids": plan.token_ids,
            "positions": plan.positions,
        }
        inputs.update(self._sampler.host_inputs(plan, self.vocab_size))
        output = self._graphs.run(shape, inputs)
        phase_events = self._dspark_phase_events.get(shape)
        if phase_events is not None:
            start, draft_end, verify_end, step_end = phase_events
            self._last_dspark_phase_ms = {
                "draft": start.elapsed_time(draft_end),
                "target_verify": draft_end.elapsed_time(verify_end),
                "commit_and_sampling": verify_end.elapsed_time(step_end),
                "device_total": start.elapsed_time(step_end),
            }
        if not self.speculative_tokens:
            return output
        block_size = self.speculative_tokens
        drafts = [row[:block_size] for row in output]
        targets = [row[block_size:] for row in output]
        return self._accept_speculative(
            drafts,
            targets,
            torch.tensor(plan.positions, dtype=torch.long, device=self.device),
        )

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        row_count = plan.num_sequences
        if row_count == 0 or row_count > self.max_batch_size:
            raise ValueError("DeepSeek-V4 batch size is outside runner capacity")
        indptr = tuple(plan.sequence_indptr)
        if (
            len(indptr) != row_count + 1
            or indptr[0] != 0
            or indptr[-1] != len(plan.token_ids)
            or any(left >= right for left, right in zip(indptr, indptr[1:]))
        ):
            raise ValueError("DeepSeek-V4 sequence metadata is malformed")
        if len(plan.context_lengths) != row_count:
            raise ValueError("DeepSeek-V4 requires one context length per sequence")
        if len(set(plan.sequence_ids)) != row_count:
            raise ValueError("DeepSeek-V4 sequence IDs must be unique within a batch")
        check_token_ids(plan.token_ids, self.vocab_size)
        check_positions(plan.positions, self.max_seq_len)
        rows = tuple(zip(indptr, indptr[1:]))
        for row, (start, end) in enumerate(rows):
            context = plan.context_lengths[row]
            if list(plan.positions[start:end]) != list(
                range(context, context + end - start)
            ):
                raise ValueError("DeepSeek-V4 positions must continue each row context")

        if plan.phase == BatchPhase.PREFILL:
            if self._active_sequence_ids:
                raise RuntimeError(
                    "DeepSeek-V4 cannot admit prefills while decode state is active"
                )
            if any(context != 0 for context in plan.context_lengths):
                raise ValueError("DeepSeek-V4 does not support continuation prefill")
        elif plan.phase == BatchPhase.DECODE:
            if any(end - start != 1 for start, end in rows):
                raise ValueError("DeepSeek-V4 decode rows must contain one token")
            if list(plan.sequence_ids) != self._active_sequence_ids[:row_count]:
                raise ValueError("DeepSeek-V4 decode rows do not own compact state")
        else:
            raise ValueError("DeepSeek-V4 plan phase must be prefill or decode")
        return rows

    def _prefill(
        self, plan: BatchPlan, rows: tuple[tuple[int, int], ...]
    ) -> torch.Tensor:
        """Prefill rows independently, staging their recurrent state compactly."""

        logits: list[torch.Tensor | None] = [None] * len(rows)
        # Row zero is the model's eager single-row target. Filling slots in
        # reverse means every completed row is copied out before row zero is
        # reused, with the final request naturally left in slot zero.
        for slot in reversed(range(len(rows))):
            start, end = rows[slot]
            input_ids = torch.tensor(
                plan.token_ids[start:end], dtype=torch.long, device=self.device
            ).unsqueeze(0)
            if getattr(self, "speculative_tokens", 0):
                row_logits, main_hidden = self._model.forward_target(input_ids)
                if main_hidden is None:
                    raise RuntimeError("target did not produce DSpark hidden states")
                self._model.prefill_dspark(main_hidden)
                logits[slot] = row_logits
            else:
                logits[slot] = self._model(input_ids, 0)
            self._copy_model_state_row(0, slot)
        return torch.cat([row for row in logits if row is not None], dim=0)

    def _broadcast_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel.enabled:
            dist.broadcast(
                token_ids,
                src=0,
                group=self.tensor_parallel.process_group,
            )
        return token_ids

    def _sample_target(
        self, logits: torch.Tensor, plan: BatchPlan, step: int
    ) -> torch.Tensor:
        return self._broadcast_tokens(
            self._sampler(
                logits,
                plan,
                offset_delta=step * SAMPLING_OFFSET_STRIDE,
            )
        )

    def _accept_speculative(
        self,
        draft_rows: list[list[int]],
        target_rows: list[list[int]],
        positions: torch.Tensor,
    ) -> list[list[int]]:
        batch_size = len(draft_rows)
        acceptance = accept_speculative_tokens(draft_rows, target_rows)
        processed_counts = torch.tensor(
            acceptance.processed_counts,
            dtype=torch.int32,
            device=self.device,
        )
        self._model.clear_rejected_speculative_state(
            positions,
            processed_counts,
            self.speculative_tokens + 1,
        )
        self._speculative_blocks += batch_size
        self._speculative_drafts += batch_size * self.speculative_tokens
        self._speculative_accepted += sum(acceptance.accepted_counts)
        self._speculative_emitted += sum(map(len, acceptance.token_rows))
        return acceptance.token_rows

    def _speculative_decode(self, plan: BatchPlan) -> list[list[int]]:
        """Draft greedily and verify a fixed block against target samples."""

        batch_size = plan.num_sequences
        anchors = torch.tensor(
            plan.token_ids, dtype=torch.long, device=self.device
        ).view(batch_size, 1)
        positions = torch.tensor(plan.positions, dtype=torch.long, device=self.device)
        graph_seq_len = max(plan.positions) + self.speculative_tokens + 1
        drafts = self._broadcast_tokens(self._model.draft(anchors, positions))
        verify_ids = torch.cat((anchors, drafts), dim=1)
        verify_logits, verify_hidden = self._model.forward_verify(
            verify_ids,
            positions,
            graph_seq_len,
        )
        if verify_hidden is None:
            raise RuntimeError("target did not produce DSpark verification states")
        self._model.commit_dspark_main_block(verify_hidden, positions)

        target_columns = [
            self._sample_target(verify_logits[:, index], plan, index)
            for index in range(self.speculative_tokens + 1)
        ]
        return self._accept_speculative(
            drafts.tolist(),
            torch.stack(target_columns, dim=1).tolist(),
            positions,
        )

    def forward(self, plan: BatchPlan) -> list[int] | list[list[int]]:
        """Run one dynamically batched prefill or decode sampling step."""

        self._graphs.check()
        rows = self._validate_plan(plan)
        if plan.phase == BatchPhase.DECODE:
            graph_output = self._graph_forward(plan)
            if graph_output is not None:
                return graph_output
        elif self._graphs.shapes:
            self._graphs.fallback("prefill")
        try:
            with (
                torch.inference_mode(),
                _default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if plan.phase == BatchPhase.PREFILL:
                    logits = self._prefill(plan, rows)
                elif self.speculative_tokens:
                    return self._speculative_decode(plan)
                else:
                    input_ids = torch.tensor(
                        plan.token_ids, dtype=torch.long, device=self.device
                    ).view(plan.num_sequences, 1)
                    positions = torch.tensor(
                        plan.positions, dtype=torch.long, device=self.device
                    )
                    logits = self._model.forward_decode(
                        input_ids, positions, max(plan.positions) + 1
                    )
                output = self._sample_target(logits, plan, 0)
            if plan.phase == BatchPhase.PREFILL:
                self._active_sequence_ids = list(plan.sequence_ids)
            return output.tolist()
        except BaseException:
            torch.cuda.synchronize(self.device)
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Release rows and compact survivors without changing their order."""

        retired = set(sequence_ids)
        survivors = tuple(
            (row, sequence_id)
            for row, sequence_id in enumerate(self._active_sequence_ids)
            if sequence_id not in retired
        )
        if len(survivors) == len(self._active_sequence_ids):
            return
        self._compact_model_state(tuple(row for row, _ in survivors))
        self._active_sequence_ids = [sequence_id for _, sequence_id in survivors]


__all__ = ["DeepSeekV4FlashCudaRunner"]
