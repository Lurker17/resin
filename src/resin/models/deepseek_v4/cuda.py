"""Native DeepSeek-V4-Flash-0731 execution on Blackwell with TP sharding.

Resin owns the model hierarchy, kernels, request lifetime, sampling, and
scheduling. The first cache implementation owns one static sequence slot; a
later DP-attention/prefix-cache implementation can replace that narrow state
contract without treating compressed and recurrent state as fungible pages.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.distributed as dist

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
from resin.backend.cuda.sampling import FlashInferSampler
from resin.models.deepseek_v4.checkpoint import (
    load_config,
    resolve_model_path,
)
from resin.models.deepseek_v4.weights import load_deepseek_v4_weights

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


class DeepSeekV4FlashCudaRunner:
    """One-sequence runner over Resin's DeepSeek-V4 target implementation."""

    supports_continuation_prefill = False

    def __init__(
        self,
        model: str | Path,
        *,
        max_seq_len: int,
        tensor_parallel: TensorParallelContext,
        device: str | torch.device | None = None,
        cuda_graph_decode_shapes: object = None,
        fast_allreduce: bool = True,
    ) -> None:
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
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
        if max_seq_len > 1_048_576:
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.max_seq_len = max_seq_len
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = 1
        self._sampler = FlashInferSampler(torch)
        self._active_sequence_id: int | None = None
        decode_graph_shapes = graph_shapes(cuda_graph_decode_shapes, dimensions=1)
        if any(shape[0] > max_seq_len for shape in decode_graph_shapes):
            raise ValueError("CUDA graph decode capacity exceeds max_seq_len")
        self._graphs = CudaGraphCache(
            torch,
            self.device,
            decode_graph_shapes,
            build=self._new_decode_graph_step,
            synchronize_failure=lambda: torch.cuda.synchronize(self.device),
        )

        # Import only after CUDA validation: importing this module initializes
        # cuTile and is intentionally not part of Resin's CPU package path.
        from resin.models.deepseek_v4 import model as native_model

        inference_config_path = self.model_path / "inference/config.json"
        try:
            inference_values = json.loads(inference_config_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError("DeepSeek-V4 inference/config.json is invalid") from error
        if not isinstance(inference_values, dict):
            raise ValueError("DeepSeek-V4 inference config must contain an object")

        inference_values.update(
            {
                "max_batch_size": 1,
                "max_seq_len": max_seq_len,
            }
        )
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
            max_token_num=1,
            hidden_dim=self.config.hidden_size,
            dtype=torch.float32,
            enabled=fast_allreduce,
        )
        self.allreduce_backend = self._allreduce.backend
        self._model.set_allreduce(self._allreduce.sum)
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

    @staticmethod
    def _graph_capture_inputs(
        shape: GraphShape, vocab_size: int = 129_280
    ) -> dict[str, tuple]:
        """Return deterministic synthetic inputs for one decode graph."""

        if shape[0] < 2:
            raise ValueError("DeepSeek-V4 decode graphs require capacity >= 2")
        return {
            "token_ids": (0,),
            "positions": (1,),
            "sampling_temperatures": (1.0,),
            "sampling_top_ps": (1.0,),
            "sampling_top_ks": (vocab_size,),
            "sampling_seeds": (0,),
            "sampling_offsets": (0,),
            "sampling_stochastic_mask": (False,),
        }

    def _reset_model_state(self) -> None:
        """Restore static recurrent/cache buffers before a new sequence."""

        with torch.no_grad():
            for name, buffer in self._model.named_buffers():
                if name.endswith("score_state"):
                    buffer.fill_(float("-inf"))
                elif name.endswith(("kv_cache", "kv_state")):
                    buffer.zero_()

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
        graph_seq_len = shape[0]
        token_ids = torch.empty((1, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(1, dtype=torch.long, device=self.device)
        output_ids = torch.empty(1, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(1, self.device)

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

    def _graph_forward(self, plan: BatchPlan, start_pos: int) -> list[int] | None:
        if not self._graphs.shapes:
            return None
        shape = self._graphs.match(lambda candidate: candidate[0] >= start_pos + 1)
        if shape is None:
            self._graphs.fallback("capacity")
            return None
        inputs: dict[str, object] = {
            "token_ids": plan.token_ids,
            "positions": plan.positions,
        }
        inputs.update(self._sampler.host_inputs(plan, self.vocab_size))
        return self._graphs.run(shape, inputs)

    def _validate_plan(self, plan: BatchPlan) -> int:
        if plan.num_sequences != 1:
            raise ValueError("DeepSeek-V4 static attention state supports one sequence")
        if len(plan.token_ids) == 0:
            raise ValueError("DeepSeek-V4 received an empty sequence")
        if list(plan.sequence_indptr) != [0, len(plan.token_ids)]:
            raise ValueError("DeepSeek-V4 requires one contiguous token row")
        check_token_ids(plan.token_ids, self.vocab_size)
        check_positions(plan.positions, self.max_seq_len)
        start_pos = plan.positions[0]
        expected_positions = list(range(start_pos, start_pos + len(plan.token_ids)))
        if list(plan.positions) != expected_positions:
            raise ValueError("DeepSeek-V4 positions must be contiguous")
        sequence_id = plan.sequence_ids[0]
        if start_pos == 0:
            if self._active_sequence_id is not None:
                raise RuntimeError("DeepSeek-V4 already owns an active sequence")
            self._active_sequence_id = sequence_id
        else:
            if len(plan.token_ids) != 1:
                raise ValueError("DeepSeek-V4 does not support continuation prefill")
            if self._active_sequence_id != sequence_id:
                raise ValueError(
                    "DeepSeek-V4 decode sequence does not own static state"
                )
        return start_pos

    def forward(self, plan: BatchPlan) -> list[int]:
        """Run one prefill or decode step and sample through shared FlashInfer."""

        self._graphs.check()
        start_pos = self._validate_plan(plan)
        if start_pos > 0:
            graph_output = self._graph_forward(plan, start_pos)
            if graph_output is not None:
                return graph_output
        elif self._graphs.shapes:
            self._graphs.fallback("prefill")
        input_ids = torch.tensor(
            plan.token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        try:
            with (
                torch.inference_mode(),
                _default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                logits = self._model(input_ids, start_pos)
                output = self._sampler(logits, plan)
                if self.tensor_parallel.enabled:
                    # Sampling is a request-level decision, not a replicated
                    # tensor-parallel computation.  Broadcast rank 0's draw so
                    # tiny rank-local numerical differences cannot fork the
                    # autoregressive sequence under stochastic sampling.
                    dist.broadcast(
                        output,
                        src=0,
                        group=self.tensor_parallel.process_group,
                    )
            return output.tolist()
        except BaseException:
            torch.cuda.synchronize(self.device)
            raise

    def release(self, sequence_ids: list[int]) -> None:
        """Release the sole logical slot; the next prefill overwrites its buffers."""

        if self._active_sequence_id in sequence_ids:
            self._active_sequence_id = None
            self._reset_model_state()


__all__ = ["DeepSeekV4FlashCudaRunner"]
