from __future__ import annotations

import torch
import torch.nn.functional as F

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.cuda.sampling import SAMPLING_OFFSET_STRIDE, SamplingTensors
from resin_models_cuda.kimi_k25.cuda import KimiK25CudaRunner
from resin_models_cuda.kimi_k25.model import DenseMlp, ParallelHead


def test_dense_mlp_packs_gate_and_up_without_changing_cpu_output() -> None:
    torch.manual_seed(0)
    mlp = DenseMlp(8, 12, TensorParallelContext(), lambda tensor: tensor)
    gate = torch.randn_like(mlp.gate_weight)
    up = torch.randn_like(mlp.up_weight)
    down = torch.randn_like(mlp.down_proj.weight)
    with torch.no_grad():
        mlp.gate_weight.copy_(gate)
        mlp.up_weight.copy_(up)
        mlp.down_proj.weight.copy_(down)

    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = F.linear(F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down)

    torch.testing.assert_close(mlp.partial(hidden), expected)
    assert mlp.gate_weight.untyped_storage().data_ptr() == (
        mlp.up_weight.untyped_storage().data_ptr()
    )


def test_parallel_head_uses_compact_bf16_logits_for_local_argmax() -> None:
    torch.manual_seed(1)
    head = ParallelHead(16, 8, TensorParallelContext())
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    local = F.linear(hidden, head.weight)

    assert head.weight.dtype == torch.bfloat16
    assert head(hidden).dtype == torch.float32
    torch.testing.assert_close(head(hidden), local.float())
    torch.testing.assert_close(head.argmax(hidden), local.argmax(dim=-1))


def test_graph_verification_sampling_flattens_every_step_once() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.tensors: SamplingTensors | None = None

        def sample_tensors(
            self, logits: torch.Tensor, tensors: SamplingTensors
        ) -> torch.Tensor:
            self.tensors = tensors
            return torch.arange(logits.shape[0], device=logits.device)

    runner = object.__new__(KimiK25CudaRunner)
    runner.device = torch.device("cpu")
    runner._sampler = Sampler()
    runner._broadcast_tokens = lambda token_ids: token_ids
    sampling = SamplingTensors(
        temperatures=torch.tensor([0.7, 0.8]),
        top_ps=torch.tensor([0.9, 0.95]),
        top_ks=torch.tensor([10, 20], dtype=torch.int32),
        seeds=torch.tensor([3, 4], dtype=torch.int64),
        offsets=torch.tensor([5, 7], dtype=torch.int64),
        stochastic_mask=torch.tensor([True, False]),
    )

    sampled = runner._graph_sample_steps(torch.zeros(2, 4, 11), sampling)

    assert sampled.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert runner._sampler.tensors is not None
    assert runner._sampler.tensors.seeds.is_contiguous()
    assert runner._sampler.tensors.offsets.tolist() == [
        5,
        5 + SAMPLING_OFFSET_STRIDE,
        5 + 2 * SAMPLING_OFFSET_STRIDE,
        5 + 3 * SAMPLING_OFFSET_STRIDE,
        7,
        7 + SAMPLING_OFFSET_STRIDE,
        7 + 2 * SAMPLING_OFFSET_STRIDE,
        7 + 3 * SAMPLING_OFFSET_STRIDE,
    ]
