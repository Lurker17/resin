"""One-device expert-TP shard arithmetic; distributed ownership is tested separately."""

from dataclasses import replace

import pytest
import torch
from test_glm53_model_gpu import (
    SyntheticReader,
    _assert_error,
    _capture,
    _moe_reference,
)
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash.model import Bf16Experts, Nvfp4Experts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100 Blackwell and the pinned CUDA kernels",
)


class SlicedReader(SyntheticReader):
    def read(self, name, shape=None, dtype=None, *, cpu=False, slices=None):
        value = self.tensors[name]
        if shape is not None:
            assert tuple(value.shape) == tuple(shape)
        if dtype is not None:
            assert value.dtype == dtype
        if slices is not None:
            value = value[slices]
        return value if cpu else value.to(self.device)


def _reference_reader(reader, start, stop, *, quantized):
    result = object.__new__(SyntheticReader)
    result.tensors = {}
    for name, value in reader.tensors.items():
        if name.endswith((".weight", ".weight_scale")):
            pack = 16 if name.endswith(".weight_scale") else 2 if quantized else 1
            value = (
                value[:, start // pack : stop // pack]
                if ".down_proj." in name
                else value[start:stop]
            )
        result.tensors[name] = value
    return result


def _bf16_reference(reader, hidden, ids, weights, first, count):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(first, first + count):
        gate = (
            F.linear(hidden.float(), reader.read(f"{expert}.gate_proj.weight").float())
            .bfloat16()
            .float()
            .clamp(max=10)
        )
        up = (
            F.linear(hidden.float(), reader.read(f"{expert}.up_proj.weight").float())
            .bfloat16()
            .float()
            .clamp(-10, 10)
        )
        contribution = (
            F.linear(
                (F.silu(gate) * up).bfloat16().float(),
                reader.read(f"{expert}.down_proj.weight").float(),
            )
            .bfloat16()
            .float()
        )
        output += contribution * torch.where(ids == expert, weights, 0).sum(-1)[:, None]
    return output.bfloat16()


@torch.inference_mode()
@pytest.mark.parametrize("quantized", [True, False])
@pytest.mark.parametrize("ep_size,tp_size", [(1, 4), (2, 2)])
def test_expert_tp_shards_match_independent_local_references_and_graph_replay(
    quantized, ep_size, tp_size
):
    reader = SlicedReader(intermediate=2048)
    if not quantized:
        generator = torch.Generator().manual_seed(159)
        for name, value in list(reader.tensors.items()):
            if name.endswith(".weight"):
                reader.tensors[name] = (
                    torch.randn(value.shape[0], value.shape[1] * 2, generator=generator)
                    / 4
                ).bfloat16()
    config = replace(
        Glm53Config(),
        hidden_size=256,
        moe_intermediate_size=2048,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    kind = Nvfp4Experts if quantized else Bf16Experts
    # EP2's second owner also checks that expert0's global calibration survives.
    ep_rank = ep_size - 1
    shards = [
        kind(
            config,
            reader,
            "",
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_tokens=128,
            expert_tensor_parallel=TensorParallelLayout(rank, tp_size),
        )
        for rank in range(tp_size)
    ]
    references = [
        _reference_reader(
            reader,
            shard.intermediate_shard.start,
            shard.intermediate_shard.stop,
            quantized=quantized,
        )
        for shard in shards
    ]
    generator = torch.Generator(device="cuda").manual_seed(144)
    hidden = torch.randn(17, 256, device="cuda", generator=generator).bfloat16() * 3
    row = torch.arange(17, device="cuda")
    ids = torch.stack((row % 8, (row + 3) % 8), -1).int()
    weights = torch.rand(17, 2, device="cuda", generator=generator).softmax(-1) * 2.5
    reference = _moe_reference if quantized else _bf16_reference
    for shard, source in zip(shards, references, strict=True):
        expected = reference(source, hidden, ids, weights, shard.start, shard.count)
        _assert_error(
            shard(hidden, ids, weights),
            expected,
            relative_l2=0.025,
            atol=1 if not quantized else 0.5,
        )

    # Graph all local shards on one device without claiming NCCL combine coverage.
    def local_outputs():
        return torch.stack([shard(hidden, ids, weights) for shard in shards])

    graph, output = _capture(local_outputs)
    for inactive in (True, False):
        if inactive:
            ids.fill_(8)
            weights.zero_()
        else:
            hidden.mul_(-0.5)
            ids[:, 0].fill_(7)
            ids[:, 1].fill_(4)
            weights.fill_(1.25)
        expected = local_outputs()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        if inactive:
            assert not output.any()
