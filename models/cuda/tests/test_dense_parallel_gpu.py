"""Real NCCL and captured BF16/NVFP4 dense MLP tensor parallelism."""

from __future__ import annotations

import copy
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin_models_cuda.dense_parallel import shard_dense_mlp
from resin_models_cuda.nvfp4 import (
    Nvfp4LinearWeights,
    prepare_nvfp4_modules,
    select_nvfp4_backend,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices"
)


class _MLP(nn.Module):
    def __init__(self, quantized, activation):
        super().__init__()
        self.activation = activation

        def linear(inputs, outputs):
            if not quantized:
                return nn.Linear(
                    inputs, outputs, bias=True, device="cuda", dtype=torch.bfloat16
                )
            layer = Nvfp4LinearWeights(inputs, outputs).cuda()
            layer.weight.copy_(
                torch.randint(
                    0, 256, layer.weight.shape, dtype=torch.uint8, device="cuda"
                )
            )
            layer.weight_scale.fill_(0.03125)
            layer.weight_scale_2.fill_(0.25)
            layer.input_scale.fill_(1.0)
            return layer

        self.gate_proj = linear(256, 512)
        self.up_proj = linear(256, 512)
        self.down_proj = linear(512, 256)

    def forward(self, hidden):
        gate = self.gate_proj(hidden)
        gate = (
            F.silu(gate)
            if self.activation == "silu"
            else F.gelu(gate, approximate="tanh")
        )
        return self.down_proj(gate * self.up_proj(hidden))


def _worker(rank, port, directory):
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        timeout=timedelta(minutes=5),
        device_id=torch.device(f"cuda:{rank}"),
    )
    results = []
    try:
        parallel = TensorParallelContext(
            TensorParallelLayout(rank, 2), dist.group.WORLD
        )
        for quantized in (False, True):
            for activation in ("silu", "gelu"):
                torch.manual_seed(87)
                original = _MLP(quantized, activation).eval()
                sharded = shard_dense_mlp(copy.deepcopy(original), parallel)
                if quantized:
                    backend = select_nvfp4_backend(
                        "auto", torch.backends.cudnn.version()
                    )
                    original = prepare_nvfp4_modules(original, backend)
                    sharded = prepare_nvfp4_modules(sharded, backend)
                hidden = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda") * 0.1
                with torch.inference_mode():
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            expected, actual = original(hidden), sharded(hidden)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = sharded(hidden)
                    torch.cuda.current_stream().wait_stream(stream)
                    maximum = 0.0
                    for shift in (0.0, 0.01, -0.02):
                        hidden.add_(shift)
                        expected, actual = original(hidden), sharded(hidden)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
                        difference = (
                            actual.float() - expected.float()
                        ).norm() / expected.float().norm()
                        maximum = max(maximum, difference.item())
                        assert difference < 0.02, difference.item()
                    graph.reset()
                results.append(
                    dict(
                        quantized=quantized,
                        activation=activation,
                        relative_l2=maximum,
                        replay_exact=True,
                    )
                )
        Path(directory, f"{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_dense_tp_matches_unsplit_and_replays_changed_inputs(tmp_path):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 validation requires Blackwell")  # ty: ignore[too-many-positional-arguments]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(
        _worker, args=(port, str(tmp_path)), nprocs=2, join=True
    )
    for rank in range(2):
        results = json.loads((tmp_path / f"{rank}.json").read_text())
        assert len(results) == 4
        assert all(
            result["replay_exact"] and result["relative_l2"] < 0.02
            for result in results
        )
