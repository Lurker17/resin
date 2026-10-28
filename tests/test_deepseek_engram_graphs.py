"""Engram projection buckets preserve live rows and request-local gating."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_cuda_prefill_graphs import _runner
from torch import nn

from resin import BatchPhase
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.deepseek_v41.engram import Engram


def _engram() -> Any:
    module: Any = Engram.__new__(Engram)
    nn.Module.__init__(module)
    module.dim, module.hc_mult, module.eps = 4, 2, 1e-5
    module.wkv = nn.Linear(192, 12, bias=False, dtype=torch.bfloat16)
    module.q_weight = nn.Parameter(torch.ones(2, 4))
    module.k_weight = nn.Parameter(torch.ones(2, 4))
    module.prefill_projection = None
    return module


def _configured(monkeypatch, *, policy="breakable"):
    runner: Any = _runner(prefill=policy)
    runner.prefill_chunk_size = 1024
    runner._prefill_compute_context = nullcontext
    modules = (_engram(), _engram())
    runner._model = SimpleNamespace(
        layers=[
            SimpleNamespace(layer_id=1, engram=modules[0]),
            SimpleNamespace(layer_id=2, engram=None),
            SimpleNamespace(layer_id=14, engram=modules[1]),
        ]
    )
    names = [tuple(module.state_dict()) for module in modules]
    parameters = [tuple(map(id, module.parameters())) for module in modules]
    DeepSeekV41CudaRunner._configure_prefill_engram_graphs(runner)
    assert [tuple(module.state_dict()) for module in modules] == names
    assert [tuple(map(id, module.parameters())) for module in modules] == parameters
    calls = []

    def replay(mode, shape, inputs):
        assert mode == "prefill_engram"
        graphs = runner._prefill_tensor_graphs[mode]
        step = graphs.build(shape)
        assert step.inputs is inputs
        calls.append((shape, inputs["0"].clone()))
        with torch.inference_mode():
            step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    return runner, modules, calls


def test_engram_prefill_uses_bounded_buckets_and_clears_previous_tail(monkeypatch):
    runner, (first, second), calls = _configured(monkeypatch)
    graphs = runner._prefill_tensor_graphs["prefill_engram"]
    assert graphs.buckets == tuple(2**p for p in range(11))
    assert graphs.pad_up_to == runner.prefill_chunk_size
    assert graphs._buffers(1024, 1) is graphs._buffers(1024, 2)
    generator = torch.Generator().manual_seed(71)
    saved = []
    with torch.inference_mode(), runner._prefill_graphs():
        for rows, module in ((763, first), (779, second), (155, first), (1, second)):
            hidden = torch.randn(1, rows, 2, 4, generator=generator).bfloat16()
            embedding = torch.randn(1, rows, 6, 32, generator=generator).bfloat16()
            mask = torch.arange(rows)[None] % 3 != 0
            expected = module(hidden, token_mask=mask, embed_hook=lambda: embedding)
            output = module(
                hidden,
                token_mask=mask,
                embed_hook=lambda: embedding,
                prefill=True,
            )
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(output[:, ~mask[0]], hidden[:, ~mask[0]])
            saved.append((output, output.clone()))
            bucket, staged = calls[-1][0][0], calls[-1][1]
            assert bucket == 1 << (rows - 1).bit_length()
            torch.testing.assert_close(
                staged[:rows], embedding.flatten(0, 1).flatten(1)
            )
            assert torch.count_nonzero(staged[rows:]) == 0
    assert [shape for shape, _ in calls] == [(1024, 1), (1024, 2), (256, 1), (1, 2)]
    # Gate results must not alias a projection buffer reused by the next layer.
    for output, expected in saved:
        torch.testing.assert_close(output, expected)
    assert runner.cuda_graph_stats["prefill_engram"]["fallback_counts"] == {}


@pytest.mark.parametrize("policy", ["breakable", "disabled"])
def test_engram_decode_and_disabled_prefill_keep_direct_projection(monkeypatch, policy):
    runner, (module, _), calls = _configured(monkeypatch, policy=policy)
    hidden, embedding = torch.ones(1, 3, 2, 4), torch.ones(1, 3, 6, 32).bfloat16()
    with torch.inference_mode(), runner._prefill_graphs():
        expected = module(hidden, embed_hook=lambda: embedding)
        assert not calls
        if policy == "disabled":
            torch.testing.assert_close(
                module(hidden, embed_hook=lambda: embedding, prefill=True), expected
            )
            assert not calls


def test_no_engram_adds_no_graph_mode(monkeypatch):
    runner: Any = _runner()
    runner._model = SimpleNamespace(layers=[SimpleNamespace(engram=None)])
    DeepSeekV41CudaRunner._configure_prefill_engram_graphs(runner)
    assert "prefill_engram" not in runner._prefill_tensor_graphs


def test_engram_mode_requires_capture_and_releases_its_buffers(monkeypatch):
    runner, _, _ = _configured(monkeypatch)
    runner._configure_prefill_graphs((), hidden_size=4, max_tokens=1024)
    with pytest.raises(RuntimeError, match="22 configured CUDA graph shapes"):
        runner._check_cuda_graphs(BatchPhase.PREFILL)
    graphs = runner._prefill_tensor_graphs["prefill_engram"]
    graphs._buffers(1024, 1)
    assert graphs.buffers
    runner.close_cuda_graphs()
    assert not graphs.buffers
