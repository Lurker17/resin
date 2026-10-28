"""Ragged prefill uses captured row-independent attention projections."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_cuda_prefill_graphs import _runner

from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner


@pytest.mark.parametrize("with_index", [False, True])
def test_attention_regions_pad_ragged_rows_and_slice_live_outputs(
    monkeypatch, with_index
):
    import resin_models_cuda.deepseek_v41.attention as implementation

    runner: Any = _runner()
    runner.prefill_chunk_size = 2048
    runner._prefill_compute_context = nullcontext
    attention = SimpleNamespace(
        dim=4,
        q_lora_rank=8,
        n_local_heads=1,
        n_local_groups=1,
        head_dim=128,
        _project=lambda hidden: (hidden.repeat(1, 1, 2), hidden.repeat(1, 1, 32)),
        project_query=lambda qr: (
            qr.repeat(1, 16),
            qr.repeat(1, 4) if with_index else None,
        ),
    )
    if with_index:
        attention.indexer = SimpleNamespace(local_heads=1, head_dim=32)

    def finish(_attention, output):
        values, scales = output
        return (values[:, 0, :4].float() + scales[:, 0, :1]).bfloat16()

    monkeypatch.setattr(implementation, "finish_output", finish)
    runner._model = SimpleNamespace(
        layers=[SimpleNamespace(layer_id=2, attn=attention)]
    )
    DeepSeekV41CudaRunner._configure_prefill_attention_graphs(runner)
    graphs = runner._prefill_tensor_graphs["prefill_attention"]
    assert graphs.pad_up_to == runner.prefill_chunk_size
    assert graphs.buckets == tuple(2**p for p in range(12))
    # Local projection capacities do not inherit a larger expert/DP bucket.
    runner._distributed_prefill_bucket = 2048
    runner._distributed_prefill_scope = True
    calls = []

    def replay(mode, shape, inputs):
        assert mode == "prefill_attention"
        step = graphs.build(shape)
        assert step.inputs is inputs
        calls.append((shape, {key: value.clone() for key, value in inputs.items()}))
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    generator = torch.Generator().manual_seed(103)
    with torch.inference_mode(), runner._prefill_graphs():
        for rows in (763, 779, 1877, 155, 1):
            hidden = torch.randn(1, rows, 4, generator=generator).bfloat16()
            expected_qr, expected_kv = attention._project(hidden)
            qr, kv = attention.prefill_project(hidden)
            torch.testing.assert_close(qr, expected_qr)
            torch.testing.assert_close(kv, expected_kv)
            q, index = attention.prefill_query(qr)
            expected_q, expected_index = attention.project_query(expected_qr[0])
            torch.testing.assert_close(q, expected_q[None])
            if with_index:
                assert index is not None and expected_index is not None
                torch.testing.assert_close(index, expected_index[None])
            else:
                assert index is None
            # Both FP8 values and packed scales must keep live rows and zero
            # the padded tail when a smaller request reuses a graph bucket.
            values = torch.randint(-4, 5, (rows, 1, 128), generator=generator).to(
                torch.float8_e4m3fn
            )
            scales = torch.randint(
                1, 8, (rows, 1, 1), generator=generator, dtype=torch.int32
            )
            positions = 4096 + torch.arange(rows)
            output = attention.prefill_finish((values, scales), positions)
            torch.testing.assert_close(
                output, finish(attention, (values, scales))[None]
            )
            torch.testing.assert_close(positions, 4096 + torch.arange(rows))
            bucket = 1 << (rows - 1).bit_length()
            for (shape, staged), region in zip(calls[-3:], (1, 2, 3), strict=True):
                assert shape == (bucket, region)
                for value in staged.values():
                    assert torch.count_nonzero(value[rows:].float()) == 0
            torch.testing.assert_close(calls[-1][1]["0"][:rows].float(), values.float())
            torch.testing.assert_close(calls[-1][1]["1"][:rows], scales)
    assert runner.cuda_graph_stats["prefill_attention"]["fallback_counts"] == {}
