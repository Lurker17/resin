"""Exact Torch oracles for fused staging and KDA state/input preparation."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _entries(kernel):
    return len(kernel.device_caches[torch.cuda.current_device()][0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_projection_stage_exact_strides_padding_and_graph_replay(dtype):
    from resin_models_cuda.glm5_3_flash.staging_cuda import _stage_rows, stage_rows

    before = None
    seen_strides = set()
    for rows in (1, 2, 7, 16, 40, 64):
        # The KDA BMM operand starts at a nonzero, non-256-aligned offset.
        parent = torch.randn(rows, 24896, dtype=dtype, device="cuda")
        source = parent[:, -256:].reshape(rows, 2, 128).transpose(0, 1)
        seen_strides.add(source.stride())
        target_parent = torch.empty(64, 24896, dtype=dtype, device="cuda")
        destination = target_parent[:, -256:].reshape(64, 2, 128).transpose(0, 1)
        reference = torch.zeros_like(destination)
        reference[:, :rows].copy_(source)
        stage_rows(source, destination, 1)
        assert torch.equal(destination, reference)
        if before is None:
            before = _entries(_stage_rows)
        else:
            assert _entries(_stage_rows) == before
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage_rows(source, destination, 1)
        for value in (1.25, -3.5):
            parent.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            reference.zero_()
            reference[:, :rows].copy_(source)
            assert torch.equal(destination, reference)
    # Singleton reshape may compact the otherwise packed token stride. Both
    # layouts must use the same compiled staging operation.
    assert len(seen_strides) > 1


def test_beta_exhaustive_bf16_and_strided_float32_match_torch_exactly():
    from resin_models_cuda.glm5_3_flash.state_cuda import beta_sigmoid

    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).reshape(-1, 64)
    torch.testing.assert_close(
        beta_sigmoid(values), values.float().sigmoid(), rtol=0, atol=0, equal_nan=True
    )
    parent = torch.randn(2049, 192, device="cuda") * 12
    values = parent[:, 64:128]
    torch.testing.assert_close(beta_sigmoid(values), values.sigmoid(), rtol=0, atol=0)


def test_beta_changed_rows_stride_graph_inputs_and_compilation():
    from resin_models_cuda.glm5_3_flash.state_cuda import _beta_sigmoid, beta_sigmoid

    before = None
    for rows, stride in ((1, 24896), (8, 24896), (65, 64), (127, 192), (2049, 64)):
        parent = torch.randn(rows, stride, dtype=torch.bfloat16, device="cuda")
        values = parent[:, :64]
        beta_sigmoid(values)
        if before is None:
            before = _entries(_beta_sigmoid)
        else:
            assert _entries(_beta_sigmoid) == before
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = beta_sigmoid(values)
        for scale in (0.75, -1.25):
            parent.mul_(scale)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(result, values.float().sigmoid())


@pytest.mark.parametrize("rows", [1, 2, 8])
def test_idle_repair_preserves_active_output_and_initial_owned_state(rows):
    from resin_models_cuda.glm5_3_flash.state_cuda import preserve_idle

    initial = torch.randn(rows, 64, 128, 128, device="cuda")
    initial_copy = initial.clone()
    final = torch.randn_like(initial)
    produced = final.clone()
    offsets = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    preserve_idle(initial, final, offsets)
    assert torch.equal(final, produced)
    assert torch.equal(initial, initial_copy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = preserve_idle(initial, final, offsets)
    assert returned is final
    for lengths in ([0] * rows, [r % 2 for r in range(rows)], [2] * rows):
        host_offsets = [0]
        for length in lengths:
            host_offsets.append(host_offsets[-1] + length)
        offsets.copy_(torch.tensor(host_offsets, device="cuda", dtype=torch.int32))
        final.copy_(produced)
        graph.replay()
        torch.cuda.synchronize()
        active = offsets[1:] > offsets[:-1]
        expected = torch.where(active[:, None, None, None], produced, initial_copy)
        assert torch.equal(final, expected)
        assert torch.equal(initial, initial_copy)
    with pytest.raises(ValueError, match="independent"):
        preserve_idle(initial, initial, offsets)


def test_idle_repair_handles_original_strided_state_view():
    from resin_models_cuda.glm5_3_flash.state_cuda import preserve_idle

    backing = torch.randn(2, 3, 4, 8, 8, device="cuda")
    initial = backing[:, 1]
    final = torch.full(initial.shape, float("nan"), device="cuda")
    offsets = torch.zeros(3, device="cuda", dtype=torch.int32)
    assert preserve_idle(initial, final, offsets) is final
    assert torch.equal(final, initial)
