"""Numerical and replay qualification for Qwen's native CUDA fast paths."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper or newer CUDA",
)
def test_qwen_fused_norms_grouped_delta_and_paged_attention():
    from deploy.qwen3_5_kernel_probe import run_probe

    assert run_probe()["ok"] is True
