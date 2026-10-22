"""Run the same bounded CUDA qualification used by the Modal deployment."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell CUDA",
)
def test_native_v41_quantization_hc_and_shared_paged_attention():
    from deploy.deepseek_v41_validation import run_probes

    assert run_probes()["ok"] is True
