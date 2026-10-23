"""Real vocabulary/Markov head shapes against independent FP32 GEMMs."""

from types import SimpleNamespace

import torch
from torch.nn import functional as F

from deploy.deepinfer_projection_probe import _graph
from resin.backend.cuda.parallel import TensorParallelContext
from resin_models_cuda.deepseek_v41.linear import PreparedVocabHead


@torch.inference_mode()
def run_probe():
    torch.manual_seed(4143)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = []
    for name, dim, row_counts in (
        ("vocabulary", 5120, (1, 1024)),
        ("markov", 256, (1, 256)),
    ):
        # Like the checkpoint loader, retain BF16 checkpoint values in FP32
        # for the previous SGEMM path. Preparation recovers the exact BF16 bits.
        source = SimpleNamespace(
            weight=(
                torch.randn(129280, dim, device="cuda", dtype=torch.bfloat16) * 0.02
            ).float(),
            tensor_parallel=TensorParallelContext(),
        )
        head = PreparedVocabHead(source)
        if not torch.equal(head.weight.float(), source.weight):
            raise AssertionError("head preparation changed checkpoint weight values")
        for rows in row_counts:
            hidden = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)

            def prepared():
                return head(hidden, full_logits=True)

            def reference():
                return F.linear(hidden.float(), source.weight)

            expected, actual = reference(), prepared()
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            difference = actual - expected
            relative_l2 = (difference.norm() / expected.norm().clamp_min(1e-20)).item()
            if relative_l2 > 2e-5:
                raise AssertionError(f"{name} rows={rows}: relative L2 {relative_l2}")
            result.append(
                {
                    "name": name,
                    "rows": rows,
                    "hidden": dim,
                    "vocabulary": 129280,
                    "output_dtype": str(actual.dtype),
                    "relative_l2": relative_l2,
                    "max_abs": difference.abs().max().item(),
                    "prepared": _graph(prepared),
                    "previous_sgemm": _graph(reference),
                    "top1_agreement": (actual.argmax(-1) == expected.argmax(-1))
                    .float()
                    .mean()
                    .item(),
                }
            )
    return {"gpu": torch.cuda.get_device_name(), "heads": result}
