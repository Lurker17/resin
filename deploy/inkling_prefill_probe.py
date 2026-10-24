"""Weight-free CUDA parity, graph, JIT-reuse, and timing checks for Inkling sconv."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reference(inputs, state, weight):
    combined = torch.cat((state, inputs), dim=1)
    convolved = F.conv1d(
        combined.transpose(1, 2).float(), weight, groups=inputs.shape[-1]
    ).transpose(1, 2)
    return (convolved + inputs.float()).to(inputs.dtype), combined[:, -3:]


def _elapsed(invoke, *, repetitions=100):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repetitions):
        invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


def run_probe():
    import cuda.tile as ct

    from resin_models_cuda.inkling.cutile import (
        _short_convolution_kernel,
        short_convolution,
    )

    torch.manual_seed(71)
    cases = []
    for batch, tokens, width in (
        (1, 1, 1024),
        (2, 2, 1024),
        (2, 3, 1024),
        (2, 127, 1024),
        (1, 4096, 1024),
        (1, 4096, 4096),
        (2, 8191, 4096),
    ):
        inputs = torch.randn(batch, tokens, width, dtype=torch.bfloat16, device="cuda")
        state = torch.randn(batch, 3, width, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(width, 1, 4, dtype=torch.float32, device="cuda") * 0.2
        # Use noncontiguous state views as the model does after its state split.
        storage = torch.empty(batch, 3, width * 2, dtype=state.dtype, device="cuda")
        storage[..., :width].copy_(state)
        state = storage[..., :width]

        def invoke():
            return short_convolution(inputs, state, weight)

        expected, expected_state = _reference(inputs, state, weight)
        actual, actual_state = invoke()
        # Both implementations accumulate in FP32 before the final BF16 boundary;
        # cuDNN may change the last FP32 rounding bit through its reduction order.
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.00003)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        if tokens >= 3:
            assert actual_state.data_ptr() == inputs[:, -3:].data_ptr()
        if tokens > 3:
            cut = tokens // 2
            first, mid_state = short_convolution(inputs[:, :cut], state, weight)
            second, final_state = short_convolution(inputs[:, cut:], mid_state, weight)
            torch.testing.assert_close(
                torch.cat((first, second), 1), actual, rtol=0, atol=0
            )
            torch.testing.assert_close(final_state, actual_state, rtol=0, atol=0)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                invoke()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            captured, captured_state = invoke()
        for _ in range(3):
            inputs.normal_()
            state.normal_()
            graph.replay()
            eager, eager_state = invoke()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            torch.testing.assert_close(captured_state, eager_state, rtol=0, atol=0)
        cases.append(
            {
                "batch": batch,
                "tokens": tokens,
                "width": width,
                "fused_us": _elapsed(invoke),
                "reference_us": _elapsed(lambda: _reference(inputs, state, weight)),
                "graph_replays_exact": 3,
            }
        )

    # An unseen channel width isolates compile accounting from the cases above.
    # Odd lengths share one divisibility class, while all power-of-two prefill
    # buckets share another.
    compile_signatures = []
    compile_kernel = ct.kernel._compile

    def counted_compile(self, signature, context):
        if self is _short_convolution_kernel:
            compile_signatures.append(str(signature))
        return compile_kernel(self, signature, context)

    # Changing the compile hook counts cache misses without timing filesystem or
    # device initialization. Already-warmed signatures may add zero misses.
    ct.kernel._compile = counted_compile
    try:
        width = 2048  # Not compiled by the parity cases above.
        state = torch.zeros(1, 3, width, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(width, 1, 4, device="cuda")
        classes = []
        for lengths in ((127, 255, 1023, 4095, 8191), (128, 256, 1024, 4096, 8192)):
            per_length = []
            for tokens in lengths:
                inputs = torch.randn(
                    1, tokens, width, device="cuda", dtype=torch.bfloat16
                )
                before = len(compile_signatures)
                short_convolution(inputs, state, weight)
                torch.cuda.synchronize()
                per_length.append(len(compile_signatures) - before)
            assert sum(per_length[1:]) == 0, (lengths, per_length)
            classes.append({"tokens": lengths, "compile_misses": per_length})
        assert classes[0]["compile_misses"][0] == 1, classes
    finally:
        ct.kernel._compile = compile_kernel

    return {"cases": cases, "jit_reuse": classes}


if __name__ == "__main__":
    import json

    print(json.dumps(run_probe(), indent=2))
