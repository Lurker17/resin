"""Dense projection numerical, capture and timing checks on a Blackwell GPU."""

import torch
from torch.nn import functional as F

from deploy.deepseek_v41_validation import _fp8_reference
from resin_models_cuda.deepseek_v41.linear import Linear, PreparedLinear, quantize_fp8


def _source(inputs, outputs):
    with torch.device("cuda"):
        source = Linear(inputs, outputs)
    weight = torch.randn(outputs, inputs, device="cuda", dtype=torch.bfloat16) * 0.02
    blocks = weight.float().reshape((outputs + 31) // 32, 32, inputs // 32, 32)
    scales = (blocks.abs().amax((1, 3)).clamp_min(1e-4) / 448).log2().ceil().exp2()
    source.weight.copy_(
        (blocks / scales[:, None, :, None])
        .reshape(outputs, inputs)
        .to(source.weight.dtype)
    )
    source.scale.copy_(scales.to(source.scale.dtype))
    restored = source.weight.float() * scales.repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    return source, restored


def _graph(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        expected = call().clone()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(output, expected):
        raise AssertionError("prepared projection graph differs from eager")
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(20):
        graph.replay()
    stop.record()
    stop.synchronize()
    return {
        "replays": 5,
        "bitwise_equal": True,
        "graph_ms": start.elapsed_time(stop) / 20,
    }


@torch.inference_mode()
def run_probe():
    """Include small decode and B256 × five verification rows per GPU."""
    from resin_models_cuda.deepseek_v41.attention import QueryProjection

    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("prepared block32 DeepGEMM projections require SM10x")
    torch.manual_seed(4141)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = []
    for name, inputs, widths in (
        ("q_a_kv", 5120, (1024, 512)),
        ("q_b_index", 1024, (128 * 512, 64 * 128)),
    ):
        sources, weights = zip(
            *(_source(inputs, width) for width in widths), strict=True
        )
        prepared = PreparedLinear(*sources)
        query = (
            QueryProjection(sources[0], 128, 512, sources[1])
            if name == "q_b_index"
            else None
        )
        weight = torch.cat(weights)
        for rows in (1, 5, 1280):
            hidden = torch.randn(rows, inputs, device="cuda", dtype=torch.bfloat16)
            values, scales = _fp8_reference(hidden)
            actual_values, actual_scales = quantize_fp8(hidden)
            restored_scales = (
                actual_scales.contiguous()
                .view(torch.uint8)
                .reshape(rows, -1)[:, : inputs // 32]
                .view(torch.float8_e8m0fnu)
                .float()
            )
            torch.testing.assert_close(
                actual_values.float(), values.float(), rtol=0, atol=0
            )
            torch.testing.assert_close(restored_scales, scales, rtol=0, atol=0)
            expected = F.linear(
                (values.float() * scales.repeat_interleave(32, -1)).bfloat16(),
                weight.bfloat16(),
            )
            actual = prepared(hidden)
            delta = actual.float() - expected.float()
            relative_l2 = (
                delta.norm() / expected.float().norm().clamp_min(1e-20)
            ).item()
            if relative_l2 > 0.005:
                raise AssertionError(f"{name} rows={rows}: relative L2 {relative_l2}")
            result.append(
                {
                    "name": name,
                    "rows": rows,
                    "inputs": inputs,
                    "outputs": sum(widths),
                    "relative_l2": relative_l2,
                    "max_abs": delta.abs().max().item(),
                    "prepared": _graph(lambda: prepared(hidden)),
                    "checkpoint_separate": _graph(
                        lambda: torch.cat([source(hidden) for source in sources], -1)
                    ),
                }
            )
            if query is not None and rows == 5:
                for query_kind, projection in (
                    ("merged_query", query),
                    ("unmerged_query", QueryProjection(sources[0], 128, 512)),
                ):
                    main, index = projection.project(hidden)
                    # Invert FlashMLA's Q-B lane layout independently. The
                    # optional index suffix stays in ordinary head-major order.
                    restored_main = (
                        main.reshape(rows, 32, 128, 16)
                        .transpose(1, 2)
                        .reshape(rows, -1)
                    )
                    parts = [("main", restored_main, expected[:, : 128 * 512])]
                    if index is not None:
                        parts.append(("index", index, expected[:, 128 * 512 :]))
                    errors = {}
                    for label, actual_part, expected_part in parts:
                        difference = actual_part.float() - expected_part.float()
                        error = (
                            difference.norm() / expected_part.float().norm()
                        ).item()
                        if error > 0.005:
                            raise AssertionError(
                                f"{query_kind} {label}: relative L2 {error}"
                            )
                        errors[label] = {
                            "relative_l2": error,
                            "max_abs": difference.abs().max().item(),
                        }

                    def captured_query(projection=projection):
                        main, index = projection.project(hidden)
                        return main if index is None else torch.cat((main, index), -1)

                    result[-1][query_kind] = {
                        **errors,
                        "capture": _graph(captured_query),
                    }
    return {"gpu": torch.cuda.get_device_name(), "projections": result}
