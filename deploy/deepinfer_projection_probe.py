"""Dense projection numerical, capture and timing checks on a Blackwell GPU."""

from types import SimpleNamespace

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


def _metrics(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-20)).item(),
        "max_abs": delta.abs().max().item(),
    }


def _output_boundary_probe():
    from resin_models_cuda.deepseek_v41.attention import OutputProjection, finish_output

    # Exact released geometry: H64 x D512, eight O-A groups of rank1024,
    # then O-B projects the flattened 8192 channels back to hidden5120.
    groups, heads_per_group, rank, width = 8, 8, 1024, 4096
    source_a, weight_a = _source(width, groups * rank)
    source_b, weight_b = _source(groups * rank, 5120)
    projected = PreparedLinear(source_b)
    fused = SimpleNamespace(
        wo_a=OutputProjection(source_a, groups, rank), wo_b=projected
    )
    fallback = SimpleNamespace(
        wo_a=OutputProjection(source_a, groups, rank, fp8_output=False), wo_b=projected
    )
    results = []
    for rows in (1, 5, 1024, 1280):
        context = torch.randn(
            rows, groups * heads_per_group, 512, device="cuda", dtype=torch.bfloat16
        )
        # Vary every head's scale so group boundaries and packed-SF addressing
        # cannot accidentally pass by using uniform exponents.
        context.mul_(
            torch.logspace(-1, 1, groups * heads_per_group, device="cuda")[
                None, :, None
            ]
        )
        values, scales = _fp8_reference(context.flatten(1))
        restored = (values.float() * scales.repeat_interleave(32, -1)).reshape(
            rows, groups, width
        )
        lane_values = (
            values.reshape(rows, groups, heads_per_group, 16, 32)
            .transpose(2, 3)
            .reshape(rows, groups, width)
        )
        lane_scales = (
            scales.to(torch.float8_e8m0fnu)
            .reshape(rows, groups, heads_per_group, 16)
            .transpose(2, 3)
            .contiguous()
            .view(torch.int32)
            .reshape(rows, groups, width // 128)
        )
        packed_context = lane_values, lane_scales
        bf16_a = fallback.wo_a(*packed_context)
        expected_a = torch.einsum(
            "mgr,gdr->mgd", restored, weight_a.reshape(groups, rank, width)
        ).bfloat16()
        grouped_error = _metrics(bf16_a, expected_a)
        if grouped_error["relative_l2"] > 0.003:
            raise AssertionError(f"output projection rows={rows}: {grouped_error}")

        direct_values, direct_sf = fused.wo_a(*packed_context)
        # Pinned DeepGEMM epilogue/operators.cuh rounds its accumulators to
        # BF16 before amax/FP8 casting, matching this standalone cast exactly.
        expected_values, expected_sf = quantize_fp8(bf16_a.flatten(1))
        if not torch.equal(
            direct_values.flatten(1).view(torch.uint8),
            expected_values.view(torch.uint8),
        ):
            raise AssertionError(
                f"direct FP8 epilogue differs from BF16 + cast, rows={rows}"
            )
        if not torch.equal(direct_sf, expected_sf):
            raise AssertionError(f"direct FP8 output scales differ, rows={rows}")
        actual = finish_output(fused, packed_context)
        previous = finish_output(fallback, packed_context)
        if not torch.equal(actual, previous):
            raise AssertionError(
                f"output boundary differs from prior path, rows={rows}"
            )

        reference_values, reference_scales = _fp8_reference(expected_a.flatten(1))
        reference = F.linear(
            (
                reference_values.float() * reference_scales.repeat_interleave(32, -1)
            ).bfloat16(),
            weight_b.bfloat16(),
        )
        dense_error = _metrics(actual, reference)
        if dense_error["relative_l2"] > 0.02:
            raise AssertionError(
                f"complete output projection rows={rows}: {dense_error}"
            )
        results.append(
            {
                "rows": rows,
                "groups": groups,
                "rank": rank,
                "group_input": width,
                "output_shape": list(actual.shape),
                "packed_scale_shape": list(direct_sf.shape),
                "packed_scale_stride": list(direct_sf.stride()),
                "grouped_projection_oracle": grouped_error,
                "complete_projection_oracle": dense_error,
                "fp8_epilogue_bitwise_equal": True,
                "packed_scales_bitwise_equal": True,
                "previous_path_bitwise_equal": True,
                "direct_fp8": _graph(lambda: finish_output(fused, packed_context)),
                "bf16_then_cast": _graph(
                    lambda: finish_output(fallback, packed_context)
                ),
            }
        )
    return results


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
    return {
        "gpu": torch.cuda.get_device_name(),
        "projections": result,
        "output_boundary": _output_boundary_probe(),
    }
