"""No-checkpoint GPU checks and graph timings for GPT-OSS and V4 primitives."""

import torch

from resin_models_cuda.deepseek_v4.cache import DeviceMapping
from resin_models_cuda.deepseek_v4.cutile import (
    finish_index_topk,
    paged_cache_rows,
    rope_last_block_inplace,
)
from resin_models_cuda.gpt_oss.cutile import top4_softmax


def _graph_us(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10, output


def _previous_rope(values, frequencies, positions):
    output = values.clone()
    block = torch.view_as_complex(
        output[..., -64:].float().reshape(*output.shape[:-1], 32, 2)
    )
    block = block * torch.view_as_complex(frequencies)[positions].unsqueeze(2)
    output[..., -64:] = torch.view_as_real(block).flatten(-2).to(output.dtype)
    return output


def _fused_rope(values, frequencies, positions):
    return rope_last_block_inplace(values.clone(), frequencies, positions)


def _previous_rows(indices, mapping):
    physical = paged_cache_rows(
        mapping.parent_ids,
        mapping.child_slots,
        mapping.first_blocks,
        indices.clamp_min(0) * 4,
        block_tokens=128,
        divisor=4,
        parent_rows=mapping.parent_rows,
        packing=mapping.packing,
        row_scale=mapping.row_scale,
        layer_offset=mapping.layer_offset,
    )
    return torch.where(indices >= 0, physical, -1).int()


@torch.inference_mode()
def run_probe():
    torch.manual_seed(749)
    routing, indexer = [], []
    for experts in (32, 128):
        for rows in (8, 128, 1024):
            logits = (
                torch.rand(rows, experts, device="cuda").argsort(-1).bfloat16() / 64
            )

            def previous():
                values, indices = logits.topk(4, dim=-1)
                return indices.int(), values.softmax(-1).float()

            previous_us, expected = _graph_us(previous)
            fused_us, actual = _graph_us(lambda: top4_softmax(logits))
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
            routing.append(
                dict(
                    experts=experts,
                    rows=rows,
                    previous_us=previous_us,
                    fused_us=fused_us,
                )
            )

    for context in (4096, 32768, 131072):
        for queries in (1, 128, 1024):
            positions = torch.arange(
                context - queries, context, dtype=torch.int32, device="cuda"
            )[None]
            width = (context + 127) // 128
            mapping = DeviceMapping(
                parent_ids=torch.randint(
                    1, 128, (1, width), dtype=torch.int32, device="cuda"
                ),
                child_slots=torch.randint(
                    0, 2, (1, width), dtype=torch.int32, device="cuda"
                ),
                first_blocks=torch.zeros(1, dtype=torch.int32, device="cuda"),
                block_tokens=128,
                divisor=4,
                parent_rows=512,
                packing=2,
                row_scale=1,
                layer_offset=64,
            )
            indices = (
                torch.arange(192, device="cuda", dtype=torch.int32)[None, None]
                .expand(1, queries, -1)
                .clone()
            )
            indices *= 137
            indices %= (positions + 1)[..., None] // 4
            indices[..., ::7] = -1
            previous_us, expected = _graph_us(lambda: _previous_rows(indices, mapping))
            fused_us, actual = _graph_us(
                lambda: finish_index_topk(
                    indices, positions, 4, 0, cache_mapping=mapping
                )
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

            query = torch.randn(
                1, queries, 16, 128, device="cuda", dtype=torch.bfloat16
            )
            angles = torch.randn(context, 32, device="cuda")
            frequencies = torch.stack((angles.cos(), angles.sin()), dim=-1)
            old_rope_us, expected_rope = _graph_us(
                lambda: _previous_rope(query, frequencies, positions)
            )
            new_rope_us, actual_rope = _graph_us(
                lambda: _fused_rope(query, frequencies, positions)
            )
            # Complex multiply and CUDA FMA can differ by one BF16 ULP.
            torch.testing.assert_close(
                actual_rope, expected_rope, atol=0.03125, rtol=0.008
            )
            error = (
                actual_rope.float() - expected_rope.float()
            ).norm() / expected_rope.float().norm()
            assert error.item() < 0.003
            indexer.append(
                dict(
                    context=context,
                    queries=queries,
                    old_rows_us=previous_us,
                    new_rows_us=fused_us,
                    old_rope_us=old_rope_us,
                    new_rope_us=new_rope_us,
                    rope_relative_l2=error.item(),
                )
            )
    return dict(device=torch.cuda.get_device_name(), routing=routing, indexer=indexer)
