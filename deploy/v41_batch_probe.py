"""Exact Engine page equations and mutable CUDA graph checks for V4.1 batches."""

from dataclasses import fields

import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.deepseek_v41.batch import ForwardMetadata, _reference
from resin_models_cuda.deepseek_v41.cache import CacheGeometry, DeviceCache


def _case(batch, queries, context, *, ca_width=None):
    anchors = torch.arange(batch) % 7 + context
    positions = anchors[:, None] + torch.arange(queries)
    packings = (4, 8, 2, 16)
    groups = []
    for index, packing in enumerate(packings):
        width = (
            (int(positions.max()) // 128 + 3 if ca_width is None else ca_width)
            if index == 1
            else max(5, (queries + 127) // 128 + 3)
        )
        ordinal = torch.arange(batch * width).reshape(batch, width) * 3 + 17
        first = torch.zeros(batch, dtype=torch.int32)
        if index != 1:
            first = (anchors // 128 - 1).clamp_min(0).int()
        groups.append(
            DevicePageGroup(
                (ordinal // packing + 1).int(), (ordinal % packing).int(), first
            )
        )
    geometry = CacheGeometry(
        attention_layers=6,
        kv_sources=(1, 3),
        ratios=(0, 2, 2, 1, 1, 0),
        compressed_offsets=(None, 0, 0, 0, 0, None),
        index_offsets=(None, 0, 0, 0, 0, None),
        compressor_sources=(1,),
        packings=packings,
        parent_bytes=4194304,
        max_write_tokens=queries,
        window_size=128,
        draft_tokens=8,
        max_batch_size=batch,
    )
    cache = DeviceCache(geometry, tuple(groups), torch.empty(0, dtype=torch.uint8))
    gpu = DeviceCache(
        geometry,
        tuple(
            DevicePageGroup(*(tensor.cuda() for tensor in _group(group)))
            for group in groups
        ),
        torch.empty(0, dtype=torch.uint8, device="cuda"),
    )
    mask = torch.arange(batch * queries).remainder(11) != 0
    selected = {}
    for ratio in (1, 2):
        values = torch.arange(512).expand(batch, queries, -1).clone()
        values *= 257
        values.remainder_(max(1, context // ratio))
        values[..., 0] = -1
        values[..., 1] = 0
        values[..., 2] = (positions + 1) // ratio - 1
        values[..., 3] = (positions + 1) // ratio
        values[..., 4] = (positions.max() // 128 + 9) * (128 // ratio)
        selected[ratio] = values.int()
    return cache, gpu, positions, mask, selected


def _group(group):
    return group.parent_ids, group.child_slots, group.first_blocks


def _assert_owned(cache, positions, draft):
    # Match Engine admission: every proposed write/state row and visible SWA
    # history remains owned even when a replay moves across a page boundary.
    for index in (0, 2):
        group = cache.groups[index]
        columns = positions // 128 - group.first_blocks[:, None]
        assert ((columns >= 0) & (columns < group.parent_ids.shape[1])).all()
    end = positions[:, :1] if draft else positions + 1
    start = (end - cache.geometry.window_size).clamp_min(0)
    first = start // 128 - cache.groups[0].first_blocks[:, None]
    assert (first >= 0).all()


def _tensors(frame):
    result = {}
    for field in fields(ForwardMetadata):
        value = getattr(frame, field.name)
        if isinstance(value, torch.Tensor):
            result[field.name] = value
        elif field.name != "_selected":
            for ratio, tensor in value.items():
                result[f"{field.name}_{ratio}"] = tensor
    return result


def _prepare(cache, positions, mask, selected, draft, *, reference=False):
    frame = (
        _reference(cache, positions, (1, 2), draft, mask)
        if reference
        else cache.prepare_metadata(positions, draft=draft, valid_tokens=mask)
    )
    result = _tensors(frame)
    for ratio, logical in selected.items():
        physical = frame.compressed_indices(3, ratio, logical)
        assert frame.compressed_indices(3, ratio, logical) is physical
        result[f"physical_{ratio}"] = physical
    return result


def _equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        value, target = actual[name].cpu(), expected[name].cpu()
        if not torch.equal(value, target):
            where = torch.nonzero(value != target)[0].tolist()
            raise AssertionError(
                f"{name}: mismatch at {where}: "
                f"{value[tuple(where)].item()} != {target[tuple(where)].item()}"
            )


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def _latency(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


def _growing_table_probe():
    from resin_models_cuda.deepseek_v41.upstream.cuda.batch_metadata import (
        prepare_engine_metadata_tl,
    )
    from resin_models_cuda.deepseek_v41.upstream.cuda.flash_mla_utils import (
        convert_kv_index_tl,
    )

    # Only runtime table tensors change. These handles must stay identical even
    # when the last table grows to a 128K history; no new specialization is needed.
    static = (4, 5, 5, (4, 8, 2), 128, 8, (1, 2), "int64", False, True)
    kernel = prepare_engine_metadata_tl(*static)
    converters = {ratio: convert_kv_index_tl(4, 512, 128 // ratio) for ratio in (1, 2)}
    results = []
    for width, context in ((1, 0), (3, 126), (1025, 131070)):
        cpu, gpu, positions, mask, selected = _case(3, 4, context, ca_width=width)
        device_positions, device_mask = positions.cuda(), mask.cuda()
        device_selected = {ratio: value.cuda() for ratio, value in selected.items()}

        def call():
            return _prepare(gpu, device_positions, device_mask, device_selected, False)

        expected = _prepare(cpu, positions, mask, selected, False)
        _assert_owned(cpu, positions, False)
        _equal(call(), expected)
        graph, output = _capture(call)
        graph.replay()
        torch.cuda.synchronize()
        _equal(output, expected)
        assert prepare_engine_metadata_tl(*static) is kernel
        for ratio in (1, 2):
            assert convert_kv_index_tl(4, 512, 128 // ratio) is converters[ratio]
        results.append({"width_ca": width, "exact_fields": len(output)})
    return {"same_compiled_factories": True, "cases": results}


@torch.inference_mode()
def run_probe():
    results = []
    for batch, queries, context, draft, masked, int32 in (
        (2, 1, 0, False, False, True),
        (3, 5, 126, False, True, False),
        (3, 3, 126, True, True, True),
        (256, 4, 131070, False, False, False),
        (256, 4, 131070, False, True, False),
        (256, 3, 131070, True, True, False),
        (1, 1024, 130048, False, True, True),
    ):
        cpu, gpu, positions, mask, selected = _case(batch, queries, context)
        if int32:
            positions = positions.int()
        if not masked:
            mask = None
        gpu_positions = positions.cuda()
        gpu_mask = None if mask is None else mask.cuda()
        gpu_selected = {ratio: values.cuda() for ratio, values in selected.items()}

        def call():
            return _prepare(gpu, gpu_positions, gpu_mask, gpu_selected, draft)

        _assert_owned(cpu, positions, draft)
        expected = _prepare(cpu, positions, mask, selected, draft)
        _equal(call(), expected)
        graph, output = _capture(call)
        initial_positions = positions.clone()
        original_first = [group.first_blocks.clone() for group in cpu.groups]
        for iteration in range(5):
            positions.copy_(initial_positions + iteration)
            if mask is not None:
                mask.copy_(
                    (torch.arange(batch * queries) + iteration).remainder(11) != 0
                )
            for index, (source, target) in enumerate(zip(cpu.groups, gpu.groups)):
                source.parent_ids.add_(3)
                source.child_slots.add_(1).remainder_(cpu.geometry.packings[index])
                if index != 1:
                    source.first_blocks.copy_(
                        (original_first[index] - iteration % 2).clamp_min(0)
                    )
                for value, device_value in zip(_group(source), _group(target)):
                    device_value.copy_(value)
            for ratio, values in selected.items():
                values[..., 2] = (positions + 1) // ratio - 1
                values[..., 3] = (positions + 1) // ratio
                gpu_selected[ratio].copy_(values)
            gpu_positions.copy_(positions)
            if gpu_mask is not None:
                gpu_mask.copy_(mask)
            _assert_owned(cpu, positions, draft)
            eager = call()
            graph.replay()
            torch.cuda.synchronize()
            _equal(output, eager)
            _equal(output, _prepare(cpu, positions, mask, selected, draft))
            for ratio in selected:
                assert torch.equal(gpu_selected[ratio].cpu(), selected[ratio])
        reference_graph, _ = _capture(
            lambda: _prepare(
                gpu, gpu_positions, gpu_mask, gpu_selected, draft, reference=True
            )
        )
        results.append(
            {
                "batch": batch,
                "queries": queries,
                "context": context,
                "draft": draft,
                "masked": masked,
                "position_dtype": str(positions.dtype),
                "exact_fields": len(output),
                "dynamic_graph_replays": 5,
                "native_ms": _latency(graph),
                "torch_shared_metadata_ms": _latency(reference_graph),
            }
        )
        print(f"[metadata-probe] {results[-1]}", flush=True)
    return {
        "gpu": torch.cuda.get_device_name(),
        "cases": results,
        "growing_tables": _growing_table_probe(),
    }
