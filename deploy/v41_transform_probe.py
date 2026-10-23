"""Fused norm/RoPE/packed-store checks on strided Engine-style cache pages."""

from types import SimpleNamespace

import torch


def _reference(value, positions, attention, compressed):
    from deploy.flashmla_validation import _rotate

    if not compressed:
        value = (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
            * attention.kv_norm.weight.float()
        ).bfloat16()
    return _rotate(value[:, None], positions, attention.cos_sin)[:, 0]


def _case(rows, compressed):
    from deploy.flashmla_validation import _decode
    from resin_models_cuda.deepseek_v41.attention import store_attention_kv

    width = 288 if compressed else 528
    pages = (rows + 63) // 64
    stride = 64 * width + 256
    storage = torch.full((pages * stride + 256,), 173, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((pages, 64, 1, width), (stride, width, width, 1), 128)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    # Negative slots are how unfinished compression windows skip their store.
    if compressed:
        slots = slots.masked_fill(slots.remainder(2) == 0, -1)
    active = slots >= 0
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) % 4096
    angles = torch.randn(4096, 32, device="cuda")
    attention = SimpleNamespace(
        head_dim=512,
        rope_head_dim=64,
        cos_sin=torch.cat((angles.cos(), angles.sin()), -1),
        kv_norm=SimpleNamespace(
            weight=torch.randn(512, device="cuda", dtype=torch.bfloat16), eps=1e-6
        ),
    )
    value = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)

    def invoke():
        store_attention_kv(
            attention, value, positions, cache, slots, compressed=compressed
        )

    def compare():
        selected = cache[slots[active] // 64, slots[active] % 64, 0]
        actual = _decode(selected)
        expected = _reference(value[active], positions[active], attention, compressed)
        relative = (actual.float() - expected.float()).norm() / expected.float().norm()
        limit = 0.15 if compressed else 0.04
        assert relative.item() < limit, (rows, compressed, relative.item())
        # Neither unused child slots nor padding between parent slabs may change.
        changed = torch.zeros_like(storage, dtype=torch.bool)
        locations = (
            128
            + slots[active, None] // 64 * stride
            + slots[active, None] % 64 * width
            + torch.arange(width, device="cuda")
        )
        changed[locations] = True
        assert (storage[~changed] == 173).all().item()
        return relative.item()

    invoke()
    eager = compare()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        invoke()
    for _ in range(3):
        value.normal_()
        positions.add_(7).remainder_(4096)
        graph.replay()
        captured = cache.clone()
        compare()
        invoke()
        torch.testing.assert_close(cache, captured, rtol=0, atol=0)
    return {
        "rows": rows,
        "compressed": compressed,
        "relative_l2": eager,
        "graph_replays_exact": 3,
        "untouched_padding": True,
    }


def run_probe():
    torch.manual_seed(41)
    return [
        _case(rows, compressed)
        for rows in (1, 5, 1280)
        for compressed in (False, True)
        if not (rows == 1 and compressed)
    ]
