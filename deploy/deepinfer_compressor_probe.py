"""Compressor pooling and rollback oracle over real Engine slab addresses."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from deploy.deepinfer_projection_probe import _graph
from deploy.deepseek_v41_validation import _check
from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.deepseek_v41.cache import CacheArena, CacheGeometry
from resin_models_cuda.deepseek_v41.compressor import (
    PreparedCompressor,
    compress_projected,
)
from resin_models_cuda.deepseek_v41.linear import Linear


def _cache():
    args = SimpleNamespace(
        n_layers=3,
        n_mtp_layers=0,
        dspark_block_size=4,
        compress_ratios=(0, 2, 2),
        kv_source_layers=(1,),
        window_size=128,
    )
    geometry = CacheGeometry.build(args, prefill_chunk_size=260, max_batch_size=2)
    groups, parent = [], 1
    for packing in geometry.packings:
        ordinal = torch.tensor([[4, 1, 5], [2, 0, 3]], device="cuda")
        groups.append(
            DevicePageGroup(
                (parent + ordinal // packing).int(),
                (ordinal % packing).int(),
                torch.zeros(2, dtype=torch.int32, device="cuda"),
            )
        )
        parent += math.ceil(6 / packing)
    arena = CacheArena("cuda", geometry.layout(parent), geometry, dtype=torch.bfloat16)
    arena.storage.zero_()
    return arena._make_graph_cache(tuple(groups))


def _reference(scores, values):
    return (
        (values.unflatten(1, (-1, 2)) * scores.unflatten(1, (-1, 2)).softmax(2))
        .sum(2)
        .bfloat16()
    )


@torch.inference_mode()
def run_probe():
    torch.manual_seed(4142)
    torch.backends.cuda.matmul.allow_tf32 = False
    values = torch.randn(2, 260, 512, device="cuda")
    scores = torch.randn_like(values)
    expected = _reference(scores, values)
    result = []
    for chunk in (1, 3, 128, 260):
        cache = _cache()
        comparisons = []
        for start in range(0, 260, chunk):
            end = min(start + chunk, 260)
            positions = torch.arange(start, end, device="cuda").expand(2, -1)
            actual = compress_projected(
                scores[:, start:end], values[:, start:end], positions, cache, 1, 2
            )
            complete = (positions[0] + 1) % 2 == 0
            if complete.any():
                comparisons.append(
                    _check(
                        actual[:, complete],
                        expected[:, positions[0, complete] // 2],
                        atol=0.0078125,
                        rtol=0.005,
                    )
                )
            if (actual[:, ~complete] != 0).any():
                raise AssertionError("incomplete C2 windows must emit zeros")
        result.append({"chunk": chunk, "comparisons": comparisons})

    cache = _cache()
    positions = torch.arange(125, device="cuda").expand(2, -1)
    compress_projected(scores[:, :125], values[:, :125], positions, cache, 1, 2)
    previous_rows = cache.compressor_rows(1, positions)
    previous = [cache.compressor_values[rows].clone() for rows in previous_rows]
    positions = torch.arange(125, 130, device="cuda").expand(2, -1)

    def decode():
        return compress_projected(
            scores[:, 125:130], values[:, 125:130], positions, cache, 1, 2
        )

    graph = _graph(decode)
    for rows, before in zip(previous_rows, previous, strict=True):
        torch.testing.assert_close(
            cache.compressor_values[rows], before, rtol=0, atol=0
        )
    actual = decode()
    completed = (positions[0] + 1) % 2 == 0
    rollback = _check(
        actual[:, completed],
        expected[:, positions[0, completed] // 2],
        atol=0.0078125,
        rtol=0.005,
    )

    with torch.device("cuda"):
        source = SimpleNamespace(
            layer=1,
            ratio=2,
            norm=nn.Identity(),
            wkv=Linear(128, 512, dtype=torch.float32),
            wgate=Linear(128, 512, dtype=torch.float32),
        )
    source.wkv.weight.copy_(torch.randn_like(source.wkv.weight).bfloat16() * 0.02)
    source.wgate.weight.copy_(torch.randn_like(source.wgate.weight).bfloat16() * 0.02)
    prepared = PreparedCompressor(source)
    hidden = torch.randn(2, 130, 128, device="cuda", dtype=torch.bfloat16)
    # Evaluate the explicitly prepared BF16 weights with an independent FP32 GEMM.
    values = hidden.float() @ source.wkv.weight.bfloat16().float().T
    scores = hidden.float() @ source.wgate.weight.bfloat16().float().T
    positions = torch.arange(130, device="cuda").expand(2, -1)
    cache = _cache()
    projected = _check(
        prepared(hidden, positions, cache)[:, 1::2],
        _reference(scores, values),
        atol=0.0078125,
        rtol=0.005,
    )
    return {
        "gpu": torch.cuda.get_device_name(),
        "chunking": result,
        "decode_graph": graph,
        "rollback": rollback,
        "merged_projection": projected,
    }
