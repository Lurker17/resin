"""Verification blocks preserve request RNG streams without column copies."""

from dataclasses import fields, replace
from importlib import import_module

import pytest
import torch

from resin import BatchPlan, SamplingParams
from resin.backend.cuda.sampling import FusedSampler, SamplingTensors
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def sampler():
    pytest.importorskip("triton")
    return FusedSampler()


def _inputs(rows=5, width=4, vocab=257, *, dtype=torch.float32, strided=False):
    generator = torch.Generator().manual_seed(171)
    physical_vocab = 2 * vocab + 3 if strided else vocab
    storage = torch.randn(rows, width, physical_vocab, generator=generator).to(
        device="cuda", dtype=dtype
    )
    logits = storage[..., 1 : 2 * vocab + 1 : 2] if strided else storage
    row = torch.arange(rows, device="cuda", dtype=torch.int64)
    sampling = SamplingTensors(
        temperatures=0.5 + (row % 3).float() * 0.5,
        top_ps=torch.where(row % 3 == 0, 1.0, 0.87),
        top_ks=torch.where(row % 2 == 0, vocab, 9).int(),
        seeds=(1 << 62) + 17 + row * (1 << 32),
        offsets=(1 << 42) + 97 + row * (1 << 36),
        stochastic_mask=row % 4 != 3,
    )
    return logits, sampling


def _columns(sampler, logits, sampling, start_step=0):
    return torch.stack(
        [
            sampler.sample_tensors(
                logits[:, column],
                replace(
                    sampling,
                    offsets=sampling.offsets
                    + (start_step + column) * SAMPLING_OFFSET_STRIDE,
                ),
            )
            for column in range(logits.shape[1])
        ],
        dim=1,
    )


@pytest.mark.parametrize("vocab", [257, 2051])
@pytest.mark.parametrize("start_step", [0, 1])
def test_plan_block_matches_eager_columns(sampler, vocab, start_step):
    logits, _ = _inputs(rows=4, width=4, vocab=vocab, strided=True)
    plan = BatchPlan(
        sequence_ids=[10, 11, 12, 13],
        context_lengths=[3, 7, 11, 15],
        token_ids=[1, 2, 3, 4],
        token_indptr=[0, 1, 2, 3, 4],
        sampling_params=[
            SamplingParams(temperature=0.7, top_p=0.8, top_k=7, seed=31 + row)
            for row in range(4)
        ],
    )
    expected = torch.stack(
        [
            sampler(
                logits[:, step],
                plan,
                offset_delta=(start_step + step) * SAMPLING_OFFSET_STRIDE,
            )
            for step in range(4)
        ],
        dim=1,
    )
    output = torch.empty((4, 4), dtype=torch.int64, device="cuda")
    actual = sampler(
        logits, plan, offset_delta=start_step * SAMPLING_OFFSET_STRIDE, out=output
    )
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [1, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_block_matches_columns_and_preserves_rng(sampler, width, dtype, strided):
    logits, sampling = _inputs(width=width, dtype=dtype, strided=strided)
    original = {
        field.name: getattr(sampling, field.name).clone() for field in fields(sampling)
    }
    expected = _columns(sampler, logits, sampling, start_step=1)
    output = torch.empty(logits.shape[:2], device="cuda", dtype=torch.int64)
    actual = sampler.sample_block(logits, sampling, start_step=1, out=output)
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in original.items():
        torch.testing.assert_close(getattr(sampling, name), value, rtol=0, atol=0)


def test_block_reordering_and_padding_preserve_requests(sampler):
    logits, sampling = _inputs(rows=5, width=6)
    expected = sampler.sample_block(logits, sampling)
    order = torch.tensor([4, 1, 3, 0, 2], device="cuda")
    reordered = replace(
        sampling,
        **{
            field.name: getattr(sampling, field.name)[order]
            for field in fields(sampling)
        },
    )
    torch.testing.assert_close(
        sampler.sample_block(logits[order], reordered), expected[order], rtol=0, atol=0
    )
    padded_logits, padded = _inputs(rows=8, width=6)
    padded_logits[:5].copy_(logits)
    for field in fields(sampling):
        getattr(padded, field.name)[:5].copy_(getattr(sampling, field.name))
    padded.stochastic_mask[5:] = False
    actual = sampler.sample_block(padded_logits, padded)
    torch.testing.assert_close(actual[:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[5:], padded_logits[5:].argmax(-1), rtol=0, atol=0)


def test_block_passes_a_view_of_strided_vocabulary(sampler, monkeypatch):
    logits, sampling = _inputs(strided=True)
    seen = []
    original = sampler.sample_tensors

    def record(flat, parameters, *, out=None):
        seen.append((flat.untyped_storage().data_ptr(), flat.stride(), flat.shape))
        return original(flat, parameters, out=out)

    monkeypatch.setattr(sampler, "sample_tensors", record)
    sampler.sample_block(logits, sampling)
    assert seen == [
        (
            logits.untyped_storage().data_ptr(),
            (logits.stride(1), logits.stride(2)),
            (20, 257),
        )
    ]


def _cache_entries():
    result = {}
    device = torch.cuda.current_device()
    for name in ("sampling_blocks", "logits_sampling", "filtered_sampling"):
        module = import_module("resin.backend.cuda." + name)
        for symbol, value in vars(module).items():
            if hasattr(value, "device_caches"):
                result[name + "." + symbol] = set(value.device_caches[device][0])
    return result


def test_block_graph_replays_live_metadata_without_new_jit(sampler):
    logits, sampling = _inputs(rows=7, width=4, strided=True)
    output = torch.empty((7, 4), device="cuda", dtype=torch.int64)
    expected = _columns(sampler, logits, sampling, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sampler.sample_block(logits, sampling, start_step=1, out=output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        sampler.sample_block(logits, sampling, start_step=1, out=output)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    before = _cache_entries()
    logits.mul_(-0.7)
    sampling.temperatures.fill_(1.3)
    sampling.top_ps.copy_(
        torch.tensor([1.0, 0.5, 0.9, 1.0, 0.8, 0.95, 1.0], device="cuda")
    )
    sampling.top_ks.copy_(
        torch.tensor([257, 7, 31, 257, 2, 9, 257], device="cuda", dtype=torch.int32)
    )
    sampling.seeds.add_(1 << 39)
    sampling.offsets.add_(1 << 44)
    sampling.stochastic_mask.logical_not_()
    expected = _columns(sampler, logits, sampling, 1)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # Policy values, width, row count, metadata strides and pointer offsets do
    # not add metadata-expansion variants or specialize the sampling kernels.
    for rows, width in ((1, 1), (3, 6), (8, 4)):
        values, params = _inputs(rows=rows, width=width, strided=True)
        shifted = {}
        for field in fields(params):
            value = getattr(params, field.name)
            storage = torch.empty(2 * rows + 1, device="cuda", dtype=value.dtype)
            view = storage[1::2]
            view.copy_(value)
            shifted[field.name] = view
        sampler.sample_block(values, replace(params, **shifted), start_step=0)
    torch.cuda.synchronize()
    assert _cache_entries() == before


@pytest.mark.parametrize("field", ["seeds", "offsets", "logits"])
def test_block_rejects_output_alias_before_writing(sampler, field):
    logits, sampling = _inputs(rows=3, width=4, vocab=256)
    if field == "logits":
        output = logits.view(torch.int64).flatten()[:12].view(3, 4)
        original = logits.clone()
    else:
        output = torch.empty((3, 4), device="cuda", dtype=torch.int64)
        alias = output.flatten()[:3]
        alias.copy_(getattr(sampling, field))
        sampling = replace(sampling, **{field: alias})
        original = alias.clone()
    with pytest.raises(ValueError, match="overlap"):
        sampler.sample_block(logits, sampling, out=output)
    torch.testing.assert_close(
        logits if field == "logits" else getattr(sampling, field),
        original,
        rtol=0,
        atol=0,
    )


def test_block_metadata_validation_and_empty_rows(sampler):
    logits, sampling = _inputs(rows=3, width=4)
    for step in (-1, 0.5, True):
        with pytest.raises(ValueError, match="start_step"):
            sampler.sample_block(logits, sampling, start_step=step)
    transposed, transposed_params = _inputs(rows=4, width=3)
    with pytest.raises(ValueError, match="flatten without copying"):
        sampler.sample_block(transposed.transpose(0, 1), sampling)
    with pytest.raises(ValueError, match="block logits"):
        sampler.sample_block(logits[:, :0], sampling)
    with pytest.raises(ValueError, match="block output"):
        sampler.sample_block(
            logits,
            sampling,
            out=torch.empty((3, 8), device="cuda", dtype=torch.int64)[:, ::2],
        )
    empty_logits, empty_sampling = _inputs(rows=0, width=4)
    assert sampler.sample_block(empty_logits, empty_sampling).shape == (0, 4)


@pytest.mark.parametrize("vocab", [257, 2051])
def test_block_capture_launch_count_is_independent_of_width(sampler, vocab):
    counts = []
    for width in (1, 4, 6):
        logits, sampling = _inputs(rows=5, width=width, vocab=vocab)
        expected = _columns(sampler, logits, sampling)
        sampler.sample_block(logits, sampling)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = sampler.sample_block(logits, sampling)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            graph.replay()
            torch.cuda.synchronize()
        names = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        # Native filtering can use several selection passes, but metadata and
        # the tiled draw run once for the entire block, regardless of width.
        for symbol in ("_expand_metadata", "_partials", "_finish"):
            assert names.count(symbol) == 1, names
        counts.append(len(names))
        assert output.shape == (5, width)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(set(counts)) == 1, counts
