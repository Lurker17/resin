"""Conservative top-k mass guard for the native nucleus selector."""

import pytest
import torch

from resin.backend.cuda.sampling import FusedSampler
from resin.backend.cuda.sampling_native import air_top_p_cutoff

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _mass_partials(weights, keep, block=1024):
    rows, vocab = weights.shape
    parts = (vocab + block - 1) // block
    padded = torch.zeros((rows, parts * block), dtype=torch.float64, device="cuda")
    padded[:, :vocab] = weights
    retained = padded.clone()
    retained[:, :vocab] *= keep
    return torch.stack(
        (padded.view(rows, parts, block).sum(-1),
         retained.view(rows, parts, block).sum(-1)),
        dim=-1,
    )


def test_air_guard_skips_only_distant_redundant_nuclei():
    weights = torch.zeros((7, 2051), device="cuda")
    weights[:, :3] = torch.tensor([0.60, 0.25, 0.15], device="cuda")
    weights[4, :4] = torch.tensor([0.30, 0.30, 0.30, 0.10], device="cuda")
    ps = torch.tensor([0.9, 0.65, 0.6005, 0.5995, 0.8, 0.65, 0.9], device="cuda")
    active = torch.ones(7, dtype=torch.bool, device="cuda")
    active[-1] = False
    guard_active = torch.ones_like(active)
    guard_active[-2] = False
    k_cutoff = torch.tensor([0.60, 0.25, 0.60, 0.60, 0.30, 0.60, 0.60], device="cuda")
    partials = _mass_partials(weights, weights >= k_cutoff[:, None])
    # Neither p-only nor inactive rows may consume their unwritten partials.
    partials[-2:] = torch.nan
    baseline = air_top_p_cutoff(weights, ps, active)
    output = torch.full((7, 3), -1.0, device="cuda")[:, 1]
    actual = air_top_p_cutoff(
        weights, ps, active, out=output,
        guard_mass=partials, guard_active=guard_active,
    )
    expected = baseline.clone()
    expected[0] = 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual[1:6] > 0).all()
    assert actual[-1] == 0


def test_air_guard_keeps_small_p_on_the_original_path():
    weights = torch.linspace(1.0, 0.5, 4096, device="cuda")[None, :]
    ps = torch.tensor([0.0009], device="cuda")
    active = torch.ones(1, dtype=torch.bool, device="cuda")
    partials = _mass_partials(weights, weights == weights.max())
    baseline = air_top_p_cutoff(weights, ps, active)
    actual = air_top_p_cutoff(
        weights, ps, active, guard_mass=partials, guard_active=active,
    )
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    assert actual[0] > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_guarded_preparation_retains_the_same_sampling_support(dtype):
    from resin.backend.cuda.filtered_sampling import prepare_filters
    from resin.backend.cuda.logits_sampling import sample_logits

    generator = torch.Generator().manual_seed(875)
    logits = torch.randn(8, 2051, generator=generator).to("cuda", dtype=dtype)
    temperature = torch.ones(8, device="cuda")
    ks = torch.full((8,), 40, dtype=torch.int32, device="cuda")
    ps = torch.full((8,), 0.9, device="cuda")
    active = torch.ones(8, dtype=torch.bool, device="cuda")
    state = prepare_filters(logits, temperature, ks, ps, active)
    assert (state[:, 1] == 0).all()
    weights = (logits.float() - state[:, 2, None]).exp()
    original = state.clone()
    air_top_p_cutoff(weights, ps, active, out=original[:, 1])
    assert (original[:, 1] > 0).all()
    seeds = torch.arange(8, device="cuda", dtype=torch.int64) + 2**62
    offsets = torch.full_like(seeds, 2**40)
    actual = sample_logits(logits, temperature, seeds, offsets, active, filter_state=state)
    expected = sample_logits(logits, temperature, seeds, offsets, active, filter_state=original)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_one_graph_replays_guard_taken_boundary_and_disabled_rows():
    from test_sampling_filters_cuda import _inputs, _reference

    from resin.backend.cuda.filtered_sampling import prepare_filters

    logits = torch.full((16, 2051), -torch.inf, device="cuda")
    logits[:, :3] = torch.tensor([0.60, 0.25, 0.15], device="cuda").log()
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.fill_(1)
    inputs.top_ps.fill_(0.9)
    inputs.stochastic_mask.fill_(True)

    def prepare():
        return prepare_filters(
            logits, inputs.temperatures, inputs.top_ks,
            inputs.top_ps, inputs.stochastic_mask,
        )

    sampler = FusedSampler()
    prepare()
    sampler.sample_tensors(logits, inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state = prepare()
        output = sampler.sample_tensors(logits, inputs)
    for k, p, skipped in (
        (1, 0.9, True),
        (2, 0.65, False),
        (2, 0.8505, False),  # Only .0005 beyond K mass: stay inside the margin.
        (2051, 0.9, False),
        (1, 1.0, True),
        (1, 0.9, True),
    ):
        inputs.top_ks.fill_(k)
        inputs.top_ps.fill_(p)
        inputs.offsets.add_(32)
        expected = _reference(logits, inputs)
        graph.replay()
        assert bool((state[:, 1] == 0).all()) == skipped
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_guard_metadata_and_output_alias_rejected_before_launch():
    weights = torch.ones((2, 2051), device="cuda")
    ps = torch.full((2,), 0.9, device="cuda")
    active = torch.ones(2, dtype=torch.bool, device="cuda")
    mass = torch.ones((2, 3, 2), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="provided together"):
        air_top_p_cutoff(weights, ps, active, guard_mass=mass)
    with pytest.raises(ValueError, match="FP64"):
        air_top_p_cutoff(weights, ps, active, guard_mass=mass.float(), guard_active=active)
    output = mass.view(torch.float32).flatten()[:2]
    before = mass.clone()
    with pytest.raises(ValueError, match="overlap"):
        air_top_p_cutoff(
            weights, ps, active, out=output, guard_mass=mass, guard_active=active,
        )
    torch.testing.assert_close(mass, before, atol=0, rtol=0)
