"""Ordinary decode may reuse owned gathers, never Engine pages or tape inputs."""

from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.attention import KdaAttention

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def call(data, cache, bound=-5.0, **kwargs):
    return kernels.kda(
        data["q"],
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        data["a"],
        data["bias"],
        bound,
        cache,
        **kwargs,
    )


def owned_cache(data):
    return SimpleNamespace(
        recurrent=data["initial"],
        cu_seqlens=data["offsets"],
        cu_seqlens_cpu=None,
        initial_state_owned=True,
        initial_state_arena=torch.empty(1, device=data["initial"].device),
    )


@pytest.mark.parametrize("projection_rows", [0, 8])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("tape_attached", [False, True])
def test_attention_only_reuses_owned_decode_without_a_tape(
    monkeypatch, projection_rows, owned, tape_attached
):
    from test_glm53_attention import _kda_cache, _kda_config, _kda_weights

    model = KdaAttention(_kda_config(), 0, _kda_weights())
    cache = _kda_cache([1, 1])
    cache.initial_state_owned = owned
    if tape_attached:
        cache.speculative_tape = SimpleNamespace(
            record=lambda *args: None, reuse_initial_state=False
        )
    observed = []

    def observe(q, *args, **kwargs):
        observed.append(kwargs)
        return torch.zeros_like(q), None

    monkeypatch.setattr(kernels, "kda", observe)
    model(torch.randn(2, 8), cache.positions, cache, projection_rows=projection_rows)
    assert len(observed) == 1
    assert bool(observed[0].get("reuse_initial_state", False)) == (
        bool(projection_rows) and owned and not tape_attached
    )


@CUDA
@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("bound", [None, -5.0])
@pytest.mark.parametrize("rows,heads", [(9, 4), (9, 64), (64, 64)])
@torch.inference_mode()
def test_decode_reuse_matches_public_fla_core_and_state(width, bound, rows, heads):
    from test_glm53_kda_state import fixture

    data = fixture(rows, width, heads=heads)
    cache = owned_cache(data)
    before = data["initial"].clone()
    expected, independent = call(data, cache, bound)
    torch.testing.assert_close(data["initial"], before, rtol=0, atol=0)
    actual, final = call(data, cache, bound, reuse_initial_state=True)
    assert final is data["initial"]
    assert independent.data_ptr() != final.data_ptr()
    active_tokens = int(data["offsets"][-1].item())
    torch.testing.assert_close(
        actual[:active_tokens], expected[:active_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(final, independent, rtol=0, atol=0)
    torch.testing.assert_close(final[-1], before[-1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["marker", "arena", "alias", "dtype", "strides", "tape"]
)
def test_decode_reuse_rejects_unowned_or_unsupported_initial_state(invalid):
    cache = SimpleNamespace(
        recurrent=torch.zeros(3, 2, 8, 8),
        initial_state_owned=True,
        initial_state_arena=torch.empty(1),
    )
    if invalid == "marker":
        cache.initial_state_owned = False
    elif invalid == "arena":
        cache.initial_state_arena = None
    elif invalid == "alias":
        cache.initial_state_arena = cache.recurrent.view_as(cache.recurrent)
    elif invalid == "dtype":
        cache.recurrent = cache.recurrent.bfloat16()
    elif invalid == "strides":
        cache.recurrent = cache.recurrent.transpose(-1, -2)
    else:
        cache.speculative_tape = object()
    before = cache.recurrent.clone()
    with pytest.raises(ValueError):
        kernels._owned_recurrent_state(cache)
    torch.testing.assert_close(cache.recurrent, before, rtol=0, atol=0)


def test_decode_ownership_accepts_disjoint_gathered_layer_views():
    gathered = torch.zeros(2, 3, 2, 8, 8)
    arena = torch.empty_like(gathered)
    for layer in gathered:
        cache = SimpleNamespace(
            recurrent=layer, initial_state_owned=True, initial_state_arena=arena
        )
        assert kernels._owned_recurrent_state(cache) is layer


@CUDA
@torch.inference_mode()
def test_decode_reuse_preserves_all_idle_sequences():
    from test_glm53_kda_state import fixture

    data = fixture(8, 1, heads=64)
    data["offsets"].zero_()
    before = data["initial"].clone()
    _, final = call(data, owned_cache(data), reuse_initial_state=True)
    assert final is data["initial"]
    torch.testing.assert_close(final, before, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("invalid", ["verification", "chunk"])
@torch.inference_mode()
def test_decode_reuse_rejects_verification_and_chunk_prefill(invalid):
    from test_glm53_kda_state import fixture

    data = fixture(3, 9 if invalid == "chunk" else 4)
    cache = owned_cache(data)
    before = data["initial"].clone()
    kwargs = {"output_final_state": False} if invalid == "verification" else {}
    with pytest.raises(ValueError):
        call(data, cache, reuse_initial_state=True, **kwargs)
    torch.testing.assert_close(data["initial"], before, rtol=0, atol=0)


@CUDA
@torch.inference_mode()
def test_decode_graph_refreshes_owned_gathers_and_leaves_engine_pages_untouched():
    from test_glm53_cache_cuda import _fixture, _reference_gather
    from test_glm53_kda_state import fixture

    from resin_models_cuda.glm5_3_flash.cache import CacheGeometry

    rows, width = 5, 1
    geometry = CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention", "linear_attention"),
        hidden_size=32,
        latent_dim=8,
        index_dim=4,
        heads=64,
        head_dim=128,
        conv_history=3,
        state_tokens=64,
    )
    arena, cache = _fixture("cuda", rows, geometry=geometry)
    arena.recurrent_states.normal_()
    contexts = torch.full((rows,), 65, dtype=torch.int32, device="cuda")
    positions = contexts.clone()
    sequences = torch.arange(rows, dtype=torch.int32, device="cuda")
    data = [fixture(rows, width, heads=64, seed=seed) for seed in (194, 711)]

    def run():
        batch = cache.batch(positions, sequences, data[0]["offsets"], contexts)
        outputs, states = [], []
        for index, layer_id in enumerate(geometry.linear_layers):
            layer = batch.layer(layer_id)
            assert layer.initial_state_owned
            core, final = call(data[index], layer, reuse_initial_state=True)
            assert final is layer.recurrent
            outputs.append(core)
            states.append(final)
        return outputs, states

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs, states = run()
    for final in states:
        assert (
            final.untyped_storage().data_ptr()
            != arena.storage.untyped_storage().data_ptr()
        )
    for turn in range(4):
        arena.recurrent_states.normal_()
        contexts.fill_(64 if turn % 2 else 65)
        contexts[0] = 0  # A newly admitted request must gather zero state.
        positions.copy_(contexts)
        # Change the packed active/idle pattern without changing graph addresses.
        lengths = torch.tensor(
            [1, turn % 2, 1, 1 - turn % 2, 0], device="cuda", dtype=torch.int32
        )
        data[0]["offsets"][1:].copy_(lengths.cumsum(0))
        data[1]["offsets"].copy_(data[0]["offsets"])
        group = cache.groups[1]
        group.slab_ids.copy_(group.slab_ids.flip(0))
        before = arena.recurrent_states.clone()
        initial, _, _ = _reference_gather(cache, contexts)
        expected = []
        for index, entry in enumerate(data):
            for name in ("q", "k", "v", "g", "beta"):
                entry[name].normal_()
            source = SimpleNamespace(
                recurrent=initial[:, index].contiguous(),
                cu_seqlens=data[0]["offsets"],
                cu_seqlens_cpu=None,
            )
            expected.append(call(entry, source))
        graph.replay()
        active_tokens = int(data[0]["offsets"][-1].item())
        for actual, final, (reference, reference_final) in zip(
            outputs, states, expected, strict=True
        ):
            torch.testing.assert_close(
                actual[:active_tokens], reference[:active_tokens], rtol=0, atol=0
            )
            torch.testing.assert_close(final, reference_final, rtol=0, atol=0)
        torch.testing.assert_close(arena.recurrent_states, before, rtol=0, atol=0)
