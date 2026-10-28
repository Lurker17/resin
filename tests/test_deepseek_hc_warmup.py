"""Finite HC startup tactics cover ragged rows without touching model state."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.deepseek_v41.hc import prefill_hc_tactics, warm_prefill_hc


def test_b300_prefill_tactics_include_intermediate_split_regimes():
    assert prefill_hc_tactics(5120, 148, 8192) == (
        (1, 40),
        (193, 27),
        (321, 20),
        (449, 16),
    )
    assert prefill_hc_tactics(5120, 148, 128) == ((1, 40),)
    assert prefill_hc_tactics(5120, 148, 320) == ((1, 40), (193, 27))
    assert prefill_hc_tactics(5120, 148, 321)[-1] == (321, 20)
    assert prefill_hc_tactics(5120, 148, 1 << 20) == prefill_hc_tactics(5120, 148, 8192)


@pytest.mark.parametrize("hidden", [1024, 5120, 7168, 8192])
@pytest.mark.parametrize("num_sms", [1, 15, 16, 17, 32, 64, 132, 148, 160, 256])
def test_representatives_cover_all_rows_for_different_sm_counts(hidden, num_sms):
    # Compare the bounded enumeration with every row's upstream heuristic,
    # including partial tiles. A power-of-two-only warmup misses some regimes.
    expected = {}
    for rows in range(1, 8193):
        m_blocks = (rows + 63) // 64
        cap = min(64, max(16, num_sms // m_blocks))
        k_blocks = hidden // 64
        longest_task = (k_blocks + cap - 1) // cap
        splits = (k_blocks + longest_task - 1) // longest_task
        expected.setdefault(splits, rows)
    tactics = prefill_hc_tactics(hidden, num_sms, 8192)
    assert tactics == tuple((rows, splits) for splits, rows in expected.items())
    assert len(tactics) <= 49
    assert prefill_hc_tactics(hidden, num_sms, 8192, deterministic=True) == ((1, 16),)


@pytest.mark.parametrize(
    "args",
    [
        (0, 148, 8192),
        (1536, 148, 8192),
        (5120, 0, 8192),
        (5120, 148, 0),
        (5120, 148, (1 << 20) + 1),
    ],
)
def test_invalid_kernel_geometry_is_rejected(args):
    with pytest.raises(ValueError):
        prefill_hc_tactics(*args)


@pytest.mark.parametrize("deterministic", [False, True])
def test_warmup_uses_real_weights_but_only_scratch_bf16_state(
    monkeypatch, deterministic
):
    args = SimpleNamespace(
        dim=5120, hc_mult=4, norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20
    )
    block = SimpleNamespace(
        args=args,
        hc_attn_fn=torch.ones(24, 4 * args.dim),
        hc_attn_scale=torch.ones(3),
        hc_attn_base=torch.zeros(24),
        attn_norm=SimpleNamespace(weight=torch.ones(args.dim, dtype=torch.bfloat16)),
    )
    # There is deliberately no FFN, cache, or collective state on this block.
    calls = []
    weights = (
        block.hc_attn_fn,
        block.hc_attn_scale,
        block.hc_attn_base,
        block.attn_norm.weight,
    )
    saved = tuple(value.clone() for value in weights)

    def mega_mhc(output, residual, pre, post, comb, **kwargs):
        rows = output.shape[0]
        assert output.dtype == residual.dtype == torch.bfloat16
        assert residual.shape == (rows, 4, args.dim)
        assert pre.dtype == post.dtype == comb.dtype == torch.float32
        assert torch.count_nonzero(output) == torch.count_nonzero(residual) == 0
        assert kwargs["fn"] is block.hc_attn_fn
        assert kwargs["mix_scales"] is block.hc_attn_scale
        assert kwargs["mix_bases"] is block.hc_attn_base
        assert kwargs["rmsnorm_weight"] is block.attn_norm.weight
        assert not any(
            key in kwargs
            for key in ("y_fp8", "y_gemm_sf", "y_routed_sf", "y_shared_sf")
        )
        assert not torch.is_grad_enabled()
        for key, source in (
            ("new_residual", residual),
            ("new_prev_mix", pre),
            ("new_post_mix", post),
            ("new_comb_res_mix", comb),
            ("y_bf16", output),
        ):
            target = kwargs[key]
            assert target.data_ptr() != source.data_ptr()
            target.fill_(7)
        calls.append(rows)

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(get_num_sms=lambda: 148, mega_mhc=mega_mhc),
    )
    tactics = warm_prefill_hc(block, 8192, deterministic=deterministic)
    assert calls == ([1] if deterministic else [1, 193, 321, 449])
    assert tuple(rows for rows, _ in tactics) == tuple(calls)
    for actual, original in zip(weights, saved, strict=True):
        torch.testing.assert_close(actual, original)


def test_warmup_refuses_cuda_capture_before_allocating_scratch(monkeypatch):
    block = SimpleNamespace(
        args=SimpleNamespace(hc_mult=4),
        hc_attn_fn=SimpleNamespace(device=torch.device("cuda:0")),
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        warm_prefill_hc(block, 8192)
