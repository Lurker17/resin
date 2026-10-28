import pytest
import torch
import torch.nn.functional as F

from resin_models_cuda.qwen3_5.gdn_decode import RecurrentStatePool, gdn_decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_in_place_decode_matches_fla_recurrence():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    torch.manual_seed(0)
    B, H, HV, K, V, L = 5, 16, 32, 128, 128, 3
    slots = 2 * B + 1
    state_elems = L * HV * K * V
    arena = torch.randn(slots, state_elems, device="cuda", dtype=torch.float32) * 0.1
    states = arena.view(slots, L, HV, K, V)
    layer = 1
    source = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    # Row 2 crosses a state block: it writes a different slot than it reads.
    destination = torch.tensor([1, 2, 9, 4, 5], device="cuda")
    pool = RecurrentStatePool(
        states, source * state_elems, destination * state_elems, HV * K * V
    )
    mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16)

    initial = states[source, layer].clone()
    q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    ref_o, ref_state = fused_recurrent_gated_delta_rule(
        q.view(B, 1, H, K),
        k.view(B, 1, H, K),
        v.view(B, 1, HV, V),
        g=g[:, None],
        beta=beta[:, None],
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    untouched = states[6].clone()
    out = gdn_decode(
        mixed[:, None],
        a[:, None],
        b[:, None],
        A_log,
        dt_bias,
        pool,
        layer,
        num_key_heads=H,
        num_value_heads=HV,
        key_dim=K,
        value_dim=V,
    )
    torch.testing.assert_close(
        out[:, 0].float(), ref_o[:, 0].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        states[destination, layer], ref_state, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(states[6], untouched)


def test_conv_decode_matches_torch_and_shifts_window():
    from resin_models_cuda.qwen3_5.gdn_decode import conv_decode

    torch.manual_seed(1)
    B, C, W, L, slots = 4, 8192, 4, 3, 9
    conv_states = torch.randn(slots, L, C, W - 1, device="cuda", dtype=torch.bfloat16)
    layer = 2
    source = torch.tensor([1, 2, 3, 4], device="cuda")
    destination = torch.tensor([1, 2, 7, 4], device="cuda")
    per_slot = L * C * (W - 1)
    states = torch.zeros(1, device="cuda")
    pool = RecurrentStatePool(
        states,
        source,
        destination,
        0,
        convolution_states=conv_states,
        convolution_source=source * per_slot,
        convolution_destination=destination * per_slot,
        convolution_layer_stride=C * (W - 1),
    )
    x = torch.randn(B, C, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(C, 1, W, device="cuda", dtype=torch.bfloat16)
    window = torch.cat((conv_states[source, layer], x[:, :, None]), dim=-1)
    ref = F.silu(F.conv1d(window, weight, groups=C))[:, :, 0]
    out = conv_decode(x, weight, pool, layer)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(conv_states[destination, layer], window[..., 1:])


def test_two_token_verify_and_commit_match_sequential_recurrence():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from resin_models_cuda.qwen3_5.gdn_decode import gdn_commit

    torch.manual_seed(4)
    B, H, HV, K, V, L, T = 4, 16, 32, 128, 128, 2, 2
    state_elems = L * HV * K * V
    for accepted_value in (0, 1):
        arena = torch.randn(8, state_elems, device="cuda", dtype=torch.float32) * 0.1
        states = arena.view(8, L, HV, K, V)
        source = torch.tensor([0, 1, 2, 3], device="cuda")
        stored = torch.tensor([4, 5, 6, 7], device="cuda")
        committed = (
            stored if accepted_value == 0 else torch.tensor([4, 5, 6, 7], device="cuda")
        )
        pending = torch.empty(L, B, T - 1, HV, K + V + 2, device="cuda")
        pool = RecurrentStatePool(
            states,
            source * state_elems,
            stored * state_elems,
            HV * K * V,
            pending=pending,
        )
        mixed = torch.randn(
            B, T, 2 * H * K + HV * V, device="cuda", dtype=torch.bfloat16
        )
        a = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        dt_bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        layer = 1
        initial = states[source, layer].clone()
        out = gdn_decode(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            pool,
            layer,
            num_key_heads=H,
            num_value_heads=HV,
            key_dim=K,
            value_dim=V,
        )
        q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        ref_o, ref_state = fused_recurrent_gated_delta_rule(
            q.reshape(B, T, H, K),
            k.reshape(B, T, H, K),
            v.reshape(B, T, HV, V),
            g=g,
            beta=b.sigmoid(),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        _, first_state = fused_recurrent_gated_delta_rule(
            q[:, :1].reshape(B, 1, H, K),
            k[:, :1].reshape(B, 1, H, K),
            v[:, :1].reshape(B, 1, HV, V),
            g=g[:, :1],
            beta=b[:, :1].sigmoid(),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(out.float(), ref_o.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            states[stored, layer], first_state, atol=1e-4, rtol=1e-4
        )
        accepted = torch.full((B,), accepted_value, device="cuda", dtype=torch.long)
        gdn_commit(
            pool,
            L,
            pool.destination,
            committed * state_elems,
            accepted,
            num_value_heads=HV,
            key_dim=K,
            value_dim=V,
        )
        expected = ref_state if accepted_value else first_state
        torch.testing.assert_close(
            states[committed, layer], expected, atol=1e-4, rtol=1e-4
        )
