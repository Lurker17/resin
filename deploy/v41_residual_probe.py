"""Independent residual equations and strict CUDA graph checks at decode sizes."""

import torch
from torch import nn

from resin_models_cuda.deepseek_v41.engram import Engram
from resin_models_cuda.deepseek_v41.hc import HCLazyPost, materialize


def _post_reference(output, state):
    # Explicit input -> output matrix contraction, without the implementation's
    # broadcast expression or the native kernel's accumulation order.
    mixed = torch.einsum(
        "...ij,...id->...jd", state.comb.double(), state.residual.double()
    )
    return (
        mixed + state.post.double()[..., None] * output.double()[..., None, :]
    ).bfloat16()


def _post_previous(output, state):
    return (
        state.post[..., None] * output[..., None, :]
        + (state.comb[..., None] * state.residual.unsqueeze(-2)).sum(-3)
    ).bfloat16()


def _gate_reference(hidden, projection, weight, eps):
    hc, dim = hidden.shape[-2:]
    kv = projection.double().unflatten(-1, (hc + 1, dim))
    key, value, trunk = kv[..., :hc, :], kv[..., hc, :], hidden.double()
    dot = torch.einsum("...hd,hd,...hd->...h", trunk, weight.double(), key)
    dot *= (trunk.square().mean(-1) + eps).rsqrt()
    dot *= (key.square().mean(-1) + eps).rsqrt() * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (trunk + gate[..., None] * value[..., None, :]).bfloat16()


def _gate_previous(hidden, projection, weight, eps):
    hc, dim = hidden.shape[-2:]
    kv = projection.float().unflatten(-1, (hc + 1, dim))
    key, value, trunk = kv[..., :hc, :], kv[..., hc, :], hidden.float()
    scale = (trunk.square().mean(-1) + eps).rsqrt()
    scale *= (key.square().mean(-1) + eps).rsqrt()
    dot = (trunk * weight * key).sum(-1) * scale * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (trunk + gate[..., None] * value[..., None, :]).bfloat16()


def _check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)
    difference = actual.float() - expected.float()
    relative_l2 = (difference.norm() / expected.float().norm().clamp_min(1e-20)).item()
    if relative_l2 > 0.003:
        raise AssertionError(f"residual relative L2 error {relative_l2}")
    return {"relative_l2": relative_l2, "max_abs": difference.abs().max().item()}


def _graph(call, change, reference):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    torch.cuda.current_stream().wait_stream(stream)
    for iteration in range(5):
        change(iteration)
        expected = call().clone()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(result, expected):
            raise AssertionError("residual graph replay differs from eager")
        _check(result, reference())
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
        "dynamic_replays": 5,
        "bitwise_equal": True,
        "graph_ms": start.elapsed_time(stop) / 20,
    }


class _Projection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, _):
        return self.value


@torch.inference_mode()
def run_probe():
    torch.manual_seed(4244)
    results = []
    for batch, queries, hc, dim in (
        (1, 1, 4, 1024),
        (2, 3, 4, 5120),
        (256, 4, 4, 5120),
    ):
        hidden = torch.randn(
            batch, queries, hc, dim, dtype=torch.bfloat16, device="cuda"
        )
        output = torch.randn(batch, queries, dim, dtype=torch.bfloat16, device="cuda")
        state = HCLazyPost(
            hidden,
            torch.randn(batch, queries, hc, device="cuda"),
            torch.rand(batch, queries, hc, device="cuda") * 2,
            torch.softmax(torch.randn(batch, queries, hc, hc, device="cuda"), -1),
        )

        def post():
            return materialize(output, state)

        def reference():
            return _post_reference(output, state)

        hc_result = {
            "operator": "mhc_post",
            "shape": list(hidden.shape),
            **_check(post(), reference()),
            "native": _graph(post, lambda _: output.mul_(0.9375), reference),
            "previous_torch": _graph(
                lambda: _post_previous(output, state), lambda _: None, reference
            ),
        }
        results.append(hc_result)

        # Exercise the real Engram.forward boundary without allocating its huge
        # checkpoint embedding table: only the gate's already projected KV varies.
        projection = torch.randn(
            batch, queries, (hc + 1) * dim, dtype=torch.bfloat16, device="cuda"
        )
        module = Engram.__new__(Engram)
        nn.Module.__init__(module)
        module.hc_mult, module.dim, module.eps = hc, dim, 1e-20
        module.q_weight = nn.Parameter(
            torch.randn(hc, dim, dtype=torch.bfloat16, device="cuda")
        )
        module.k_weight = nn.Parameter(
            torch.randn(hc, dim, dtype=torch.bfloat16, device="cuda")
        )
        module.register_buffer("_gate_weight", None, persistent=False)
        module.add_module("wkv", _Projection(projection))
        module.prepare_gate()
        expected_weight = module.q_weight.float() * module.k_weight.float()
        if not torch.equal(module._gate_weight, expected_weight):
            raise AssertionError("prepared gate weights changed checkpoint values")
        embedding = hidden.new_empty(batch, queries, 1, 1)

        def gate():
            return module(hidden, embed_hook=lambda: embedding)

        def gate_reference():
            return _gate_reference(hidden, projection, expected_weight, module.eps)

        gate_result = {
            "operator": "engram_gate",
            "shape": list(hidden.shape),
            **_check(gate(), gate_reference()),
            "native": _graph(gate, lambda _: hidden.mul_(0.9375), gate_reference),
            "previous_torch": _graph(
                lambda: _gate_previous(hidden, projection, expected_weight, module.eps),
                lambda _: None,
                gate_reference,
            ),
        }
        mask = (
            torch.arange(batch * queries, device="cuda")
            .reshape(batch, queries)
            .remainder(2)
            == 0
        )
        masked = module(hidden, token_mask=mask, embed_hook=lambda: embedding)
        torch.testing.assert_close(masked[~mask], hidden[~mask], rtol=0, atol=0)
        _check(masked[mask], gate_reference()[mask])
        gate_result["mask_preserved"] = True
        results.append(gate_result)
    return {"gpu": torch.cuda.get_device_name(), "residuals": results}
