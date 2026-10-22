"""Small, independent numerical checks for pinned DeepGEMM mega kernels."""

from __future__ import annotations

import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.cuda.nccl
import torch.distributed as dist
from torch.nn import functional as F


def _command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def _compare(actual, expected, *, atol=2e-5, rtol=2e-4):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-20)).item(),
        "atol": atol,
        "rtol": rtol,
    }


def _capture(call, outputs):
    # DeepGEMM's persistent barriers belong to a CUDA stream. Initialize on the
    # actual capture stream before capture, rather than relying on default warmup.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
        expected = {name: value.clone() for name, value in outputs.items()}
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    for name, value in outputs.items():
        if not torch.equal(value, expected[name]):
            raise AssertionError(
                f"captured {name} differs from eager on the same inputs"
            )
    return {"replays": 5, "bitwise_equal": True}


def _gate(deep_gemm, rows, hidden=5120, experts=384, topk=6):
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    )
    bias = torch.randn(experts, device="cuda")
    mask = torch.arange(rows, device="cuda") % 7 != 0
    mask[0] = True
    outputs = {
        "indices": torch.empty(rows, topk, device="cuda", dtype=torch.int64),
        "weights": torch.empty(rows, topk, device="cuda"),
    }

    def call():
        deep_gemm.bf16_mega_gate(
            x,
            weight,
            topk,
            False,
            0,
            1.5,
            0,
            bias=bias,
            mask=mask,
            out=(outputs["indices"], outputs["weights"]),
        )

    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    scores = F.softplus(x.float() @ weight.float().T).sqrt()
    selected = outputs["indices"][mask]
    reference = scores[mask].gather(1, selected)
    reference = reference / (reference.sum(-1, keepdim=True) + 1e-20) * 1.5
    weights = _compare(outputs["weights"][mask], reference)
    ranking = scores[mask] + bias
    boundary = ranking.gather(1, selected).min(-1).values
    crossing = ranking.scatter(1, selected, -torch.inf).max(-1).values - boundary
    if not (crossing <= 2e-5).all():
        raise AssertionError(
            f"gate selected outside numerical top-k ties: {crossing.max().item()}"
        )
    if (
        not (outputs["indices"][~mask] == -1).all()
        or not (outputs["weights"][~mask] == 0).all()
    ):
        raise AssertionError("gate emitted a route for a masked token")
    return {
        "rows": rows,
        "hidden": hidden,
        "experts": experts,
        "topk": topk,
        "compile_seconds": compile_seconds,
        "weights": weights,
        "topk_max_crossing": crossing.max().item(),
        "capture": _capture(call, outputs),
    }


def _mhc_reference(inputs):
    x, residual = inputs["x"], inputs["residual"]
    updated = (
        x.float()[:, None] * inputs["post_mix"]
        + torch.einsum("bij,bih->bjh", inputs["comb_res_mix"], residual.float())
    ).bfloat16()
    flat = updated.flatten(1).float()
    mixes = (flat @ inputs["fn"].T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + inputs["hc_norm_eps"]
    )
    scales = inputs["mix_scales"].repeat_interleave(
        torch.tensor([4, 4, 16], device=x.device)
    )
    mixes = mixes * scales + inputs["mix_bases"]
    pre = mixes[:, :4].sigmoid()[..., None] + inputs["hc_pre_eps"]
    post = mixes[:, 4:8].sigmoid()[..., None] * inputs["hc_post_scale"]
    comb = mixes[:, 8:].reshape(-1, 4, 4).softmax(-1) + inputs["sinkhorn_eps"]
    comb /= comb.sum(-2, keepdim=True) + inputs["sinkhorn_eps"]
    for _ in range(inputs["num_sinkhorn_iters"] - 1):
        comb /= comb.sum(-1, keepdim=True) + inputs["sinkhorn_eps"]
        comb /= comb.sum(-2, keepdim=True) + inputs["sinkhorn_eps"]
    previous = inputs["shifted_prev_mix"]
    collapsed = (
        (updated.float() * (pre if previous is None else previous))
        .sum(1)
        .bfloat16()
        .float()
    )
    normalized = (
        collapsed
        * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + inputs["rmsnorm_eps"])
        * (inputs["rmsnorm_weight"].float() * inputs["rmsnorm_scale"])
    ).bfloat16()
    outputs = dict(
        new_residual=updated,
        new_post_mix=post,
        new_comb_res_mix=comb,
        y_bf16=normalized,
    )
    if previous is not None:
        outputs["new_prev_mix"] = pre
    return outputs


def _mhc(deep_gemm, rows, shifted, hidden=5120):
    def randn(*shape):
        return torch.randn(*shape, device="cuda")

    inputs = dict(
        x=randn(rows, hidden).bfloat16(),
        residual=randn(rows, 4, hidden).bfloat16(),
        shifted_prev_mix=randn(rows, 4, 1).sigmoid() if shifted else None,
        post_mix=randn(rows, 4, 1).sigmoid(),
        comb_res_mix=randn(rows, 4, 4).softmax(-1),
        fn=randn(24, 4 * hidden) * 0.01,
        mix_scales=randn(3) * 0.1,
        mix_bases=randn(24) * 0.1,
        rmsnorm_weight=(randn(hidden) * 0.1 + 1).bfloat16(),
        hc_mult=4,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=10,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
    )
    reference = _mhc_reference(inputs)
    outputs = {name: torch.empty_like(value) for name, value in reference.items()}

    def call():
        deep_gemm.mega_mhc(
            **inputs, **outputs, **({} if shifted else {"new_prev_mix": None})
        )

    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    # BF16 rounding may move individual values by one ULP. Retain the upstream
    # normalized L2 bound as well as reporting absolute error for every output.
    comparisons = {}
    for name, expected in reference.items():
        actual = outputs[name]
        delta = actual.float() - expected.float()
        diff = 1 - 2 * (actual.float() * expected.float()).sum() / (
            actual.float().square().sum() + expected.float().square().sum()
        ).clamp_min(1e-20)
        if diff.abs() >= 5e-5:
            raise AssertionError(f"mHC {name} differs from reference: {diff.item()}")
        comparisons[name] = {
            "max_abs": delta.abs().max().item(),
            "normalized_squared_error": diff.item(),
            "limit": 5e-5,
        }
    return {
        "rows": rows,
        "hidden": hidden,
        "shifted": shifted,
        "compile_seconds": compile_seconds,
        "outputs": comparisons,
        "capture": _capture(call, outputs),
    }


@torch.inference_mode()
def run_probe():
    import deep_gemm

    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("these pinned mega kernels require a Blackwell SM10x GPU")
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # The development environment has Resin's older SGL package; this probe
    # intentionally loads and checks the separately pinned upstream API.
    getattr(deep_gemm, "use_deterministic_algorithms")(True)
    for name in ("bf16_mega_gate", "mega_mhc", "SymmBuffer", "fp8_fp4_mega_moe"):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError(f"pinned DeepGEMM is missing {name}")
    nccl_libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libnccl.so" in line
        }
    )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "deep_gemm": {"version": deep_gemm.__version__, "path": deep_gemm.__file__},
        "hardware": {
            "nvcc": _command("nvcc", "--version"),
            "compiler": _command("g++-13", "--version"),
            "topology": _command("nvidia-smi", "topo", "-m"),
            "nvlink": _command("nvidia-smi", "nvlink", "-s"),
            "ibstat": _command("ibstat"),
            "infiniband_devices": [
                str(path) for path in Path("/sys/class/infiniband").glob("*")
            ],
            "infiniband_device_nodes": [
                str(path) for path in Path("/dev/infiniband").glob("*")
            ],
            "nccl_version": torch.cuda.nccl.version(),
            "nccl_libraries": nccl_libraries,
            "gdaki_runtime_test": "not run: requires a pinned DeepEP communicator probe",
        },
        "mega_gate": [_gate(deep_gemm, rows) for rows in (1, 65)],
        "mega_mhc": [
            _mhc(deep_gemm, rows, shifted)
            for shifted in (False, True)
            for rows in (1, 65)
        ],
    }


# Quantization and symmetric-buffer layout follow DeepGEMM's MIT-licensed
# tests/test_mega_moe.py at 1fc9c66975ee322d628b3288534ca0e9ce12a64d.
# The reference FFNs below use ordinary Torch GEMMs and decoded quantized data.
def quantize_fp8(value):
    from deep_gemm.utils import cast_back_from_fp8, per_token_cast_to_fp8

    q, scales = per_token_cast_to_fp8(
        value, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    restored = cast_back_from_fp8(q, scales, gran_k=32, use_packed_ue8m0=True)
    return q, scales, restored


def quantize_fp4_raw(source):
    from deep_gemm.utils import cast_back_from_fp4, per_token_cast_to_fp4

    groups, n, k = source.shape
    q, scales = per_token_cast_to_fp4(source.flatten(0, 1), use_ue8m0=True, gran_k=32)
    restored = cast_back_from_fp4(q, scales, gran_k=32).reshape(source.shape)
    q = q.reshape(groups, n, k // 2)
    return q, scales.reshape(groups, n, k // 32), restored


def quantize_fp4_weights(deep_gemm, source):
    q, scales, restored = quantize_fp4_raw(source)
    groups, n, k = source.shape
    scales = deep_gemm.transform_sf_into_required_layout(scales, n, k, (1, 32), groups)
    return (q, scales), restored


def tma_scale(scale):
    result = torch.empty_strided(
        scale.shape,
        (1, (scale.shape[0] + 3) // 4 * 4),
        dtype=scale.dtype,
        device=scale.device,
    )
    return result.copy_(scale)


def reference_ffn(value, w1, w2, route_weight=None):
    projected = (value @ w1.T).bfloat16().float()
    gate, up = projected.chunk(2, -1)
    activated = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    if route_weight is not None:
        activated *= route_weight[:, None]
    _, _, quantized = quantize_fp8(activated)
    return (quantized @ w2.T).bfloat16().float()


def _mega_moe_rank(rank, rendezvous, directory, tokens_per_rank):
    import importlib

    deep_gemm: Any = importlib.import_module("deep_gemm")

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        world_size=2,
        rank=rank,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(minutes=5),
    )
    torch.manual_seed(4141 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    getattr(deep_gemm, "use_deterministic_algorithms")(True)
    rows, hidden, intermediate, experts, topk = 8, 5120, 2304, 16, 6
    buffer = deep_gemm.SymmBuffer(
        dist.group.WORLD,
        experts,
        16,
        topk,
        hidden,
        intermediate,
        num_shared_experts=1,
        mma_type="fp8xfp4",
    )

    def fp4_weights(shape):
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        return quantize_fp4_weights(deep_gemm, source)

    first, ref_first = fp4_weights((experts // 2, 2 * intermediate, hidden))
    second, ref_second = fp4_weights((experts // 2, hidden, intermediate))
    first, second = deep_gemm.transform_weights_for_mega_moe(first, second)
    shared_first = quantize_fp8(
        torch.randn(2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    shared_second = quantize_fp8(
        torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    )

    shared = deep_gemm.transform_weights_for_mega_moe(
        (shared_first[0], tma_scale(shared_first[1])),
        (shared_second[0], tma_scale(shared_second[1])),
    )
    original = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    x, x_scale, ref_x = quantize_fp8(original)
    indices = torch.randn(rows, experts, device="cuda").topk(topk, dim=-1).indices
    weights = torch.randn(rows, topk, device="cuda").softmax(-1)
    results = []

    try:
        for name in ("regular", "changed_routing", "idle_rank"):
            count = 0 if name == "idle_rank" and rank == 1 else rows
            selected = indices.clone()
            if name != "regular":
                selected = (selected + 3) % experts
                selected[0, 0] = -1
            route_weights = weights.masked_fill(selected < 0, 0)
            global_x = [torch.empty_like(ref_x) for _ in range(2)]
            global_idx = [torch.empty_like(selected) for _ in range(2)]
            global_w = [torch.empty_like(route_weights) for _ in range(2)]
            if not count:
                selected.fill_(-1)
                route_weights.zero_()
            dist.all_gather(global_x, ref_x)
            dist.all_gather(global_idx, selected)
            dist.all_gather(global_w, route_weights)
            all_x, all_indices, all_weights = map(
                torch.cat, (global_x, global_idx, global_w)
            )
            expected = torch.zeros(2 * rows, hidden, device="cuda")
            for expert in range(experts // 2):
                token, slot = (all_indices == rank * (experts // 2) + expert).nonzero(
                    as_tuple=True
                )
                if token.numel():
                    value = reference_ffn(
                        all_x[token],
                        ref_first[expert],
                        ref_second[expert],
                        all_weights[token, slot],
                    )
                    expected.index_add_(0, token, value)
            dist.all_reduce(expected)
            expected = expected[rank * rows : rank * rows + count]
            if count:
                expected += reference_ffn(
                    ref_x[:count], shared_first[2], shared_second[2]
                )
            expected = expected.bfloat16()
            shared_sf = torch.zeros_like(buffer.shared_l1_acts_sf)
            block = deep_gemm.get_block_m_for_mega_moe(
                2, experts, buffer.num_max_tokens_per_rank, count, topk, "fp8xfp4"
            )
            for token in range(count):
                offset = token % block
                row = (
                    token // block * ((block + 127) // 128 * 128)
                    + offset // 128 * 128
                    + offset % 32 * 4
                    + offset % 128 // 32
                )
                shared_sf[row].copy_(x_scale[token])
            output = torch.empty(count, hidden, device="cuda", dtype=torch.bfloat16)
            stats = torch.empty(experts // 2, device="cuda", dtype=torch.int32)

            def call():
                buffer.x[:count].copy_(x[:count])
                buffer.x_sf[:count].copy_(x_scale[:count])
                buffer.shared_l1_acts_sf.copy_(shared_sf)
                buffer.topk_idx[:count].copy_(selected[:count])
                buffer.topk_weights[:count].copy_(route_weights[:count])
                stats.zero_()
                deep_gemm.fp8_fp4_mega_moe(
                    y=output,
                    l1_weights=first,
                    l2_weights=second,
                    sym_buffer=buffer,
                    shared_l1_weights=shared[0],
                    shared_l2_weights=shared[1],
                    cumulative_local_expert_recv_stats=stats,
                    activation_clamp=10,
                    fast_math=False,
                )

            call()
            torch.cuda.synchronize()
            if count:
                error = (
                    output.float() - expected.float()
                ).norm() / expected.float().norm().clamp_min(1e-20)
                if error >= 0.02:
                    raise AssertionError(
                        f"MegaMoE {name} quantized reference relative L2={error.item()}"
                    )
                comparison = {
                    "relative_l2": error.item(),
                    "limit": 0.02,
                    "max_abs": (output.float() - expected.float()).abs().max().item(),
                }
            else:
                comparison = {"empty_output": list(output.shape)}
            received = all_indices.flatten() - rank * (experts // 2)
            received = received[(received >= 0) & (received < experts // 2)]
            torch.testing.assert_close(
                stats.long(), received.bincount(minlength=experts // 2)
            )
            results.append(
                {
                    "case": name,
                    "tokens": count,
                    "reference": comparison,
                    "capture": _capture(call, {"output": output, "received": stats}),
                }
            )
        dist.barrier()
        buffer.destroy()
        from deploy.mega_moe_validation import run_probe as adapter_probe

        adapter = adapter_probe(dist.group.WORLD, rows=tokens_per_rank)
        import json

        Path(directory, f"rank{rank}.json").write_text(
            json.dumps({"rank": rank, "cases": results, "adapter": adapter})
        )
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
        dist.destroy_process_group()


def run_mega_moe(tokens_per_rank: int = 8):
    import json
    import socket

    if torch.cuda.device_count() != 2:
        raise ValueError("the MegaMoE probe requires exactly two visible NVLink GPUs")
    if tokens_per_rank < 1:
        raise ValueError("tokens_per_rank must be positive")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        rendezvous = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    with tempfile.TemporaryDirectory(prefix="mega-moe-") as directory:
        torch.multiprocessing.spawn(
            _mega_moe_rank,
            args=(rendezvous, directory, tokens_per_rank),
            nprocs=2,
            join=True,
        )
        return {
            "world_size": 2,
            "tokens_per_rank": tokens_per_rank,
            "hidden": 5120,
            "intermediate": 2304,
            "experts": 16,
            "topk": 6,
            "shared_experts": 1,
            "ranks": [
                json.loads(Path(directory, f"rank{rank}.json").read_text())
                for rank in range(2)
            ],
        }


def _memory_rank(rank, rendezvous, directory, packed_attention):
    import json

    from deploy.deepseek_v41_validation import run_probes
    from deploy.engram_validation import run_probe as engram_probe

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        world_size=2,
        rank=rank,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(minutes=5),
    )
    try:
        result = {"rank": rank}
        if packed_attention:
            result["packed_attention"] = run_probes()
            print(
                json.dumps(
                    {"rank": rank, "packed_attention": result["packed_attention"]}
                ),
                flush=True,
            )
        result["engram"] = engram_probe(dist.group.WORLD)
        Path(directory, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


def run_memory(*, packed_attention=True):
    import json
    import socket

    if torch.cuda.device_count() != 2:
        raise ValueError("the memory probe requires two visible GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        rendezvous = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    with tempfile.TemporaryDirectory(prefix="packed-memory-") as directory:
        torch.multiprocessing.spawn(
            _memory_rank,
            args=(rendezvous, directory, packed_attention),
            nprocs=2,
            join=True,
        )
        return {
            "world_size": 2,
            "ranks": [
                json.loads(Path(directory, f"rank{rank}.json").read_text())
                for rank in range(2)
            ],
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "mode",
        choices=("kernels", "mega-moe", "memory", "engram"),
        nargs="?",
        default="kernels",
    )
    parser.add_argument("--tokens-per-rank", type=int, default=8)
    args = parser.parse_args()
    if args.tokens_per_rank < 1:
        parser.error("--tokens-per-rank must be positive")
    if args.mode in ("memory", "engram"):
        result = run_memory(packed_attention=args.mode == "memory")
    elif args.mode == "mega-moe":
        result = run_mega_moe(args.tokens_per_rank)
    else:
        result = run_probe()
    args.output.write_text(json.dumps(result, indent=2))
