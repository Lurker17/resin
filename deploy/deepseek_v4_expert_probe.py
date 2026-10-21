"""Isolate FP4 expert ownership, intermediate sharding, and kernel numerics.

Use synthetic tensors or a few contiguous real checkpoint experts without
loading the model. Run on Blackwell through the existing DeepSeek Modal image.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _Weights:
    up_gate: torch.Tensor
    up_gate_scales: torch.Tensor
    down: torch.Tensor
    down_scales: torch.Tensor

    def clone(self):
        return _Weights(*(tensor.clone() for tensor in vars(self).values()))

    def experts(self, start, stop):
        return _Weights(
            *(tensor[start:stop].contiguous().clone() for tensor in vars(self).values())
        )

    def intermediate(self, start, stop):
        width = self.up_gate.shape[1] // 2
        rows = torch.cat(
            (
                torch.arange(start, stop, device=self.up_gate.device),
                torch.arange(width + start, width + stop, device=self.up_gate.device),
            )
        )
        return _Weights(
            self.up_gate.view(torch.uint8)
            .index_select(1, rows)
            .view(self.up_gate.dtype),
            self.up_gate_scales.view(torch.uint8)
            .index_select(1, rows)
            .view(self.up_gate_scales.dtype),
            self.down.view(torch.uint8)[..., start // 2 : stop // 2]
            .contiguous()
            .clone()
            .view(self.down.dtype),
            self.down_scales.view(torch.uint8)[..., start // 32 : stop // 32]
            .contiguous()
            .clone()
            .view(self.down_scales.dtype),
        )


def _dequantize(weights, scales):
    packed = weights.view(torch.uint8)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weights.device)
    values = levels[(nibbles & 7).long()] * torch.where(nibbles < 8, 1, -1)
    exponent = scales.view(torch.uint8).int() - 127
    return values * torch.exp2(exponent.float()).repeat_interleave(32, dim=-1)


def _mxfp8(values):
    blocks = values.float().unflatten(-1, (-1, 32))
    maximum = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(maximum / 448)))
    rounded = (blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (rounded * scales).flatten(-2)


def _metrics(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "max_abs": float(difference.abs().max()),
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": float(
            difference.norm() / expected.float().norm().clamp_min(1e-12)
        ),
        "actual_rms": float(actual.float().square().mean().sqrt()),
        "reference_rms": float(expected.float().square().mean().sqrt()),
    }


def _references(hidden, ids, probabilities, raw, clamp):
    up_gate = _dequantize(raw.up_gate, raw.up_gate_scales)
    down = _dequantize(raw.down, raw.down_scales)
    results = {
        name: torch.zeros_like(hidden, dtype=torch.float32)
        for name in ("fp32_up_gate", "fp32_gate_up", "mxfp8_up_gate", "bf16_up_gate")
    }
    for expert in range(raw.up_gate.shape[0]):
        probability = ((ids == expert) * probabilities.float()).sum(1, keepdim=True)
        first, second = F.linear(hidden.float(), up_gate[expert]).chunk(2, -1)
        for name, up, gate in (
            ("fp32_up_gate", first, second),
            ("fp32_gate_up", second, first),
        ):
            activated = F.silu(gate.clamp_max(clamp)) * up.clamp(-clamp, clamp)
            results[name] += F.linear(activated, down[expert]) * probability
            if name == "fp32_up_gate":
                results["mxfp8_up_gate"] += (
                    F.linear(_mxfp8(activated), down[expert]) * probability
                )
        up, gate = F.linear(hidden.bfloat16(), up_gate[expert].bfloat16()).chunk(2, -1)
        activated = (
            F.silu(gate.float().clamp_max(clamp)) * up.float().clamp(-clamp, clamp)
        ).bfloat16()
        results["bf16_up_gate"] += (
            F.linear(activated, down[expert].bfloat16()).float() * probability
        )
    return results


def _checkpoint_weights(checkpoint_path, experts, device):
    from resin.backend.safetensors import SafetensorReader, indexed_tensor_files

    root = Path(checkpoint_path).expanduser().resolve()
    config = json.loads((root / "inference/config.json").read_text())
    if config.get("expert_dtype") != "fp4" or config.get("scale_dtype", "fp8") != "fp8":
        raise ValueError("the expert probe requires FP4 weights and UE8M0 scales")
    tensor_files = indexed_tensor_files(root, require_index=True)
    pattern = re.compile(r"(.+\.experts)\.(\d+)\.w1\.weight$")
    prefixes = {
        match[1]
        for name in tensor_files
        if (match := pattern.fullmatch(name)) is not None
    }
    if not prefixes:
        raise ValueError("checkpoint has no routed expert weights")
    prefix = min(prefixes, key=lambda name: tuple(map(int, re.findall(r"\d+", name))))
    top_k = int(config["n_activated_experts"])
    experts = max(experts, top_k + top_k % 2)
    if experts > int(config["n_routed_experts"]):
        raise ValueError("requested more experts than the checkpoint contains")
    loaded = []
    with SafetensorReader() as reader:
        for expert in range(experts):
            tensors = {}
            for projection in ("w1", "w2", "w3"):
                for kind in ("weight", "scale"):
                    name = f"{prefix}.{expert}.{projection}.{kind}"
                    tensor = reader.read(tensor_files[name], name)
                    dtype = (
                        torch.float4_e2m1fn_x2
                        if kind == "weight"
                        else torch.float8_e8m0fnu
                    )
                    tensors[projection, kind] = tensor.view(dtype).to(device)
            loaded.append(
                _Weights(
                    *(
                        torch.cat(
                            (
                                tensors["w3", kind].view(torch.uint8),
                                tensors["w1", kind].view(torch.uint8),
                            ),
                            dim=0,
                        ).view(tensors["w1", kind].dtype)
                        for kind in ("weight", "scale")
                    ),
                    tensors["w2", "weight"],
                    tensors["w2", "scale"],
                )
            )
    raw = _Weights(
        *(
            torch.stack(
                [getattr(weights, field).view(torch.uint8) for weights in loaded]
            )
            .view(getattr(loaded[0], field).dtype)
            .contiguous()
            for field in _Weights.__dataclass_fields__
        )
    )
    return raw, config, {"path": str(root), "layer": prefix, "experts": [0, experts]}


def _deep_gemm_precision(raw, generator):
    from deep_gemm import fp8_fp4_gemm_nt, m_grouped_fp8_fp4_gemm_nt_contiguous

    from resin_models_cuda.deepseek_v4.cutile import act_quant

    device = raw.down.device
    rows = 128
    width = raw.down.shape[-1] * 2
    columns = raw.down.shape[1]
    values = torch.randn(
        (2 * rows, width), device=device, dtype=torch.bfloat16, generator=generator
    )
    quantized, scales = act_quant(values, 128, "ue8m0", torch.float8_e8m0fnu)
    weights = raw.down[:2].view(torch.int8)
    weight_scales = raw.down_scales[:2].float()
    dequantized_weights = _dequantize(raw.down[:2], raw.down_scales[:2])
    oracle = F.linear(
        quantized[:rows].float() * scales[:rows].repeat_interleave(128, -1),
        dequantized_weights[0],
    )
    report = {"deep_gemm": version("sgl-deep-gemm"), "ordinary": {}, "grouped": {}}

    def ordinary(output, *, start=0, stop=width):
        fp8_fp4_gemm_nt(
            (
                quantized[:rows, start:stop].contiguous(),
                # A row subset retains the full grouped batch's scale stride.
                # Repack MN-major even when only one scale column remains.
                scales[:rows, start // 128 : stop // 128]
                .T.clone(memory_format=torch.contiguous_format)
                .T,
            ),
            (
                weights[0, :, start // 2 : stop // 2].contiguous(),
                weight_scales[0, :, start // 32 : stop // 32].contiguous(),
            ),
            output,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )

    outputs = {}
    for dtype in (torch.bfloat16, torch.float32):
        name = str(dtype).removeprefix("torch.")
        output = torch.empty((rows, columns), dtype=dtype, device=device)
        try:
            ordinary(output)
            outputs[dtype] = output
            report["ordinary"][name] = {
                "supported": True,
                "vs_dequantized_reference": _metrics(output, oracle),
            }
        except RuntimeError as error:
            report["ordinary"][name] = {"supported": False, "error": str(error)}

    if torch.float32 in outputs:
        full = outputs[torch.float32]
        report["ordinary"]["float32"]["non_bf16_rms"] = float(
            (full - full.bfloat16().float()).square().mean().sqrt()
        )
        for dtype in (torch.bfloat16, torch.float32):
            partials = [
                torch.empty((rows, columns), dtype=dtype, device=device)
                for _ in range(2)
            ]
            ordinary(partials[0], stop=width // 2)
            ordinary(partials[1], start=width // 2)
            report["ordinary"][f"split_k_{dtype}"] = _metrics(
                partials[0].float() + partials[1].float(), full
            )

        # Reuse prewarmed descriptors/configurations; replay must read updated
        # input storage and preserve FP32 output rather than upcasting BF16.
        graph = torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph):
                ordinary(full)
            quantized.copy_((quantized.float() * 0.5).to(quantized.dtype))
            graph.replay()
            replay_oracle = F.linear(
                quantized[:rows].float() * scales[:rows].repeat_interleave(128, -1),
                dequantized_weights[0],
            )
            report["ordinary"]["graph_replay"] = {
                "supported": True,
                "vs_dequantized_reference": _metrics(full, replay_oracle),
            }
        except RuntimeError as error:
            report["ordinary"]["graph_replay"] = {
                "supported": False,
                "error": str(error),
            }
        finally:
            torch.cuda.synchronize(device)
            graph.reset()

    layout = torch.arange(2, device=device, dtype=torch.int32).repeat_interleave(rows)
    grouped_oracle = torch.cat(
        [
            F.linear(
                quantized[index * rows : (index + 1) * rows].float()
                * scales[index * rows : (index + 1) * rows].repeat_interleave(128, -1),
                dequantized_weights[index],
            )
            for index in range(2)
        ]
    )
    for dtype in (torch.bfloat16, torch.float32):
        name = str(dtype).removeprefix("torch.")
        output = torch.empty((2 * rows, columns), dtype=dtype, device=device)
        try:
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (quantized, scales),
                (weights, weight_scales),
                output,
                layout,
                recipe_a=(1, 128),
                recipe_b=(1, 32),
            )
            report["grouped"][name] = {
                "supported": True,
                "vs_dequantized_reference": _metrics(output, grouped_oracle),
            }
        except RuntimeError as error:
            report["grouped"][name] = {"supported": False, "error": str(error)}
    print(f"[expert-probe] DeepGEMM precision {report}", flush=True)
    return report


@torch.inference_mode()
def run_probe(
    *,
    device="cuda:0",
    hidden_size=512,
    intermediate_size=256,
    experts=4,
    top_k=2,
    row_counts=(1, 8, 32, 128),
    clamp=7.0,
    checkpoint_path=None,
    probe_deep_gemm=False,
):
    """Report full-vs-EP/ETP errors and independent dequantized references.

    EP/ETP comparisons use identical raw tensors and input quantization. All
    layout preparation happens after slicing, exactly as checkpoint loading.
    Independent dense references are diagnostic; they intentionally expose
    activation rounding/layout assumptions rather than imposing a new model
    tolerance. ``checks_passed`` uses fixed composition limits of max_abs .25
    and relative L2 .02, and does not modify the engine validation tolerances.
    """

    from resin_models_cuda.deepseek_v4.cutile import moe_act_quant
    from resin_models_cuda.deepseek_v4.gemm import (
        fp4_fused_moe,
        prepare_fp4_fused_moe_weights,
    )

    device = torch.device(device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("the FP4 expert probe requires Blackwell")
    source = {"kind": "synthetic"}
    route_scale = 1.0
    if checkpoint_path is not None:
        raw, config, source = _checkpoint_weights(checkpoint_path, experts, device)
        source["kind"] = "checkpoint"
        hidden_size = int(config["dim"])
        intermediate_size = int(config["moe_inter_dim"])
        experts = raw.up_gate.shape[0]
        top_k = int(config["n_activated_experts"])
        clamp = float(config["swiglu_limit"])
        route_scale = float(config.get("route_scale", 1.0))
    if hidden_size % 512 or intermediate_size % 256 or experts % 2:
        raise ValueError("use H512-aligned, I256-aligned, and an even expert count")
    if not 0 < top_k <= experts:
        raise ValueError("top_k must be positive and no larger than expert count")
    generator = torch.Generator(device=device).manual_seed(1729)

    def packed(shape):
        return torch.randint(
            0, 256, shape, device=device, dtype=torch.uint8, generator=generator
        ).view(torch.float4_e2m1fn_x2)

    def scales(shape):
        return torch.full(shape, 122, device=device, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

    if checkpoint_path is None:
        raw = _Weights(
            packed((experts, 2 * intermediate_size, hidden_size // 2)),
            scales((experts, 2 * intermediate_size, hidden_size // 32)),
            packed((experts, hidden_size, intermediate_size // 2)),
            scales((experts, hidden_size, intermediate_size // 32)),
        )
    precision = _deep_gemm_precision(raw, generator) if probe_deep_gemm else None
    prepared = [
        raw.clone(),
        raw.experts(0, experts // 2),
        raw.experts(experts // 2, experts),
        raw.intermediate(0, intermediate_size // 2),
        raw.intermediate(intermediate_size // 2, intermediate_size),
    ]
    for weights in prepared:
        prepare_fp4_fused_moe_weights(*vars(weights).values())

    def run(hidden, ids, probabilities, weights, *, offset=0):
        quantized, sf = moe_act_quant(hidden, 128, "ue8m0", torch.float8_e8m0fnu)
        count = weights.up_gate.shape[0]
        return fp4_fused_moe(
            quantized,
            sf,
            ids,
            probabilities,
            *vars(weights).values(),
            torch.ones(count, device=device, dtype=torch.float32),
            torch.full((count,), clamp, device=device, dtype=torch.float32),
            num_experts=experts,
            local_expert_offset=offset,
            scale_dtype=torch.float8_e8m0fnu,
        )

    records = []
    for rows in row_counts:
        hidden = torch.randn(
            (rows, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ) / (4 if checkpoint_path is None else 1)
        ids = (
            (
                torch.arange(rows, device=device)[:, None]
                + torch.arange(top_k, device=device)[None, :]
            )
            % experts
        ).int()
        routing = (
            torch.tensor([0.375, 0.625], device=device)
            if top_k == 2
            else torch.arange(1, top_k + 1, device=device).float()
        )
        probabilities = (
            (routing / routing.sum() * route_scale)
            .bfloat16()
            .expand(rows, top_k)
            .contiguous()
        )
        full = run(hidden, ids, probabilities, prepared[0])
        ep = (
            run(hidden, ids, probabilities, prepared[1]).float()
            + run(hidden, ids, probabilities, prepared[2], offset=experts // 2).float()
        ).bfloat16()
        etp = (
            run(hidden, ids, probabilities, prepared[3]).float()
            + run(hidden, ids, probabilities, prepared[4]).float()
        ).bfloat16()
        duplicated = run(
            hidden.repeat(2, 1),
            ids.repeat(2, 1),
            probabilities.repeat(2, 1),
            prepared[0],
        )[:rows]
        padded_rows = max(128, 2 * rows)
        padded_hidden = F.pad(hidden, (0, 0, 0, padded_rows - rows))
        padded_ids = F.pad(ids, (0, 0, 0, padded_rows - rows))
        padded_probabilities = F.pad(probabilities, (0, 0, 0, padded_rows - rows))
        padded_full = run(padded_hidden, padded_ids, padded_probabilities, prepared[0])[
            :rows
        ]
        padded_etp = (
            run(padded_hidden, padded_ids, padded_probabilities, prepared[3]).float()
            + run(padded_hidden, padded_ids, padded_probabilities, prepared[4]).float()
        )[:rows].bfloat16()
        quantized, sf = moe_act_quant(hidden, 128, "ue8m0", torch.float8_e8m0fnu)
        quantized_hidden = quantized.float() * torch.exp2(
            sf.view(torch.uint8).int().float() - 127
        ).repeat_interleave(32, -1)
        references = _references(quantized_hidden, ids, probabilities, raw, clamp)
        checks = {
            "ep2_vs_full": _metrics(ep, full),
            "etp2_vs_full": _metrics(etp, full),
            "duplicated_batch_vs_full": _metrics(duplicated, full),
            "padded_batch_vs_full": _metrics(padded_full, full),
            "padded_etp2_vs_full": _metrics(padded_etp, full),
        }
        records.append(
            {
                "rows": rows,
                "composition": checks,
                "full_vs_reference": {
                    name: _metrics(full, value) for name, value in references.items()
                },
                "etp2_vs_reference": {
                    name: _metrics(etp, value) for name, value in references.items()
                },
                "checks_passed": all(
                    value["max_abs"] <= 0.25 and value["relative_l2"] <= 0.02
                    for value in checks.values()
                ),
            }
        )
        print(f"[expert-probe] rows={rows} {records[-1]}", flush=True)
    torch.cuda.synchronize(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "flashinfer": version("flashinfer-python"),
        "shape": {
            "experts": experts,
            "hidden": hidden_size,
            "intermediate": intermediate_size,
            "top_k": top_k,
        },
        "clamp": clamp,
        "route_scale": route_scale,
        "source": source,
        "deep_gemm_precision": precision,
        "records": records,
        "checks_passed": all(record["checks_passed"] for record in records),
    }
