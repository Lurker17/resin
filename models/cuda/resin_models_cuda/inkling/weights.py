"""Stream Inkling target and DSpark tensors from their original checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.inkling.dspark import DSparkDraft
from resin_models_cuda.inkling.model import (
    DenseMlp,
    InklingForCausalLM,
    Nvfp4Experts,
    SparseMlp,
)


def _copy(torch: Any, destination: Any, source: Any, name: str) -> int:
    if source.numel() == destination.numel() and source.shape != destination.shape:
        source = source.reshape(destination.shape)
    if tuple(source.shape) != tuple(destination.shape):
        raise ValueError(
            f"Inkling tensor {name!r} has shape {tuple(source.shape)}, expected "
            f"{tuple(destination.shape)}"
        )
    if source.dtype != destination.dtype:
        source = source.to(destination.dtype)
    with torch.no_grad():
        destination.copy_(source.to(destination.device))
    return source.numel() * source.element_size()


def _target_names(model: InklingForCausalLM) -> set[str]:
    names = {
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
        "model.visual.final_norm.weight",
    }
    for index, layer in enumerate(model.vision.layers):
        names.add(f"model.visual.layers.linear_{index}.weight")
        if layer.norm is not None:
            names.add(f"model.visual.layers.norm_{index}.weight")
    for index, layer in enumerate(model.target.layers):
        prefix = f"model.llm.layers.{index}"
        names.update(
            {
                f"{prefix}.attn_norm.weight",
                f"{prefix}.mlp_norm.weight",
                f"{prefix}.attn.wq_du.weight",
                f"{prefix}.attn.wk_dv.weight",
                f"{prefix}.attn.wv_dv.weight",
                f"{prefix}.attn.wr_du.weight",
                f"{prefix}.attn.wo_ud.weight",
                f"{prefix}.attn.q_norm.weight",
                f"{prefix}.attn.k_norm.weight",
                f"{prefix}.attn.rel_logits_proj.proj",
                f"{prefix}.attn.k_sconv.weight",
                f"{prefix}.attn.v_sconv.weight",
                f"{prefix}.attn_sconv.weight",
                f"{prefix}.mlp_sconv.weight",
            }
        )
        if isinstance(layer.mlp, DenseMlp):
            names.update(
                {
                    f"{prefix}.mlp.w13_dn.weight",
                    f"{prefix}.mlp.w2_md.weight",
                    f"{prefix}.mlp.global_scale",
                }
            )
            continue
        names.update(
            {
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.gate.bias",
                f"{prefix}.mlp.gate.global_scale",
                f"{prefix}.mlp.shared_experts.shared_w13_weight",
                f"{prefix}.mlp.shared_experts.shared_w2_weight",
                f"{prefix}.mlp.experts.w13_weight",
                f"{prefix}.mlp.experts.w2_weight",
            }
        )
        if isinstance(layer.mlp.routed, Nvfp4Experts):
            for projection in ("w13", "w2"):
                for suffix in ("input_amax", "scale", "scale2"):
                    names.add(f"{prefix}.mlp.experts.{projection}_weight.{suffix}")
    return names


def _target_destination(model: InklingForCausalLM, name: str):
    target = model.target
    direct = {
        "model.llm.embed.weight": target.embed.weight,
        "model.llm.embed_norm.weight": target.embed_norm.weight,
        "model.llm.norm.weight": target.norm.weight,
        "model.llm.unembed.weight": target.unembed.weight,
    }
    if name in direct:
        return direct[name]
    if name == "model.visual.final_norm.weight":
        return model.vision.final_norm.weight
    if name.startswith("model.visual.layers."):
        suffix = name.removeprefix("model.visual.layers.")
        kind, separator, tail = suffix.partition("_")
        index_text, dot, parameter = tail.partition(".")
        if separator != "_" or dot != "." or parameter != "weight":
            raise ValueError(f"unrecognized Inkling vision tensor: {name!r}")
        try:
            layer = model.vision.layers[int(index_text)]
        except (ValueError, IndexError) as error:
            raise ValueError(f"unrecognized Inkling vision tensor: {name!r}") from error
        if kind == "linear":
            return layer.projection.weight
        if kind == "norm" and layer.norm is not None:
            return layer.norm.weight
        raise ValueError(f"unrecognized Inkling vision tensor: {name!r}")
    parts = name.split(".")
    if len(parts) < 6 or parts[:3] != ["model", "llm", "layers"]:
        raise ValueError(f"unrecognized Inkling target tensor: {name!r}")
    layer = target.layers[int(parts[3])]
    suffix = ".".join(parts[4:])
    common = {
        "attn_norm.weight": layer.attn_norm.weight,
        "mlp_norm.weight": layer.mlp_norm.weight,
        "attn.wq_du.weight": layer.attn.wq.weight,
        "attn.wk_dv.weight": layer.attn.wk.weight,
        "attn.wv_dv.weight": layer.attn.wv.weight,
        "attn.wr_du.weight": layer.attn.wr.weight,
        "attn.wo_ud.weight": layer.attn.wo.weight,
        "attn.q_norm.weight": layer.attn.q_norm.weight,
        "attn.k_norm.weight": layer.attn.k_norm.weight,
        "attn.rel_logits_proj.proj": layer.attn.rel_proj,
        "attn.k_sconv.weight": layer.attn.k_sconv.weight,
        "attn.v_sconv.weight": layer.attn.v_sconv.weight,
        "attn_sconv.weight": layer.attn_sconv.weight,
        "mlp_sconv.weight": layer.mlp_sconv.weight,
    }
    if suffix in common:
        return common[suffix]
    if isinstance(layer.mlp, DenseMlp):
        dense = {
            "mlp.w13_dn.weight": layer.mlp.w13,
            "mlp.w2_md.weight": layer.mlp.w2,
            "mlp.global_scale": layer.mlp.global_scale,
        }
        try:
            return dense[suffix]
        except KeyError as error:
            raise ValueError(f"unrecognized Inkling dense tensor: {name!r}") from error
    if not isinstance(layer.mlp, SparseMlp):
        raise TypeError("unexpected Inkling MLP implementation")
    sparse = {
        "mlp.gate.weight": layer.mlp.router.weight,
        "mlp.gate.bias": layer.mlp.router.bias,
        "mlp.gate.global_scale": layer.mlp.router.global_scale,
        "mlp.shared_experts.shared_w13_weight": layer.mlp.shared.w13,
        "mlp.shared_experts.shared_w2_weight": layer.mlp.shared.w2,
        "mlp.experts.w13_weight": layer.mlp.routed.w13,
        "mlp.experts.w2_weight": layer.mlp.routed.w2,
    }
    if isinstance(layer.mlp.routed, Nvfp4Experts):
        sparse.update(
            {
                "mlp.experts.w13_weight.input_amax": (layer.mlp.routed.w13_input_amax),
                "mlp.experts.w13_weight.scale": layer.mlp.routed.w13_scales,
                "mlp.experts.w13_weight.scale2": layer.mlp.routed.w13_scale2,
                "mlp.experts.w2_weight.input_amax": layer.mlp.routed.w2_input_amax,
                "mlp.experts.w2_weight.scale": layer.mlp.routed.w2_scales,
                "mlp.experts.w2_weight.scale2": layer.mlp.routed.w2_scale2,
            }
        )
    try:
        return sparse[suffix]
    except KeyError as error:
        raise ValueError(f"unrecognized Inkling sparse tensor: {name!r}") from error


def _indexed_target_files(root: Path, expected: set[str]) -> dict[str, Path]:
    """Resolve text-target tensors without requiring the unused MTP shard."""

    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid Inkling checkpoint index: {index_path}") from error
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid Inkling checkpoint weight map: {index_path}")
    if missing := expected - set(weight_map):
        raise ValueError(f"Inkling target is missing tensors: {sorted(missing)[:8]}")

    files: dict[str, Path] = {}
    for name in expected:
        filename = weight_map[name]
        if not isinstance(filename, str):
            raise ValueError(f"invalid Inkling shard name for {name!r}")
        relative = Path(filename)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"invalid Inkling shard name: {filename!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Inkling checkpoint shard: {path}")
        files[name] = path
    return files


def load_target_weights(
    model_path: str | Path, model: InklingForCausalLM
) -> dict[str, int]:
    """Load the text target and image tower, excluding bundled MTP and audio."""

    import torch

    root = Path(model_path).expanduser().resolve()
    expected = _target_names(model)
    files = _indexed_target_files(root, expected)
    loaded_bytes = 0
    with SafetensorReader() as reader:
        ordered = sorted(expected, key=lambda key: (files[key].name, key))
        for name in ordered:
            source = reader.read(files[name], name)
            loaded_bytes += _copy(torch, _target_destination(model, name), source, name)
    return {"tensors": len(expected), "bytes": loaded_bytes}


def _draft_names(model: DSparkDraft) -> set[str]:
    names = {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
    for index in range(len(model.layers)):
        prefix = f"layers.{index}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return names


def _draft_destination(model: DSparkDraft, name: str):
    direct = {
        "fc.weight": model.fc.weight,
        "hidden_norm.weight": model.hidden_norm.weight,
        "norm.weight": model.norm.weight,
        "markov_head.markov_w1.weight": model.markov_w1.weight,
        "markov_head.markov_w2.weight": model.markov_w2.weight,
        "confidence_head.proj.weight": model.confidence.weight,
        "confidence_head.proj.bias": model.confidence.bias,
    }
    if name in direct:
        return direct[name]
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "layers":
        raise ValueError(f"unrecognized Inkling DSpark tensor: {name!r}")
    layer = model.layers[int(parts[1])]
    suffix = ".".join(parts[2:])
    layer_tensors = {
        "input_layernorm.weight": layer.input_norm.weight,
        "post_attention_layernorm.weight": layer.post_norm.weight,
        "self_attn.q_proj.weight": layer.attn.q_proj.weight,
        "self_attn.k_proj.weight": layer.attn.k_proj.weight,
        "self_attn.v_proj.weight": layer.attn.v_proj.weight,
        "self_attn.o_proj.weight": layer.attn.o_proj.weight,
        "self_attn.q_norm.weight": layer.attn.q_norm.weight,
        "self_attn.k_norm.weight": layer.attn.k_norm.weight,
        "mlp.gate_proj.weight": layer.mlp.gate.weight,
        "mlp.up_proj.weight": layer.mlp.up.weight,
        "mlp.down_proj.weight": layer.mlp.down.weight,
    }
    try:
        return layer_tensors[suffix]
    except KeyError as error:
        raise ValueError(f"unrecognized Inkling DSpark tensor: {name!r}") from error


def load_draft_weights(model_path: str | Path, model: DSparkDraft) -> dict[str, int]:
    """Load the external BF16 DSpark checkpoint without duplicated target weights."""

    import torch

    root = Path(model_path).expanduser().resolve()
    files = indexed_tensor_files(root)
    expected = _draft_names(model)
    if missing := expected - set(files):
        raise ValueError(f"Inkling DSpark is missing tensors: {sorted(missing)[:8]}")
    loaded_bytes = 0
    with SafetensorReader() as reader:
        for name in sorted(expected):
            loaded_bytes += _copy(
                torch,
                _draft_destination(model, name),
                reader.read(files[name], name),
                name,
            )
    return {"tensors": len(expected), "bytes": loaded_bytes}


__all__ = ["load_draft_weights", "load_target_weights"]
