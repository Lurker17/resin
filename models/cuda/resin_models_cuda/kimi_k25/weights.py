"""Stream NVIDIA Kimi NVFP4 target and LightSeek EAGLE3 draft weights."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.kimi_k25.model import KimiForCausalLM, KimiMoE
from resin_models_cuda.kimi_k25.vision import KimiVisionEncoder

_TARGET_LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.(.+)")
_EXPERT = re.compile(
    r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down|gate|up)_proj\.(weight|weight_scale|input_scale|weight_scale_2)"
)


def _axis_shard(tensor: Any, axis: int, tp: TensorParallelContext, name: str):
    if not tp.enabled:
        return tensor
    shard = tp.layout.even_shard(tensor.shape[axis], name=name)
    return tensor.narrow(axis, shard.start, shard.size).contiguous()


def _copy(torch: Any, destination: Any, source: Any, name: str) -> int:
    if source.numel() == destination.numel() and source.shape != destination.shape:
        source = source.reshape(destination.shape)
    if tuple(source.shape) != tuple(destination.shape):
        raise ValueError(
            f"Kimi tensor {name!r} has shape {tuple(source.shape)}, expected "
            f"{tuple(destination.shape)}"
        )
    if source.dtype != destination.dtype:
        source = source.to(destination.dtype)
    with torch.no_grad():
        destination.copy_(source.to(destination.device))
    return source.numel() * source.element_size()


def _target_names(model: KimiForCausalLM) -> set[str]:
    names = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    }
    for layer_id, layer in enumerate(model.target.layers):
        prefix = f"language_model.model.layers.{layer_id}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
            }
        )
        if isinstance(layer.mlp, KimiMoE):
            names.update(
                {
                    f"{prefix}.mlp.gate.weight",
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                }
            )
            experts = layer.mlp.experts
            for expert in range(
                experts.local_expert_offset,
                experts.local_expert_offset + experts.local_experts,
            ):
                for projection in ("down", "gate", "up"):
                    for field in (
                        "weight",
                        "weight_scale",
                        "input_scale",
                        "weight_scale_2",
                    ):
                        names.add(
                            f"{prefix}.mlp.experts.{expert}.{projection}_proj.{field}"
                        )
        else:
            names.update(
                {
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
    return names


def _target_destination(model: KimiForCausalLM, name: str):
    if name == "language_model.model.embed_tokens.weight":
        return model.target.embed_tokens.weight
    if name == "language_model.model.norm.weight":
        return model.target.norm.weight
    if name == "language_model.lm_head.weight":
        return model.target.lm_head.weight
    match = _TARGET_LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized Kimi target tensor: {name!r}")
    layer = model.target.layers[int(match.group(1))]
    suffix = match.group(2)
    direct = {
        "input_layernorm.weight": layer.input_layernorm.weight,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        "self_attn.q_a_proj.weight": layer.self_attn.q_a_proj.weight,
        "self_attn.q_a_layernorm.weight": layer.self_attn.q_a_layernorm.weight,
        "self_attn.q_b_proj.weight": layer.self_attn.q_b_proj.weight,
        "self_attn.kv_a_proj_with_mqa.weight": (
            layer.self_attn.kv_a_proj_with_mqa.weight
        ),
        "self_attn.kv_a_layernorm.weight": layer.self_attn.kv_a_layernorm.weight,
        "self_attn.kv_b_proj.weight": layer.self_attn.kv_b_proj.weight,
        "self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
    }
    if suffix in direct:
        return direct[suffix]
    if not isinstance(layer.mlp, KimiMoE):
        dense = {
            "mlp.gate_proj.weight": layer.mlp.gate_weight,
            "mlp.up_proj.weight": layer.mlp.up_weight,
            "mlp.down_proj.weight": layer.mlp.down_proj.weight,
        }
        try:
            return dense[suffix]
        except KeyError as error:
            raise ValueError(f"unrecognized Kimi dense tensor: {name!r}") from error
    moe = layer.mlp
    routed = {
        "mlp.gate.weight": moe.experts.gate.weight,
        "mlp.gate.e_score_correction_bias": (moe.experts.e_score_correction_bias),
        "mlp.shared_experts.gate_proj.weight": moe.shared_experts.gate_weight,
        "mlp.shared_experts.up_proj.weight": moe.shared_experts.up_weight,
        "mlp.shared_experts.down_proj.weight": moe.shared_experts.down_proj.weight,
    }
    if suffix in routed:
        return routed[suffix]
    expert_match = _EXPERT.fullmatch(name)
    if expert_match is None:
        raise ValueError(f"unrecognized Kimi MoE tensor: {name!r}")
    global_expert = int(expert_match.group(2))
    local_expert = global_expert - moe.experts.local_expert_offset
    projection, field = expert_match.group(3), expert_match.group(4)
    intermediate = moe.experts.intermediate_size
    if projection in {"up", "gate"}:
        projection_index = 0 if projection == "up" else 1
        if field == "weight":
            start = projection_index * intermediate
            return moe.experts.w13[local_expert, start : start + intermediate]
        if field == "weight_scale":
            start = projection_index * intermediate
            return moe.experts.w13_scales[local_expert, start : start + intermediate]
        if field == "input_scale":
            return moe.experts.w13_input_scales[local_expert, projection_index]
        return moe.experts.w13_weight_scales[local_expert, projection_index]
    if field == "weight":
        return moe.experts.w2[local_expert]
    if field == "weight_scale":
        return moe.experts.w2_scales[local_expert]
    if field == "input_scale":
        return moe.experts.w2_input_scales[local_expert]
    return moe.experts.w2_weight_scales[local_expert]


def _target_shard(name: str, tensor: Any, tp: TensorParallelContext):
    if name in {
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }:
        return _axis_shard(tensor, 0, tp, name)
    if ".self_attn.q_b_proj." in name or ".self_attn.kv_b_proj." in name:
        return _axis_shard(tensor, 0, tp, name)
    if ".self_attn.o_proj." in name:
        return _axis_shard(tensor, 1, tp, name)
    if ".mlp.shared_experts." in name or ".layers.0.mlp." in name:
        axis = 1 if ".down_proj." in name else 0
        return _axis_shard(tensor, axis, tp, name)
    return tensor


def load_target_weights(
    model_path: str | Path,
    model: KimiForCausalLM,
    *,
    tp: TensorParallelContext,
) -> dict[str, int]:
    """Load one TP/EP rank directly from NVIDIA's original target shards."""

    import torch

    root = Path(model_path).expanduser().resolve()
    files = indexed_tensor_files(root, require_index=True)
    expected = _target_names(model)
    if missing := expected - set(files):
        raise ValueError(f"Kimi target is missing tensors: {sorted(missing)[:8]}")
    loaded_bytes = 0
    shard_count = len({files[name] for name in expected})
    shard_index = 0
    current_shard = None
    with SafetensorReader() as reader:
        ordered = sorted(expected, key=lambda key: (files[key].name, key))
        for tensor_index, name in enumerate(ordered):
            shard = files[name]
            if shard != current_shard:
                current_shard = shard
                shard_index += 1
                if tp.rank == 0 and (shard_index == 1 or shard_index % 8 == 0):
                    print(
                        f"[kimi] loading target shard {shard_index}/{shard_count} "
                        f"({tensor_index}/{len(ordered)} tensors)",
                        flush=True,
                    )
            source = reader.read(files[name], name)
            source = _target_shard(name, source, tp)
            loaded_bytes += _copy(torch, _target_destination(model, name), source, name)
    return {"tensors": len(expected), "bytes": loaded_bytes}


def load_vision_weights(
    model_path: str | Path, vision: KimiVisionEncoder
) -> dict[str, int]:
    """Load the replicated MoonViT tower and multimodal projector."""

    import torch

    root = Path(model_path).expanduser().resolve()
    files = indexed_tensor_files(root, require_index=True)
    destinations = vision.state_dict()
    expected = set(destinations)
    if missing := expected - set(files):
        raise ValueError(
            f"Kimi vision target is missing tensors: {sorted(missing)[:8]}"
        )
    loaded_bytes = 0
    with SafetensorReader() as reader:
        for name in sorted(expected, key=lambda key: (files[key].name, key)):
            loaded_bytes += _copy(
                torch,
                destinations[name],
                reader.read(files[name], name),
                name,
            )
    return {"tensors": len(expected), "bytes": loaded_bytes}


_DRAFT_NAMES = {
    "embed_tokens.weight",
    "fc.weight",
    "layers.0.hidden_norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.kv_a_layernorm.weight",
    "layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "layers.0.self_attn.kv_b_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_a_layernorm.weight",
    "layers.0.self_attn.q_a_proj.weight",
    "layers.0.self_attn.q_b_proj.weight",
    "lm_head.weight",
    "norm.weight",
}


def _draft_destination(model: KimiForCausalLM, name: str):
    draft = model.draft
    direct = {
        "embed_tokens.weight": draft.embed_tokens.weight,
        "fc.weight": draft.fc.weight,
        "layers.0.hidden_norm.weight": draft.hidden_norm.weight,
        "layers.0.input_layernorm.weight": draft.input_layernorm.weight,
        "layers.0.mlp.down_proj.weight": draft.mlp.down_proj.weight,
        "layers.0.mlp.gate_proj.weight": draft.mlp.gate_weight,
        "layers.0.mlp.up_proj.weight": draft.mlp.up_weight,
        "layers.0.post_attention_layernorm.weight": (
            draft.post_attention_layernorm.weight
        ),
        "layers.0.self_attn.kv_a_layernorm.weight": (
            draft.self_attn.kv_a_layernorm.weight
        ),
        "layers.0.self_attn.kv_a_proj_with_mqa.weight": (
            draft.self_attn.kv_a_proj_with_mqa.weight
        ),
        "layers.0.self_attn.kv_b_proj.weight": draft.self_attn.kv_b_proj.weight,
        "layers.0.self_attn.o_proj.weight": draft.self_attn.o_proj.weight,
        "layers.0.self_attn.q_a_layernorm.weight": (
            draft.self_attn.q_a_layernorm.weight
        ),
        "layers.0.self_attn.q_a_proj.weight": draft.self_attn.q_a_proj.weight,
        "layers.0.self_attn.q_b_proj.weight": draft.self_attn.q_b_proj.weight,
        "lm_head.weight": draft.lm_head.weight,
        "norm.weight": draft.norm.weight,
    }
    if name.startswith("fc_norm."):
        if draft.fc_norm is None:
            raise ValueError("draft checkpoint has fc_norm but model does not")
        return draft.fc_norm[int(name.split(".")[1])].weight
    try:
        return direct[name]
    except KeyError as error:
        raise ValueError(f"unrecognized Kimi draft tensor: {name!r}") from error


def _draft_shard(name: str, tensor: Any, tp: TensorParallelContext):
    if name in {"embed_tokens.weight", "lm_head.weight", "fc.weight"}:
        return _axis_shard(tensor, 0, tp, name)
    if ".self_attn.q_b_proj." in name or ".self_attn.kv_b_proj." in name:
        return _axis_shard(tensor, 0, tp, name)
    if ".self_attn.o_proj." in name or ".mlp.down_proj." in name:
        return _axis_shard(tensor, 1, tp, name)
    if ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
        return _axis_shard(tensor, 0, tp, name)
    return tensor


def load_draft_weights(
    model_path: str | Path,
    model: KimiForCausalLM,
    *,
    tp: TensorParallelContext,
) -> dict[str, int]:
    """Load one TP rank from a BF16 LightSeek MLA draft."""

    import torch

    root = Path(model_path).expanduser().resolve()
    files = indexed_tensor_files(root)
    expected = set(_DRAFT_NAMES)
    if model.draft.fc_norm is not None:
        expected.update(
            f"fc_norm.{index}.weight" for index in range(len(model.draft.fc_norm))
        )
    if missing := expected - set(files):
        raise ValueError(f"Kimi draft is missing tensors: {sorted(missing)[:8]}")
    if unexpected := set(files) - expected:
        raise ValueError(f"Kimi draft has unexpected tensors: {sorted(unexpected)[:8]}")
    loaded_bytes = 0
    with SafetensorReader() as reader:
        for name in sorted(expected):
            source = _draft_shard(name, reader.read(files[name], name), tp)
            loaded_bytes += _copy(torch, _draft_destination(model, name), source, name)
    return {"tensors": len(expected), "bytes": loaded_bytes}


__all__ = ["load_draft_weights", "load_target_weights", "load_vision_weights"]
