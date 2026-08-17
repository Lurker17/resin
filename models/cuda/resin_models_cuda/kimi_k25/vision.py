"""MoonViT tower and projector loaded without constructing the text model."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import Tensor, nn


def _flash_attention_4(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_cu_seqlens: Tensor | None = None,
    k_cu_seqlens: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """Adapt the checkpoint's FlashAttention-2 call shape to FlashAttention-4."""

    from flash_attn.cute.interface import flash_attn_varlen_func

    # FA4 inserted ``qv`` before the varlen metadata, so using the checkpoint's
    # positional FA2 call would silently bind cu_seqlens as an MLA tensor.  The
    # total packed length is a valid upper bound for every member sequence and
    # avoids synchronizing a GPU scalar back to Python in every vision layer.
    output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=q_cu_seqlens,
        cu_seqlens_k=k_cu_seqlens,
        max_seqlen_q=query.shape[0],
        max_seqlen_k=key.shape[0],
        causal=False,
        deterministic=bool(kwargs.get("deterministic", False)),
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.flatten(start_dim=-2)


def _checkpoint_modules(root: Path) -> tuple[Any, Any]:
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package_name = f"_resin_kimi_k25_{suffix}"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    text_module_name = f"{package_name}.modeling_deepseek"
    if text_module_name not in sys.modules:
        text_module = ModuleType(text_module_name)
        text_module.DeepseekV3ForCausalLM = object
        sys.modules[text_module_name] = text_module
    configuration = importlib.import_module(f"{package_name}.configuration_kimi_k25")
    modeling = importlib.import_module(f"{package_name}.modeling_kimi_k25")
    return configuration, modeling


class KimiVisionEncoder(nn.Module):
    """Checkpoint-native Kimi vision tower followed by its patch merger."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__()
        root = Path(model_path).expanduser().resolve()
        configuration, module = _checkpoint_modules(root)
        values = json.loads((root / "config.json").read_text())
        config = configuration.KimiK25Config(**values)
        vision_config = module.VisionTowerConfig(config.vision_config)
        # MoonViT selects its attention function from this checkpoint-local
        # table.  Use FA4 directly instead of falling back to the quadratic
        # eager path when Transformers cannot discover FlashAttention-2.
        module.VL_VISION_ATTENTION_FUNCTIONS["flash_attention_2"] = (
            _flash_attention_4
        )
        # Transformers rejects FlashAttention for this remote-code model before
        # MoonViT gets to its own attention dispatch.  Construct without asking
        # Transformers to dispatch, then select FA4 on every checkpoint block.
        vision_config._attn_implementation = "eager"
        self.vision_tower = module.MoonViT3dPretrainedModel(vision_config)
        for block in self.vision_tower.encoder.blocks:
            block.attn_implementation = "flash_attention_2"
        projector_config = module.ProjectorConfig(config.vision_config)
        projector_types: dict[str, Any] = {
            "identity": module.IdentityMap,
            "mlp": module.MLP,
            "patchmerger": module.PatchMergerMLP,
        }
        try:
            projector_type = projector_types[projector_config.mm_projector_type]
        except KeyError as error:
            raise ValueError(
                f"unsupported Kimi projector: {projector_config.mm_projector_type}"
            ) from error
        self.mm_projector = (
            projector_type()
            if projector_config.mm_projector_type == "identity"
            else projector_type(projector_config)
        )

    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> tuple[Tensor, ...]:
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        features = self.vision_tower(pixel_values.to(dtype), grid_thws)
        projected = self.mm_projector(features)
        if not isinstance(projected, (list, tuple)):
            raise TypeError("Kimi projector must return one tensor per image")
        return tuple(projected)


__all__ = ["KimiVisionEncoder"]
