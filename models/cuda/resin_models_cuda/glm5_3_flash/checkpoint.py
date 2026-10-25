"""Pinned GLM-5.3-Flash NVFP4 checkpoint preparation for EP/attention-DP."""

from pathlib import Path

import torch

from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash.model import (
    Bf16Experts,
    DecoderLayer,
    DenseMLP,
    Glm53Model,
    MtpLayer,
    Nvfp4Experts,
    SparseMLP,
)
from resin_models_cuda.glm5_3_flash.weights import WeightReader

DEFAULT_MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
DEFAULT_REVISION = "da920bb0b9f4a06727223a349e55468e38352348"


def _download_model(repo_id, revision):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*MODEL_FILES, "*.json", "*.jinja", "tokenizer*"],
        )
    )


def resolve_model_path(model=DEFAULT_MODEL, revision=None):
    if revision is None and str(model) == DEFAULT_MODEL:
        revision = DEFAULT_REVISION
    return resolve_model(model, revision, _download_model)


def expert_partition(config, parallel_groups):
    if parallel_groups is None:
        return 1, 0
    groups = parallel_groups
    if groups.attn_tp.size != 1 or groups.cp.size != 1 or groups.expert_tp.size != 1:
        raise ValueError(
            "GLM-5.3-Flash currently supports EP with attention DP, not TP/CP"
        )
    if (
        groups.ep.size != groups.device_domain.size
        or groups.attn_dp.size != groups.ep.size
    ):
        raise ValueError("GLM-5.3-Flash requires attention-DP = EP = world size")
    if config.n_routed_experts % groups.ep.size:
        raise ValueError("GLM expert count must divide evenly across EP ranks")
    return groups.ep.size, groups.ep.local_rank


def load_model(
    model_path,
    *,
    device,
    parallel_groups=None,
    include_mtp=False,
    nvfp4_backend="cute-dsl",
    max_tokens=32768,
):
    """Stream only owned expert weights, preparing each layer before the next.

    Scalar activation calibration is read globally so changing EP placement does
    not silently alter the NVFP4 quantization regime. The unquantized image tower
    is replicated with attention; video handling remains outside this loader.
    """
    from resin_models_cuda.glm5_3_flash.attention import Attention

    root = Path(model_path).expanduser().resolve()
    raw = read_config(root)
    config = Glm53Config.from_dict(raw)
    ep_size, ep_rank = expert_partition(config, parallel_groups)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("GLM-5.3-Flash NVFP4 execution requires CUDA")
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise ValueError("GLM-5.3-Flash NVFP4 currently requires SM100 Blackwell")
    if include_mtp and config.num_nextn_predict_layers != 1:
        raise ValueError("expected exactly one native MTP layer")
    reader = WeightReader(root, device)
    prefix = "model.language_model."
    try:
        embedding = reader.read(
            prefix + "embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
            torch.bfloat16,
        )
        norm = reader.read(
            prefix + "norm.weight", (config.hidden_size,), torch.bfloat16
        )
        head = reader.read(
            "lm_head.weight", (config.vocab_size, config.hidden_size), torch.bfloat16
        )
        layers = []
        mtp = None
        for layer_id in range(config.num_hidden_layers + int(include_mtp)):
            layer_prefix = prefix + f"layers.{layer_id}."
            weights = reader.prefix(layer_prefix, exclude=(".mlp.experts.",))
            mlp_weights = {
                k.removeprefix("mlp."): v
                for k, v in weights.items()
                if k.startswith("mlp.")
            }
            if layer_id < config.first_k_dense_replace:
                mlp = DenseMLP(mlp_weights, config.swiglu_limit, backend=nvfp4_backend)
            else:
                experts_type = (
                    Bf16Experts
                    if layer_id == config.num_hidden_layers
                    else Nvfp4Experts
                )
                experts = experts_type(
                    config,
                    reader,
                    layer_prefix + "mlp.experts.",
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    # Every owner can receive one copy from every attention replica.
                    max_tokens=max_tokens * ep_size,
                )
                mlp = SparseMLP(config, mlp_weights, experts, parallel_groups)
            if layer_id == config.num_hidden_layers:
                mtp = MtpLayer(config, weights, mlp, Attention)
            else:
                layers.append(DecoderLayer(config, layer_id, weights, mlp, Attention))
            del weights, mlp_weights
            print(
                f"[glm5.3] rank {ep_rank} loaded layer {layer_id + 1}/{config.num_hidden_layers + int(include_mtp)}",
                flush=True,
            )
        model = Glm53Model(config, embedding, norm, head, layers, mtp).eval()
        if "vision_config" in raw:
            from resin_models_cuda.glm5_3_flash.vision import (
                GlmVisionConfig,
                GlmVisionModel,
            )

            model.vision = GlmVisionModel(
                GlmVisionConfig.from_dict(raw["vision_config"]),
                reader.prefix("model.visual."),
            ).eval()
        model.weight_load = {
            "bytes": reader.loaded_bytes,
            "ep_size": ep_size,
            "ep_rank": ep_rank,
            "mtp": include_mtp,
        }
        return model, config, raw
    finally:
        reader.close()


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "Glm53Config",
    "load_model",
    "resolve_model_path",
]
