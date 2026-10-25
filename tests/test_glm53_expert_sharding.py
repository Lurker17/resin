"""Checkpoint-byte reconstruction for independent GLM EP and expert TP axes."""

import json
from dataclasses import replace
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.safetensors import SafetensorReader
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash import model as glm
from resin_models_cuda.glm5_3_flash.weights import WeightReader


def checkpoint(tmp_path, *, quantized, intermediate=512):
    config = replace(
        Glm53Config(),
        hidden_size=64,
        moe_intermediate_size=intermediate,
        n_routed_experts=8,
        num_experts_per_tok=2,
    )
    tensors = {}
    for expert in range(config.n_routed_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            h, m = config.hidden_size, config.moe_intermediate_size
            rows, columns = (h, m) if projection == "down_proj" else (m, h)
            base = f"{expert}.{projection}."
            salt = (
                expert * 13
                + {"gate_proj": 1, "up_proj": 5, "down_proj": 17}[projection]
            )
            if quantized:
                values = torch.arange(rows * columns // 2).reshape(rows, columns // 2)
                tensors[base + "weight"] = (values * 7 + salt).to(torch.uint8)
                scales = torch.arange(rows * columns // 16).reshape(rows, columns // 16)
                tensors[base + "weight_scale"] = (
                    (scales + salt).remainder(7).float() + 1
                ).to(torch.float8_e4m3fn)
                tensors[base + "weight_scale_2"] = torch.tensor(
                    {"gate_proj": 2.0, "up_proj": 4.0, "down_proj": 3.0}[projection]
                )
                # Global calibration must include expert0, even on other EP owners.
                tensors[base + "input_scale"] = torch.tensor(
                    (8.0 if expert == 0 else 0.5)
                    * (2 if projection == "down_proj" else 1)
                )
            else:
                values = torch.arange(rows * columns).reshape(rows, columns)
                tensors[base + "weight"] = (
                    (values * 7 + salt).remainder(251).float() / 32
                ).bfloat16()
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    return config, tensors


def equal(actual, expected):
    # Compare float8 storage too; CPU torch.equal does not implement every dtype.
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("ep_size,tp_size", [(4, 1), (1, 4), (2, 2)])
def test_expert_mesh_reconstructs_checkpoint_and_preserves_scalar_calibration(
    tmp_path, monkeypatch, quantized, ep_size, tp_size
):
    config, tensors = checkpoint(tmp_path, quantized=quantized)
    preparations = []

    def prepare(*weights):
        # The shared tested row-shuffle receives only the local, contiguous shard.
        preparations.append(tuple(value.shape for value in weights))
        assert all(value.is_contiguous() for value in weights)

    monkeypatch.setattr(glm, "prepare_nvfp4_moe_weights", prepare)
    kind = glm.Nvfp4Experts if quantized else glm.Bf16Experts
    meshes = {}
    for ep_rank in range(ep_size):
        for tp_rank in range(tp_size):
            reader = WeightReader(tmp_path, "cpu")
            original = reader.reader.read

            def no_full_body(path, name):
                if tp_size > 1:
                    assert not name.endswith((".weight", ".weight_scale"))
                return original(path, name)

            monkeypatch.setattr(reader.reader, "read", no_full_body)
            try:
                experts = kind(
                    config,
                    reader,
                    "",
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    expert_tensor_parallel=TensorParallelLayout(tp_rank, tp_size),
                    max_tokens=32,
                )
            finally:
                reader.close()
            meshes[ep_rank, tp_rank] = experts
            assert experts.count == config.n_routed_experts // ep_size
            assert experts.start == ep_rank * experts.count
            assert experts.intermediate == config.moe_intermediate_size // tp_size
            assert experts.intermediate_shard.start == tp_rank * experts.intermediate
            assert experts.w13.is_contiguous() and experts.w2.is_contiguous()
            assert experts.w13.stride() == (
                2 * experts.intermediate * experts.w13.shape[-1],
                experts.w13.shape[-1],
                1,
            )
            if quantized:
                assert isinstance(experts, glm.Nvfp4Experts)
                assert experts.input_scale_quant.item() == 1 / 8
                assert torch.equal(
                    experts.g1_alphas, torch.full((experts.count,), 16.0)
                )
                assert torch.equal(
                    experts.g2_alphas, torch.full((experts.count,), 48.0)
                )
            hidden = torch.empty(0, config.hidden_size, dtype=torch.bfloat16)
            assert (
                experts(hidden, torch.empty(0, 2), torch.empty(0, 2)).dtype
                == hidden.dtype
            )
            with pytest.raises(ValueError, match="prepared token bound"):
                experts(torch.empty(33, 64), torch.empty(33, 2), torch.empty(33, 2))

    for expert in range(config.n_routed_experts):
        ep_rank, local = divmod(expert, config.n_routed_experts // ep_size)
        parts = [meshes[ep_rank, tp_rank] for tp_rank in range(tp_size)]
        for projection, lower in (("up_proj", True), ("gate_proj", False)):
            weights = [
                value.w13[local, : value.intermediate]
                if lower
                else value.w13[local, value.intermediate :]
                for value in parts
            ]
            equal(torch.cat(weights, 0), tensors[f"{expert}.{projection}.weight"])
            if quantized:
                scales = [
                    value.w13_scales[local, : value.intermediate]
                    if lower
                    else value.w13_scales[local, value.intermediate :]
                    for value in (cast(glm.Nvfp4Experts, part) for part in parts)
                ]
                equal(
                    torch.cat(scales, 0), tensors[f"{expert}.{projection}.weight_scale"]
                )
        equal(
            torch.cat([value.w2[local] for value in parts], 1),
            tensors[f"{expert}.down_proj.weight"],
        )
        if quantized:
            equal(
                torch.cat(
                    [cast(glm.Nvfp4Experts, value).w2_scales[local] for value in parts],
                    1,
                ),
                tensors[f"{expert}.down_proj.weight_scale"],
            )
    assert len(preparations) == (ep_size * tp_size if quantized else 0)


@pytest.mark.parametrize("kind", [glm.Nvfp4Experts, glm.Bf16Experts])
@pytest.mark.parametrize(
    "ep_size,ep_rank,tp_size,intermediate,match",
    [
        (3, 0, 1, 512, "divisible by TP"),
        (4, 4, 1, 512, "rank"),
        (1, 0, 8, 512, "aligned units"),
        (1, 0, 2, 240, "divisible by 128"),
    ],
)
def test_invalid_geometry_fails_before_weight_read(
    kind, ep_size, ep_rank, tp_size, intermediate, match
):
    class NoReads:
        def read(self, *args, **kwargs):
            raise AssertionError("invalid geometry reached checkpoint data")

    config = replace(
        Glm53Config(), n_routed_experts=8, moe_intermediate_size=intermediate
    )
    with pytest.raises(ValueError, match=match):
        kind(
            config,
            NoReads(),
            "",
            ep_size=ep_size,
            ep_rank=ep_rank,
            expert_tensor_parallel=TensorParallelLayout(0, tp_size),
        )


def test_slice_reader_validates_source_and_bounds_counts_only_selected_bytes(
    tmp_path, monkeypatch
):
    config, tensors = checkpoint(tmp_path, quantized=True)
    reader = WeightReader(tmp_path, "cpu")
    name = "0.down_proj.weight"
    shape = tuple(tensors[name].shape)

    def no_full_read(*args):
        raise AssertionError("full body read")

    monkeypatch.setattr(reader.reader, "read", no_full_read)
    try:
        for slices in (
            (slice(-1, 3), slice(None)),
            (slice(None), slice(0, 257)),
            (slice(None), slice(16, 16)),
            (slice(None), slice(0, 16, 2)),
            (slice(None),),
            (0, slice(None)),
        ):
            with pytest.raises(ValueError):
                reader.read(name, shape, torch.uint8, slices=slices)
        with pytest.raises(ValueError, match="expected"):
            reader.read(name, (64, 1), torch.uint8, slices=(slice(None), slice(0, 64)))
        with pytest.raises(ValueError, match="expected"):
            reader.read(name, shape, torch.bfloat16, slices=(slice(None), slice(0, 64)))
        assert reader.loaded_bytes == 0
        part = reader.read(
            name, shape, torch.uint8, slices=(slice(None), slice(64, 128))
        )
        equal(part, tensors[name][:, 64:128])
        assert reader.loaded_bytes == config.hidden_size * 64
        # Opening another mapped file and closing the reader cannot invalidate part.
        save_file({"other": torch.zeros(4)}, str(tmp_path / "other.safetensors"))
        reader.reader.read_slice(
            tmp_path / "other.safetensors", "other", (slice(None),)
        )
    finally:
        reader.close()
    equal(part, tensors[name][:, 64:128])


def test_shared_safetensor_reader_keeps_original_full_read_and_close_contract(tmp_path):
    path = tmp_path / "a.safetensors"
    value = torch.arange(24).reshape(4, 6)
    save_file({"a": value}, str(path))
    with SafetensorReader() as reader:
        equal(reader.read(path, "a"), value)
        selected = reader.read_slice(
            path, "a", (slice(1, 3), slice(2, 5)), shape=(4, 6)
        )
        equal(selected, value[1:3, 2:5])
        equal(reader.read(path, "a"), value)
    equal(selected, value[1:3, 2:5])
