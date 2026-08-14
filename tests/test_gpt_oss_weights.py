from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(
    not SAFETENSORS_AVAILABLE and TORCH_AVAILABLE,
    reason="torch and safetensors are required",
)
class TestGptOssWeightLoader:
    def setup_method(self) -> None:
        import torch

        self.torch = torch

    def _model(self):
        torch = self.torch

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(5, 3, device="meta")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(3, 5, bias=False, device="meta")

        return Model()

    def _tensors(self, *, layers: int = 2):
        torch = self.torch
        tensors = {
            "model.embed_tokens.weight": torch.arange(15, dtype=torch.float32).view(
                5, 3
            ),
            "lm_head.weight": torch.arange(15, 30, dtype=torch.float32).view(5, 3),
        }
        for layer in range(layers):
            prefix = f"model.layers.{layer}.mlp.experts"
            tensors[f"{prefix}.blocks"] = torch.tensor(
                [layer, layer + 1], dtype=torch.uint8
            )
            tensors[f"{prefix}.scales"] = torch.tensor(
                [layer + 0.5], dtype=torch.float32
            )
        return tensors

    def _write_single(self, root: Path, tensors: dict) -> None:
        from safetensors.torch import save_file

        save_file(tensors, root / "model.safetensors")

    def _load(self, root: Path, model, install, *, layers: int = 2, dtype=None):
        from resin.models.gpt_oss.weights import load_gpt_oss_weights

        load_gpt_oss_weights(
            root,
            model,
            device="cpu",
            dtype=dtype,
            num_hidden_layers=layers,
            expert_tensor_names=("blocks", "scales"),
            install_expert_layer=install,
        )

    def test_streams_ordinary_tensors_and_one_expert_layer_at_a_time(self) -> None:
        model = self._model()
        tensors = self._tensors()
        installed = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors)
            self._load(
                root,
                model,
                lambda layer, values: installed.append((layer, dict(values))),
                dtype="float64",
            )

        assert model.model.embed_tokens.weight.dtype == self.torch.float64
        assert not model.model.embed_tokens.weight.is_meta
        self.torch.testing.assert_close(
            model.model.embed_tokens.weight,
            tensors["model.embed_tokens.weight"].double(),
        )
        assert [layer for layer, _ in installed] == [0, 1]
        assert [set(values) for _, values in installed] == [
            {"blocks", "scales"},
            {"blocks", "scales"},
        ]

    def test_sharded_checkpoint_uses_index(self) -> None:
        from safetensors.torch import save_file

        tensors = self._tensors(layers=1)
        ordinary = {"model.embed_tokens.weight", "lm_head.weight"}
        installed = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            save_file({key: tensors[key] for key in ordinary}, first)
            save_file(
                {key: value for key, value in tensors.items() if key not in ordinary},
                second,
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            key: first.name if key in ordinary else second.name
                            for key in tensors
                        }
                    }
                )
            )
            self._load(
                root,
                self._model(),
                lambda layer, values: installed.append((layer, dict(values))),
                layers=1,
            )

        assert len(installed) == 1

    def test_missing_tensor_is_rejected_before_assignment(self) -> None:
        tensors = self._tensors(layers=1)
        del tensors["lm_head.weight"]
        model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors)
            with pytest.raises(ValueError, match="missing tensors"):
                self._load(root, model, lambda _layer, _values: None, layers=1)
        assert model.model.embed_tokens.weight.is_meta

    def test_shape_mismatch_is_rejected(self) -> None:
        tensors = self._tensors(layers=1)
        tensors["model.embed_tokens.weight"] = self.torch.ones(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors)
            with pytest.raises(ValueError, match="shape mismatch"):
                self._load(
                    root,
                    self._model(),
                    lambda _layer, _values: None,
                    layers=1,
                )


class TestGptOssWeightLoaderImport:
    def test_import_loads_torch_but_not_checkpoint_stacks(self) -> None:
        script = """
import sys
import resin.models.gpt_oss.weights
assert 'torch' in sys.modules
for name in ('transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)
