from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _config(*, layers: int = 2) -> dict:
    return {
        "model_type": "gpt_oss",
        "num_hidden_layers": layers,
    }


@unittest.skipUnless(
    SAFETENSORS_AVAILABLE and TORCH_AVAILABLE,
    "torch and safetensors are required",
)
class GptOssWeightLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def _write_single(self, root: Path, tensors: dict, *, layers: int = 2) -> None:
        from safetensors.torch import save_file

        (root / "config.json").write_text(json.dumps(_config(layers=layers)))
        save_file(tensors, root / "model.safetensors")

    def test_single_file_streams_ordinary_and_experts_by_layer(self) -> None:
        from resin.backends.gpt_oss.weights import load_gpt_oss_weights

        previous: weakref.ReferenceType | None = None

        class Builder:
            def __init__(self, test) -> None:
                self.test = test
                self.events = []

            def expected_tensor_names(self, layer_index, config):
                self.test.assertEqual(config["num_hidden_layers"], 2)
                return ("blocks", "scales")

            def begin_layer(self, layer_index, config):
                self.events.append(("begin", layer_index))

            def consume_tensor(self, layer_index, name, tensor):
                nonlocal previous
                if previous is not None:
                    self.test.assertIsNone(previous())
                self.test.assertEqual(tensor.device.type, "cpu")
                self.events.append(("tensor", layer_index, name, tensor.clone()))
                previous = weakref.ref(tensor)

            def finish_layer(self, layer_index):
                self.events.append(("finish", layer_index))

        model = self._model()
        tensors = self._tensors()
        builder = Builder(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors)
            config, report = load_gpt_oss_weights(
                root,
                model,
                device="cpu",
                dtype="float64",
                expert_builder=builder,
            )

        self.assertEqual(config["model_type"], "gpt_oss")
        self.assertEqual(model.model.embed_tokens.weight.dtype, self.torch.float64)
        self.assertFalse(model.model.embed_tokens.weight.is_meta)
        self.torch.testing.assert_close(
            model.model.embed_tokens.weight,
            tensors["model.embed_tokens.weight"].double(),
        )
        self.assertEqual(
            [(event[0], event[1]) for event in builder.events],
            [
                ("begin", 0),
                ("tensor", 0),
                ("tensor", 0),
                ("finish", 0),
                ("begin", 1),
                ("tensor", 1),
                ("tensor", 1),
                ("finish", 1),
            ],
        )
        self.assertEqual(report.ordinary_tensor_count, 2)
        self.assertEqual(report.expert_tensor_count, 4)
        self.assertEqual(report.expert_layer_count, 2)
        self.assertEqual(report.meta_parameters_replaced, 2)
        self.assertFalse(report.peak_strategy.shard_dict_materialized)
        self.assertEqual(report.peak_strategy.checkpoint_tensors_in_flight, 1)
        self.assertFalse(report.peak_strategy.builder_retention_included)

    def test_index_is_checked_against_physical_shards(self) -> None:
        from safetensors.torch import save_file

        from resin.backends.gpt_oss.weights import inspect_gpt_oss_checkpoint

        tensors = self._tensors(layers=1)
        first_keys = {"model.embed_tokens.weight", "lm_head.weight"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config(layers=1)))
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            save_file({key: tensors[key] for key in first_keys}, first)
            save_file(
                {key: value for key, value in tensors.items() if key not in first_keys},
                second,
            )
            weight_map = {
                key: first.name if key in first_keys else second.name for key in tensors
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map})
            )
            manifest = inspect_gpt_oss_checkpoint(root)

        self.assertTrue(manifest.sharded)
        self.assertEqual(manifest.tensor_keys, frozenset(tensors))

    def test_duplicate_tensor_across_shards_is_rejected(self) -> None:
        from safetensors.torch import save_file

        from resin.backends.gpt_oss.weights import (
            CheckpointFormatError,
            inspect_gpt_oss_checkpoint,
        )

        tensor = self.torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config(layers=1)))
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            save_file({"duplicate": tensor}, first)
            save_file({"duplicate": tensor.clone(), "other": tensor.clone()}, second)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "duplicate": first.name,
                            "other": second.name,
                        }
                    }
                )
            )
            with self.assertRaisesRegex(
                CheckpointFormatError, "duplicate tensors across shards"
            ):
                inspect_gpt_oss_checkpoint(root)

    def test_wrong_index_location_is_rejected(self) -> None:
        from safetensors.torch import save_file

        from resin.backends.gpt_oss.weights import (
            CheckpointFormatError,
            inspect_gpt_oss_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config(layers=1)))
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            save_file({"left": self.torch.ones(1)}, first)
            save_file({"right": self.torch.ones(1)}, second)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "left": second.name,
                            "right": first.name,
                        }
                    }
                )
            )
            with self.assertRaisesRegex(CheckpointFormatError, "wrong shard"):
                inspect_gpt_oss_checkpoint(root)

    def test_missing_and_unexpected_model_coverage_is_rejected_before_assignment(
        self,
    ) -> None:
        from resin.backends.gpt_oss.weights import (
            CheckpointCoverageError,
            load_gpt_oss_weights,
        )

        class Builder:
            def expected_tensor_names(self, layer_index, config):
                return ("blocks", "scales")

            def begin_layer(self, layer_index, config):
                raise AssertionError("coverage must be checked before loading")

            def consume_tensor(self, layer_index, name, tensor):
                raise AssertionError

            def finish_layer(self, layer_index):
                raise AssertionError

        tensors = self._tensors(layers=1)
        del tensors["lm_head.weight"]
        tensors["surprise.weight"] = self.torch.ones(1)
        model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors, layers=1)
            with self.assertRaisesRegex(
                CheckpointCoverageError,
                "missing expected tensors.*lm_head.weight.*unexpected checkpoint tensors",
            ):
                load_gpt_oss_weights(
                    root, model, device="cpu", expert_builder=Builder()
                )
        self.assertTrue(model.model.embed_tokens.weight.is_meta)

    def test_shape_mismatch_is_rejected(self) -> None:
        from resin.backends.gpt_oss.weights import (
            CheckpointCoverageError,
            load_gpt_oss_weights,
        )

        class Builder:
            def expected_tensor_names(self, layer_index, config):
                return ("blocks", "scales")

            def begin_layer(self, layer_index, config):
                pass

            def consume_tensor(self, layer_index, name, tensor):
                pass

            def finish_layer(self, layer_index):
                pass

        tensors = self._tensors(layers=1)
        tensors["model.embed_tokens.weight"] = self.torch.ones(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_single(root, tensors, layers=1)
            with self.assertRaisesRegex(CheckpointCoverageError, "shape mismatch"):
                load_gpt_oss_weights(
                    root, self._model(), device="cpu", expert_builder=Builder()
                )

    def test_each_safetensors_shard_is_opened_once_during_weight_load(self) -> None:
        from safetensors import safe_open as real_safe_open
        from safetensors.torch import save_file

        from resin.backends.gpt_oss import weights as gpt_oss_weights

        class Builder:
            def expected_tensor_names(self, layer_index, config):
                return ("blocks", "scales")

            def begin_layer(self, layer_index, config):
                pass

            def consume_tensor(self, layer_index, name, tensor):
                pass

            def finish_layer(self, layer_index):
                pass

        tensors = self._tensors(layers=1)
        ordinary = {"model.embed_tokens.weight", "lm_head.weight"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config(layers=1)))
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

            opened = []

            def recording_safe_open(path, *args, **kwargs):
                opened.append(Path(path).name)
                return real_safe_open(path, *args, **kwargs)

            # The manifest audit opens every shard once, then the actual load
            # opens every shard once. Tensor count must not affect open count.
            with mock.patch.object(
                gpt_oss_weights, "_safe_open", return_value=recording_safe_open
            ):
                gpt_oss_weights.load_gpt_oss_weights(
                    root,
                    self._model(),
                    device="cpu",
                    expert_builder=Builder(),
                )

        self.assertEqual(opened.count(first.name), 2)
        self.assertEqual(opened.count(second.name), 2)


class GptOssWeightLoaderImportTests(unittest.TestCase):
    def test_import_is_tensor_and_tokenizer_free(self) -> None:
        script = """
import sys
import resin.backends.gpt_oss.weights
for name in ('torch', 'transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
