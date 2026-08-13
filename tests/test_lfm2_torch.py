from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _tiny_args():
    from resin.backends.lfm2 import Lfm2Config

    return Lfm2Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        norm_eps=1e-5,
        conv_bias=False,
        conv_cache_length=3,
        intermediate_size=32,
        multiple_of=8,
        auto_adjust_ff_dim=False,
        ffn_dim_multiplier=1.0,
        rope_theta=10_000.0,
        layer_types=("conv", "full_attention"),
    )


class Lfm2TorchImportTests(unittest.TestCase):
    def test_import_does_not_import_torch(self) -> None:
        script = """
import sys
import resin.backends.lfm2
for name in ('torch', 'transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_args_reject_non_lfm_checkpoint(self) -> None:
        from resin.backends.lfm2 import Lfm2Config

        with self.assertRaisesRegex(ValueError, "expected an LFM2"):
            Lfm2Config.from_dict({"model_type": "llama"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class Lfm2TorchTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from resin.backends.lfm2 import make_model_class

        torch.manual_seed(7)
        self.torch = torch
        self.args = _tiny_args()
        self.model = make_model_class(torch)(self.args).eval()

    def test_cached_chunks_match_full_forward(self) -> None:
        torch = self.torch
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            full = self.model(tokens, self.model.make_cache())
            for split in (1, 2, 3):
                cache = self.model.make_cache()
                self.model(tokens[:, :split], cache)
                cached = self.model(tokens[:, split:], cache)
                torch.testing.assert_close(
                    full[:, split:], cached, rtol=1e-5, atol=1e-5
                )

    def test_runner_consumes_batch_plan_and_releases_per_request_cache(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin.backends.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner.__new__(Lfm2TorchRunner)
        runner.device = self.torch.device("cpu")
        runner._torch = self.torch
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        runner.forward_calls = 0

        prefill = BatchPlan(
            [1, 2, 8],
            [0, 1, 0],
            [17, 23],
            [0, 2, 3],
            [0, 0],
        )
        first = runner.forward(prefill)
        self.assertEqual(len(first), 2)
        self.assertEqual(runner.live_cache_count, 2)

        decode = BatchPlan(
            first,
            [2, 1],
            [17, 23],
            [0, 1, 2],
            [2, 1],
            phase=BatchPhase.DECODE,
        )
        second = runner.forward(decode)
        self.assertEqual(len(second), 2)
        self.assertTrue(all(0 <= token < self.args.vocab_size for token in second))
        runner.release([17, 23])
        self.assertEqual(runner.live_cache_count, 0)

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors"), "safetensors is not installed"
    )
    def test_direct_safetensors_loading_without_transformers(self) -> None:
        from safetensors.torch import save_file

        from resin.backends.lfm2.reference import load_model

        config = {
            "model_type": "lfm2",
            "vocab_size": self.args.vocab_size,
            "hidden_size": self.args.hidden_size,
            "num_hidden_layers": self.args.num_hidden_layers,
            "num_attention_heads": self.args.num_attention_heads,
            "num_key_value_heads": self.args.num_key_value_heads,
            "norm_eps": self.args.norm_eps,
            "conv_bias": self.args.conv_bias,
            "conv_L_cache": self.args.conv_cache_length,
            "intermediate_size": self.args.intermediate_size,
            "block_ff_dim": self.args.intermediate_size,
            "block_multiple_of": self.args.multiple_of,
            "block_auto_adjust_ff_dim": self.args.auto_adjust_ff_dim,
            "block_ffn_dim_multiplier": self.args.ffn_dim_multiplier,
            "rope_parameters": {
                "rope_theta": self.args.rope_theta,
                "rope_type": "default",
            },
            "layer_types": list(self.args.layer_types),
        }
        tokens = self.torch.tensor([[1, 2, 3]])
        with self.torch.inference_mode():
            expected = self.model(tokens, self.model.make_cache())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config))
            save_file(self.model.state_dict(), path / "model.safetensors")
            torch, loaded, loaded_config = load_model(path, device="cpu")
            self.assertNotIn("transformers", sys.modules)
            self.assertNotIn("tokenizers", sys.modules)
            with torch.inference_mode():
                actual = loaded(tokens, loaded.make_cache())
        self.assertEqual(loaded_config["model_type"], "lfm2")
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    @unittest.skipUnless(
        os.environ.get("RESIN_LFM2_TORCH_MODEL"),
        "set RESIN_LFM2_TORCH_MODEL to run the pretrained-model integration test",
    )
    def test_pretrained_runner_generates_expected_tokens(self) -> None:
        from resin import Engine
        from resin.backends.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner(
            Path(os.environ["RESIN_LFM2_TORCH_MODEL"]), device="cpu"
        )
        engine = Engine(runner, eos_token_id=runner.eos_token_id)
        prompt = [
            1,
            6,
            24131,
            708,
            4083,
            938,
            768,
            11538,
            16701,
            12216,
            968,
            59929,
            14009,
            523,
            7,
            708,
            6,
            6423,
            708,
            3493,
            856,
            779,
            5706,
            803,
            4481,
            540,
            7,
            708,
            6,
            64015,
            708,
        ]
        request = engine.submit(prompt, max_new_tokens=8)
        while engine.has_work():
            engine.step()
        self.assertTrue(request.done)
        self.assertEqual(
            request.take_result(), [1098, 5706, 803, 4481, 856, 5242, 523, 7]
        )
        self.assertEqual(runner.live_cache_count, 0)


if __name__ == "__main__":
    unittest.main()
