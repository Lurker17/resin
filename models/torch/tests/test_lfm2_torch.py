from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _tiny_args():
    from resin_models_torch.lfm2.config import Lfm2Config

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


class TestLfm2TorchImport:
    def test_import_loads_torch_but_not_checkpoint_stacks(self) -> None:
        script = """
import sys
sys.path.insert(0, 'models/torch')
import resin_models_torch.lfm2
assert 'torch' in sys.modules
for name in ('transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_args_reject_non_lfm_checkpoint(self) -> None:
        from resin_models_torch.lfm2.config import Lfm2Config

        with pytest.raises(ValueError, match="expected an LFM2"):
            Lfm2Config.from_dict({"model_type": "llama"})


class TestLfm2Torch:
    def setup_method(self) -> None:
        from resin_models_torch.lfm2.model import Lfm2ForCausalLM

        torch.manual_seed(7)
        self.args = _tiny_args()
        self.model = Lfm2ForCausalLM(self.args).eval()

    def test_cached_chunks_match_full_forward(self) -> None:
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

    def test_batched_decode_matches_independent_cached_forwards(self) -> None:
        independent_states = [self.model.make_cache(), self.model.make_cache()]
        batched_states = [self.model.make_cache(), self.model.make_cache()]
        prefixes = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])]

        with torch.inference_mode():
            for prefix, independent, batched in zip(
                prefixes, independent_states, batched_states, strict=True
            ):
                self.model(prefix, independent)
                self.model(prefix, batched)

            for tokens in (torch.tensor([[6], [7]]), torch.tensor([[8], [9]])):
                expected = torch.cat(
                    [
                        self.model(tokens[index : index + 1], state)
                        for index, state in enumerate(independent_states)
                    ]
                )
                actual = self.model.forward_batch(tokens, batched_states)
                torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)

        assert [state.length for state in batched_states] == [5, 4]
        for independent, batched in zip(
            independent_states, batched_states, strict=True
        ):
            for expected_layer, actual_layer in zip(
                independent.layers, batched.layers, strict=True
            ):
                expected_tensors = vars(expected_layer).values()
                actual_tensors = vars(actual_layer).values()
                for expected_tensor, actual_tensor in zip(
                    expected_tensors, actual_tensors, strict=True
                ):
                    torch.testing.assert_close(
                        expected_tensor, actual_tensor, rtol=1e-5, atol=1e-5
                    )

    def test_checkpoint_parameter_paths_match_the_released_layout(self) -> None:
        keys = set(self.model.state_dict())
        assert "model.layers.0.conv.in_proj.weight" in keys
        assert "model.layers.1.self_attn.q_layernorm.weight" in keys
        assert "model.layers.1.feed_forward.w1.weight" in keys
        assert "model.embedding_norm.weight" in keys
        assert not any(".operator." in key for key in keys)

    def test_runner_consumes_batch_plan_and_releases_per_request_cache(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin_models_torch.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner.__new__(Lfm2TorchRunner)
        runner.device = torch.device("cpu")
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        runner.forward_calls = 0

        prefill = BatchPlan(
            [17, 23],
            [0, 0],
            [1, 2, 8],
            [0, 2, 3],
        )
        first = runner.forward(prefill)
        assert len(first) == 2
        assert runner.live_cache_count == 2

        decode = BatchPlan(
            [17, 23],
            [2, 1],
            first,
            [0, 1, 2],
            phase=BatchPhase.DECODE,
        )
        second = runner.forward(decode)
        assert len(second) == 2
        assert all(0 <= token < self.args.vocab_size for token in second)
        runner.release([17, 23])
        assert runner.live_cache_count == 0

    @pytest.mark.skipif(
        not importlib.util.find_spec("safetensors"),
        reason="safetensors is not installed",
    )
    def test_direct_safetensors_loading_without_transformers(
        self, tmp_path: Path
    ) -> None:
        from safetensors.torch import save_file

        from resin_models_torch.lfm2.reference import load_model

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
        tokens = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            expected = self.model(tokens, self.model.make_cache())

        path = tmp_path
        (path / "config.json").write_text(json.dumps(config))
        save_file(self.model.state_dict(), path / "model.safetensors")
        loaded, loaded_config = load_model(path, device="cpu")
        assert "transformers" not in sys.modules
        assert "tokenizers" not in sys.modules
        with torch.inference_mode():
            actual = loaded(tokens, loaded.make_cache())
        assert loaded_config["model_type"] == "lfm2"
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
