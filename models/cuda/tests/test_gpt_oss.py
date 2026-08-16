from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _tiny_args():
    from resin_models_cuda.gpt_oss.config import GptOssConfig

    return GptOssConfig(
        vocab_size=43,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=4,
        sliding_window=3,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        initial_context_length=32,
        rope_scaling_factor=1.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        max_position_embeddings=64,
        swiglu_limit=7.0,
        attention_bias=True,
        layer_types=("sliding_attention", "full_attention"),
    )


def _config(args, *, quantization=None):
    config = {
        "model_type": "gpt_oss",
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "num_local_experts": args.num_local_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "head_dim": args.head_dim,
        "sliding_window": args.sliding_window,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "rope_scaling": {
            "factor": args.rope_scaling_factor,
            "beta_fast": args.rope_beta_fast,
            "beta_slow": args.rope_beta_slow,
            "original_max_position_embeddings": args.initial_context_length,
            "rope_type": "yarn",
        },
        "max_position_embeddings": args.max_position_embeddings,
        "swiglu_limit": args.swiglu_limit,
        "attention_bias": args.attention_bias,
        "attention_dropout": args.attention_dropout,
        "initializer_range": args.initializer_range,
        "layer_types": list(args.layer_types),
        "tie_word_embeddings": False,
    }
    if quantization is not None:
        config["quantization_config"] = quantization
    return config


class TestGptOssImport:
    def test_configs_cover_both_released_model_shapes(self) -> None:
        from resin_models_cuda.gpt_oss.config import GptOssConfig

        base = {
            "model_type": "gpt_oss",
            "vocab_size": 201_088,
            "hidden_size": 2_880,
            "intermediate_size": 2_880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts_per_tok": 4,
            "head_dim": 64,
            "sliding_window": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 150_000,
            "rope_scaling": {
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4_096,
            },
            "max_position_embeddings": 131_072,
            "swiglu_limit": 7.0,
            "attention_bias": True,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
        }
        for layers, experts in ((24, 32), (36, 128)):
            config = dict(base)
            config.update(
                num_hidden_layers=layers,
                num_local_experts=experts,
                layer_types=[
                    "sliding_attention" if index % 2 == 0 else "full_attention"
                    for index in range(layers)
                ],
            )
            args = GptOssConfig.from_dict(config)
            assert args.num_hidden_layers == layers
            assert args.num_local_experts == experts
            assert args.num_experts_per_tok == 4
            assert args.head_dim == 64

    def test_args_reject_non_gpt_oss_checkpoint(self) -> None:
        from resin_models_cuda.gpt_oss.config import GptOssConfig

        with pytest.raises(ValueError, match="expected a GPT-OSS"):
            GptOssConfig.from_dict({"model_type": "llama"})


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"), reason="PyTorch is not installed"
)
class TestGptOssTorch:
    def setup_method(self) -> None:
        import torch

        from resin_models_cuda.gpt_oss.model import GptOssForCausalLM

        torch.manual_seed(11)
        self.torch = torch
        self.args = _tiny_args()
        self.model = GptOssForCausalLM(self.args).eval()

    def test_cached_chunks_match_full_forward(self) -> None:
        torch = self.torch
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            full = self.model(tokens, self.model.make_cache())
            for split in (1, 2, 3, 4, 5):
                cache = self.model.make_cache()
                self.model(tokens[:, :split], cache)
                cached = self.model(tokens[:, split:], cache)
                torch.testing.assert_close(
                    full[:, split:], cached, rtol=1e-5, atol=1e-6
                )

    def test_sliding_attention_ignores_old_tokens_and_trims_cache(self) -> None:
        torch = self.torch
        attention = self.model.model.layers[0].self_attn
        original = torch.randn(1, 6, self.args.hidden_size)
        changed = original.clone()
        changed[:, 0] += 100
        positions = torch.arange(6).unsqueeze(0)
        rope = self.model.model.rotary_emb(positions, original.dtype)
        first_cache = self.model.make_cache().layers[0]
        second_cache = self.model.make_cache().layers[0]
        with torch.inference_mode():
            first = attention(original, positions, rope, first_cache)
            second = attention(changed, positions, rope, second_cache)
        torch.testing.assert_close(first[:, -1], second[:, -1], rtol=0, atol=0)
        assert first_cache.stored == self.args.sliding_window - 1
        assert second_cache.stored == self.args.sliding_window - 1

    def test_topk_moe_routes_and_weights_selected_experts(self) -> None:
        torch = self.torch
        mlp = self.model.model.layers[0].mlp
        with torch.no_grad():
            mlp.router.weight.zero_()
            mlp.router.bias.copy_(torch.tensor([10.0, 9.0, -10.0, -20.0]))
            mlp.experts.gate_up_proj.zero_()
            mlp.experts.gate_up_proj_bias.zero_()
            mlp.experts.down_proj.zero_()
            mlp.experts.down_proj_bias.zero_()
            mlp.experts.down_proj_bias[0].fill_(2.0)
            mlp.experts.down_proj_bias[1].fill_(4.0)

        hidden = torch.randn(1, 1, self.args.hidden_size)
        _, weights, indices = mlp.router(hidden.reshape(1, -1))
        assert indices.tolist() == [[0, 1]]
        expected_value = 2.0 * weights[0, 0] + 4.0 * weights[0, 1]
        with torch.inference_mode():
            actual = mlp(hidden)
        torch.testing.assert_close(
            actual,
            torch.full_like(actual, expected_value.item()),
            rtol=0,
            atol=1e-6,
        )

    def test_expert_kernel_boundary_receives_all_routing_metadata(self) -> None:
        from resin_models_cuda.gpt_oss.model import GptOssForCausalLM

        torch = self.torch
        calls = []

        class RecordingExperts(torch.nn.Module):
            def forward(self, hidden, indices, weights, logits):
                calls.append((hidden.shape, indices.shape, weights.shape, logits.shape))
                return torch.zeros_like(hidden)

        model = GptOssForCausalLM(
            self.args, expert_factory=lambda _args: RecordingExperts()
        ).eval()
        with torch.inference_mode():
            output = model(torch.tensor([[1, 2, 3]]), model.make_cache())
        assert output.shape == (1, 3, self.args.vocab_size)
        assert (
            calls
            == [
                (
                    torch.Size([3, self.args.hidden_size]),
                    torch.Size([3, self.args.num_experts_per_tok]),
                    torch.Size([3, self.args.num_experts_per_tok]),
                    torch.Size([3, self.args.num_local_experts]),
                )
            ]
            * self.args.num_hidden_layers
        )

    def test_equal_length_requests_share_one_model_forward_and_keep_own_caches(
        self,
    ) -> None:
        torch = self.torch
        tokens = torch.tensor([[1, 2, 3], [8, 9, 10]])
        positions = torch.arange(3).expand(2, -1)
        batched_caches = [self.model.make_cache(), self.model.make_cache()]
        reference_caches = [self.model.make_cache(), self.model.make_cache()]
        model_calls = 0

        def count_call(_module, _inputs, _output):
            nonlocal model_calls
            model_calls += 1

        handle = self.model.register_forward_hook(count_call)
        with torch.inference_mode():
            actual = self.model(
                tokens,
                self.model.make_batch_cache(batched_caches),
                positions,
            )
        handle.remove()
        assert model_calls == 1

        with torch.inference_mode():
            expected = torch.cat(
                [
                    self.model(
                        tokens[index : index + 1],
                        reference_caches[index],
                        positions[index : index + 1],
                    )
                    for index in range(2)
                ],
                dim=0,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert [cache.length for cache in batched_caches] == [3, 3]
        assert batched_caches[0] is not batched_caches[1]
        for layer_index in range(self.args.num_hidden_layers):
            assert (
                batched_caches[0].layers[layer_index]
                is not batched_caches[1].layers[layer_index]
            )
            assert (
                batched_caches[0].layers[layer_index].keys.untyped_storage().data_ptr()
                != batched_caches[1]
                .layers[layer_index]
                .keys.untyped_storage()
                .data_ptr()
            )

        decode_tokens = torch.tensor([[4], [11]])
        decode_positions = torch.tensor([[3], [3]])
        with torch.inference_mode():
            actual_decode = self.model(
                decode_tokens,
                self.model.make_batch_cache(batched_caches),
                decode_positions,
            )
            expected_decode = torch.cat(
                [
                    self.model(
                        decode_tokens[index : index + 1],
                        reference_caches[index],
                        decode_positions[index : index + 1],
                    )
                    for index in range(2)
                ],
                dim=0,
            )
        torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-5, atol=1e-6)

    def test_mxfp4_checkpoint_has_precise_kernel_boundary_error(
        self, tmp_path: Path
    ) -> None:
        from resin_models_cuda.gpt_oss.checkpoint import load_model

        path = tmp_path
        (path / "config.json").write_text(
            json.dumps(_config(self.args, quantization={"quant_method": "mxfp4"}))
        )
        with pytest.raises(
            NotImplementedError,
            match="MXFP4 expert weights.*H100 MXFP4 expert kernel",
        ):
            load_model(path, device="cpu")

    @pytest.mark.skipif(
        not importlib.util.find_spec("safetensors"),
        reason="safetensors is not installed",
    )
    def test_direct_dense_safetensors_loading_without_transformers(
        self, tmp_path: Path
    ) -> None:
        from safetensors.torch import save_file

        from resin_models_cuda.gpt_oss.checkpoint import load_model

        tokens = self.torch.tensor([[1, 2, 3]])
        with self.torch.inference_mode():
            expected = self.model(tokens, self.model.make_cache())

        path = tmp_path
        (path / "config.json").write_text(json.dumps(_config(self.args)))
        save_file(self.model.state_dict(), path / "model.safetensors")
        torch, loaded, loaded_config = load_model(path, device="cpu")
        assert "transformers" not in sys.modules
        assert "tokenizers" not in sys.modules
        with torch.inference_mode():
            actual = loaded(tokens, loaded.make_cache())
        assert loaded_config["model_type"] == "gpt_oss"
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
