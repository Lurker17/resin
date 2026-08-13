from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _tiny_args():
    from resin.backends.gpt_oss import _GptOssArgs

    return _GptOssArgs(
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


class GptOssImportTests(unittest.TestCase):
    def test_import_is_framework_and_tokenizer_free(self) -> None:
        script = """
import sys
import resin.backends.gpt_oss
for name in ('torch', 'transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_configs_cover_both_released_model_shapes(self) -> None:
        from resin.backends.gpt_oss import _GptOssArgs

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
            args = _GptOssArgs.from_config(config)
            self.assertEqual(args.num_hidden_layers, layers)
            self.assertEqual(args.num_local_experts, experts)
            self.assertEqual(args.num_experts_per_tok, 4)
            self.assertEqual(args.head_dim, 64)

    def test_args_reject_non_gpt_oss_checkpoint(self) -> None:
        from resin.backends.gpt_oss import _GptOssArgs

        with self.assertRaisesRegex(ValueError, "expected a GPT-OSS"):
            _GptOssArgs.from_config({"model_type": "llama"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class GptOssTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from resin.backends.gpt_oss import _make_model_class

        torch.manual_seed(11)
        self.torch = torch
        self.args = _tiny_args()
        self.model = _make_model_class(torch)(self.args).eval()

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
        self.assertEqual(first_cache.stored_tokens, self.args.sliding_window - 1)
        self.assertEqual(second_cache.stored_tokens, self.args.sliding_window - 1)

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
        self.assertEqual(indices.tolist(), [[0, 1]])
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
        from resin.backends.gpt_oss import _make_model_class

        torch = self.torch
        calls = []

        class RecordingExperts(torch.nn.Module):
            def forward(self, hidden, indices, weights, logits):
                calls.append((hidden.shape, indices.shape, weights.shape, logits.shape))
                return torch.zeros_like(hidden)

        model = _make_model_class(
            torch, expert_factory=lambda _args: RecordingExperts()
        )(self.args).eval()
        with torch.inference_mode():
            output = model(torch.tensor([[1, 2, 3]]), model.make_cache())
        self.assertEqual(output.shape, (1, 3, self.args.vocab_size))
        self.assertEqual(
            calls,
            [
                (
                    torch.Size([3, self.args.hidden_size]),
                    torch.Size([3, self.args.num_experts_per_tok]),
                    torch.Size([3, self.args.num_experts_per_tok]),
                    torch.Size([3, self.args.num_local_experts]),
                )
            ]
            * self.args.num_hidden_layers,
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
        self.assertEqual(model_calls, 1)

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
        self.assertEqual([cache.offset for cache in batched_caches], [3, 3])
        self.assertIsNot(batched_caches[0], batched_caches[1])
        for layer_index in range(self.args.num_hidden_layers):
            self.assertIsNot(
                batched_caches[0].layers[layer_index],
                batched_caches[1].layers[layer_index],
            )
            self.assertNotEqual(
                batched_caches[0].layers[layer_index].keys.untyped_storage().data_ptr(),
                batched_caches[1].layers[layer_index].keys.untyped_storage().data_ptr(),
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

    def test_runner_consumes_rust_batch_plan_and_releases_caches(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin.backends.gpt_oss import GptOssRunner

        runner = GptOssRunner.__new__(GptOssRunner)
        runner.device = self.torch.device("cpu")
        runner._torch = self.torch
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}

        model_calls = 0

        def count_call(_module, _inputs, _output):
            nonlocal model_calls
            model_calls += 1

        handle = self.model.register_forward_hook(count_call)
        prefill = BatchPlan(
            [1, 2, 8, 9],
            [0, 1, 0, 1],
            [17, 23],
            [0, 2, 4],
            [0, 0],
        )
        first = runner.forward(prefill)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(runner._cache), 2)
        self.assertEqual(model_calls, 1)

        decode = BatchPlan(
            first,
            [2, 2],
            [17, 23],
            [0, 1, 2],
            [2, 2],
            phase=BatchPhase.DECODE,
        )
        second = runner.forward(decode)
        self.assertEqual(len(second), 2)
        self.assertTrue(all(0 <= token < self.args.vocab_size for token in second))
        self.assertEqual(model_calls, 2)
        runner.release([17])
        self.assertEqual(len(runner._cache), 1)
        self.assertIn(23, runner._cache)
        runner.release([23])
        self.assertEqual(len(runner._cache), 0)
        handle.remove()

    def test_runner_chunked_prefill_matches_single_chunk_and_cache_state(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin.backends.gpt_oss import GptOssRunner

        def make_runner(chunk_size):
            runner = GptOssRunner.__new__(GptOssRunner)
            runner.device = self.torch.device("cpu")
            runner._torch = self.torch
            runner._model = self.model
            runner.config = {"vocab_size": self.args.vocab_size}
            runner.eos_token_id = None
            runner.vocab_size = self.args.vocab_size
            runner.prefill_chunk_size = chunk_size
            runner._cache = {}
            return runner

        plan = BatchPlan(
            [1, 2, 3, 4, 5, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [17, 23],
            [0, 5, 10],
            [0, 0],
        )
        single_chunk = make_runner(128)
        chunked = make_runner(2)

        expected = single_chunk.forward(plan)
        actual = chunked.forward(plan)
        self.assertEqual(actual, expected)
        self.assertEqual(len(single_chunk._cache), 2)
        self.assertEqual(len(chunked._cache), 2)
        for request_id in plan.sequence_ids:
            expected_cache = single_chunk._cache[request_id]
            actual_cache = chunked._cache[request_id]
            self.assertEqual(actual_cache.offset, expected_cache.offset)
            for expected_layer, actual_layer in zip(
                expected_cache.layers, actual_cache.layers, strict=True
            ):
                self.torch.testing.assert_close(
                    actual_layer.keys, expected_layer.keys, rtol=1e-5, atol=1e-6
                )
                self.torch.testing.assert_close(
                    actual_layer.values, expected_layer.values, rtol=1e-5, atol=1e-6
                )

        decode = BatchPlan(
            expected,
            [5, 5],
            [17, 23],
            [0, 1, 2],
            [5, 5],
            phase=BatchPhase.DECODE,
        )
        self.assertEqual(chunked.forward(decode), single_chunk.forward(decode))
        single_chunk.release([17, 23])
        chunked.release([17, 23])
        self.assertEqual(len(single_chunk._cache), 0)
        self.assertEqual(len(chunked._cache), 0)

    def test_runner_executes_one_model_call_per_plan(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin.backends.gpt_oss import GptOssRunner

        runner = GptOssRunner.__new__(GptOssRunner)
        runner.device = self.torch.device("cpu")
        runner._torch = self.torch
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner.prefill_chunk_size = 2
        runner._cache = {}

        model_calls = 0

        def count_call(_module, _inputs, _output):
            nonlocal model_calls
            model_calls += 1

        handle = self.model.register_forward_hook(count_call)
        prefill = BatchPlan(
            [1, 2, 3, 4, 5, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [17, 23],
            [0, 5, 10],
            [0, 0],
        )
        first = runner.forward(prefill)
        self.assertEqual(model_calls, 1)
        self.assertEqual(
            [runner._cache[request_id].offset for request_id in (17, 23)], [5, 5]
        )

        decode = BatchPlan(
            first,
            [5, 5],
            [17, 23],
            [0, 1, 2],
            [5, 5],
            phase=BatchPhase.DECODE,
        )
        runner.forward(decode)
        self.assertEqual(model_calls, 2)
        self.assertEqual(
            [runner._cache[request_id].offset for request_id in (17, 23)], [6, 6]
        )
        handle.remove()

    def test_runner_rejects_heterogeneous_plan_instead_of_serializing(self) -> None:
        from resin import BatchPlan
        from resin.backends.gpt_oss import GptOssRunner

        runner = GptOssRunner.__new__(GptOssRunner)
        runner.device = self.torch.device("cpu")
        runner._torch = self.torch
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        plan = BatchPlan(
            [1, 2, 8],
            [0, 1, 0],
            [17, 23],
            [0, 2, 3],
            [0, 0],
        )

        with self.assertRaisesRegex(ValueError, "equal token counts"):
            runner.forward(plan)
        self.assertEqual(len(runner._cache), 0)

    def test_mxfp4_checkpoint_has_precise_kernel_boundary_error(self) -> None:
        from resin.backends.gpt_oss import _load_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps(_config(self.args, quantization={"quant_method": "mxfp4"}))
            )
            with self.assertRaisesRegex(
                NotImplementedError, "MXFP4 expert weights.*H100 MXFP4 expert kernel"
            ):
                _load_model(path, device="cpu")

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors"), "safetensors is not installed"
    )
    def test_direct_dense_safetensors_loading_without_transformers(self) -> None:
        from safetensors.torch import save_file

        from resin.backends.gpt_oss import _load_model

        tokens = self.torch.tensor([[1, 2, 3]])
        with self.torch.inference_mode():
            expected = self.model(tokens, self.model.make_cache())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(_config(self.args)))
            save_file(self.model.state_dict(), path / "model.safetensors")
            torch, loaded, loaded_config = _load_model(path, device="cpu")
            self.assertNotIn("transformers", sys.modules)
            self.assertNotIn("tokenizers", sys.modules)
            with torch.inference_mode():
                actual = loaded(tokens, loaded.make_cache())
        self.assertEqual(loaded_config["model_type"], "gpt_oss")
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
