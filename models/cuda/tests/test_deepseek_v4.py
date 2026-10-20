from __future__ import annotations

import importlib
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from resin.backend.cuda.parallel import FlashInferAllReduce
from resin.models.deepseek_v4.config import DeepSeekV4Config
from resin_models_cuda.deepseek_v4.checkpoint import resolve_model_path
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner
from resin_models_cuda.deepseek_v4.weights import (
    load_deepseek_v4_weights,
    tensor_parallel_axis,
)


def _release_config() -> dict[str, object]:
    return {
        "vocab_size": 129280,
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "qk_rope_head_dim": 64,
        "sliding_window": 128,
        "compress_ratios": [0, 0, *([4, 128] * 20), 4, 0, 0, 0],
        "index_topk": 512,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "hc_mult": 4,
        "expert_dtype": "fp4",
    }


class TestDeepSeekV4Config:
    def test_release_geometry_supports_tp4(self) -> None:
        config = DeepSeekV4Config.from_hugging_face(_release_config())

        config.validate_tensor_parallel(4)
        assert config.num_hidden_layers == 43
        assert config.compress_ratios[:6] == (0, 0, 4, 128, 4, 128)
        assert config.num_routed_experts // 4 == 64
        assert config.num_attention_heads // 4 == 16

    def test_pro_geometry_supports_tp4(self) -> None:
        values = _release_config()
        values.update(
            {
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "q_lora_rank": 1536,
                "o_groups": 16,
                "compress_ratios": [128, 128, *([4, 128] * 29), 4],
                "index_topk": 1024,
                "n_routed_experts": 384,
                "moe_intermediate_size": 3072,
                "dspark_block_size": 5,
                "dspark_noise_token_id": 128799,
                "dspark_target_layer_ids": [58, 59, 60],
                "dspark_markov_rank": 512,
            }
        )

        config = DeepSeekV4Config.from_hugging_face(values)

        config.validate_tensor_parallel(4)
        assert config.num_hidden_layers == 61
        assert config.num_routed_experts // 4 == 96
        assert config.num_attention_heads // 4 == 32
        assert config.dspark_markov_rank == 512

    def test_rejects_mismatched_layer_compression(self) -> None:
        values = _release_config()
        values["compress_ratios"] = [0]

        with pytest.raises(ValueError, match="one entry per target layer"):
            DeepSeekV4Config.from_hugging_face(values)

    def test_rejects_incompatible_tensor_parallel_size(self) -> None:
        config = DeepSeekV4Config.from_hugging_face(_release_config())

        with pytest.raises(ValueError, match="TP=3"):
            config.validate_tensor_parallel(3)

    def test_dspark_geometry_is_checkpoint_data(self) -> None:
        values = _release_config()
        values.update(
            {
                "dspark_block_size": 5,
                "dspark_noise_token_id": 128799,
                "dspark_target_layer_ids": [40, 41, 42],
                "dspark_markov_rank": 256,
            }
        )

        config = DeepSeekV4Config.from_hugging_face(values)

        assert config.dspark_block_size == 5
        assert config.dspark_noise_token_id == 128799
        assert config.dspark_target_layer_ids == (40, 41, 42)
        assert config.dspark_markov_rank == 256


class TestDeepSeekV4Collectives:
    def test_flashinfer_allreduce_uses_plain_sum_pattern(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import flashinfer.comm as flashinfer_comm

        reducer = FlashInferAllReduce.__new__(FlashInferAllReduce)
        reducer.context = SimpleNamespace(sum=lambda tensor: tensor)
        reducer.max_token_num = 1
        reducer.hidden_dim = 8
        reducer.dtype = torch.float32
        reducer.workspace = SimpleNamespace(backend="trtllm")
        calls: list[dict[str, object]] = []

        def fused(**kwargs: object) -> torch.Tensor:
            calls.append(kwargs)
            return kwargs["input"] * 2

        monkeypatch.setattr(flashinfer_comm, "allreduce_fusion", fused)
        tensor = torch.randn(1, 1, 8)

        output = reducer.sum(tensor)

        torch.testing.assert_close(output, tensor * 2)
        assert calls[0]["pattern"] == 0
        assert not calls[0]["launch_with_pdl"]
        assert calls[0]["trigger_completion_at_end"]

    def test_runner_exposes_fast_allreduce_switch(self) -> None:
        parameters = inspect.signature(DeepSeekV4CudaRunner.__init__).parameters

        assert "fast_allreduce" in parameters


class TestDeepSeekV4Checkpoint:
    @staticmethod
    def _checkpoint(tmp_path: Path) -> Path:
        inference = tmp_path / "inference"
        inference.mkdir()
        (tmp_path / "config.json").write_text(json.dumps(_release_config()))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}})
        )
        (inference / "config.json").write_text("{}")
        return tmp_path

    def test_resolver_treats_checkpoint_python_as_untrusted_data(
        self, tmp_path: Path
    ) -> None:
        checkpoint = self._checkpoint(tmp_path)
        marker = checkpoint / "inference/model.py"
        marker.write_text("raise AssertionError('checkpoint code executed')\n")

        assert resolve_model_path(checkpoint) == checkpoint.resolve()

    def test_resolver_reports_missing_checkpoint_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing files"):
            resolve_model_path(tmp_path)


class TestDeepSeekV4Weights:
    @pytest.mark.parametrize(
        ("name", "axis"),
        [
            ("embed.weight", 0),
            ("head.weight", 0),
            ("layers.2.attn.attn_sink", 0),
            ("layers.2.attn.wq_b.scale", 0),
            ("layers.2.attn.indexer.weights_proj.weight", 0),
            ("layers.2.attn.wo_b.weight", 1),
            ("layers.2.attn.wq_a.weight", None),
            ("layers.2.ffn.experts.9.w1.weight", None),
        ],
    )
    def test_published_tensor_parallel_axes(self, name: str, axis: int | None) -> None:
        assert tensor_parallel_axis(name) == axis

    def test_streaming_loader_reads_only_the_local_vocab_shards(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import resin_models_cuda.deepseek_v4.weights as weights

        class TinyRank(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(4, 2)
                self.head = torch.nn.Linear(2, 4, bias=False)

        tensors = {
            "embed.weight": torch.arange(16, dtype=torch.float32).view(8, 2),
            "head.weight": torch.arange(16, 32, dtype=torch.float32).view(8, 2),
        }

        class Reader:
            def __enter__(self) -> Reader:
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def read(self, _path: Path, name: str) -> torch.Tensor:
                return tensors[name]

        shard = tmp_path / "model.safetensors"
        monkeypatch.setattr(
            weights,
            "indexed_tensor_files",
            lambda root, **kwargs: {name: shard for name in tensors},
        )
        monkeypatch.setattr(weights, "SafetensorReader", Reader)
        model = TinyRank()

        report = load_deepseek_v4_weights(
            tmp_path,
            model,
            rank=1,
            tensor_parallel_size=2,
            device=torch.device("cpu"),
        )

        assert torch.equal(model.embed.weight, tensors["embed.weight"][4:])
        assert torch.equal(model.head.weight, tensors["head.weight"][4:])
        assert report == {"tensors": 2, "bytes": 64}

    @staticmethod
    def _native_model(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        cutile = ModuleType("resin_models_cuda.deepseek_v4.cutile")
        for name in (
            "act_quant",
            "clear_rejected_compress_state",
            "compress_verify",
            "dspark_attention_metadata",
            "fp4_act_quant",
            "finish_index_topk",
            "hadamard_transform",
            "hash_sqrt_softplus",
            "index_scores",
            "kv_norm_rope_store",
            "moe_act_quant",
            "qk_norm_rope_block",
            "qk_norm_rope_store",
            "rope_last_block_inplace",
            "swiglu",
            "top6_sqrt_softplus",
            "verify_attention_metadata",
        ):
            setattr(cutile, name, None)
        flashinfer = ModuleType("flashinfer")
        flashinfer_mhc = ModuleType("flashinfer.mhc")
        flashinfer_mhc.mhc_post = None
        flashinfer_mhc.mhc_pre_big_fuse_with_prenorm = None
        flashinfer_topk = ModuleType("flashinfer.topk")
        flashinfer_topk.top_k = None
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.tf32_hc_prenorm_gemm = None
        gemm = ModuleType("resin_models_cuda.deepseek_v4.gemm")
        for name in (
            "dynamic_gemm_rows",
            "fp4_fused_moe",
            "fp4_gemm",
            "fp8_gemm",
            "fp8_gemm_uses_packed_scales",
            "prepare_fp4_fused_moe_weights",
        ):
            setattr(gemm, name, None)
        attention = ModuleType("resin_models_cuda.deepseek_v4.attention")
        attention.sparse_decode_attention = None
        attention.sparse_topk_lengths = None
        attention.sparse_verify_attention = None
        quack = ModuleType("quack")
        quack.rmsnorm = None
        monkeypatch.setitem(sys.modules, cutile.__name__, cutile)
        monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
        monkeypatch.setitem(sys.modules, attention.__name__, attention)
        monkeypatch.setitem(sys.modules, flashinfer.__name__, flashinfer)
        monkeypatch.setitem(sys.modules, flashinfer_mhc.__name__, flashinfer_mhc)
        monkeypatch.setitem(sys.modules, flashinfer_topk.__name__, flashinfer_topk)
        monkeypatch.setitem(sys.modules, deep_gemm.__name__, deep_gemm)
        monkeypatch.setitem(sys.modules, quack.__name__, quack)

        module_path = (
            Path(__file__).parents[1] / "resin_models_cuda/deepseek_v4/model.py"
        )
        experts_spec = importlib.util.spec_from_file_location(
            "_resin_test_deepseek_v4_experts", module_path.with_name("experts.py")
        )
        assert experts_spec is not None and experts_spec.loader is not None
        experts = importlib.util.module_from_spec(experts_spec)
        experts_spec.loader.exec_module(experts)
        monkeypatch.setitem(
            sys.modules, "resin_models_cuda.deepseek_v4.experts", experts
        )
        spec = importlib.util.spec_from_file_location(
            "_resin_test_deepseek_v4_model", module_path
        )
        assert spec is not None and spec.loader is not None
        native_model = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, native_model)
        spec.loader.exec_module(native_model)
        return native_model

    def test_grouped_expert_storage_preserves_checkpoint_parameter_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        native_model = self._native_model(monkeypatch)
        topk_call: dict[str, object] = {}

        def top_k(scores: torch.Tensor, width: int, **kwargs: object):
            topk_call.update(width=width, **kwargs)
            return scores, torch.arange(width).view(1, width)

        native_model.flashinfer_top_k = top_k
        topk_indices = native_model._compact_topk_indices(torch.zeros(1, 8), 3)
        assert topk_indices.tolist() == [[0, 1, 2]]
        assert topk_call == {"width": 3, "sorted": True}
        frequencies = torch.polar(torch.ones(1, 5, 32), torch.randn(1, 5, 32))
        rank_three = torch.randn(1, 5, 64)
        rank_four = torch.randn(1, 5, 2, 64)
        assert native_model.apply_rotary_emb(rank_three, frequencies).shape == (
            1,
            5,
            64,
        )
        assert native_model.apply_rotary_emb(rank_four, frequencies).shape == (
            1,
            5,
            2,
            64,
        )
        guard_args = native_model.ModelArgs(
            max_batch_size=1,
            max_seq_len=32,
            vocab_size=256,
            dim=64,
            moe_inter_dim=64,
            n_layers=1,
            n_heads=2,
            q_lora_rank=64,
            head_dim=64,
            rope_head_dim=64,
            o_groups=2,
            o_lora_rank=64,
            window_size=4,
            compress_ratios=(0,),
            dspark_block_size=5,
        )
        attention = native_model.DSparkAttention(0, guard_args)
        assert attention.window_capacity == 10
        assert not hasattr(attention, "kv_cache")
        assert "paged_cache" not in native_model.ModelArgs.__dataclass_fields__
        native_model.world_size = 2
        native_model.rank = 1
        args = native_model.ModelArgs(
            vocab_size=256,
            dim=128,
            moe_inter_dim=128,
            n_hash_layers=0,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            expert_dtype="fp4",
        )

        moe = native_model.MoEWeights(0, args)

        assert moe.experts[0] is None
        assert moe.experts[1] is None
        assert (
            moe.experts[2].w1.weight.data_ptr()
            == moe._fused_w13_weight[0, 128:].data_ptr()
        )
        assert (
            moe.experts[2].w3.weight.data_ptr()
            == moe._fused_w13_weight[0, :128].data_ptr()
        )
        assert moe.experts[3].w2.weight.data_ptr() == moe._fused_w2_weight[1].data_ptr()
        state = moe.state_dict()
        assert "experts.2.w1.weight" in state
        assert "experts.2.w1.scale" in state
        assert "experts.3.w3.weight" in state
        assert not any("_fused" in name for name in state)
        with torch.no_grad():
            moe.experts[2].w1.weight.view(torch.uint8).fill_(17)
        assert torch.all(moe._fused_w13_weight[0, 128:].view(torch.uint8) == 17)

        monkeypatch.setattr(
            native_model,
            "bf16_linear_fp32",
            lambda inputs, weight: inputs.float() @ weight.float().T,
        )
        routing_weights, expert_ids = moe.gate(
            torch.randn(3, args.dim, dtype=torch.bfloat16),
            torch.arange(3),
        )
        assert routing_weights.dtype == torch.bfloat16
        assert expert_ids.dtype == torch.int32

    def test_prepared_routed_experts_adopt_storage_and_own_reduction(self, monkeypatch):
        native = self._native_model(monkeypatch)
        execution = sys.modules["resin_models_cuda.deepseek_v4.experts"]
        native.world_size, native.rank = 2, 1
        args = native.ModelArgs(
            vocab_size=256,
            dim=128,
            moe_inter_dim=128,
            n_hash_layers=0,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            expert_dtype="fp4",
        )
        source = native.MoEWeights(0, args)
        loaded_weight = source._fused_w13_weight
        calls = []
        monkeypatch.setattr(
            execution, "prepare_fp4_fused_moe_weights", lambda *w: calls.append("pack")
        )
        monkeypatch.setattr(execution, "moe_act_quant", lambda x, *a: (x, None))

        def kernel(x, scales, indices, weights, *storage, **options):
            assert storage[0] is loaded_weight
            assert options["num_experts"] == 4
            assert options["local_expert_offset"] == 2
            return torch.full_like(x, 2)

        def reduce(partial):
            calls.append("reduce")
            return partial * 10

        monkeypatch.setattr(execution, "fp4_fused_moe", kernel)
        with pytest.raises(RuntimeError, match="prepared"):
            source(torch.ones(2, 128), torch.zeros(2, dtype=torch.long))
        model = torch.nn.ModuleList([source])
        native.Transformer.prepare_weights(model, reduce)
        moe = model[0]
        assert moe.routed.w13_weight is loaded_weight
        assert not hasattr(moe, "experts") and not hasattr(moe, "_fused_w13_weight")
        assert calls == ["pack"]
        native.Transformer.prepare_weights(model, reduce)
        assert model[0] is moe and calls == ["pack"]
        assert source._fused_w13_weight is loaded_weight

        class Gate(torch.nn.Module):
            def forward(self, x, ids):
                return torch.ones(2, 2), torch.tensor([[2, 3], [0, 1]])

        moe.gate = Gate()
        moe.shared_experts = torch.nn.Identity()
        # Runtime ownership is captured; another model's topology cannot change it.
        native.world_size, native.rank = 1, 0
        output = moe(
            torch.ones(2, 128, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.long)
        )
        torch.testing.assert_close(output, torch.full_like(output, 21))
        assert calls == ["pack", "reduce"]

    def test_unfused_expert_shard_reduces_even_with_no_local_assignments(
        self, monkeypatch
    ):
        native = self._native_model(monkeypatch)

        class Expert(torch.nn.Module):
            def forward(self, x, weights):
                return x * weights * 3

        source = SimpleNamespace(
            n_routed_experts=4,
            experts_start_idx=2,
            experts_end_idx=3,
            _fused_fp4=False,
            experts=torch.nn.ModuleList([None, None, Expert(), None]),
        )
        partials = []

        def reduce(partial):
            partials.append(partial.clone())
            return partial + 1

        runtime = native.RoutedExperts(
            source,
            reduce=reduce,
            block_size=128,
            scale_fmt=None,
            scale_dtype=torch.float32,
        )
        hidden = torch.ones(2, 8, dtype=torch.bfloat16)
        result = runtime(
            hidden,
            torch.tensor([[2, 0], [1, 2]]),
            torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        )
        torch.testing.assert_close(
            result[:, 0], torch.tensor([1.75, 2.5], dtype=torch.bfloat16)
        )
        result = runtime(hidden, torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2))
        assert torch.equal(partials[-1], torch.zeros_like(hidden))
        assert torch.equal(result, torch.ones_like(hidden))
        assert len(partials) == 2


class TestDeepSeekV4Gemms:
    @staticmethod
    def _module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        flashinfer = ModuleType("flashinfer")
        flashinfer_gemm = ModuleType("flashinfer.gemm")
        flashinfer_gemm.gemm_fp8_nt_groupwise = None
        flashinfer_moe = ModuleType("flashinfer.fused_moe")
        flashinfer_moe.reorder_rows_for_gated_act_gemm = None
        flashinfer_moe.trtllm_fp4_block_scale_routed_moe = None
        flashinfer_quantization = ModuleType("flashinfer.quantization")
        flashinfer_quantization.block_scale_interleave = None
        flashinfer_quantization.shuffle_matrix_a = None
        flashinfer_enums = ModuleType("flashinfer.tllm_enums")

        class ActivationType:
            class Swiglu:
                value = 3

        flashinfer_enums.ActivationType = ActivationType
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_gemm_nt = None
        deep_gemm.transform_sf_into_required_layout = None
        quack = ModuleType("quack")
        quack_blockscaled = ModuleType("quack.blockscaled")
        quack_blockscaled.BlockScaledOperand = None
        quack_blockscaled.pack_scale_2d_to_blocked_contig = None
        quack_gemm = ModuleType("quack.gemm_interface")
        quack_gemm.gemm = None
        for name, module in {
            "flashinfer": flashinfer,
            "flashinfer.gemm": flashinfer_gemm,
            "flashinfer.fused_moe": flashinfer_moe,
            "flashinfer.quantization": flashinfer_quantization,
            "flashinfer.tllm_enums": flashinfer_enums,
            "deep_gemm": deep_gemm,
            "quack": quack,
            "quack.blockscaled": quack_blockscaled,
            "quack.gemm_interface": quack_gemm,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)
        module_name = "resin_models_cuda.deepseek_v4.gemm"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        return importlib.import_module(module_name)

    def test_fp8_uses_deep_gemm_packed_k128_layout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        transform_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
        gemm_calls: list[tuple[object, ...]] = []

        def transform(scales: torch.Tensor, **kwargs: object) -> torch.Tensor:
            transform_calls.append((scales, kwargs))
            return torch.ones(128, 1, dtype=torch.int32)

        def deep_gemm(*args: object, **kwargs: object) -> None:
            gemm_calls.append((*args, kwargs))

        monkeypatch.setattr(
            deepseek_gemm, "transform_sf_into_required_layout", transform
        )
        monkeypatch.setattr(deepseek_gemm, "fp8_fp4_gemm_nt", deep_gemm)
        left = torch.zeros(2, 128, dtype=torch.float8_e4m3fn)
        left_scales = torch.ones(2, 1, dtype=torch.int32)
        right = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        right_scales = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)

        output = deepseek_gemm.fp8_gemm(
            left,
            left_scales,
            right,
            right_scales,
            torch.float8_e8m0fnu,
        )

        assert output.shape == (2, 128)
        assert transform_calls[0][1] == {
            "mn": 128,
            "k": 128,
            "recipe": (1, 128, 128),
            "is_sfa": False,
        }
        assert gemm_calls[0][0][1] is left_scales
        assert gemm_calls[0][-1] == {
            "recipe_a": (1, 128),
            "recipe_b": (1, 128),
            "disable_ue8m0_cast": True,
        }

    def test_arbitrary_prefill_rows_select_runtime_m_gemm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        assert deepseek_gemm.fp8_gemm_uses_packed_scales(5)
        with deepseek_gemm.dynamic_gemm_rows():
            assert not deepseek_gemm.fp8_gemm_uses_packed_scales(5)
        assert deepseek_gemm.fp8_gemm_uses_packed_scales(5)

    def test_prefill_capture_restores_compute_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)
        runner = object.__new__(DeepSeekV4CudaRunner)
        runner.device = torch.device("meta")
        original = torch.empty(0)

        with pytest.raises(RuntimeError, match="capture failed"):
            with runner._prefill_compute_context():
                output = torch.empty(0)
                assert output.dtype == torch.bfloat16
                assert output.device.type == "meta"
                assert not deepseek_gemm.fp8_gemm_uses_packed_scales(5)
                raise RuntimeError("capture failed")

        restored = torch.empty(0)
        assert (restored.dtype, restored.device) == (
            original.dtype,
            original.device,
        )
        assert deepseek_gemm.fp8_gemm_uses_packed_scales(5)

    @pytest.mark.parametrize("rows", [132, 129])
    def test_fp8_prefill_uses_dynamic_trtllm_gemm_and_aligns_rows(
        self, monkeypatch: pytest.MonkeyPatch, rows: int
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def flashinfer_gemm(*args: object, **kwargs: object) -> torch.Tensor:
            calls.append((args, kwargs))
            return kwargs["out"]

        monkeypatch.setattr(deepseek_gemm, "gemm_fp8_nt_groupwise", flashinfer_gemm)
        monkeypatch.setattr(
            deepseek_gemm.torch, "get_default_dtype", lambda: torch.bfloat16
        )
        aligned_rows = (rows + 3) // 4 * 4
        left = torch.zeros(1, rows, 256, dtype=torch.float8_e4m3fn)
        scale_storage = (torch.arange(2 * aligned_rows).float() % 7).exp2().view(2, -1)
        left_scales = scale_storage[:, :rows].T.view(1, rows, 2)
        right = torch.zeros(128, 256, dtype=torch.float8_e4m3fn)
        right_scales = torch.ones(1, 2, dtype=torch.float8_e8m0fnu)

        output = deepseek_gemm.fp8_gemm(
            left,
            left_scales,
            right,
            right_scales,
            torch.float8_e8m0fnu,
        )

        assert output.shape == (1, rows, 128)
        assert output.dtype == torch.bfloat16
        args, kwargs = calls[0]
        assert args[0].shape == (aligned_rows, 256)
        assert torch.count_nonzero(args[0][rows:].float()) == 0
        assert args[2].shape == (2, aligned_rows)
        assert args[2].dtype == torch.float32
        assert torch.all(args[2][:, rows:] == 1)
        torch.testing.assert_close(args[2][:, :rows], scale_storage[:, :rows])
        if rows == aligned_rows:
            assert args[2].data_ptr() == left_scales.data_ptr()
        assert args[3].shape == (1, 2)
        assert args[3].dtype == torch.float32
        assert kwargs["scale_major_mode"] == "MN"
        assert kwargs["scale_granularity_mnk"] == (1, 128, 128)
        assert kwargs["out"].shape == (aligned_rows, 128)
        assert kwargs["out"].dtype == torch.bfloat16
        assert kwargs["backend"] == "trtllm"

    def test_fp4_uses_deep_gemm_with_native_asymmetric_scale_recipes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        transform_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
        gemm_calls: list[tuple[object, ...]] = []

        def transform(scales: torch.Tensor, **kwargs: object) -> torch.Tensor:
            transform_calls.append((scales, kwargs))
            return torch.ones(128, 1, dtype=torch.int32)

        def deep_gemm(*args: object, **kwargs: object) -> None:
            gemm_calls.append((*args, kwargs))

        monkeypatch.setattr(
            deepseek_gemm, "transform_sf_into_required_layout", transform
        )
        monkeypatch.setattr(deepseek_gemm, "fp8_fp4_gemm_nt", deep_gemm)
        left = torch.zeros(2, 128, dtype=torch.float8_e4m3fn)
        left_scales = torch.ones(2, 1, dtype=torch.int32)
        right = torch.empty(128, 64, dtype=torch.float4_e2m1fn_x2)
        right_scales = torch.ones(128, 4, dtype=torch.float8_e8m0fnu)

        output = deepseek_gemm.fp4_gemm(
            left,
            left_scales,
            right,
            right_scales,
            torch.float8_e8m0fnu,
        )

        assert output.shape == (2, 128)
        assert transform_calls[0][1] == {"mn": 128, "k": 128, "recipe": (1, 32)}
        assert gemm_calls[0][0][1] is left_scales
        assert gemm_calls[0][-1] == {
            "recipe_a": (1, 128),
            "recipe_b": (1, 32),
            "disable_ue8m0_cast": True,
        }

    def test_fp4_fused_moe_uses_flashinfer_exact_mixed_format_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        calls: list[dict[str, object]] = []

        def fused_moe(**kwargs: object) -> list[torch.Tensor]:
            calls.append(kwargs)
            return [torch.zeros(3, 128, dtype=torch.bfloat16)]

        monkeypatch.setattr(
            deepseek_gemm, "trtllm_fp4_block_scale_routed_moe", fused_moe
        )
        left = torch.zeros(3, 128, dtype=torch.float8_e4m3fn)
        left_scales = (
            torch.arange(12, dtype=torch.uint8).view(3, 4).view(torch.float8_e4m3fn)
        )
        topk_ids = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.int32)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)
        w13 = torch.empty(2, 256, 64, dtype=torch.float4_e2m1fn_x2)
        w13_scales = torch.ones(2, 256, 4, dtype=torch.float8_e8m0fnu)
        w2 = torch.empty(2, 128, 64, dtype=torch.float4_e2m1fn_x2)
        w2_scales = torch.ones(2, 128, 4, dtype=torch.float8_e8m0fnu)
        output_scales = torch.ones(2, dtype=torch.float32)
        clamp_limits = torch.full((2,), 10.0, dtype=torch.float32)

        output = deepseek_gemm.fp4_fused_moe(
            left,
            left_scales,
            topk_ids,
            topk_weights,
            w13,
            w13_scales,
            w2,
            w2_scales,
            output_scales,
            clamp_limits,
            num_experts=4,
            local_expert_offset=2,
            scale_dtype=torch.float8_e8m0fnu,
        )

        kwargs = calls[0]
        ids, weights = kwargs["topk_ids"]  # type: ignore[misc]
        assert ids is topk_ids
        assert weights is topk_weights
        assert kwargs["hidden_states_scale"] is left_scales
        assert kwargs["gemm1_weights"].data_ptr() == w13.data_ptr()  # type: ignore[union-attr]
        assert kwargs["gemm2_weights"].data_ptr() == w2.data_ptr()  # type: ignore[union-attr]
        assert kwargs["gemm1_clamp_limit"] is clamp_limits
        assert kwargs["output1_scale_scalar"] is output_scales
        assert kwargs["num_experts"] == 4
        assert kwargs["local_expert_offset"] == 2
        assert kwargs["local_num_experts"] == 2
        assert kwargs["activation_type"] == 3
        assert output.shape == (3, 128)

    def test_fused_moe_weight_preparation_is_in_place_and_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)
        calls: list[tuple[str, tuple[int, ...]]] = []

        def reorder(tensor: torch.Tensor) -> torch.Tensor:
            calls.append(("reorder", tuple(tensor.shape)))
            return tensor.clone()

        def shuffle_weight(tensor: torch.Tensor, tile: int) -> torch.Tensor:
            assert tile == 128
            calls.append(("weight", tuple(tensor.shape)))
            return tensor.clone()

        def shuffle_scale(tensor: torch.Tensor) -> torch.Tensor:
            calls.append(("scale", tuple(tensor.shape)))
            return tensor + 2

        monkeypatch.setattr(deepseek_gemm, "reorder_rows_for_gated_act_gemm", reorder)
        monkeypatch.setattr(deepseek_gemm, "shuffle_matrix_a", shuffle_weight)
        monkeypatch.setattr(deepseek_gemm, "block_scale_interleave", shuffle_scale)
        w13 = torch.zeros(2, 256, 64, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        w13_scales = torch.zeros(2, 256, 4, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        w2 = torch.zeros(2, 128, 64, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        w2_scales = torch.zeros(2, 128, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        deepseek_gemm.prepare_fp4_fused_moe_weights(w13, w13_scales, w2, w2_scales)
        first_calls = list(calls)
        deepseek_gemm.prepare_fp4_fused_moe_weights(w13, w13_scales, w2, w2_scales)

        assert calls == first_calls
        assert len([call for call in calls if call[0] == "reorder"]) == 1
        assert torch.all(w13.view(torch.uint8) == 0)
        assert torch.all(w2.view(torch.uint8) == 0)
        assert torch.all(w13_scales.view(torch.uint8) == 2)
        assert torch.all(w2_scales.view(torch.uint8) == 2)


class TestDeepSeekV4Attention:
    def test_flashinfer_decode_uses_dsv4_sparse_mla_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def flashinfer(*args: object, **kwargs: object) -> torch.Tensor:
            calls.append((args, kwargs))
            return torch.ones(2, 1, 2, 4)

        flashinfer_package = ModuleType("flashinfer")
        flashinfer_mla = ModuleType("flashinfer.mla")
        flashinfer_mla.trtllm_batch_decode_sparse_mla_dsv4 = flashinfer
        monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_package)
        monkeypatch.setitem(sys.modules, "flashinfer.mla", flashinfer_mla)
        module_name = "resin_models_cuda.deepseek_v4.attention"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        attention = importlib.import_module(module_name)
        query = torch.zeros(2, 1, 2, 4)
        window = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        compressed = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
        indices = torch.tensor([[[2, 1, 0, 1]], [[1, 0, -1, -1]]], dtype=torch.int32)
        sinks = torch.tensor([1.0, 1.0])
        sparse_lengths = torch.tensor([4, 2], dtype=torch.int32)
        sequence_lengths = torch.tensor([3, 2], dtype=torch.int32)
        workspace = torch.empty(8, dtype=torch.uint8)

        output = attention.sparse_decode_attention(
            query,
            window,
            compressed,
            sinks,
            indices,
            sparse_lengths,
            sequence_lengths,
            workspace,
            0.5,
        )

        args, kwargs = calls[0]
        assert args[0] is query
        assert args[1] is window
        assert args[2] is workspace
        assert torch.equal(args[3], indices.view(2, -1))
        assert kwargs["compressed_kv_cache"] is compressed
        assert kwargs["sparse_topk_lens"] is sparse_lengths
        assert kwargs["seq_lens"] is sequence_lengths
        assert kwargs["sinks"] is sinks
        assert kwargs["bmm1_scale"] == 0.5
        assert kwargs["kv_layout"] == "HND"
        assert kwargs["enable_pdl"] is True
        assert output.shape == query.shape

        unaligned_indices = torch.zeros(2, 1, 5, dtype=torch.int32)
        attention.sparse_decode_attention(
            query,
            window,
            compressed,
            sinks,
            unaligned_indices,
            sparse_lengths,
            sequence_lengths,
            workspace,
            0.5,
        )
        padded_indices = calls[1][0][3]
        assert padded_indices.shape == (2, 8)
        assert torch.equal(padded_indices[:, :5], unaligned_indices.view(2, 5))
        assert torch.equal(padded_indices[:, 5:], torch.full((2, 3), -1))

        complete = torch.tensor([0, 32, 2048])
        assert attention.sparse_topk_lengths(complete, 512, 128).tolist() == [
            128,
            160,
            640,
        ]

        with pytest.raises(ValueError, match="physical KV pools must be contiguous"):
            attention.sparse_decode_attention(
                query,
                torch.zeros(2, 4, 4)[:, :3],
                compressed,
                sinks,
                indices,
                sparse_lengths,
                sequence_lengths,
                workspace,
                0.5,
            )

    def test_flashinfer_verify_flattens_a_small_query_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def flashinfer(*args: object, **kwargs: object) -> torch.Tensor:
            calls.append((args, kwargs))
            return torch.ones(6, 1, 2, 4)

        flashinfer_package = ModuleType("flashinfer")
        flashinfer_mla = ModuleType("flashinfer.mla")
        flashinfer_mla.trtllm_batch_decode_sparse_mla_dsv4 = flashinfer
        monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_package)
        monkeypatch.setitem(sys.modules, "flashinfer.mla", flashinfer_mla)
        module_name = "resin_models_cuda.deepseek_v4.attention"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        attention = importlib.import_module(module_name)

        query = torch.zeros(2, 3, 2, 4)
        window = torch.zeros(2, 7, 4)
        compressed = torch.zeros(2, 5, 4)
        indices = torch.zeros(2, 3, 5, dtype=torch.int32)
        sparse_lengths = torch.full((2, 3), 5, dtype=torch.int32)
        sequence_lengths = torch.arange(1, 7, dtype=torch.int32).view(2, 3)
        sinks = torch.ones(2)
        workspace = torch.empty(8, dtype=torch.uint8)

        output = attention.sparse_verify_attention(
            query,
            window,
            compressed,
            sinks,
            indices,
            sparse_lengths,
            sequence_lengths,
            workspace,
            0.5,
        )

        args, kwargs = calls[0]
        assert args[0].shape == (6, 1, 2, 4)
        assert args[1] is window
        assert args[2] is workspace
        assert args[3].shape == (6, 8)
        assert kwargs["compressed_kv_cache"] is compressed
        assert torch.equal(kwargs["sparse_topk_lens"], sparse_lengths.flatten())
        assert torch.equal(kwargs["seq_lens"], sequence_lengths.flatten())
        assert kwargs["enable_pdl"] is True
        assert output.shape == query.shape

        attention.sparse_verify_attention(
            query,
            window,
            compressed,
            sinks,
            indices,
            sparse_lengths,
            sequence_lengths,
            workspace,
            0.5,
            enable_pdl=False,
        )
        assert calls[1][1]["enable_pdl"] is False


class TestDeepSeekV4Runner:
    def test_prefill_ffn_preserves_hash_inputs_and_decode_dispatch(self, monkeypatch):
        from test_cuda_runner_graph_policy import cpu_prefill_regions

        native = TestDeepSeekV4Weights._native_model(monkeypatch)
        mode = {"prefill": False}

        @contextmanager
        def prefill_context():
            previous = mode["prefill"]
            mode["prefill"] = True
            try:
                yield
            finally:
                mode["prefill"] = previous

        def ffn(hidden, ids):
            routed = ids.reshape(*hidden.shape[:-1], 1).to(hidden.dtype)
            return hidden + routed * (2 if mode["prefill"] else 1)

        block = SimpleNamespace(
            ffn=ffn,
            prefill_ffn=None,
            hc_pre_graph=lambda hidden, *_args: (hidden, None, None),
            attn_norm=SimpleNamespace(weight=None),
            ffn_norm=SimpleNamespace(weight=None),
            **dict.fromkeys(
                (
                    "hc_attn_fn",
                    "hc_attn_scale",
                    "hc_attn_base",
                    "hc_ffn_fn",
                    "hc_ffn_scale",
                    "hc_ffn_base",
                )
            ),
        )
        monkeypatch.setattr(
            native, "mhc_post", lambda x, residual, *_args: x + residual
        )

        def forward(hidden, ids):
            return native.Block.forward_attention(block, hidden, ids, lambda x: x)

        runner, (callback,), replays = cpu_prefill_regions(
            (ffn,), 4, token_ids=True, capture_context=prefill_context
        )
        for ids in (torch.tensor([[2, 7, 3]]), torch.tensor([[8, 1]])):
            hidden = torch.ones((*ids.shape, 4), dtype=torch.bfloat16)
            block.prefill_ffn = None
            eager_decode = forward(hidden, ids)
            with prefill_context():
                eager_prefill = forward(hidden, ids)
            block.prefill_ffn = callback
            torch.testing.assert_close(forward(hidden, ids), eager_decode)
            with runner._prefill_graphs():
                actual = forward(hidden, ids)
            torch.testing.assert_close(actual, eager_prefill)
            assert mode["prefill"] is False
        assert replays == [(4, 1), (2, 1)]

    def test_cuda_graphs_use_shared_static_lifecycle(self) -> None:
        from resin.backend.cuda.runner import CudaRunner

        parameters = inspect.signature(DeepSeekV4CudaRunner.__init__).parameters
        source = inspect.getsource(DeepSeekV4CudaRunner)

        assert "cuda_graph_decode_shapes" in parameters
        assert issubclass(DeepSeekV4CudaRunner, CudaRunner)
        assert hasattr(DeepSeekV4CudaRunner, "cuda_graph_stats")
        assert hasattr(DeepSeekV4CudaRunner, "capture_cuda_graphs")
        assert "self._configure_cuda_graphs(" in source
        assert "pad_up_to=0" in source
        assert "capture_context=self._prefill_compute_context" in source
        assert "def capture_cuda_graphs" not in source
        assert "CudaGraphProgram(" in source
        assert "CUDAGraph" not in source
        assert ".replay()" not in source

    def test_graph_capture_inputs_stage_position_and_sampling(self) -> None:
        inputs = DeepSeekV4CudaRunner._graph_capture_inputs((3, 128), 256)

        assert inputs["token_ids"] == (0, 0, 0)
        assert inputs["positions"] == (1, 1, 1)
        assert inputs["sampling_top_ks"] == (256, 256, 256)
        assert inputs["sampling_stochastic_mask"] == (False, False, False)
