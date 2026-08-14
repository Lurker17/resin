from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.parallel import FlashInferAllReduce
from resin.models.deepseek_v4.checkpoint import resolve_model_path
from resin.models.deepseek_v4.config import DeepSeekV4Config
from resin.models.deepseek_v4.cuda import DeepSeekV4FlashCudaRunner
from resin.models.deepseek_v4.weights import (
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

    def test_rejects_mismatched_layer_compression(self) -> None:
        values = _release_config()
        values["compress_ratios"] = [0]

        with pytest.raises(ValueError, match="one entry per target layer"):
            DeepSeekV4Config.from_hugging_face(values)

    def test_rejects_incompatible_tensor_parallel_size(self) -> None:
        config = DeepSeekV4Config.from_hugging_face(_release_config())

        with pytest.raises(ValueError, match="TP=3"):
            config.validate_tensor_parallel(3)


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
        parameters = inspect.signature(DeepSeekV4FlashCudaRunner.__init__).parameters

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
        from resin.models.deepseek_v4 import weights

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

    def test_grouped_expert_storage_preserves_checkpoint_parameter_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cutile = ModuleType("resin.models.deepseek_v4.cutile")
        for name in (
            "act_quant",
            "compress_decode",
            "fp4_act_quant",
            "hadamard_transform",
            "qk_norm_rope_store",
            "sparse_attn",
            "top6_sqrt_softplus",
        ):
            setattr(cutile, name, None)
        flashinfer = ModuleType("flashinfer")
        flashinfer_mhc = ModuleType("flashinfer.mhc")
        flashinfer_mhc.mhc_post = None
        flashinfer_mhc.mhc_pre_big_fuse_with_prenorm = None
        gemm = ModuleType("resin.models.deepseek_v4.gemm")
        for name in (
            "fp4_fused_moe",
            "fp4_gemm",
            "fp8_gemm",
            "prepare_fp4_fused_moe_weights",
        ):
            setattr(gemm, name, None)
        attention = ModuleType("resin.models.deepseek_v4.attention")
        attention.sparse_decode_attention = None
        attention.sparse_topk_lengths = None
        quack = ModuleType("quack")
        quack.rmsnorm = None
        monkeypatch.setitem(sys.modules, cutile.__name__, cutile)
        monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
        monkeypatch.setitem(sys.modules, attention.__name__, attention)
        monkeypatch.setitem(sys.modules, flashinfer.__name__, flashinfer)
        monkeypatch.setitem(sys.modules, flashinfer_mhc.__name__, flashinfer_mhc)
        monkeypatch.setitem(sys.modules, quack.__name__, quack)

        module_path = (
            Path(__file__).parents[1] / "src/resin/models/deepseek_v4/model.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_resin_test_deepseek_v4_model", module_path
        )
        assert spec is not None and spec.loader is not None
        native_model = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, native_model)
        spec.loader.exec_module(native_model)
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

        moe = native_model.MoE(0, args)

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
            "quack": quack,
            "quack.blockscaled": quack_blockscaled,
            "quack.gemm_interface": quack_gemm,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)
        module_name = "resin.models.deepseek_v4.gemm"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        return importlib.import_module(module_name)

    def test_fp8_uses_flashinfer_groupwise_layout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def flashinfer(*args: object, **kwargs: object) -> torch.Tensor:
            calls.append((args, kwargs))
            return torch.zeros(2, 128, dtype=torch.bfloat16)

        monkeypatch.setattr(deepseek_gemm, "gemm_fp8_nt_groupwise", flashinfer)
        left = torch.zeros(2, 128, dtype=torch.float8_e4m3fn)
        left_scales = torch.ones(2, 1, dtype=torch.float8_e8m0fnu)
        right = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        right_scales = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)

        output = deepseek_gemm.fp8_gemm(
            left,
            left_scales,
            right,
            right_scales,
            torch.float8_e8m0fnu,
        )

        args, kwargs = calls[0]
        assert output.shape == (2, 128)
        assert args[2].dtype == torch.float32
        assert args[3].dtype == torch.float32
        assert kwargs["scale_major_mode"] == "K"
        assert kwargs["scale_granularity_mnk"] == (1, 128, 128)
        assert kwargs["backend"] == "cutlass"

    def test_fp4_uses_quack_and_repeats_activation_scale_descriptors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deepseek_gemm = self._module(monkeypatch)

        packed_inputs: list[torch.Tensor] = []
        gemm_calls: list[tuple[object, object, dict[str, object]]] = []

        class Operand:
            def __init__(self, data: torch.Tensor, scale: torch.Tensor) -> None:
                self.data = data
                self.scale = scale

            @classmethod
            def from_parts(
                cls,
                data: torch.Tensor,
                scale: torch.Tensor,
                _format: str,
                **_kwargs: object,
            ) -> "Operand":
                return cls(data, scale)

            @property
            def mT(self) -> "Operand":
                return self

        def pack_scales(scales: torch.Tensor) -> torch.Tensor:
            packed_inputs.append(scales.clone())
            return scales

        def quack_gemm(
            activation: object, weight: object, **kwargs: object
        ) -> torch.Tensor:
            gemm_calls.append((activation, weight, kwargs))
            return torch.zeros(2, 128, dtype=torch.bfloat16)

        monkeypatch.setattr(deepseek_gemm, "BlockScaledOperand", Operand)
        monkeypatch.setattr(deepseek_gemm, "quack_gemm", quack_gemm)
        monkeypatch.setattr(
            deepseek_gemm, "pack_scale_2d_to_blocked_contig", pack_scales
        )
        left = torch.zeros(2, 128, dtype=torch.float8_e4m3fn)
        left_scales = torch.ones(2, 1, dtype=torch.float8_e8m0fnu)
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
        assert packed_inputs[0].shape == (2, 4)
        assert torch.equal(packed_inputs[0], left_scales.repeat_interleave(4, dim=-1))
        assert packed_inputs[1] is not right_scales
        assert packed_inputs[1].shape == (128, 4)
        assert gemm_calls[0][2] == {
            "out_dtype": torch.get_default_dtype(),
            "tuned": True,
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
        left_scales = torch.arange(1, 4, dtype=torch.float32).to(torch.float8_e8m0fnu)[
            :, None
        ]
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
        expected_scales = left_scales.repeat_interleave(4, dim=-1)
        assert torch.equal(
            kwargs["hidden_states_scale"].view(torch.uint8),  # type: ignore[union-attr]
            expected_scales.view(torch.uint8),
        )
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
        module_name = "resin.models.deepseek_v4.attention"
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


class TestDeepSeekV4BatchPlan:
    @staticmethod
    def _runner() -> DeepSeekV4FlashCudaRunner:
        runner = DeepSeekV4FlashCudaRunner.__new__(DeepSeekV4FlashCudaRunner)
        runner.vocab_size = 32
        runner.max_seq_len = 16
        runner.max_batch_size = 3
        runner._active_sequence_ids = []

        class StateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("kv_cache", torch.zeros(3, 2))
                self.register_buffer("kv_state", torch.zeros(3, 2))
                self.register_buffer("score_state", torch.full((3, 2), float("-inf")))

            def forward(self, input_ids: torch.Tensor, _start_pos: int) -> torch.Tensor:
                markers = input_ids[:, :1].to(torch.float32)
                rows = input_ids.shape[0]
                self.kv_cache[:rows].copy_(markers.expand(rows, 2))
                self.kv_state[:rows].copy_((markers + 10).expand(rows, 2))
                self.score_state[:rows].copy_((markers + 20).expand(rows, 2))
                return torch.nn.functional.one_hot(input_ids[:, -1], num_classes=32).to(
                    torch.float32
                )

        runner._model = StateModel()
        runner.device = torch.device("cpu")
        return runner

    def test_cuda_graphs_use_shared_static_lifecycle(self) -> None:
        parameters = inspect.signature(DeepSeekV4FlashCudaRunner.__init__).parameters
        source = inspect.getsource(DeepSeekV4FlashCudaRunner)

        assert "cuda_graph_decode_shapes" in parameters
        assert hasattr(DeepSeekV4FlashCudaRunner, "cuda_graph_stats")
        assert hasattr(DeepSeekV4FlashCudaRunner, "capture_cuda_graphs")
        assert "CudaGraphCache(" in source
        assert "CUDAGraph" not in source
        assert ".replay()" not in source

    def test_graph_capture_inputs_stage_position_and_sampling(self) -> None:
        inputs = DeepSeekV4FlashCudaRunner._graph_capture_inputs((3, 128), 256)

        assert inputs["token_ids"] == (0, 0, 0)
        assert inputs["positions"] == (1, 1, 1)
        assert inputs["sampling_top_ks"] == (256, 256, 256)
        assert inputs["sampling_stochastic_mask"] == (False, False, False)

    def test_ragged_prefill_then_decode_owns_compact_rows(self) -> None:
        runner = self._runner()
        prefill = BatchPlan(
            [2, 3, 4, 5, 6],
            [0, 1, 0, 1, 2],
            [7, 8],
            [0, 2, 5],
            [0, 0],
            phase=BatchPhase.PREFILL,
        )
        decode = BatchPlan(
            [9, 10],
            [2, 3],
            [7, 8],
            [0, 1, 2],
            [2, 3],
            phase=BatchPhase.DECODE,
        )

        assert runner._validate_plan(prefill) == ((0, 2), (2, 5))
        runner._active_sequence_ids = [7, 8]
        assert runner._validate_plan(decode) == ((0, 1), (1, 2))

        runner.release([7])
        assert runner._active_sequence_ids == [8]

    def test_rejects_continuation_prefill_and_foreign_decode(self) -> None:
        runner = self._runner()
        continuation = BatchPlan(
            [4, 5],
            [2, 3],
            [7],
            [0, 2],
            [2],
            phase=BatchPhase.PREFILL,
        )

        with pytest.raises(ValueError, match="continuation prefill"):
            runner._validate_plan(continuation)
        runner._active_sequence_ids = [7]
        foreign = BatchPlan(
            [4],
            [2],
            [8],
            [0, 1],
            [2],
            phase=BatchPhase.DECODE,
        )
        with pytest.raises(ValueError, match="do not own"):
            runner._validate_plan(foreign)

    def test_release_compacts_every_sequence_state_buffer(self) -> None:
        runner = self._runner()
        runner._active_sequence_ids = [7, 8, 9]
        runner._model.kv_cache.copy_(torch.tensor([[7, 7], [8, 8], [9, 9]]))
        runner._model.kv_state.copy_(torch.tensor([[17, 17], [18, 18], [19, 19]]))
        runner._model.score_state.copy_(torch.tensor([[27, 27], [28, 28], [29, 29]]))

        runner.release([8])

        assert runner._active_sequence_ids == [7, 9]
        assert torch.equal(
            runner._model.kv_cache, torch.tensor([[7, 7], [9, 9], [0, 0]])
        )
        assert torch.equal(
            runner._model.kv_state, torch.tensor([[17, 17], [19, 19], [0, 0]])
        )
        assert torch.equal(
            runner._model.score_state[:2], torch.tensor([[27, 27], [29, 29]])
        )
        assert torch.isneginf(runner._model.score_state[2]).all()

    def test_ragged_prefill_places_each_sequence_in_its_compact_row(self) -> None:
        runner = self._runner()
        plan = BatchPlan(
            [2, 3, 8, 9, 10],
            [0, 1, 0, 1, 2],
            [7, 8],
            [0, 2, 5],
            [0, 0],
            phase=BatchPhase.PREFILL,
        )

        logits = runner._prefill(plan, ((0, 2), (2, 5)))

        assert logits.argmax(dim=-1).tolist() == [3, 10]
        assert torch.equal(runner._model.kv_cache[:2], torch.tensor([[2, 2], [8, 8]]))
        assert torch.equal(
            runner._model.kv_state[:2], torch.tensor([[12, 12], [18, 18]])
        )
