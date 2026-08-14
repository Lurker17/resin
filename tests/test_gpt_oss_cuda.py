from __future__ import annotations

import importlib.util
import inspect
from types import SimpleNamespace

import pytest

from resin import BatchPhase
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import TorchSampler
from resin.models.gpt_oss.cuda import (
    GptOssCudaRunner,
    _route,
    _router_logits,
    plan_gpt_oss_kv_capacity,
)
from resin.models.gpt_oss.model import Attention
from resin.models.gpt_oss.plan import (
    PREFILL_CHUNK_SIZE,
    WINDOW_TABLE_WIDTH,
    build_attention_batches,
)


def _group_plan(group_index, rows, first_logical_blocks, positions, indptr):
    block_indptr = [0]
    block_ids = []
    append_block_ids = []
    append_block_offsets = []
    for sequence_index, row in enumerate(rows):
        block_ids.extend(row)
        block_indptr.append(len(block_ids))
        first = first_logical_blocks[sequence_index]
        token_start, token_end = indptr[sequence_index : sequence_index + 2]
        for position in positions[token_start:token_end]:
            append_block_ids.append(row[position // 128 - first])
            append_block_offsets.append(position % 128)
    return SimpleNamespace(
        group_index=group_index,
        block_indptr=block_indptr,
        block_ids=block_ids,
        first_logical_blocks=list(first_logical_blocks),
        append_block_ids=append_block_ids,
        append_block_offsets=append_block_offsets,
    )


def _flat_plan(
    context_lengths,
    query_lengths,
    *,
    phase=BatchPhase.PREFILL,
    cache_pool_id=41,
):
    sequence_count = len(context_lengths)
    assert len(query_lengths) == sequence_count
    sequence_indptr = [0]
    positions = []
    history_rows = []
    window_rows = []
    window_first = []
    for sequence_index, (context, query) in enumerate(
        zip(context_lengths, query_lengths, strict=True)
    ):
        assert context >= 0 and query > 0
        positions.extend(range(context, context + query))
        sequence_indptr.append(len(positions))
        post_length = context + query
        last_block = (post_length - 1) // 128
        history_rows.append(
            tuple(
                1 + sequence_index * 128 + logical for logical in range(last_block + 1)
            )
        )
        first = max(0, last_block + 1 - WINDOW_TABLE_WIDTH)
        window_first.append(first)
        window_rows.append(
            tuple(
                1025 + sequence_index * 128 + logical % WINDOW_TABLE_WIDTH
                for logical in range(first, last_block + 1)
            )
        )

    history = _group_plan(
        0,
        history_rows,
        [0] * sequence_count,
        positions,
        sequence_indptr,
    )
    window = _group_plan(
        1,
        window_rows,
        window_first,
        positions,
        sequence_indptr,
    )
    return SimpleNamespace(
        cache_pool_id=cache_pool_id,
        cache_groups=(history, window),
        token_ids=list(range(1, len(positions) + 1)),
        positions=positions,
        sequence_ids=list(range(100, 100 + sequence_count)),
        num_sequences=sequence_count,
        sequence_indptr=sequence_indptr,
        context_lengths=list(context_lengths),
        phase=phase,
        sample_sequence_indices=list(range(sequence_count)),
    )


def _prefill_plan():
    return _flat_plan([0, 130], [128, 5])


def _decode_plan():
    return _flat_plan(
        [130, 260],
        [1, 1],
        phase=BatchPhase.DECODE,
    )


class TestGptOssCudaNativePlan:
    @staticmethod
    def _binding_runner():
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner.cache_layout = object()
        binding = CachePoolBinding(runner.cache_layout, resource="GPT-OSS KV")
        runner._kv = SimpleNamespace(
            bind=binding.bind,
            check=lambda plan: binding.check(plan.cache_pool_id),
        )
        return runner

    def test_cuda_graphs_use_the_shared_static_step_lifecycle(self) -> None:
        parameters = inspect.signature(GptOssCudaRunner.__init__).parameters
        source = inspect.getsource(GptOssCudaRunner)
        assert "cuda_graph_decode_shapes" in parameters
        assert hasattr(GptOssCudaRunner, "cuda_graph_stats")
        assert hasattr(GptOssCudaRunner, "capture_cuda_graphs")
        assert "CudaGraphCache(" in source
        assert "CUDAGraph" not in source
        assert ".replay()" not in source

    def test_graph_capture_inputs_use_only_the_reserved_null_page(self) -> None:
        inputs = GptOssCudaRunner._graph_capture_inputs((2, 4))

        assert inputs["token_indices"] == (0, 1)
        assert inputs["sequence_indices"] == (0, 1)
        assert inputs["full_page_table"] == (0,) * 8
        assert inputs["sliding_page_table"] == (0,) * (2 * WINDOW_TABLE_WIDTH)
        assert inputs["full_append_slots"] == (0, 0)
        assert inputs["sliding_append_slots"] == (0, 0)

    def test_kv_capacity_plan_uses_static_fraction_and_swa_ratio(self) -> None:
        gib = 1024**3
        plan = plan_gpt_oss_kv_capacity(
            100 * gib,
            55 * gib,
            0.75,
            swa_full_tokens_ratio=0.8,
        )

        assert plan.static_memory_bytes == 75 * gib
        assert plan.kv_budget_bytes == 20 * gib
        assert plan.bytes_per_parent == 4_718_592
        assert plan.parent_capacity == 4_550
        assert plan.full_parent_target + plan.swa_parent_target == plan.parent_capacity
        assert plan.full_token_target == plan.full_parent_target * 128
        assert plan.swa_token_target == plan.swa_parent_target * 128

        tp4 = plan_gpt_oss_kv_capacity(
            100 * gib,
            55 * gib,
            0.75,
            tensor_parallel_size=4,
        )
        assert tp4.bytes_per_parent == plan.bytes_per_parent // 4
        assert tp4.parent_capacity > plan.parent_capacity

    def test_runner_exposes_only_the_flat_grouped_cache_path(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        parameters = inspect.signature(GptOssCudaRunner.__init__).parameters
        assert GptOssCudaRunner.supports_continuation_prefill is True
        assert GptOssCudaRunner.prefill_chunk_size == 8192
        assert GptOssCudaRunner.max_batch_tokens == 8192
        assert "bind_cache_pool" in source
        assert "CacheGroupLayout(_PAGE_SIZE)" in source
        assert 'retention="window"' in source
        assert "build_attention_batches" in source
        assert "tensor_parallel" in parameters
        assert "fast_allreduce" in parameters
        assert source.count("self.tensor_parallel.sum(") == 1
        assert source.count("self._allreduce_residual_rmsnorm(") == 2

    def test_cache_pool_binding_is_exact_idempotent_and_permanent(self) -> None:
        runner = self._binding_runner()

        assert runner.bind_cache_pool(41, runner.cache_layout) == 41
        assert runner.bind_cache_pool(41, runner.cache_layout) == 41
        fresh = self._binding_runner()
        with pytest.raises(ValueError, match="frozen layout object"):
            fresh.bind_cache_pool(41, object())
        with pytest.raises(ValueError, match="different cache pool"):
            runner.bind_cache_pool(42, runner.cache_layout)
        for invalid in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                fresh = self._binding_runner()
                fresh.bind_cache_pool(invalid, fresh.cache_layout)

    def test_plan_binding_requires_the_one_bound_pool(self) -> None:
        runner = self._binding_runner()
        with pytest.raises(ValueError, match="not bound"):
            runner._validate_plan_binding(_prefill_plan())
        runner.bind_cache_pool(41, runner.cache_layout)
        runner._validate_plan_binding(_prefill_plan())
        with pytest.raises(ValueError, match="does not match"):
            runner._validate_plan_binding(_flat_plan([0], [1], cache_pool_id=42))

    def test_prefill_tables_are_derived_from_history_and_window_csr(self) -> None:
        batches, decode = build_attention_batches(_prefill_plan())

        assert decode is None
        assert len(batches) == 1
        batch = batches[0]
        assert batch.phase == BatchPhase.PREFILL
        assert batch.sequence_indices == (0, 1)
        assert batch.query_lengths == (128, 5)
        assert batch.token_indices[:3] == (0, 1, 2)
        assert batch.token_indices[-3:] == (130, 131, 132)
        assert batch.full_width == 2
        assert batch.full_table == (1, 1, 129, 130)
        assert batch.full_lengths == (128, 135)
        assert batch.sliding_width == WINDOW_TABLE_WIDTH
        assert batch.sliding_table[:WINDOW_TABLE_WIDTH] == (1025,) * WINDOW_TABLE_WIDTH
        assert batch.sliding_table[WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 2] == (
            1153,
            1154,
        )
        assert batch.sliding_lengths == (128, 135)
        assert batch.full_append_slots[0] == 128
        assert batch.full_append_slots[-1] == 130 * 128 + 6
        assert batch.sliding_append_slots[-1] == 1154 * 128 + 6

    def test_one_token_continuation_remains_explicit_prefill(self) -> None:
        plan = _flat_plan([260], [1])
        prefill, decode = build_attention_batches(plan)

        assert decode is None
        assert len(prefill) == 1
        batch = prefill[0]
        assert batch.query_lengths == (1,)
        assert batch.token_indices == (0,)
        assert batch.full_table == (1, 2, 3)
        assert batch.full_lengths == (261,)
        assert batch.sliding_table[:3] == (1025, 1026, 1027)
        assert batch.sliding_lengths == (261,)

    def test_graph_padding_repeats_only_pages_owned_by_each_row(self) -> None:
        _, batch = build_attention_batches(_decode_plan())
        assert batch is not None
        assert GptOssCudaRunner._pad_page_table(batch, 4) == (
            1,
            2,
            1,
            1,
            129,
            130,
            131,
            129,
        )

    def test_decode_tables_and_append_slots_come_from_sparse_groups(self) -> None:
        _, batch = build_attention_batches(_decode_plan())

        assert batch is not None
        assert batch is not None
        assert batch.sequence_indices == (0, 1)
        assert batch.query_lengths == (1, 1)
        assert batch.full_width == 3
        assert batch.full_table == (1, 2, 1, 129, 130, 131)
        assert batch.full_lengths == (131, 261)
        assert batch.sliding_table[:3] == (1025, 1026, 1025)
        assert batch.sliding_table[WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 3] == (
            1153,
            1154,
            1155,
        )
        assert batch.sliding_lengths == (131, 261)
        assert batch.sliding_append_slots == (1026 * 128 + 2, 1155 * 128 + 4)
        assert batch.full_append_slots == (2 * 128 + 2, 131 * 128 + 4)

    def test_frozen_append_destinations_are_a_validation_oracle(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[1].append_block_ids[0] += 1
        with pytest.raises(ValueError, match="append destinations"):
            build_attention_batches(plan)

    def test_group_order_suffix_and_parent_ownership_fail_closed(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[0].group_index = 1
        with pytest.raises(ValueError, match="ordered"):
            build_attention_batches(plan)

        plan = _flat_plan([9000], [1])
        plan.cache_groups[1].first_logical_blocks[0] = 0
        with pytest.raises(ValueError, match="active suffix"):
            build_attention_batches(plan)

        plan = _flat_plan([0], [1])
        plan.cache_groups[1].block_ids[0] = plan.cache_groups[0].block_ids[0]
        plan.cache_groups[1].append_block_ids[0] = plan.cache_groups[0].block_ids[0]
        with pytest.raises(ValueError, match="cannot address"):
            build_attention_batches(plan)

        plan = _flat_plan([0], [1])
        with pytest.raises(ValueError, match="outside"):
            build_attention_batches(plan, parent_capacity=64)

    def test_multi_page_prefill_is_validated_before_device_work(self) -> None:
        prefill, decode = build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE]))

        assert decode is None
        assert prefill[0].query_lengths == (PREFILL_CHUNK_SIZE,)
        assert prefill[0].full_width == PREFILL_CHUNK_SIZE // 128
        assert prefill[0].sliding_width == WINDOW_TABLE_WIDTH
        assert len(set(prefill[0].sliding_append_slots)) == PREFILL_CHUNK_SIZE

        with pytest.raises(ValueError, match="exceeds 8192"):
            build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE + 1]))

    def test_failed_eager_work_must_quiesce_or_be_post_accept(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        synchronized = []
        runner.device = "cuda:0"
        runner._torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda device: synchronized.append(device))
        )
        runner._synchronize_failed_eager_forward()
        assert synchronized == ["cuda:0"]

        runner._torch.cuda.synchronize = lambda device: (_ for _ in ()).throw(
            RuntimeError(f"{device} synchronize failed")
        )
        with pytest.raises(RuntimeError, match="quarantined") as caught:
            runner._synchronize_failed_eager_forward()
        assert getattr(caught.value, "_resin_post_accept", None) is True

    def test_cuda_attention_has_no_fp32_promotion_or_kv_head_expansion(self) -> None:
        source = inspect.getsource(GptOssCudaRunner._attention)
        assert ".float(" not in source
        assert "repeat_interleave" not in source
        assert "repeat(" not in source
        assert source.count("qkv_proj(") == 1
        assert "attention.q_proj(" not in source
        assert "attention.k_proj(" not in source
        assert "attention.v_proj(" not in source
        assert "index_copy_" in source

    def test_runner_has_no_python_request_cache_or_page_allocator(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        assert "_request_pages" not in source
        assert "_request_lengths" not in source
        assert "_free_pages" not in source
        assert "_logical_pages" not in source
        assert "_sequence_cache" not in source

    def test_one_zeroed_parent_allocation_backs_both_group_views(self) -> None:
        from resin.backend.cuda.paged_kv import PagedKvArena

        source = inspect.getsource(GptOssCudaRunner.__init__)
        arena_source = inspect.getsource(PagedKvArena.__init__)
        assert "self._kv = PagedKvArena(" in source
        assert arena_source.count("torch.zeros(") == 1
        assert "self.capacity + 1" in arena_source
        assert "self.keys = self.storage[0]" in arena_source
        assert "self.values = self.storage[1]" in arena_source


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"), reason="PyTorch is not installed"
)
class TestGptOssCudaTinyOracle:
    def test_flashinfer_allreduce_fuses_residual_and_rmsnorm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._allreduce_workspace = SimpleNamespace(backend="trtllm")
        partial = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        residual = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        weight = torch.randn(8, dtype=torch.bfloat16)
        eps = 1e-5
        calls = []

        def fused(**kwargs):
            calls.append(kwargs)
            combined = kwargs["input"] + kwargs["residual_in"]
            kwargs["residual_out"].copy_(combined)
            normalized = combined.float()
            normalized *= torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + kwargs["rms_eps"]
            )
            kwargs["norm_out"].copy_((normalized * kwargs["rms_gamma"]).bfloat16())
            return kwargs["norm_out"]

        monkeypatch.setattr("resin.models.gpt_oss.cuda.allreduce_fusion", fused)
        normalized, combined = runner._allreduce_residual_rmsnorm(
            partial, residual, weight, eps
        )

        expected_combined = partial + residual
        expected_normalized = expected_combined.float()
        expected_normalized *= torch.rsqrt(
            expected_normalized.square().mean(-1, keepdim=True) + eps
        )
        expected_normalized = (expected_normalized * weight).bfloat16()
        torch.testing.assert_close(combined, expected_combined)
        torch.testing.assert_close(normalized, expected_normalized)
        assert "use_oneshot" not in calls[0]
        assert not calls[0]["launch_with_pdl"]
        assert calls[0]["trigger_completion_at_end"]

    def test_packed_qkv_matches_three_loaded_projections(self) -> None:
        import torch

        attention = SimpleNamespace(
            q_proj=torch.nn.Linear(8, 12, bias=True),
            k_proj=torch.nn.Linear(8, 4, bias=True),
            v_proj=torch.nn.Linear(8, 4, bias=True),
        )
        hidden = torch.randn(7, 8)
        expected = torch.cat(
            (
                attention.q_proj(hidden),
                attention.k_proj(hidden),
                attention.v_proj(hidden),
            ),
            dim=-1,
        )

        Attention.pack_qkv_projection(attention)

        torch.testing.assert_close(attention.qkv_proj(hidden), expected)
        assert not hasattr(attention, "q_proj")
        assert not hasattr(attention, "k_proj")
        assert not hasattr(attention, "v_proj")

    def test_router_uses_tinygemm_through_128_rows_then_linear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        router = SimpleNamespace(
            weight=torch.randn(16, 64, dtype=torch.bfloat16),
            bias=torch.randn(16, dtype=torch.bfloat16),
        )
        calls = []

        def tinygemm(input, weight, output, bias, *, use_pdl):
            calls.append((input.shape[0], use_pdl))
            output.copy_(torch.nn.functional.linear(input, weight, bias))

        monkeypatch.setattr("resin.models.gpt_oss.cuda.tinygemm_bf16", tinygemm)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128):
            hidden = torch.randn(rows, 64, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(hidden, router.weight, router.bias)
            torch.testing.assert_close(
                _router_logits(router, hidden), expected, rtol=0, atol=0
            )

        hidden = torch.randn(129, 64, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(hidden, router.weight, router.bias)
        torch.testing.assert_close(
            _router_logits(router, hidden), expected, rtol=0, atol=0
        )

        assert calls == [
            (1, False),
            (2, False),
            (4, False),
            (8, False),
            (16, False),
            (32, False),
            (64, False),
            (128, False),
        ]

    def test_official_top4_router_uses_cutile_and_other_layouts_fall_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        official = SimpleNamespace(top_k=4)
        logits = torch.randn(32, 128, dtype=torch.bfloat16)
        expected_values, expected_indices = torch.topk(logits, 4, dim=-1, sorted=True)
        expected_weights = torch.softmax(expected_values, dim=-1)
        calls = []

        def top4_softmax(actual_logits):
            calls.append(actual_logits.shape)
            return expected_indices.to(torch.int32), expected_weights.to(torch.float32)

        with monkeypatch.context() as context:
            context.setattr("resin.models.gpt_oss.cuda.top4_softmax", top4_softmax)
            indices, weights = _route(official, logits)

        assert calls == [torch.Size([32, 128])]
        torch.testing.assert_close(indices, expected_indices.to(torch.int32))
        torch.testing.assert_close(weights, expected_weights.to(torch.float32))

        generic = SimpleNamespace(top_k=2)
        indices, weights = _route(generic, logits[:, :16])
        values, expected_indices = torch.topk(logits[:, :16], 2, dim=-1, sorted=True)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, torch.softmax(values, dim=-1))

        large_logits = torch.randn(8192, 128, dtype=torch.bfloat16)
        large_values, large_indices = torch.topk(large_logits, 4, dim=-1, sorted=True)
        fused_calls = []

        def fused(actual_logits):
            fused_calls.append(actual_logits)
            return large_indices.int(), torch.softmax(large_values, dim=-1)

        with monkeypatch.context() as context:
            context.setattr("resin.models.gpt_oss.cuda.top4_softmax", fused)
            _route(official, large_logits)
        assert len(fused_calls) == 1
        assert fused_calls[0] is large_logits

    def test_static_decode_step_restores_rust_sequence_order_before_sampling(
        self,
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = torch
        runner._sampler = TorchSampler(torch)
        runner.device = "cpu"
        runner.hidden_size = 3
        runner._model = SimpleNamespace(lm_head=lambda hidden: hidden)
        runner._forward_batch = lambda *_args: torch.tensor(
            [[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]], dtype=torch.bfloat16
        )
        step = runner._new_decode_graph_step((2, 1))
        step.inputs["sequence_indices"].copy_(torch.tensor([1, 0]))

        step.launch()

        assert step.output.tolist() == [0, 1]

    def test_rotary_path_matches_reference_formula_without_dtype_promotion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = torch
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -3.0]],
                [[-2.0, 1.0, 0.25, 0.75], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=torch.bfloat16,
        )
        cos = torch.tensor([[[1.0, 0.5], [0.25, 1.0]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[0.0, 0.5], [1.0, 0.0]]], dtype=torch.bfloat16)

        def apply_rotary(hidden, cos, sin, *, interleaved, inplace):
            assert not interleaved
            assert inplace
            first, second = hidden.chunk(2, dim=-1)
            hidden.copy_(
                torch.cat(
                    (
                        first * cos.unsqueeze(0).unsqueeze(2)
                        - second * sin.unsqueeze(0).unsqueeze(2),
                        second * cos.unsqueeze(0).unsqueeze(2)
                        + first * sin.unsqueeze(0).unsqueeze(2),
                    ),
                    dim=-1,
                )
            )
            return hidden

        first, second = hidden.chunk(2, dim=-1)
        expected = torch.cat(
            (
                first * cos.squeeze(0).unsqueeze(1)
                - second * sin.squeeze(0).unsqueeze(1),
                second * cos.squeeze(0).unsqueeze(1)
                + first * sin.squeeze(0).unsqueeze(1),
            ),
            dim=-1,
        )

        monkeypatch.setattr("resin.models.gpt_oss.cuda.apply_rotary", apply_rotary)
        actual = runner._apply_yarn_rotary(hidden, cos, sin)

        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
