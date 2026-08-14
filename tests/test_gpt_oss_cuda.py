from __future__ import annotations

import importlib.util
import inspect
import math
import unittest
from types import SimpleNamespace

from resin import BatchPhase
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import GreedySampler
from resin.models.gpt_oss.cuda import (
    GptOssCudaRunner,
    _invoke_fa4_b22,
    plan_gpt_oss_kv_capacity,
)
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


class GptOssCudaNativePlanTests(unittest.TestCase):
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
        self.assertIn("cuda_graph_decode_shapes", parameters)
        self.assertTrue(hasattr(GptOssCudaRunner, "cuda_graph_stats"))
        self.assertTrue(hasattr(GptOssCudaRunner, "capture_cuda_graphs"))
        self.assertIn("CudaGraphCache(", source)
        self.assertNotIn("CUDAGraph", source)
        self.assertNotIn(".replay()", source)

    def test_graph_capture_inputs_use_only_the_reserved_null_page(self) -> None:
        inputs = GptOssCudaRunner._graph_capture_inputs((2, 4))

        self.assertEqual(inputs["token_indices"], (0, 1))
        self.assertEqual(inputs["sequence_indices"], (0, 1))
        self.assertEqual(inputs["full_page_table"], (0,) * 8)
        self.assertEqual(inputs["sliding_page_table"], (0,) * (2 * WINDOW_TABLE_WIDTH))
        self.assertEqual(inputs["full_append_slots"], (0, 0))
        self.assertEqual(inputs["sliding_append_slots"], (0, 0))

    def test_kv_capacity_plan_uses_static_fraction_and_swa_ratio(self) -> None:
        gib = 1024**3
        plan = plan_gpt_oss_kv_capacity(
            100 * gib,
            55 * gib,
            0.75,
            swa_full_tokens_ratio=0.8,
        )

        self.assertEqual(plan.static_memory_bytes, 75 * gib)
        self.assertEqual(plan.kv_budget_bytes, 20 * gib)
        self.assertEqual(plan.bytes_per_parent, 4_718_592)
        self.assertEqual(plan.parent_capacity, 4_550)
        self.assertEqual(
            plan.full_parent_target + plan.swa_parent_target,
            plan.parent_capacity,
        )
        self.assertEqual(plan.full_token_target, plan.full_parent_target * 128)
        self.assertEqual(plan.swa_token_target, plan.swa_parent_target * 128)

    def test_runner_exposes_only_the_flat_grouped_cache_path(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        self.assertIs(GptOssCudaRunner.supports_continuation_prefill, True)
        self.assertEqual(GptOssCudaRunner.prefill_chunk_size, 8192)
        self.assertEqual(GptOssCudaRunner.max_batch_tokens, 8192)
        self.assertIn("bind_cache_pool", source)
        self.assertIn("CacheGroupLayout(_PAGE_SIZE)", source)
        self.assertIn('retention="window"', source)
        self.assertIn("build_attention_batches", source)

    def test_cache_pool_binding_is_exact_idempotent_and_permanent(self) -> None:
        runner = self._binding_runner()

        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        fresh = self._binding_runner()
        with self.assertRaisesRegex(ValueError, "frozen layout object"):
            fresh.bind_cache_pool(41, object())
        with self.assertRaisesRegex(ValueError, "different cache pool"):
            runner.bind_cache_pool(42, runner.cache_layout)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fresh = self._binding_runner()
                fresh.bind_cache_pool(invalid, fresh.cache_layout)

    def test_plan_binding_requires_the_one_bound_pool(self) -> None:
        runner = self._binding_runner()
        with self.assertRaisesRegex(ValueError, "not bound"):
            runner._validate_plan_binding(_prefill_plan())
        runner.bind_cache_pool(41, runner.cache_layout)
        runner._validate_plan_binding(_prefill_plan())
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner._validate_plan_binding(_flat_plan([0], [1], cache_pool_id=42))

    def test_prefill_tables_are_derived_from_history_and_window_csr(self) -> None:
        batches, decode = build_attention_batches(_prefill_plan())

        self.assertIsNone(decode)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.phase, BatchPhase.PREFILL)
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.query_lengths, (128, 5))
        self.assertEqual(batch.token_indices[:3], (0, 1, 2))
        self.assertEqual(batch.token_indices[-3:], (130, 131, 132))
        self.assertEqual(batch.full_width, 2)
        self.assertEqual(batch.full_table, (1, 1, 129, 130))
        self.assertEqual(batch.full_lengths, (128, 135))
        self.assertEqual(batch.sliding_width, WINDOW_TABLE_WIDTH)
        self.assertEqual(
            batch.sliding_table[:WINDOW_TABLE_WIDTH],
            (1025,) * WINDOW_TABLE_WIDTH,
        )
        self.assertEqual(
            batch.sliding_table[WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 2],
            (1153, 1154),
        )
        self.assertEqual(batch.sliding_lengths, (128, 135))
        self.assertEqual(batch.full_append_slots[0], 128)
        self.assertEqual(batch.full_append_slots[-1], 130 * 128 + 6)
        self.assertEqual(batch.sliding_append_slots[-1], 1154 * 128 + 6)

    def test_one_token_continuation_remains_explicit_prefill(self) -> None:
        plan = _flat_plan([260], [1])
        prefill, decode = build_attention_batches(plan)

        self.assertIsNone(decode)
        self.assertEqual(len(prefill), 1)
        batch = prefill[0]
        self.assertEqual(batch.query_lengths, (1,))
        self.assertEqual(batch.token_indices, (0,))
        self.assertEqual(batch.full_table, (1, 2, 3))
        self.assertEqual(batch.full_lengths, (261,))
        self.assertEqual(batch.sliding_table[:3], (1025, 1026, 1027))
        self.assertEqual(batch.sliding_lengths, (261,))

    def test_graph_padding_repeats_only_pages_owned_by_each_row(self) -> None:
        _, batch = build_attention_batches(_decode_plan())
        assert batch is not None
        self.assertEqual(
            GptOssCudaRunner._pad_page_table(batch, 4),
            (1, 2, 1, 1, 129, 130, 131, 129),
        )

    def test_decode_tables_and_append_slots_come_from_sparse_groups(self) -> None:
        _, batch = build_attention_batches(_decode_plan())

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.query_lengths, (1, 1))
        self.assertEqual(batch.full_width, 3)
        self.assertEqual(batch.full_table, (1, 2, 1, 129, 130, 131))
        self.assertEqual(batch.full_lengths, (131, 261))
        self.assertEqual(batch.sliding_table[:3], (1025, 1026, 1025))
        self.assertEqual(
            batch.sliding_table[WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 3],
            (1153, 1154, 1155),
        )
        self.assertEqual(batch.sliding_lengths, (131, 261))
        self.assertEqual(
            batch.sliding_append_slots,
            (1026 * 128 + 2, 1155 * 128 + 4),
        )
        self.assertEqual(batch.full_append_slots, (2 * 128 + 2, 131 * 128 + 4))

    def test_frozen_append_destinations_are_a_validation_oracle(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[1].append_block_ids[0] += 1
        with self.assertRaisesRegex(ValueError, "append destinations"):
            build_attention_batches(plan)

    def test_group_order_suffix_and_parent_ownership_fail_closed(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[0].group_index = 1
        with self.assertRaisesRegex(ValueError, "ordered"):
            build_attention_batches(plan)

        plan = _flat_plan([9000], [1])
        plan.cache_groups[1].first_logical_blocks[0] = 0
        with self.assertRaisesRegex(ValueError, "active suffix"):
            build_attention_batches(plan)

        plan = _flat_plan([0], [1])
        plan.cache_groups[1].block_ids[0] = plan.cache_groups[0].block_ids[0]
        plan.cache_groups[1].append_block_ids[0] = plan.cache_groups[0].block_ids[0]
        with self.assertRaisesRegex(ValueError, "cannot address"):
            build_attention_batches(plan)

        plan = _flat_plan([0], [1])
        with self.assertRaisesRegex(ValueError, "outside"):
            build_attention_batches(plan, parent_capacity=64)

    def test_multi_page_prefill_is_validated_before_device_work(self) -> None:
        prefill, decode = build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE]))

        self.assertIsNone(decode)
        self.assertEqual(prefill[0].query_lengths, (PREFILL_CHUNK_SIZE,))
        self.assertEqual(prefill[0].full_width, PREFILL_CHUNK_SIZE // 128)
        self.assertEqual(prefill[0].sliding_width, WINDOW_TABLE_WIDTH)
        self.assertEqual(len(set(prefill[0].sliding_append_slots)), PREFILL_CHUNK_SIZE)

        with self.assertRaisesRegex(ValueError, "exceeds 8192"):
            build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE + 1]))

    def test_failed_eager_work_must_quiesce_or_be_post_accept(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        synchronized = []
        runner.device = "cuda:0"
        runner._torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda device: synchronized.append(device))
        )
        runner._synchronize_failed_eager_forward()
        self.assertEqual(synchronized, ["cuda:0"])

        runner._torch.cuda.synchronize = lambda device: (_ for _ in ()).throw(
            RuntimeError(f"{device} synchronize failed")
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined") as caught:
            runner._synchronize_failed_eager_forward()
        self.assertIs(getattr(caught.exception, "_resin_post_accept", None), True)

    def test_cuda_attention_has_no_fp32_promotion_or_kv_head_expansion(self) -> None:
        source = inspect.getsource(GptOssCudaRunner._attention)
        self.assertNotIn(".float(", source)
        self.assertNotIn("repeat_interleave", source)
        self.assertNotIn("repeat(", source)
        self.assertIn("index_copy_", source)

    def test_runner_has_no_python_request_cache_or_page_allocator(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        self.assertNotIn("_request_pages", source)
        self.assertNotIn("_request_lengths", source)
        self.assertNotIn("_free_pages", source)
        self.assertNotIn("_logical_pages", source)
        self.assertNotIn("_sequence_cache", source)

    def test_one_zeroed_parent_allocation_backs_both_group_views(self) -> None:
        from resin.backend.cuda.paged_kv import PagedKvArena

        source = inspect.getsource(GptOssCudaRunner.__init__)
        arena_source = inspect.getsource(PagedKvArena.__init__)
        self.assertIn("self._kv = PagedKvArena(", source)
        self.assertEqual(arena_source.count("torch.zeros("), 1)
        self.assertIn("self.capacity + 1", arena_source)
        self.assertIn("self.keys = self.storage[0]", arena_source)
        self.assertIn("self.values = self.storage[1]", arena_source)


class GptOssCudaFa4CallTests(unittest.TestCase):
    def test_exact_b22_surface_for_full_and_sliding_attention(self) -> None:
        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return ("output", None)

        common = {
            "cu_seqlens_q": object(),
            "max_seqlen_q": 1,
            "page_table": object(),
            "seqused_k": object(),
            "min_seqlen_k": 129,
            "max_seqlen_k": 8193,
            "softmax_scale": 1 / math.sqrt(64),
            "learnable_sink": object(),
        }
        self.assertEqual(
            _invoke_fa4_b22(kernel, "q", "k", "v", sliding=False, **common),
            "output",
        )
        self.assertEqual(calls[-1][0], ("q", "k", "v"))
        full = calls[-1][1]
        self.assertTrue(full["causal"])
        self.assertEqual(full["window_size"], (None, None))
        self.assertTrue(full["pack_gqa"])
        self.assertEqual(full["num_splits"], 1)
        self.assertIs(full["learnable_sink"], common["learnable_sink"])
        self.assertEqual(full["softmax_scale"], 0.125)
        self.assertIsNone(full["cu_seqlens_k"])

        _invoke_fa4_b22(kernel, "q", "k", "v", sliding=True, **common)
        sliding = calls[-1][1]
        self.assertEqual(sliding["window_size"], (127, 0))
        self.assertTrue(sliding["causal"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class GptOssCudaTinyOracleTests(unittest.TestCase):
    def test_static_decode_step_restores_rust_sequence_order_before_sampling(
        self,
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._torch = torch
        runner._sampler = GreedySampler(torch)
        runner.device = "cpu"
        runner.hidden_size = 3
        runner._model = SimpleNamespace(lm_head=lambda hidden: hidden)
        runner._forward_batch = lambda *_args: torch.tensor(
            [[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]], dtype=torch.bfloat16
        )
        step = runner._new_decode_graph_step((2, 1))
        step.inputs["sequence_indices"].copy_(torch.tensor([1, 0]))

        step.launch()

        self.assertEqual(step.output.tolist(), [0, 1])

    def test_rotary_path_matches_reference_formula_without_dtype_promotion(
        self,
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

        actual = runner._apply_yarn_rotary(hidden, cos, sin)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
