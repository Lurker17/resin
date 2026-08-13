from __future__ import annotations

import importlib.util
import inspect
import math
import subprocess
import sys
import unittest
from types import SimpleNamespace

from resin import BatchPhase
from resin.backends.gpt_oss.cuda import (
    GptOssCudaRunner,
    _invoke_fa4_b22,
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
                1 + sequence_index * 16 + logical for logical in range(last_block + 1)
            )
        )
        first = max(0, last_block - 1)
        window_first.append(first)
        window_rows.append(
            tuple(
                129 + sequence_index * 16 + logical % 2
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


class GptOssCudaImportTests(unittest.TestCase):
    def test_import_is_lazy(self) -> None:
        script = """
import sys
import resin.backends.gpt_oss.cuda
for name in ('torch', 'flash_attn', 'flashinfer', 'transformers', 'tokenizers'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True)


class GptOssCudaNativePlanTests(unittest.TestCase):
    def test_runner_exposes_only_the_flat_grouped_cache_path(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        self.assertIs(GptOssCudaRunner.supports_continuation_prefill, True)
        self.assertIn("bind_cache_pool", source)
        self.assertIn("CacheGroupLayout(_PAGE_SIZE)", source)
        self.assertIn('retention="window"', source)
        self.assertIn("plan.cache_groups", source)

    def test_cache_pool_binding_is_exact_idempotent_and_permanent(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner.cache_layout = object()
        runner._cache_pool_id = None

        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        self.assertEqual(runner.bind_cache_pool(41, runner.cache_layout), 41)
        fresh = GptOssCudaRunner.__new__(GptOssCudaRunner)
        fresh.cache_layout = object()
        fresh._cache_pool_id = None
        with self.assertRaisesRegex(ValueError, "frozen cache_layout object"):
            fresh.bind_cache_pool(41, object())
        with self.assertRaisesRegex(ValueError, "different cache pool"):
            runner.bind_cache_pool(42, runner.cache_layout)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fresh = GptOssCudaRunner.__new__(GptOssCudaRunner)
                fresh.cache_layout = object()
                fresh._cache_pool_id = None
                fresh.bind_cache_pool(invalid, fresh.cache_layout)

    def test_plan_binding_requires_the_one_bound_pool(self) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        runner._cache_pool_id = None
        with self.assertRaisesRegex(ValueError, "not bound"):
            runner._validate_plan_binding(_prefill_plan())
        runner._cache_pool_id = 41
        runner._validate_plan_binding(_prefill_plan())
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner._validate_plan_binding(_flat_plan([0], [1], cache_pool_id=42))

    def test_prefill_tables_are_derived_from_history_and_window_csr(self) -> None:
        batches, decode = GptOssCudaRunner._attention_batches(_prefill_plan())

        self.assertIsNone(decode)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual((batch.phase, batch.round_index), (BatchPhase.PREFILL, 0))
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.q_lengths, (128, 5))
        self.assertEqual(batch.token_indices[:3], (0, 1, 2))
        self.assertEqual(batch.token_indices[-3:], (130, 131, 132))
        self.assertEqual(batch.full_page_table_width, 2)
        self.assertEqual(batch.full_page_table, (1, 1, 17, 18))
        self.assertEqual(batch.full_seqused_k, (128, 135))
        self.assertEqual(batch.sliding_page_table_width, 2)
        self.assertEqual(batch.sliding_page_table, (129, 129, 145, 146))
        self.assertEqual(batch.sliding_seqused_k, (128, 135))
        self.assertEqual(batch.full_append_slots[0], 128)
        self.assertEqual(batch.full_append_slots[-1], 18 * 128 + 6)
        self.assertEqual(batch.sliding_append_slots[-1], 146 * 128 + 6)

    def test_one_token_continuation_remains_explicit_prefill(self) -> None:
        plan = _flat_plan([260], [1])
        prefill, decode = GptOssCudaRunner._attention_batches(plan)

        self.assertIsNone(decode)
        self.assertEqual(len(prefill), 1)
        batch = prefill[0]
        self.assertEqual(batch.q_lengths, (1,))
        self.assertEqual(batch.token_indices, (0,))
        self.assertEqual(batch.full_page_table, (1, 2, 3))
        self.assertEqual(batch.full_seqused_k, (261,))
        self.assertEqual(batch.sliding_page_table, (130, 129))
        self.assertEqual(batch.sliding_seqused_k, (133,))

    def test_decode_tables_and_append_slots_come_from_sparse_groups(self) -> None:
        batch = GptOssCudaRunner._decode_batch(_decode_plan())

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.sequence_indices, (0, 1))
        self.assertEqual(batch.q_lengths, (1, 1))
        self.assertEqual(batch.full_page_table_width, 3)
        self.assertEqual(batch.full_page_table, (1, 2, 1, 17, 18, 19))
        self.assertEqual(batch.full_seqused_k, (131, 261))
        self.assertEqual(batch.sliding_page_table, (129, 130, 146, 145))
        self.assertEqual(batch.sliding_seqused_k, (131, 133))
        self.assertEqual(batch.full_append_slots, (2 * 128 + 2, 19 * 128 + 4))
        self.assertEqual(
            batch.sliding_append_slots,
            (130 * 128 + 2, 145 * 128 + 4),
        )

    def test_frozen_append_destinations_are_a_validation_oracle(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[1].append_block_ids[0] += 1
        with self.assertRaisesRegex(ValueError, "append destinations"):
            GptOssCudaRunner._attention_batches(plan)

    def test_group_order_suffix_and_parent_ownership_fail_closed(self) -> None:
        plan = _prefill_plan()
        plan.cache_groups[0].group_index = 1
        with self.assertRaisesRegex(ValueError, "ordered"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([260], [1])
        plan.cache_groups[1].first_logical_blocks[0] = 0
        with self.assertRaisesRegex(ValueError, "active suffix"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([0], [1])
        plan.cache_groups[1].block_ids[0] = plan.cache_groups[0].block_ids[0]
        plan.cache_groups[1].append_block_ids[0] = plan.cache_groups[0].block_ids[0]
        with self.assertRaisesRegex(ValueError, "cannot be interpreted"):
            GptOssCudaRunner._attention_batches(plan)

        plan = _flat_plan([0], [1])
        with self.assertRaisesRegex(ValueError, "outside"):
            GptOssCudaRunner._attention_batches(plan, parent_capacity=64)

    def test_window_prefill_boundary_is_validated_before_device_work(self) -> None:
        plan = _flat_plan([127], [2])
        with self.assertRaisesRegex(ValueError, "within one logical"):
            GptOssCudaRunner._attention_batches(plan)

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
        source = inspect.getsource(GptOssCudaRunner.__init__)
        self.assertEqual(source.count("torch.zeros("), 1)
        self.assertIn("self.cache_parent_capacity + 1", source)
        self.assertIn("self._cache_parent_storage = torch.zeros(", source)
        self.assertIn(
            "self._window_k_cache = self._history_k_cache",
            source,
        )
        self.assertIn(
            "self._window_v_cache = self._history_v_cache",
            source,
        )


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
