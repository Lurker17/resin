# Adapted from DeepInfer 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01.
# SPDX-License-Identifier: Apache-2.0
# Source: deepinfer/model_executor/layers/tilelang_ops/cuda/flash_mla_utils.py
# Engine adaptation: separate output preserves logical indices; -1/causal guards.
from typing import Any

import tilelang
import tilelang.language as _language

T: Any = _language


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def convert_kv_index_tl(
    s: int,
    k: int,
    block_size: int,
) -> Any:
    b = T.dynamic("b")
    n = T.dynamic("n")

    @T.prim_func
    def _convert_kv_index(
        selected: T.Tensor([b, s, k], T.int32),  # ty: ignore[invalid-type-form]
        context_lens: T.Tensor([b, s], T.int32),  # ty: ignore[invalid-type-form]
        block_tables: T.Tensor([b, n], T.int32),  # ty: ignore[invalid-type-form]
        output: T.Tensor([b, s, k], T.int32),  # ty: ignore[invalid-type-form]
    ):
        with T.Kernel(s, b, threads=128) as (sid, bid):
            T.pdl_sync()
            topk_idxs_fragment = T.alloc_fragment((k,), T.int32)
            T.copy(selected[bid, sid, :], topk_idxs_fragment)
            for k_i in T.Parallel(k):
                x = topk_idxs_fragment[k_i]
                if 0 <= x < context_lens[bid, sid] and x // block_size < n:
                    page = block_tables[bid, x // block_size]
                    topk_idxs_fragment[k_i] = T.if_then_else(
                        page >= 0, page * block_size + x % block_size, -1
                    )
                else:
                    topk_idxs_fragment[k_i] = -1
            T.copy(topk_idxs_fragment, output[bid, sid, :])

    return _convert_kv_index
