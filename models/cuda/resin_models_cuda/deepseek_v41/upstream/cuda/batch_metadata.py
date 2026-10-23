# Adapted from DeepInfer 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01.
# SPDX-License-Identifier: Apache-2.0
# Source: deepinfer/model_executor/layers/tilelang_ops/cuda/prepare_decode_metadata.py
# The per-request preparation and per-ratio stores are retained. The address
# adapter reads Engine parent/child tables instead of DeepInfer's fixed SWA ring.
from typing import Any

import tilelang
import tilelang.language as _language

T: Any = _language

_PASS_CONFIGS: dict[str, Any] = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


@tilelang.jit(
    out_idx=None,
    pass_configs=_PASS_CONFIGS,
)
def prepare_engine_metadata_tl(
    queries: int | None,
    packings: tuple[int, ...],
    window_size: int,
    draft_capacity: int,
    ca_ratios: tuple[int, ...],
    position_dtype: str,
    draft: bool,
    has_valid_mask: bool,
) -> Any:
    # Ordinary query counts vary with every prompt and retained suffix. Only
    # the draft's private attention rectangle needs a static query dimension.
    queries = T.dynamic("queries") if queries is None else queries
    batch = T.dynamic("batch")
    width_swa = T.dynamic("width_swa")
    width_ca = T.dynamic("width_ca")
    width_state = T.dynamic("width_state")
    num_ratios = max(len(ca_ratios), 1)
    padded_draft = (queries + 7) // 8 * 8 if draft else 1
    ratio_items = tuple(enumerate(ca_ratios))

    @T.macro
    def physical_row(parents, slots, first, width, packing, ratio, bid, pos):
        logical = pos // 128 - first[bid]
        column = T.min(T.max(logical, 0), width - 1)
        parent = parents[bid, column]
        page = parent * packing + slots[bid, column]
        return T.if_then_else(
            pos >= 0 and 0 <= logical < width and parent > 0,
            page * (128 // ratio) + (pos % 128) // ratio,
            -1,
        )

    @T.macro
    def write_compressed(
        ci,
        ratio,
        bid,
        sid,
        pos,
        valid,
        parents,
        slots,
        first,
        out_pos,
        out_slot,
        out_len,
    ):
        complete = valid and (pos + 1) % ratio == 0
        out_pos[ci, bid, sid] = T.if_then_else(complete, pos // ratio * ratio, 0)
        out_slot[ci, bid, sid] = T.if_then_else(
            complete,
            physical_row(parents, slots, first, width_ca, packings[1], ratio, bid, pos),
            -1,
        )
        out_len[ci, bid, sid] = T.if_then_else(valid, (pos + 1) // ratio, 0)

    @T.prim_func
    def prepare_engine_metadata(
        positions_in: T.Tensor((batch, queries), position_dtype),  # ty: ignore[invalid-type-form]
        valid_in: T.Tensor((batch * queries,), T.bool),  # ty: ignore[invalid-type-form]
        swa_parents: T.Tensor((batch, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        swa_slots: T.Tensor((batch, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        swa_first: T.Tensor((batch,), T.int32),  # ty: ignore[invalid-type-form]
        ca_parents: T.Tensor((batch, width_ca), T.int32),  # ty: ignore[invalid-type-form]
        ca_slots: T.Tensor((batch, width_ca), T.int32),  # ty: ignore[invalid-type-form]
        ca_first: T.Tensor((batch,), T.int32),  # ty: ignore[invalid-type-form]
        state_parents: T.Tensor((batch, width_state), T.int32),  # ty: ignore[invalid-type-form]
        state_slots: T.Tensor((batch, width_state), T.int32),  # ty: ignore[invalid-type-form]
        state_first: T.Tensor((batch,), T.int32),  # ty: ignore[invalid-type-form]
        positions: T.Tensor((batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        context_lens: T.Tensor((batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        valid_token_mask: T.Tensor((batch * queries,), T.bool),  # ty: ignore[invalid-type-form]
        slot_mapping_store_swa: T.Tensor((batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        state_block_idx: T.Tensor((batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        query_indices: T.Tensor((batch * queries,), T.int32),  # ty: ignore[invalid-type-form]
        positions_c: T.Tensor((num_ratios, batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        slot_mapping_store_ca: T.Tensor((num_ratios, batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        klen: T.Tensor((num_ratios, batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        swa_offset_len: T.Tensor((batch * queries, 2), T.int32),  # ty: ignore[invalid-type-form]
        swa_tables: T.Tensor((batch * queries, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        block_tables: T.Tensor((batch * queries, width_ca), T.int32),  # ty: ignore[invalid-type-form]
        draft_slots: T.Tensor((batch, queries), T.int32),  # ty: ignore[invalid-type-form]
        draft_indices: T.Tensor((batch, queries, padded_draft), T.int32),  # ty: ignore[invalid-type-form]
    ):
        with T.Kernel(batch, threads=128) as bid:
            T.pdl_sync()
            # One scalar predicate per lane also supports long masked prefills;
            # vectorized boolx8 query temporaries have no CUDA representation.
            for query_tile in T.serial(T.ceildiv(queries, 128)):
                for lane in T.Parallel(128):
                    sid = query_tile * 128 + lane
                    if sid < queries:
                        q_off = bid * queries + sid
                        pos = T.cast(positions_in[bid, sid], T.int32)
                        valid = T.bool(True)
                        if has_valid_mask:
                            valid = valid_in[q_off]
                        positions[bid, sid] = pos
                        context_lens[bid, sid] = pos + 1
                        valid_token_mask[q_off] = valid
                        query_indices[q_off] = bid
                        slot_mapping_store_swa[bid, sid] = T.if_then_else(
                            valid,
                            physical_row(
                                swa_parents,
                                swa_slots,
                                swa_first,
                                width_swa,
                                packings[0],
                                1,
                                bid,
                                pos,
                            ),
                            -1,
                        )
                        # Even inactive rows retain their Engine-provided scratch state
                        # page: the C2 kernel requires a mapped page for every query.
                        state_row = physical_row(
                            state_parents,
                            state_slots,
                            state_first,
                            width_state,
                            packings[2],
                            1,
                            bid,
                            pos,
                        )
                        state_block_idx[bid, sid] = T.max(state_row // 128, 0)
                        [
                            write_compressed(
                                ci,
                                ratio,
                                bid,
                                sid,
                                pos,
                                valid,
                                ca_parents,
                                ca_slots,
                                ca_first,
                                positions_c,
                                slot_mapping_store_ca,
                                klen,
                            )
                            for ci, ratio in ratio_items
                        ]

                        # The copied index-expansion kernel resolves this logical range
                        # through a per-query Engine SWA table, including cross-page windows.
                        end = pos + 1
                        if draft:
                            end = T.cast(positions_in[bid, 0], T.int32)
                        length = T.if_then_else(
                            valid, T.min(T.max(end, 0), window_size), 0
                        )
                        start = end - length
                        offset = q_off * width_swa * 128 + start - swa_first[bid] * 128
                        swa_offset_len[q_off, 0] = T.if_then_else(length > 0, offset, 0)
                        swa_offset_len[q_off, 1] = length
                        draft_slots[bid, sid] = T.if_then_else(
                            draft and valid, bid * draft_capacity + sid, -1
                        )

            # Decode copies its few rows in this CTA. Long prefills expand the
            # same tables in a separate grid, as DeepInfer's copy_block_tables.
            if queries <= 8:
                for tile in T.serial(T.ceildiv(width_ca, 128)):
                    for lane in T.Parallel(128):
                        column = tile * 128 + lane
                        if column < width_ca:
                            page = (
                                ca_parents[bid, column] * packings[1]
                                + ca_slots[bid, column]
                            )
                            for sid in T.serial(queries):
                                block_tables[bid * queries + sid, column] = (
                                    T.if_then_else(
                                        ca_parents[bid, column] > 0, page, -1
                                    )
                                )
                for tile in T.serial(T.ceildiv(width_swa, 128)):
                    for lane in T.Parallel(128):
                        column = tile * 128 + lane
                        if column < width_swa:
                            page = (
                                swa_parents[bid, column] * packings[0]
                                + swa_slots[bid, column]
                            )
                            for sid in T.serial(queries):
                                swa_tables[bid * queries + sid, column] = (
                                    T.if_then_else(
                                        swa_parents[bid, column] > 0, page, -1
                                    )
                                )
            if draft:
                for sid, column in T.Parallel(queries, padded_draft):
                    valid = T.bool(True)
                    key_valid = column < queries
                    if has_valid_mask:
                        valid = valid_in[bid * queries + sid]
                        key_valid = (
                            column < queries
                            and valid_in[bid * queries + T.min(column, queries - 1)]
                        )
                    draft_indices[bid, sid, column] = T.if_then_else(
                        valid and key_valid,
                        bid * draft_capacity + column,
                        -1,
                    )
            else:
                for sid in T.Parallel(queries):
                    draft_indices[bid, sid, 0] = -1

    return prepare_engine_metadata


@tilelang.jit(out_idx=None, pass_configs=_PASS_CONFIGS)
def copy_engine_tables_tl(packings: tuple[int, int]) -> Any:
    """Expand prefill tables without specializing on prompt or history length."""
    rows = T.dynamic("rows")
    batch = T.dynamic("batch")
    width_swa = T.dynamic("width_swa")
    width_ca = T.dynamic("width_ca")

    @T.prim_func
    def copy_engine_tables(
        swa_parents: T.Tensor((batch, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        swa_slots: T.Tensor((batch, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        ca_parents: T.Tensor((batch, width_ca), T.int32),  # ty: ignore[invalid-type-form]
        ca_slots: T.Tensor((batch, width_ca), T.int32),  # ty: ignore[invalid-type-form]
        swa_tables: T.Tensor((rows, width_swa), T.int32),  # ty: ignore[invalid-type-form]
        block_tables: T.Tensor((rows, width_ca), T.int32),  # ty: ignore[invalid-type-form]
    ):
        with T.Kernel(rows, threads=128) as row:
            queries = rows // batch
            bid, sid = row // queries, row % queries
            T.pdl_sync()
            # DeepGEMM selects a request's table by its first flattened query
            # offset; the IDs only identify runs of queries belonging together.
            for tile in T.serial(T.ceildiv(width_ca, 128)):
                for lane in T.Parallel(128):
                    column = tile * 128 + lane
                    if column < width_ca:
                        page = (
                            ca_parents[bid, column] * packings[1]
                            + ca_slots[bid, column]
                        )
                        block_tables[row, column] = T.if_then_else(
                            ca_parents[bid, column] > 0, page, -1
                        )
            for tile in T.serial(T.ceildiv(width_swa, 128)):
                for lane in T.Parallel(128):
                    column = tile * 128 + lane
                    if column < width_swa:
                        page = (
                            swa_parents[bid, column] * packings[0]
                            + swa_slots[bid, column]
                        )
                        swa_tables[bid * queries + sid, column] = T.if_then_else(
                            swa_parents[bid, column] > 0, page, -1
                        )

    return copy_engine_tables
